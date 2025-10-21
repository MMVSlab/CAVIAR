#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAVIAR 1.0 alpha — pooled PCA → pooled tICA (PyEMMA)
→ heatmap TIC1/TIC2 + CSV
→ CV selection (cluster-aware: TIC1 top, TIC1 second, TIC2 top)
→ stability report (split-half cos² + component-wise cosine)

NOTE: Questa versione riscrive SOLO la parte di stability report,
      senza cambiare altra logica del programma.
"""

import os, re, json, argparse, logging
from datetime import datetime
import numpy as np
# Compat shim for PyEMMA on NumPy >= 1.24 (uses deprecated np.bool)
if not hasattr(np, "bool"):
    np.bool = bool
import mdtraj as md
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from joblib import Parallel, delayed

import pyemma
from sklearn.utils.extmath import randomized_svd

_USE_SEABORN = True
try:
    import seaborn as sns
except Exception:
    _USE_SEABORN = False

DEFAULTS = dict(
    traj_name='gamd.nc',
    top_name='topologia_stripped.prmtop',
    log_name='gamd.log',
    systemA='WT_GTP',
    systemB='WT_GDP',
    N_USE=None,
    temperature_K=300.0,
    lag_frames=10,
    min_seq_separation=5,
    n_pca_components=60,
    top_k_tica=50,
    residue_label_offset=21,
    regularization=1e-8,
    target_total_frames=14000,
    chunk_size=8000,
    n_jobs=None,
    select_mode='TIC12',        # 'TIC12' (TIC1 + TIC2) or 'TIC1x2' (two distances from TIC1)
    enforce_diversity=True,     # cluster-wise diversity constraint
    cluster_cut_A=8.0,          # Distance cutoff (Å) to define residue clusters from average Cα–Cα separations
    split_report_components=200,
    pca_randomized=False,
    vamp_multipliers='0.001,0.25,0.5,0.75,1,1.25,1.5,1.75,2',   # lag multipliers (τ)
)

AA1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G',
       'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
       'THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

# ========= Setup =========
def setup_run_dir(base_dir="runs"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = os.path.join(base_dir, ts); os.makedirs(rd, exist_ok=True)
    return rd

def setup_logging(run_dir):
    lp = os.path.join(run_dir, "run.log")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(lp, mode='w'), logging.StreamHandler()])
    return lp

def count_non_comment_lines(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    n = 0
    with open(path, 'r') as fh:
        for line in fh:
            s = line.strip()
            if (not s) or s.startswith('#'):
                continue
            n += 1
    return n

# ========= Distances =========
def compute_distances_parallel(X_xyz, pairs, n_jobs=None, chunk_size=8000):
    if n_jobs is None:
        n_jobs = max(cpu_count()-1, 1)
    chunks = [pairs[k:k+chunk_size] for k in range(0, len(pairs), chunk_size)]
    def _chunk(chunk):
        i = np.fromiter((p[0] for p in chunk), dtype=int)
        j = np.fromiter((p[1] for p in chunk), dtype=int)
        d = X_xyz[:, i, :] - X_xyz[:, j, :]
        return np.linalg.norm(d, axis=2)
    mats = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_chunk)(ch) for ch in chunks)
    return np.concatenate(mats, axis=1)

# ========= Labels & CSV =========
def resnum_from_label(label):
    return int(re.findall(r'(\d+)$', label)[0])

def residue_labels(top, ca_atoms, offset):
    res1, res3 = {}, {}
    for idx, ai in enumerate(ca_atoms):
        r = top.atom(ai).residue
        r3 = r.name.upper(); r1 = AA1.get(r3, 'X')
        res1[idx] = f"{r1}{r.resSeq + offset}"
        res3[idx] = f"{r3}{r.resSeq + offset}"
    return res1, res3

def save_tic_csv(path, sel_pairs, tic_vec, res3_labels):
    order = np.argsort(np.abs(tic_vec))[::-1]
    with open(path, "w") as fh:
        fh.write("rank,pair,res1,res2,abs_loading\n")
        for k, idx in enumerate(order, 1):
            i, j = sel_pairs[idx]
            fh.write(f"{k},{res3_labels[i]}-{res3_labels[j]},{res3_labels[i]},{res3_labels[j]},{abs(tic_vec[idx]):.6f}\n")

def make_heatmap(tag, tic_vec, sel_pairs, res1_labels, cfg, out_dir):
    order = np.argsort(np.abs(tic_vec))[::-1][:cfg['top_k_tica']]
    top_pairs = [sel_pairs[i] for i in order]
    top_loads = [tic_vec[i] for i in order]
    residues_involved = set()
    for i, j in top_pairs:
        residues_involved.update([res1_labels[i], res1_labels[j]])
    residues_sorted = sorted(residues_involved, key=resnum_from_label)
    r2i = {r: k for k, r in enumerate(residues_sorted)}
    mat = np.full((len(residues_sorted), len(residues_sorted)), np.nan)
    for (i, j), l in zip(top_pairs, top_loads):
        a, b = res1_labels[i], res1_labels[j]
        ia, ib = r2i[a], r2i[b]
        v = abs(l); mat[ia, ib] = v; mat[ib, ia] = v
    plt.figure(figsize=(10,8))
    if _USE_SEABORN:
        sns.heatmap(mat, xticklabels=residues_sorted, yticklabels=residues_sorted,
                    cmap="YlOrRd", square=True, cbar_kws={"label": "Importance (|loading|)"})
    else:
        im = plt.imshow(mat, aspect='equal'); plt.colorbar(im, label="Importance (|loading|)")
        plt.xticks(range(len(residues_sorted)), residues_sorted, rotation=90)
        plt.yticks(range(len(residues_sorted)), residues_sorted)
    plt.title(f"Pooled tICA {tag} — {cfg['systemA']}+{cfg['systemB']}")
    plt.xlabel("Residue"); plt.ylabel("Residue")
    plt.tight_layout()
    out = os.path.join(out_dir, f"pooled_heatmap_{tag}.png")
    plt.savefig(out, dpi=300); plt.close()
    return out

# ========= Cluster-aware helpers =========
def build_residue_components(n_res, pair_means_nm, nm_cut):
    adj = [[] for _ in range(n_res)]
    for (i, j), dnm in pair_means_nm.items():
        if dnm < nm_cut:
            adj[i].append(j); adj[j].append(i)
    comp = [-1]*n_res; cid = 0
    for r in range(n_res):
        if comp[r] != -1: continue
        stack = [r]; comp[r] = cid
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if comp[v] == -1:
                    comp[v] = cid; stack.append(v)
        cid += 1
    return comp

def diverse_enough(p_ref, p_new, comp, cfg):
    if not cfg['enforce_diversity']:
        return True
    if len(set(p_ref) & set(p_new)) > 0:
        return False
    cref = set([comp[p_ref[0]], comp[p_ref[1]]])
    cnew = set([comp[p_new[0]], comp[p_new[1]]])
    return len(cref & cnew) == 0

# ========= Stability helper =========
def _split_half_cosines(Y, ncomp=10):
    """Split-half stability sulle coordinate tICA Y.
    - metà pari vs metà dispari
    - mean cos^2 dalle SVD dei sottospazi
    - component-wise |cos| tra le stesse componenti i
    """
    import numpy as _np
    if Y is None or Y.ndim != 2 or Y.shape[0] < 4:
        return float('nan'), []
    T, K = Y.shape
    ncomp = int(min(max(1, ncomp), K))
    idx = _np.arange(T)
    h1 = idx[::2]
    h2 = idx[1::2] if T > 1 else idx
    A = Y[h1, :ncomp].copy()
    B = Y[h2, :ncomp].copy()
    # center columns
    A -= A.mean(axis=0, keepdims=True)
    B -= B.mean(axis=0, keepdims=True)
    if A.shape[0] < 2 or B.shape[0] < 2:
        return float('nan'), [0.0]*ncomp
    Ua, _, _ = _np.linalg.svd(A, full_matrices=False)
    Ub, _, _ = _np.linalg.svd(B, full_matrices=False)
    r = min(Ua.shape[1], Ub.shape[1], ncomp)
    if r == 0:
        return float('nan'), [0.0]*ncomp
    M = Ua[:, :r].T @ Ub[:, :r]
    s = _np.linalg.svd(M, compute_uv=False)  # cos(theta_i)
    mean_cos2 = float(_np.mean((s**2))) if s.size else float('nan')
    comp_cos = []
    for i in range(ncomp):
        ai = A[:, i]; bi = B[:, i]
        den = _np.linalg.norm(ai) * _np.linalg.norm(bi)
        c = 0.0 if den == 0 else float(abs(ai.dot(bi) / den))
        comp_cos.append(c)
    return mean_cos2, comp_cos

def _write_stability_report(path, frames, nfeat, lag, eigvals, mean_cos2, comp_cos):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"Frames: {frames}, Features selected: {nfeat}\n")
        fh.write(f"Lag: {lag}\n")
        if eigvals is not None and len(eigvals):
            top8 = ", ".join(f"{v:.6f}" for v in list(eigvals[:8]))
            fh.write(f"Eigenvalues (first 8): {top8}\n")
        fh.write(f"Split-half mean cos^2 principal angles: {mean_cos2:.6f}\n")
        if comp_cos:
            series = ", ".join(f"{c:.4f}" for c in comp_cos)
            fh.write("Component-wise |cosine| (half1 vs half2): " + series + "\n")

# ================= CORE =================
def run(cfg, dirA, dirB):
    run_dir = setup_run_dir(); setup_logging(run_dir)
    logging.info(f"Run directory: {run_dir}")

    # ---- Load (tail-aligned) per sistema ----
    def load_system(basedir, tag, stride):
        traj_path = os.path.join(basedir, cfg['traj_name'])
        top_path  = os.path.join(basedir, cfg['top_name'])
        log_path  = os.path.join(basedir, cfg['log_name'])
        if not (os.path.exists(traj_path) and os.path.exists(top_path) and os.path.exists(log_path)):
            raise FileNotFoundError(f"Missing required files in {basedir}")
        n_use_auto = count_non_comment_lines(log_path)
        n_use_eff = int(cfg['N_USE']) if (cfg.get('N_USE') is not None) else n_use_auto

        stride = max(1, int(stride))
        logging.info(f"[{tag}] Streaming load (stride={stride}) ...")
        chunks = []
        for tr in md.iterload(traj_path, top=top_path, stride=stride, chunk=5000):
            tr.xyz = tr.xyz.astype(np.float32, copy=False)
            chunks.append(tr)
        if not chunks:
            raise RuntimeError(f"Nessun frame letto da {traj_path}")
        traj = md.join(chunks)
        n_avail = traj.n_frames

        if n_use_eff > n_avail * stride:
            logging.warning(f"[{tag}] n_use richiesto={n_use_eff} > disponibili post-stride≈{n_avail*stride}; uso n_use={n_avail*stride}")
            n_use_eff = n_avail * stride
        if n_use_eff <= 0:
            raise ValueError(f"[{tag}] n_use non positivo ({n_use_eff})")

        n_use_strided = max(1, n_use_eff // stride)
        tail = traj[-n_use_strided:]
        logging.info(f"[{tag}] frames (STRIDED tail) = {tail.n_frames}  | stride={stride}  | n_use={n_use_eff} (auto={n_use_auto}, override={cfg.get('N_USE')})")
        return dict(traj_full=tail, top=tail.topology, n_use=n_use_eff)

    # ---- Stride pre-load (stima da log) per arrivare ~ target_total_frames (A+B) ----
    nA_full = count_non_comment_lines(os.path.join(dirA, cfg['log_name']))
    nB_full = count_non_comment_lines(os.path.join(dirB, cfg['log_name']))
    tot_full = nA_full + nB_full
    target_tot = max(1, int(cfg['target_total_frames']))
    stride = 1 if tot_full <= target_tot else int(np.ceil(tot_full / target_tot))
    logging.info(f"Stride scelto (pre-load) = {stride}")

    A = load_system(dirA, cfg['systemA'], stride)
    B = load_system(dirB, cfg['systemB'], stride)

    trajA = A['traj_full']
    trajB = B['traj_full']

    # ---- CA atoms & pair list ----
    caA = trajA.topology.select('name CA')
    caB = trajB.topology.select('name CA')
    if len(caA) != len(caB):
        raise ValueError("Number of Cα atoms differs tra A e B")
    n_res = len(caA)
    res1A, res3A = residue_labels(trajA.topology, caA, cfg['residue_label_offset'])

    pairs = [(i, j) for i in range(n_res) for j in range(i+cfg['min_seq_separation'], n_res)]
    logging.info(f"Candidate residue pairs: {len(pairs)}")

    # ---- Pooled Distances ----
    XA = trajA.atom_slice(caA).xyz
    XB = trajB.atom_slice(caB).xyz
    distA = compute_distances_parallel(XA, pairs, n_jobs=cfg['n_jobs'], chunk_size=cfg['chunk_size'])
    distB = compute_distances_parallel(XB, pairs, n_jobs=cfg['n_jobs'], chunk_size=cfg['chunk_size'])
    logging.info(f"distA shape = {distA.shape} | distB shape = {distB.shape}")

    # ---- PCA + prefilter (solo dedup strutturale) ----
    mu_pool = (distA.mean(axis=0) + distB.mean(axis=0)) / 2.0
    X = np.vstack([distA - mu_pool, distB - mu_pool]).astype(np.float32, copy=False)

    if cfg['pca_randomized']:
        U, S, Vt = randomized_svd(X, n_components=min(cfg['n_pca_components'], X.shape[1]))
        V = Vt
    else:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        V = Vt[:cfg['n_pca_components'], :]

    N_PC_USE = min(50, V.shape[0])
    TOP_PER_PC = 200
    candidates = []
    for pc in range(N_PC_USE):
        load = V[pc]
        top_idx = np.argsort(np.abs(load))[-TOP_PER_PC:][::-1]
        candidates.extend(top_idx.tolist())
    seen = set(); candidate_idx = []
    for idx in candidates:
        if idx not in seen:
            seen.add(idx); candidate_idx.append(idx)
    struct_seen = set(); cand_struct = []
    for idx in candidate_idx:
        i, j = pairs[idx]; key = (min(i, j), max(i, j))
        if key not in struct_seen:
            struct_seen.add(key); cand_struct.append(idx)
    final_idx = np.array(cand_struct, dtype=int)
    sel_pairs = [pairs[i] for i in final_idx]
    logging.info(f"[PCA preselezione] candidati={len(candidates)} → unici dopo dedup={len(final_idx)}")

    # ---- tICA con PyEMMA (pooled input come lista [A, B]) ----
    distA_sel = distA[:, final_idx]
    distB_sel = distB[:, final_idx]
    mu_pool_sel = (distA_sel.mean(axis=0) + distB_sel.mean(axis=0)) / 2.0
    XA = distA_sel - mu_pool_sel
    XB = distB_sel - mu_pool_sel

    # --- VAMP-2 cross-validation helper (run-local) ------
    def _vamp2_cv_score(X_list, tau, k=None, eps=1e-10):
        """Cross-validated VAMP-2 (even/odd) su lista di matrici (frames x features)."""
        pairs_loc = []
        for Xloc in X_list:
            Tloc = Xloc.shape[0]
            if Tloc <= tau:
                continue
            X0 = Xloc[:-tau, :]
            Xt = Xloc[tau:, :]
            pairs_loc.append((X0, Xt))
        if not pairs_loc:
            return -np.inf

        def _stack_even_odd(ps, parity):
            X0s, Xts = [], []
            for X0, Xt in ps:
                n = X0.shape[0]
                idx = np.arange(n)
                sel = (idx % 2 == parity)
                X0s.append(X0[sel]); Xts.append(Xt[sel])
            return np.vstack(X0s), np.vstack(Xts)

        def _cov_blocks(X0, Xt):
            n = X0.shape[0]
            C00 = (X0.T @ X0) / max(1, n)
            Ctt = (Xt.T @ Xt) / max(1, n)
            C0t = (X0.T @ Xt) / max(1, n)
            return C00, Ctt, C0t

        def _invsqrt(C):
            w, V = np.linalg.eigh(C + eps*np.eye(C.shape[0]))
            w = np.clip(w, eps, None)
            return (V @ np.diag(1.0/np.sqrt(w)) @ V.T)

        scores = []
        for parity_train in (0, 1):
            X0_tr, Xt_tr = _stack_even_odd(pairs_loc, parity_train)
            X0_va, Xt_va = _stack_even_odd(pairs_loc, 1 - parity_train)

            C00_tr, Ctt_tr, C0t_tr = _cov_blocks(X0_tr, Xt_tr)
            C00_va, Ctt_va, C0t_va = _cov_blocks(X0_va, Xt_va)

            W0 = _invsqrt(C00_tr)
            Wt = _invsqrt(Ctt_tr)

            M_va = W0 @ C0t_va @ Wt
            s = np.linalg.svd(M_va, compute_uv=False)
            if k is not None:
                s = s[:k]
            scores.append(float(np.sum(s**2)))
        return float(np.mean(scores))

    min_len = min(XA.shape[0], XB.shape[0])
    if min_len <= 1:
        raise ValueError("numero di frame insufficiente per tICA/VAMP-2")

    tau0 = int(cfg['lag_frames'])
    max_tau = max(1, min_len // 2)

    def _parse_multipliers_str(s):
        toks = re.split(r'[,_;\s]+', str(s).strip())
        vals = []
        for t in toks:
            if not t:
                continue
            try:
                vals.append(float(t))
            except ValueError:
                pass
        return vals or [0.5, 1.0, 1.5, 2.0]

    mults = _parse_multipliers_str(cfg.get('vamp_multipliers', '0.5,1,2'))
    taus = {int(max(1, min(max_tau, round(tau0 * m)))) for m in mults}
    taus.add(min(max_tau, max(1, tau0)))
    tau_grid = sorted(taus)

    k_vamp = None
    X_list = [XA, XB]
    vamp_scores = {int(t): _vamp2_cv_score(X_list, int(t), k=k_vamp) for t in tau_grid}
    best_tau = int(max(vamp_scores, key=vamp_scores.get))

    with open(os.path.join(run_dir, "vamp2_lag_scan.json"), "w") as fh:
        json.dump({"tau_grid": tau_grid, "scores": vamp_scores, "best_tau": best_tau}, fh, indent=2)

    if min_len <= best_tau:
        raise ValueError("lag selezionata da VAMP-2 troppo grande per i frame disponibili")
    tica_model = pyemma.coordinates.tica([XA, XB], lag=best_tau, var_cutoff=1.0)
    eigvals, eigvecs = tica_model.eigenvalues, tica_model.eigenvectors
    Y_list = tica_model.get_output()
    Y = np.vstack(Y_list)
    np.save(os.path.join(run_dir, "tica_Y.npy"), Y)

    # ---- Heatmap & CSV (TIC1/TIC2) ----
    tic1 = eigvecs[:, 0]
    heatmap1 = make_heatmap("TIC1", tic1, sel_pairs, res1A, cfg, run_dir)
    csv1 = os.path.join(run_dir, "pooled_distances_TIC1.csv"); save_tic_csv(csv1, sel_pairs, tic1, res3A)

    tic2 = None
    if eigvecs.shape[1] >= 2:
        tic2 = eigvecs[:, 1]
        _ = make_heatmap("TIC2", tic2, sel_pairs, res1A, cfg, run_dir)
        csv2 = os.path.join(run_dir, "pooled_distances_TIC2.csv"); save_tic_csv(csv2, sel_pairs, tic2, res3A)

    # ---- Cluster-aware CV selection ----
    meanP_nm = 0.5*(distA.mean(axis=0) + distB.mean(axis=0))
    pair_mean_nm = {pairs[k]: float(meanP_nm[k]) for k in range(len(pairs))}
    nm_cut = cfg['cluster_cut_A']/10.0
    comp = build_residue_components(n_res, pair_mean_nm, nm_cut)

    order1 = np.argsort(np.abs(tic1))[::-1]
    pair_TIC1_top = sel_pairs[order1[0]]

    pair_TIC1_second = None
    for idx in order1[1:]:
        cand = sel_pairs[idx]
        if diverse_enough(pair_TIC1_top, cand, comp, cfg):
            pair_TIC1_second = cand; break
    if pair_TIC1_second is None:
        pair_TIC1_second = sel_pairs[order1[1]]

    pair_TIC2_top = pair_TIC1_second
    if tic2 is not None and cfg['select_mode'].upper() == 'TIC12':
        order2 = np.argsort(np.abs(tic2))[::-1]
        for idx in order2:
            cand = sel_pairs[idx]
            if cand != pair_TIC1_top and diverse_enough(pair_TIC1_top, cand, comp, cfg):
                pair_TIC2_top = cand; break

    cv_meta = dict(TIC1_top=pair_TIC1_top, TIC1_second=pair_TIC1_second, TIC2_top=pair_TIC2_top)
    with open(os.path.join(run_dir, "cv_selected.json"), "w") as fh: json.dump(cv_meta, fh, indent=2)

    # ---- Stability report (formato originale) ----
    mean_cos2, comp_cos = _split_half_cosines(Y, ncomp=min(cfg['split_report_components'], Y.shape[1]))
    stab_path = os.path.join(run_dir, "stability_tica.txt")
    _write_stability_report(
        stab_path,
        frames=int(XA.shape[0] + XB.shape[0]),
        nfeat=len(final_idx),
        lag=cfg['lag_frames'],  # lasciato invariato
        eigvals=eigvals,
        mean_cos2=mean_cos2,
        comp_cos=comp_cos,
    )
    logging.info(f"Stability report written: {stab_path}")

# =============== Single-system mode (stessa pipeline su A) ===============
def run_single(cfg, dirA):
    run_dir = setup_run_dir(); setup_logging(run_dir)
    logging.info(f"Run directory: {run_dir}")

    traj_path = os.path.join(dirA, cfg['traj_name'])
    top_path  = os.path.join(dirA, cfg['top_name'])
    log_path  = os.path.join(dirA, cfg['log_name'])
    if not (os.path.exists(traj_path) and os.path.exists(top_path) and os.path.exists(log_path)):
        raise FileNotFoundError(f"Missing required files in {dirA} (trajectory/topology/gamd.log)")

    n_use_auto = count_non_comment_lines(log_path)
    n_use_eff = int(cfg.get('N_USE') or n_use_auto)
    traj = md.load(traj_path, top=top_path)
    n_avail = traj.n_frames
    if n_use_eff > n_avail:
        logging.warning(f"[{cfg['systemA']}] requested n_use={n_use_eff} > available={n_avail}; using n_use={n_avail}")
        n_use_eff = n_avail
    if n_use_eff <= 0:
        raise ValueError(f"[{cfg['systemA']}] non-positive n_use ({n_use_eff})")
    tail = traj[-n_use_eff:]
    logging.info(f"[{cfg['systemA']}] frames (FULL tail) = {tail.n_frames} | n_use={n_use_eff} (auto={n_use_auto}, override={cfg.get('N_USE')})")

    tot_full = tail.n_frames
    target_tot = max(1, int(cfg['target_total_frames']))
    stride = 1 if tot_full <= target_tot else int(np.ceil(tot_full / target_tot))
    nA = int(np.ceil(tot_full / stride))
    logging.info(f"Chosen stride = {stride} → post-stride frames: {nA} (target={target_tot})")

    trajA = tail[::stride]
    caA = trajA.topology.select('name CA')
    n_res = len(caA)
    res1A, res3A = residue_labels(trajA.topology, caA, cfg['residue_label_offset'])

    pairs = [(i, j) for i in range(n_res) for j in range(i + cfg['min_seq_separation'], n_res)]
    logging.info(f"Candidate residue pairs: {len(pairs)}")

    XA = trajA.atom_slice(caA).xyz
    distA = compute_distances_parallel(XA, pairs, n_jobs=cfg.get('n_jobs'), chunk_size=cfg['chunk_size']).astype(np.float32, copy=False)

    # PCA preselection (identica)
    muA = distA.mean(axis=0)
    XcA = distA - muA
    U, S, Vt = np.linalg.svd(XcA, full_matrices=False)
    V = Vt[:cfg['n_pca_components'], :]
    N_PC_USE = min(50, V.shape[0]); TOP_PER_PC = 200
    candidates = []
    for pc in range(N_PC_USE):
        load = V[pc]
        top_idx = np.argsort(np.abs(load))[-TOP_PER_PC:][::-1]
        candidates.extend(top_idx.tolist())
    seen = set(); candidate_idx = []
    for idx in candidates:
        if idx not in seen:
            seen.add(idx); candidate_idx.append(idx)
    struct_seen = set(); final_idx = []
    for idx in candidate_idx:
        i, j = pairs[idx]; key = (min(i, j), max(i, j))
        if key not in struct_seen:
            struct_seen.add(key); final_idx.append(idx)
    final_idx = np.array(final_idx, dtype=int)
    sel_pairs = [pairs[i] for i in final_idx]
    logging.info(f"[PCA preselection] candidates={len(candidates)} → unique after dedup={len(final_idx)}")

    # tICA (single)
    distA_sel = distA[:, final_idx]
    Xsel = distA_sel - distA_sel.mean(axis=0)
    if Xsel.shape[0] <= cfg['lag_frames']:
        raise ValueError("Lag is too large for the available frames.")

    # Implementazione tICA semplice (identica a quella già usata qui)
    def tica_unweighted_symmetric(X, lag, reg=1e-8):
        X0, Xt = X[:-lag], X[lag:]
        mu = X0.mean(axis=0)
        X0c, Xtc = X0 - mu, Xt - mu
        T0 = X0c.shape[0]
        C0 = (X0c.T @ X0c) / T0
        Ctau = (Xtc.T @ X0c) / T0
        C0r = C0 + reg*np.eye(C0.shape[0])
        M = np.linalg.solve(C0r, Ctau)
        Ms = 0.5*(M + M.T)
        eigvals, eigvecs = np.linalg.eigh(Ms)
        idx = np.argsort(np.abs(eigvals))[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        proj = (X - mu) @ eigvecs
        return eigvals, eigvecs, proj

    eigvals, eigvecs, Y = tica_unweighted_symmetric(Xsel, cfg['lag_frames'], reg=cfg['regularization'])

    # Output accessori (come prima)
    try:
        tic1 = eigvecs[:, 0]
        _ = make_heatmap("TIC1", tic1, sel_pairs, res1A, cfg, run_dir)
        csv1 = os.path.join(run_dir, "distances_TIC1.csv"); save_tic_csv(csv1, sel_pairs, tic1, res3A)
        if eigvecs.shape[1] >= 2:
            tic2 = eigvecs[:, 1]
            _ = make_heatmap("TIC2", tic2, sel_pairs, res1A, cfg, run_dir)
            csv2 = os.path.join(run_dir, "distances_TIC2.csv"); save_tic_csv(csv2, sel_pairs, tic2, res3A)
    except Exception as e:
        logging.warning(f"Plot/CSV generation skipped: {e}")

    # Stability report (formato originale)
    mean_cos2, comp_cos = _split_half_cosines(Y, ncomp=min(cfg['split_report_components'], Y.shape[1]))
    stab_path = os.path.join(run_dir, "stability_tica.txt")
    _write_stability_report(
        stab_path,
        frames=Xsel.shape[0],
        nfeat=len(final_idx),
        lag=cfg['lag_frames'],  # lasciato invariato
        eigvals=eigvals,
        mean_cos2=mean_cos2,
        comp_cos=comp_cos,
    )
    logging.info(f"Stability report written: {stab_path}")
    return run_dir

# ================= CLI =================

def parse_args():
    p = argparse.ArgumentParser(description="CAVIAR 1.0a — PCA → tICA → CV selection (pooled or single-system)")
    p.add_argument('--dirA', default=DEFAULTS['systemA'])
    # In single-system mode, omit both --systemB and --dirB (defaults None)
    p.add_argument('--dirB', default=None)
    p.add_argument('--systemA', default=DEFAULTS['systemA'])
    p.add_argument('--systemB', default=None)
    p.add_argument('--nuse', type=int, default=None)
    p.add_argument('--total-frames', type=int, default=DEFAULTS['target_total_frames'])
    p.add_argument('--tempK', type=float, default=DEFAULTS['temperature_K'])
    p.add_argument('--lag', type=int, default=DEFAULTS['lag_frames'])
    p.add_argument('--minsep', type=int, default=DEFAULTS['min_seq_separation'])
    p.add_argument('--npc', type=int, default=DEFAULTS['n_pca_components'])
    p.add_argument('--topk_tica', type=int, default=DEFAULTS['top_k_tica'])
    p.add_argument('--jobs', type=int, default=-1)
    p.add_argument('--select-mode', choices=['TIC12','TIC1x2'], default=DEFAULTS['select_mode'])
    p.add_argument('--no-diversity', action='store_true')
    p.add_argument('--cluster-cut', type=float, default=DEFAULTS['cluster_cut_A'])
    p.add_argument('--split-report-components', type=int, default=DEFAULTS['split_report_components'])
    p.add_argument('--pca-randomized', action='store_true')
    p.add_argument('--vamp-mults', default=None,
                   help="Moltiplicatori attorno a --lag, es. '0.5,1,2' o '0.75,1,1.25'")
    return p

def main():
    ap = parse_args(); args = ap.parse_args()
    cfg = DEFAULTS.copy()
    if getattr(args, 'vamp_mults', None): cfg['vamp_multipliers'] = args.vamp_mults
    if args.jobs is not None and args.jobs > 0: cfg['n_jobs'] = args.jobs
    if args.nuse is not None:  cfg['N_USE'] = args.nuse
    if args.total_frames is not None: cfg['target_total_frames'] = int(args.total_frames)
    if args.tempK is not None: cfg['temperature_K'] = args.tempK
    if args.lag is not None:   cfg['lag_frames'] = args.lag
    if args.minsep is not None: cfg['min_seq_separation'] = args.minsep
    if args.npc is not None:   cfg['n_pca_components'] = args.npc
    if args.topk_tica is not None: cfg['top_k_tica'] = args.topk_tica
    if args.no_diversity:      cfg['enforce_diversity'] = False
    if args.cluster_cut is not None: cfg['cluster_cut_A'] = args.cluster_cut
    if args.split_report_components is not None: cfg['split_report_components'] = args.split_report_components
    if args.pca_randomized:    cfg['pca_randomized'] = True

    # Decide the mode:
    sysB = (None if args.systemB is None else str(args.systemB).strip())
    dirB = (None if args.dirB is None else str(args.dirB).strip())
    single_mode = (not sysB) and (not dirB)

    if single_mode:
        run_single(cfg, args.dirA)
    else:
        if not sysB or not dirB:
            raise SystemExit("Pooled mode richiesto: specifica sia --systemB sia --dirB (oppure omettili entrambi per la single-system mode).")
        run(cfg, args.dirA, dirB)

if __name__ == "__main__":
    main()
