#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAVIAR 0.2.3 — (no-weights) pooled PCA → pooled tICA (PyEMMA)
→ heatmap TIC1/TIC2 + CSV
→ CV selection (cluster-aware: TIC1 top, TIC1 second, TIC2 top)
→ stability report (split-half cos² + component-wise cosine)
"""

import os, re, json, argparse, logging
from datetime import datetime
import numpy as np
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
    select_mode='TIC12',        # 'TIC12' (TIC1 + TIC2) or 'TIC1x2' (two from TIC1)
    enforce_diversity=True,     # diversity constraint for clusters
    cluster_cut_A=8.0,          # threshold [Å] to define residue clusters based on average Cα–Cα distances
    split_report_components=200,
    pca_randomized=False
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
    import numpy as _np
    T, K = Y.shape
    ncomp = int(min(ncomp, K))
    idx = _np.arange(T)
    h1 = idx[::2]
    h2 = idx[1::2] if T > 1 else idx
    A = Y[h1, :ncomp].copy()
    B = Y[h2, :ncomp].copy()
    if A.shape[0] < 2 or B.shape[0] < 2:
        return 0.0, [0.0]*ncomp
    A -= A.mean(axis=0, keepdims=True)
    B -= B.mean(axis=0, keepdims=True)
    Ua, _, _ = _np.linalg.svd(A, full_matrices=False)
    Ub, _, _ = _np.linalg.svd(B, full_matrices=False)
    r = min(Ua.shape[1], Ub.shape[1], ncomp)
    if r == 0:
        return 0.0, [0.0]*ncomp
    M = Ua[:, :r].T @ Ub[:, :r]
    s = _np.linalg.svd(M, compute_uv=False)  # cos(theta_i)
    mean_cos2 = float(_np.mean((s**2))) if s.size else 0.0
    comp_cos = []
    for i in range(ncomp):
        ai = A[:, i]; bi = B[:, i]
        den = _np.linalg.norm(ai) * _np.linalg.norm(bi)
        c = 0.0 if den == 0 else float(abs(ai.dot(bi) / den))
        comp_cos.append(c)
    return mean_cos2, comp_cos

# ================= CORE =================
def run(cfg, dirA, dirB):
    run_dir = setup_run_dir(); setup_logging(run_dir)
    logging.info(f"Run dir: {run_dir}")

    # ---- Load (tail-aligned) per system. ----
    
    def load_system(basedir, tag, stride):
        traj_path = os.path.join(basedir, cfg['traj_name'])
        top_path  = os.path.join(basedir, cfg['top_name'])
        log_path  = os.path.join(basedir, cfg['log_name'])
        if not (os.path.exists(traj_path) and os.path.exists(top_path) and os.path.exists(log_path)):
            raise FileNotFoundError(f"File mancanti in {basedir}")
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


    
    # ---- Pre-load stride (estimated from the log) to reach ~target_total_frames (A+B) ----
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
        raise ValueError("Mismatch numero di Cα tra A e B")
    n_res = len(caA)
    res1A, res3A = residue_labels(trajA.topology, caA, cfg['residue_label_offset'])

    pairs = [(i, j) for i in range(n_res) for j in range(i+cfg['min_seq_separation'], n_res)]
    logging.info(f"coppie candidate: {len(pairs)}")

    # ---- pooled Distances ----
    XA = trajA.atom_slice(caA).xyz
    XB = trajB.atom_slice(caB).xyz
    distA = compute_distances_parallel(XA, pairs, n_jobs=cfg['n_jobs'], chunk_size=cfg['chunk_size'])
    distB = compute_distances_parallel(XB, pairs, n_jobs=cfg['n_jobs'], chunk_size=cfg['chunk_size'])
    logging.info(f"distA shape = {distA.shape} | distB shape = {distB.shape}")

    # ---- PCA + preselection (structural dedup only) ----
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

    # ---- tICA with PyEMMA ----
    distA_sel = distA[:, final_idx]
    distB_sel = distB[:, final_idx]
    mu_pool_sel = (distA_sel.mean(axis=0) + distB_sel.mean(axis=0)) / 2.0
    Xsel_pool = np.vstack([distA_sel - mu_pool_sel, distB_sel - mu_pool_sel])

    if Xsel_pool.shape[0] <= cfg['lag_frames']:
        raise ValueError("lag troppo grande per i frame analisi pooled")

    tica_model = pyemma.coordinates.tica([Xsel_pool], lag=cfg['lag_frames'], var_cutoff=1.0)
    eigvals, eigvecs = tica_model.eigenvalues, tica_model.eigenvectors
    Y = tica_model.get_output()[0]
    np.save(os.path.join(run_dir, "tica_Y.npy"), Y)

    # ---- Heatmap & CSV (TIC1/TIC2) ----
    tic1 = eigvecs[:, 0]
    heatmap1 = make_heatmap("TIC1", tic1, sel_pairs, res1A, cfg, run_dir)
    csv1 = os.path.join(run_dir, "pooled_distances_TIC1.csv"); save_tic_csv(csv1, sel_pairs, tic1, res3A)

    tic2 = None; heatmap2 = None; csv2 = None
    if eigvecs.shape[1] >= 2:
        tic2 = eigvecs[:, 1]
        heatmap2 = make_heatmap("TIC2", tic2, sel_pairs, res1A, cfg, run_dir)
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

    # ---- Stability report ----
    stab_path = os.path.join(run_dir, "stability_tica.txt")
    with open(stab_path, "w", encoding="utf-8") as fh:
        fh.write(f"Frames pooled: {Xsel_pool.shape[0]}, Features selected: {len(final_idx)}\n")
        fh.write(f"Lag: {cfg['lag_frames']}\n")
        fh.write("Eigenvalues (first 8): " + ", ".join(f"{v:.6f}" for v in eigvals[:8]) + "\n")
        try:
            mean_cos2, comp_cos = _split_half_cosines(Y, ncomp=min(cfg['split_report_components'], Y.shape[1]))
            fh.write(f"Split-half mean cos^2 principal angles: {mean_cos2:.6f}\n")
            fh.write("Component-wise |cosine| (half1 vs half2): " + ", ".join(f"{c:.4f}" for c in comp_cos) + "\n")
        except Exception as e:
            fh.write(f"[WARN] Split-half stability skipped: {type(e).__name__}: {e}\n")
    logging.info(f"Stability report scritto: {stab_path}")

# ================= CLI =================
def parse_args():
    p = argparse.ArgumentParser(description="CAVIAR 0.2.1 — PCA → tICA (PyEMMA) → heatmap + CV selection (no FEL)")
    p.add_argument('--dirA', default=DEFAULTS['systemA'])
    p.add_argument('--dirB', default=DEFAULTS['systemB'])
    p.add_argument('--systemA', default=DEFAULTS['systemA'])
    p.add_argument('--systemB', default=DEFAULTS['systemB'])
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
    return p

def main():
    ap = parse_args(); args = ap.parse_args()
    cfg = DEFAULTS.copy()
    if args.jobs is not None and args.jobs > 0:
        cfg['n_jobs'] = args.jobs
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
    run(cfg, args.dirA, args.dirB)

if __name__ == "__main__":
    main()
