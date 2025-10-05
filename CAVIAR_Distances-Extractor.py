#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distance_extractor_cpptraj.py

Estrae distanze Cα–Cα da una traiettoria MD usando **cpptraj** (AmberTools),
senza dipendenze Python (pytraj/mdtraj). È memory‑friendly (streaming).

Formato coppie accettato (anche mischiati nello stesso file):
  - "ASP25-GLU216"
  - "dist 2: ALA15-GLY26" (lo script ignora il prefisso "dist 2:")
  - "ALA15 - GLY26" (spazi consentiti)

Numerazione:
  - sequential  : usa l'indice sequenziale Amber (1..N). Se usi etichette in stile CAVIAR,
                  imposta --offset così che label = seq + offset. Esempio: se CAVIAR etichetta
                  ASP25 ma il residuo reale in Amber è 4, allora offset=21 (25=4+21).
  - resSeq      : usa i numeri residuo PDB (richiede topologia PDB coerente). Usa selettori
                  cpptraj "resid <N>@CA".

Output:
  CSV con prima colonna di tempo/frame (da cpptraj) e una colonna per coppia.

Esempio:
  python Distance_extractor_cpptraj.py \
      --top WT_GTP.parm7 --traj WT_GTP.nc \
      --pairs "ASP25-GLU216,ALA15-GLY26" \
      --numbering sequential --offset 21 \
      --stride 5 --out dists_WT_GTP.csv
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import List, Tuple

PAIR_RE = re.compile(r"([A-Z]{3})(\d+)\s*-\s*([A-Z]{3})(\d+)", re.IGNORECASE)
DIST_LINE_RE = re.compile(r"dist\s*\d*\s*:\s*([A-Z]{3}\s*\d+)\s*-\s*([A-Z]{3}\s*\d+)", re.IGNORECASE)

# ------------------------------- parsing coppie ---------------------------------

def _parse_pair_token(token: str) -> List[Tuple[str, int, str, int]]:
    token = token.strip()
    m = PAIR_RE.search(token)
    if m:
        return [(m.group(1).upper(), int(m.group(2)), m.group(3).upper(), int(m.group(4)))]
    # supporta righe tipo: "dist 2: ALA15-GLY26"
    m2 = DIST_LINE_RE.search(token)
    if m2:
        left = m2.group(1).replace(" ", "")
        right = m2.group(2).replace(" ", "")
        mL = PAIR_RE.search(f"{left}-{right}")
        if mL:
            return [(mL.group(1).upper(), int(mL.group(2)), mL.group(3).upper(), int(mL.group(4)))]
    return []

def parse_pairs_str(s: str) -> List[Tuple[str, int, str, int]]:
    out: List[Tuple[str, int, str, int]] = []
    for tok in s.split(','):
        out.extend(_parse_pair_token(tok))
    return out

def read_pairs(pairs: str, pairs_file: str) -> List[Tuple[str, int, str, int]]:
    items: List[Tuple[str, int, str, int]] = []
    if pairs:
        items.extend(parse_pairs_str(pairs))
    if pairs_file:
        with open(pairs_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                items.extend(_parse_pair_token(line))
    # dedup preservando ordine
    seen = set()
    uniq: List[Tuple[str, int, str, int]] = []
    for p in items:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    if not uniq:
        sys.exit("Nessuna coppia valida trovata (es. ASP25-GLU216).")
    return uniq

# ----------------------------- utilità varie ------------------------------------

def which(prog: str) -> str:
    path = shutil.which(prog)
    return path or ""

# ------------------------------- main -------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Distanze Cα–Cα via cpptraj (streaming, no pytraj/mdtraj).")
    ap.add_argument('--top', required=True, help='Topologia (parm7/prmtop/pdb)')
    ap.add_argument('--traj', required=True, help='Traccia MD (nc/dcd/xtc/trr/mdcrd...)')
    ap.add_argument('--pairs-file', help='File testo con coppie tipo ASP25-GLU216')
    ap.add_argument('--pairs', help='Coppie comma-separated, es: "ASP25-GLU216,ALA15-GLY26"')
    ap.add_argument('--numbering', default='sequential', choices=['sequential','resSeq'],
                    help="sequential: indice Amber 1..N con --offset; resSeq: numerazione PDB (richiede PDB)")
    ap.add_argument('--offset', type=int, default=0, help='Offset etichette CAVIAR (solo con sequential)')
    ap.add_argument('--stride', type=int, default=1, help='Leggi 1 frame ogni N (default 1)')
    ap.add_argument('--out', default='distances_ca.csv', help='Output CSV')
    args = ap.parse_args()

    if not which('cpptraj'):
        sys.exit("cpptraj non trovato nel PATH. Esegui: export AMBERHOME=/home/accelrys/ambertools25 && source $AMBERHOME/amber.sh")

    pairs = read_pairs(args.pairs or '', args.pairs_file or '')

    # Prepara selettori e nomi colonne
    ds_names = []
    colnames = []
    cpptraj_cmds = []

    # righe iniziali
    cpptraj_cmds.append(f'parm "{args.top}"')
    # Compatibilità con cpptraj <= 6.x: niente keyword 'stride', usa l'argomento 'offset' (terzo numero)
    if args.stride and args.stride != 1:
        cpptraj_cmds.append(f'trajin "{args.traj}" 1 last {args.stride}')
    else:
        cpptraj_cmds.append(f'trajin "{args.traj}" 1 last')

    for i, (a_name, a_lab, b_name, b_lab) in enumerate(pairs, start=1):
        if args.numbering == 'sequential':
            a_idx = a_lab - args.offset
            b_idx = b_lab - args.offset
            if a_idx < 1 or b_idx < 1:
                sys.exit(f"[ERR] label-offset < 1 per {a_name}{a_lab}-{b_name}{b_lab} (offset={args.offset})")
            selA = f":{a_idx}@CA"
            selB = f":{b_idx}@CA"
        else:  # resSeq
            # Richiede top PDB con resSeq coerenti
            selA = f"resid {a_lab}@CA"
            selB = f"resid {b_lab}@CA"

        ds = f"d{i:04d}"
        ds_names.append(ds)
        colnames.append(f"{a_name}{a_lab}-{b_name}{b_lab}")
        cpptraj_cmds.append(f"distance {ds} {selA} {selB}")

    cpptraj_cmds.append("run")

    # csv temporaneo prodotto da cpptraj (header: Time,d0001,d0002,...)
    tmp_csv = "cpptraj_tmp_dist.csv"
    # scriviamo tutte le serie in un unico file
    joined_ds = " ".join(ds_names)
    cpptraj_cmds.append(f"writedata \"{tmp_csv}\" {joined_ds} delim comma")

    # eseguiamo cpptraj
    with tempfile.NamedTemporaryFile('w', delete=False, prefix='cpptraj_input_', suffix='.in') as tf:
        tf.write("\n".join(cpptraj_cmds) + "\n")
        tmp_in = tf.name

    try:
        proc = subprocess.run(['cpptraj', '-i', tmp_in], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
            sys.exit(f"[cpptraj ERR] returncode {proc.returncode}")

        if not os.path.exists(tmp_csv):
            sys.exit("[ERR] cpptraj non ha prodotto il CSV atteso.")

        # Rinomina intestazioni d**** con nomi umani e salva il CSV finale
        with open(tmp_csv, 'r', newline='') as fin:
            rows = list(csv.reader(fin))
        if not rows:
            sys.exit("[ERR] CSV vuoto.")

        header = rows[0]
        # Mantieni la prima colonna (Time/Frame), rimpiazza le successive
        expected = 1 + len(colnames)
        if len(header) != expected:
            # cpptraj in alcune versioni può includere anche Step; prendiamo le ultime N colonne come serie
            base_cols = len(header) - len(colnames)
            if base_cols < 1:
                sys.exit(f"[ERR] Intestazione inattesa dal CSV di cpptraj: {header}")
            new_header = header[:base_cols] + colnames
        else:
            new_header = [header[0]] + colnames
        rows[0] = new_header

        with open(args.out, 'w', newline='') as fout:
            w = csv.writer(fout)
            w.writerows(rows)

        # pulizia
        os.remove(tmp_csv)
        print(f"[DONE] Salvato CSV: {args.out}")
        print(f"[INFO] Colonne: {', '.join(colnames)}")

    finally:
        try:
            os.remove(tmp_in)
        except Exception:
            pass

if __name__ == "__main__":
    main()
