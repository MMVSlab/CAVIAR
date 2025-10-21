# CAVIAR 1.0 alpha — README

**C**ollective **v**Ariable b**I**as‑free **A**utomated **R**anking (CAVIAR) is a pipeline to discover and rank distance‑based collective variables (CVs) from MD simulations using PCA pre‑selection and tICA. It works in **pooled mode** (compare two systems A vs B) or in **single‑system mode** (system A only).

> This README describes inputs, options, outputs, and the internal workflow for the current `CAVIAR-1.0_alpha.py`.

---

## 1) Inputs & Folder Layout

CAVIAR expects each system directory to contain **three files** with fixed names (defaults):

- Trajectory: `gamd.nc`
- Topology: `topologia_stripped.prmtop`
- GaMD log (for frame counting/metadata): `gamd.log`

You provide the base folders with `--dirA` (and optionally `--dirB`).

> **Note:** In the current alpha, trajectory/topology file names are **not** passed via CLI; the script reads the files above from `--dirA`/`--dirB`.

---

## 2) Modes of Operation

### Single‑system mode (A only)
Omit both `--systemB` **and** `--dirB`:

```bash
python CAVIAR-1.0_alpha.py   --dirA WT_GTP --systemA WT_GTP
```

- Loads `WT_GTP/gamd.nc`, `WT_GTP/topologia_stripped.prmtop`, `WT_GTP/gamd.log`.
- Performs PCA pre‑selection, then **tICA on A only** using the specified `--lag`.
- Writes the stability report (split‑half cos² and component‑wise cosines), heatmaps, CSVs, and `tica_Y.npy`.

### Pooled mode (A vs B)
Specify **both** `--dirB` and `--systemB`:

```bash
python CAVIAR-1.0_alpha.py   --dirA WT_GTP --systemA WT_GTP   --dirB WT_GDP --systemB WT_GDP
```

- Loads both systems, aligns sample sizes with an adaptive stride, and **pools** them.
- PCA pre‑selection on pooled centered distances.
- **tICA via PyEMMA** with **VAMP‑2 lag selection** over a grid of lag multipliers.
- Writes stability report, pooled heatmaps/CSVs, `tica_Y.npy`, and the VAMP‑2 scan JSON.

> If you pass one of `--dirB` or `--systemB` but not both, the script exits with an error.

---

## 3) Command‑line Options

```
usage: CAVIAR-1.0_alpha.py [-h] --dirA DIR [--systemA NAME]
                           [--dirB DIR] [--systemB NAME]
                           [--nuse INT] [--total-frames INT]
                           [--tempK FLOAT] [--lag INT]
                           [--minsep INT] [--npc INT]
                           [--topk_tica INT] [--jobs INT]
                           [--select-mode {TIC12,TIC1x2}] [--no-diversity]
                           [--cluster-cut FLOAT]
                           [--split-report-components INT]
                           [--pca-randomized]
                           [--vamp-mults STR]
```

**Core**
- `--dirA` (**required**): Folder for system A.
- `--systemA` (default: `WT_GTP`): Label for system A (used in logs/plots).
- `--dirB` (omit for single‑system): Folder for system B.
- `--systemB` (omit for single‑system): Label for system B.

**Data & sampling**
- `--nuse INT` (default: auto from `gamd.log`): Number of frames to use (tail). If omitted, CAVIAR infers it from non‑comment lines in `gamd.log`.
- `--total-frames INT` (default: 14000): Target total frames (A (+B)) after stride. The script picks an integer stride to approach this target.
- `--lag INT` (default: 10): tICA lag (frames). In pooled mode, this is the **center** of the lag grid; the final lag is selected by VAMP‑2.
- `--vamp-mults STR` (pooled only): Comma/space/semicolon‑separated multipliers for the lag grid (e.g., "0.5,1,1.5,2").

**Feature construction**
- `--minsep INT` (default: 5): Minimum sequence separation for Cα–Cα pairs.
- `--npc INT` (default: 60): Number of PCA components considered in pre‑selection.
- `--pca-randomized` (flag): Use randomized SVD for PCA pre‑selection.

**tICA ranking and CV selection**
- `--topk_tica INT` (default: 50): How many top |loadings| to visualize/consider.
- `--select-mode {TIC12,TIC1x2}` (default: `TIC12`):
  - `TIC12`: pick CVs from TIC1 and TIC2 (with a diversity constraint).
  - `TIC1x2`: pick two CVs from TIC1 respecting diversity.
- `--no-diversity` (flag): Disable cluster‑aware diversity when picking CVs.
- `--cluster-cut FLOAT` (default: 8.0 Å): Contact cutoff (mean Cα–Cα distance in Å) to build residue clusters used for diversity.

**Stability report**
- `--split-report-components INT` (default: 200): How many component‑wise |cosines| to print.

**Performance**
- `--jobs INT` (default: auto): Parallelism for distance computation; negative or 0 → auto.
- `--tempK FLOAT` (default: 300.0): Stored for metadata; not used by tICA.

---

## 4) Outputs (per run)
CAVIAR creates a new folder under `runs/` named with a timestamp, e.g., `runs/20251021_143805/`, containing:

- **Logs**
  - `run.log`: detailed log of the run.

- **tICA outputs**
  - `tica_Y.npy`: tICA coordinates (stacked A+B in pooled; A only in single).
  - `stability_tica.txt`: text report with:
    - `Frames: …, Features selected: …`
    - `Lag: …`
    - `Eigenvalues (first 8): …`
    - `Split‑half mean cos^2 principal angles: …`
    - `Component‑wise |cosine| (half1 vs half2): c1, c2, …`

- **Lag selection (pooled)**
  - `vamp2_lag_scan.json`: scanned lag grid, VAMP‑2 scores, and selected best lag.

- **CV ranking/visualization**
  - `pooled_heatmap_TIC1.png`, `pooled_heatmap_TIC2.png` *(names may include "pooled" even in single mode; cosmetic)*
  - `pooled_distances_TIC1.csv`, `pooled_distances_TIC2.csv` *(or `distances_TIC*.csv` in single mode)*
  - `cv_selected.json`: chosen CV pairs (TIC1_top, TIC1_second, TIC2_top).

> Heatmaps display |loading| of the top TICs across involved residue pairs; CSVs list ranked pairs with residue labels.

---

## 5) Internal Workflow (high level)

1. **Frame budgeting & stride**
   - Estimate usable frames from `gamd.log` non‑comment lines.
   - Choose a stride so total frames (A(+B)) ≈ `--total-frames`.

2. **Distance features**
   - Build Cα–Cα pairs with `|i−j| ≥ --minsep`.
   - Compute per‑frame distances in parallel; center by pooled (or single) mean.

3. **PCA pre‑selection**
   - SVD/PCA on the centered matrix; for the first PCs, keep pairs with the largest |loadings|; deduplicate by residue pair.

4. **tICA**
   - **Pooled**: PyEMMA tICA on `[XA, XB]` with **VAMP‑2 lag selection** from the multiplier grid around `--lag`.
   - **Single**: internal symmetric tICA using the specified `--lag`.

5. **Stability**
   - Split even/odd frames; compute **principal angles** between subspaces and report **mean cos²**.
   - Also print **component‑wise |cosine|** for the first `--split-report-components` coordinates.

6. **CV selection (cluster‑aware)**
   - Build residue clusters from average distances (cutoff `--cluster-cut` Å).
   - Choose diverse top pairs from TIC1 (and TIC2 if `TIC12`).

---

## 6) Tips & Troubleshooting

- **Single vs pooled:** If you omit one of `--dirB`/`--systemB` but not both, the script stops with an explicit error. Omit **both** for single mode.
- **Lag choice:** In pooled mode, the effective lag is the VAMP‑2 winner (`vamp2_lag_scan.json`). In single mode it’s exactly `--lag`.
- **Odd number of frames:** Stability split handles odd T by trimming one frame to equalize halves.
- **NumPy warning (`np.bool`)**: harmless; a compatibility shim maps `np.bool → bool`.
- **Performance:** Increase `--jobs` (or leave auto), and consider enabling `--pca-randomized` for very large feature spaces.
- **File names:** Some plot/CSV names may contain the word `pooled` even in single mode; this is cosmetic and does not affect content.

---

## 7) Reproducibility

- The run directory includes JSON artifacts (`cv_selected.json`, `vamp2_lag_scan.json`) and `tica_Y.npy` to facilitate downstream analyses and comparisons.
- Keep the same `--lag`/`--vamp-mults`, stride (implicitly derived), and selection parameters to obtain comparable results across runs.

---

## 8) License & Citation

- **License:** Apache 2.0
- **Please cite:** CAVIAR_1.0alpha "Collective vAriable bIas-free Automated Ranking"
