# CAVIAR
"Collective vAriable bIas-free Automated Ranking". An automated pipeline to select and rank the most informative collective variables from molecular simulations.
CAVIAR is a Python script that identifies, from molecular dynamics trajectories (MD, GaMD, etc.), the inter-residue distances that best discriminate protein conformational states. 
The selected distances are ranked and proposed as collective variables (CVs) for constructing free-energy landscapes—i.e., potential of mean force (PMF) maps.
Originally designed to find the CVs that represent two systems in order to generate FELs that allows the comparison between two different state of a protein (WTvsMUTANT, GDP- vs GDP-bound, Liganded protein vs apo form etc.), it can be used to analyze the one system too. 

Pipeline structure
1) trajectory representation. Measurement of the distance between each C-alpha of the protein with all the other C-alpha with a default sequence separation >5 residues (MDTraj)

2) Loading-based PCA pre-selection of distances. By default the first 200 distances of the top 50 principal components are selected

3) Loading-based PCA pre-selection — Unweighted Principal Component Analysis (PCA) [10.1007/b98835] was performed on the pooled, mean-centered distance matrix. From each of the first 50 PCs, the top 200 features by absolute loading were collected and merged. Duplicate residue pairs were removed, ensuring that each inter–Cα distance appeared only once. No weighted statistics or greedy correlation filter were applied in this version. When needed for scale, the optional randomized PCA route noted above can be enabled.

4) CV screening by time-lagged independent component analysis (tICA) — The reduced feature set was then subjected to time-lagged independent component analysis (tICA) [10.1063/1.4811489] using PyEMMA [10.1021/acs.jctc.5b00743]. The lag time was set to τ = 10 frames, with regularization 10⁻⁸. Distances were ranked by absolute loadings in TIC1 and TIC2. Ranked lists were exported as CSV files, and the top 50 pairs were visualized as heatmaps, highlighting residues involved in the slowest conformational motions.

5) Cluster-aware CV selection — To ensure diversity, a cluster-aware selection strategy was adopted. Residues were grouped into clusters based on mean inter–Cα distances < 8 Å using connected components analysis. Candidate CVs were required to involve different clusters to avoid redundancy. Three final CVs were proposed: the top distance from TIC1, the second-best distance from TIC1, and the top distance from TIC2. The selected set was exported in JSON format for downstream use.

6) Output and stability report — The pipeline output included (i) ranked CSV lists of distance loadings for TIC1 and TIC2, (ii) heatmaps of the top 50 residue pairs per TIC, (iii) a JSON file containing the selected CVs, and (iv) a stability report. Stability is quantified by split-half analysis of the tICA projections: cos² principal angles between subspaces and component-wise cosine similarities were reported.


