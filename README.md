# CAVIAR
"Collective vAriable bIas-free Automated Ranking". An automated pipeline to select and rank the most informative collective variables from molecular simulations.
CAVIAR is a Python script that identifies, from molecular dynamics trajectories (MD, GaMD, etc.), the inter-residue distances that best discriminate protein conformational states. 
The selected distances are ranked and proposed as collective variables (CVs) for constructing free-energy landscapes—i.e., potential of mean force (PMF) maps.
Originally designed to find the CVs that represent two systems in order to generate FELs that allows the comparison between two different state of a protein (WTvsMUTANT, GDP- vs GDP-bound, Liganded protein vs apo form etc.), it can be used to analyze the one system too. 

## Pubblicazione su GitHub

Per condividere le modifiche locali (incluse quelle introdotte da `CAVIAR_0.2.3.py` quando `--dirB` è omesso) su un repository GitHub, procedere così:

1. **Verificare le modifiche locali**
   ```bash
   git status
   ```
   Se il file modificato (`CAVIAR_0.2.3.py`) risulta "modified", procedere con lo staging.

2. **Aggiungere i file al commit**
   ```bash
   git add CAVIAR_0.2.3.py
   ```
   È possibile usare `git add .` per includere tutte le modifiche desiderate.

3. **Creare il commit locale**
   ```bash
   git commit -m "Descrivere in breve la modifica"
   ```
   Confermare il commit con `git log --oneline` per assicurarsi che sia presente nella cronologia.

4. **Configurare (o aggiornare) il repository remoto**
   - Se non esiste un remoto: `git remote add origin https://github.com/<utente>/<repo>.git`
   - Per modificare l'URL di un remoto esistente: `git remote set-url origin https://github.com/<utente>/<repo>.git`
   Verificare il risultato con `git remote -v`.

5. **Pubblicare il branch corrente su GitHub**
   ```bash
   git push -u origin <nome-branch>
   ```
   Sostituire `<nome-branch>` con il branch attivo (`main`, `master`, o un branch dedicato). L'opzione `-u` memorizza l'associazione per i push futuri.

6. **Controllare su GitHub**
   Visitare il repository su github.com per verificare che `CAVIAR_0.2.3.py` sia aggiornato. Da lì si può aprire una Pull Request se necessario.

Ricordarsi di utilizzare un [token personale](https://docs.github.com/it/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) se l'autenticazione tramite password non è disponibile (ad esempio da riga di comando su HTTPS).

## Applicare una patch con `git apply`

Se qualcuno ti fornisce un *diff* (ad esempio in formato `.patch` o `.diff`) puoi applicarlo direttamente al repository locale con `git apply`. Non è necessario creare file speciali oltre alla patch stessa. Il flusso tipico è il seguente:

1. **Salvare la patch su disco**
   - Se il diff è stato condiviso come testo, copialo in un file, ad esempio `modifiche.patch`, usando un editor di testo o un redirect da shell:
     ```bash
     cat > modifiche.patch <<'EOF'
     # incolla qui il contenuto della patch
     EOF
     ```
   - In alternativa, scarica direttamente il file `.patch` se è stato allegato da GitHub o inviato via email.

2. **Verificare cosa farà la patch** (opzionale ma consigliato)
   ```bash
   git apply --stat modifiche.patch
   git apply --check modifiche.patch
   ```
   `--stat` mostra un riepilogo delle modifiche, mentre `--check` conferma che la patch può essere applicata senza conflitti.

3. **Applicare la patch**
   ```bash
   git apply modifiche.patch
   ```
   Dopo il comando, i file risultano modificati nella working tree (non ancora committati).

4. **Verificare e proseguire normalmente**
   ```bash
   git status
   ```
   A questo punto puoi ispezionare i file, eseguire test e creare un commit con `git add` + `git commit`.

Se la patch non si applica perché il tuo branch è troppo diverso, sincronizza prima il repository (`git pull` o `git fetch` + `git merge/rebase`) o chiedi una patch aggiornata. Per annullare un'applicazione non voluta, usa `git reset --hard` (attenzione: elimina le modifiche locali non committate).

Pipeline structure
1) trajectory representation. Measurement of the distance between each C-alpha of the protein with all the other C-alpha with a default sequence separation >5 residues (MDTraj)

2) Loading-based PCA pre-selection of distances. By default the first 200 distances of the top 50 principal components are selected

3) Loading-based PCA pre-selection — Unweighted Principal Component Analysis (PCA) [10.1007/b98835] was performed on the pooled, mean-centered distance matrix. From each of the first 50 PCs, the top 200 features by absolute loading were collected and merged. Duplicate residue pairs were removed, ensuring that each inter–Cα distance appeared only once. No weighted statistics or greedy correlation filter were applied in this version. When needed for scale, the optional randomized PCA route noted above can be enabled.

4) CV screening by time-lagged independent component analysis (tICA) — The reduced feature set was then subjected to time-lagged independent component analysis (tICA) [10.1063/1.4811489] using PyEMMA [10.1021/acs.jctc.5b00743]. The lag time was set to τ = 10 frames, with regularization 10⁻⁸. Distances were ranked by absolute loadings in TIC1 and TIC2. Ranked lists were exported as CSV files, and the top 50 pairs were visualized as heatmaps, highlighting residues involved in the slowest conformational motions.

5) Cluster-aware CV selection — To ensure diversity, a cluster-aware selection strategy was adopted. Residues were grouped into clusters based on mean inter–Cα distances < 8 Å using connected components analysis. Candidate CVs were required to involve different clusters to avoid redundancy. Three final CVs were proposed: the top distance from TIC1, the second-best distance from TIC1, and the top distance from TIC2. The selected set was exported in JSON format for downstream use.

6) Output and stability report — The pipeline output included (i) ranked CSV lists of distance loadings for TIC1 and TIC2, (ii) heatmaps of the top 50 residue pairs per TIC, (iii) a JSON file containing the selected CVs, and (iv) a stability report. Stability is quantified by split-half analysis of the tICA projections: cos² principal angles between subspaces and component-wise cosine similarities were reported.


