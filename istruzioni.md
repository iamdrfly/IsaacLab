
# Come Creare un Fork e Aggiornare il Tuo Repository

## Creazione del Fork

1. **Fai Fork**: Vai al repository originale su GitHub e fai clic sul pulsante "Fork" per creare una copia del repository sul tuo account.
  
2. **Clona il tuo Fork**: Esegui il comando per clonare il tuo fork sul tuo computer:
   ```bash
   git clone https://github.com/tuo_utente/tuo_repo.git
   ```

3. **Aggiungi il Remoto Upstream**: Collega il tuo fork al repository originale con il comando:
   ```bash
   git remote add upstream https://github.com/isaac-sim/IsaacLab.git
   ```

4. **Verifica i Remoti**: Controlla dove sta puntando Git:
   ```bash
   git remote -v
   ```

   Dovresti vedere qualcosa di simile:
   ```
   origin  https://github.com/tuo_utente/tuo_repo.git (fetch)
   origin  https://github.com/tuo_utente/tuo_repo.git (push)
   upstream  https://github.com/isaac-sim/IsaacLab.git (fetch)
   upstream  https://github.com/isaac-sim/IsaacLab.git (push)
   ```

   - `origin` deve essere la tua personale copia ottenuta dal fork e deve avere `fetch` e `push`.
   - `upstream` deve puntare al repository originale (quello da cui hai fatto il fork) e deve avere solo `fetch`.

5. **Rimuovi il Push da Upstream (se necessario)**: Se `upstream` ha anche un `push`, rimuovilo con il comando:
   ```bash
   git remote set-url --push upstream NO_PUSH
   ```

## Aggiornamento da IsaacLab

1. **Recupera le Modifiche da IsaacLab**:
   ```bash
   git fetch upstream
   ```

2. **Unisci le Modifiche nel Tuo Branch Principale Locale**:
   ```bash
   git checkout main
   git merge upstream/main
   ```

3. **Torna al Tuo Branch di Lavoro**:
   ```bash
   git checkout nome_del_tuo_branch
   ```

4. **Unisci le Modifiche dal Tuo Branch Principale nel Tuo Branch di Lavoro**:
   ```bash
   git merge main
   ```

## Riepilogo

Seguendo questi passaggi, sarai in grado di mantenere il tuo fork aggiornato e di lavorare sulle tue modifiche senza problemi.


## Delete rami remoti
git push -d origino nome_del_branch_remoto

## Delete rami locali
git branch -d nome_branch_locale
