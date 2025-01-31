# Train (RSL)
#### File python:
```
IsaacLab/source/standalone/workflows/rsl_rl/train.py
```
#### Esempio parametri:
```
--task pos-grace-rough-direct --num_envs 10 --resume True --load_run 2025-01-28_13-18-54 --checkpoint model_0.pt
```

### Spiegazione parametri:
- `--task` : task da eseguire
- `--num_envs` : numero di ambienti in parallelo
- `--headless` : non apre la finestra di rendering della scena
- `--resume`: continua allenamento con checkpoint dato (transfer-learning)
- `--load_run` : nome cartella dove cerca il checkpoint (creata dentro <i>logs/rsl_rl/TASK_NAME/</i>)
- `--checkpoint` : nome checkpoint (file pt) da eseguire che deve essere all'interno della run
---
# Play (RSL)
#### File python:
```
IsaacLab/source/standalone/workflows/rsl_rl/play.py
```
#### Esempio parametri:
```
--task pos-grace-rough-direct --num_envs 10 --load_run 2024-12-04_10-51-04 --checkpoint model_0.pt
```

### Spiegazione parametri:
- `--task` : task da eseguire
- `--num_envs` : numero di ambienti in parallelo
- `--headless` : non apre la finestra di rendering della scena
- `--load_run` : nome cartella dove cerca il checkpoint (creata dentro <i>logs/rsl_rl/TASK_NAME/</i>)
- `--checkpoint` : nome checkpoint (file pt) da eseguire che deve essere all'interno della run
