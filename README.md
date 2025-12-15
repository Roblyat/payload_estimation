# payload_estimation
payload estimation submodule

## Top-level orchestration

- payload_estimation/docker-compose.yml — defines delan, lstm, evaluation services, shared volumes, and networks.
- payload_estimation/.env — shared settings (ports, data/model paths).

### Shared assets

- payload_estimation/shared/data/raw/ — synced robot logs; mount read-only into all services.
- payload_estimation/shared/data/processed/ — feature-engineered/normalized sets produced by delan and reused by lstm.
- payload_estimation/shared/models/delan/ — trained DeLaN weights/checkpoints.
- payload_estimation/shared/models/lstm/ — trained LSTM checkpoints.
- payload_estimation/shared/config/ — YAML/JSON for dataset splits, normalization stats, and loss hyperparams.
- payload_estimation/shared/logs/ — tensorboard/runs; mount per-service.

### DeLaN service

- payload_estimation/services/delan/Dockerfile
- payload_estimation/services/delan/requirements.txt (or environment.yml)
- payload_estimation/services/delan/src/ — training/inference code for Stage 1 (nominal inverse dynamics).
- payload_estimation/services/delan/scripts/ — entrypoints: train_delan.py, export_checkpoint.py.
- payload_estimation/services/delan/tests/ — unit/smoke tests.
- payload_estimation/services/delan/config/ — model sizes, optimizer, data paths.

### LSTM service

- payload_estimation/services/lstm/Dockerfile
- payload_estimation/services/lstm/requirements.txt
- payload_estimation/services/lstm/src/ — sequence residual model (Stage 2) + data loader consuming DeLaN outputs.
- payload_estimation/services/lstm/scripts/ — train_lstm.py, predict_residuals.py.
- payload_estimation/services/lstm/tests/
- payload_estimation/services/lstm/config/

### Evaluation service

- payload_estimation/services/evaluation/Dockerfile
- payload_estimation/services/evaluation/requirements.txt
- payload_estimation/services/evaluation/src/ — evaluation pipeline: load DeLaN + LSTM checkpoints, combine torques, map via Jacobian, compute wrench residuals, generate metrics/plots.
- payload_estimation/services/evaluation/scripts/ — run_eval.py, make_report.py.
- payload_estimation/services/evaluation/tests/
- payload_estimation/services/evaluation/config/

### Utility/ops

- payload_estimation/scripts/ — convenience wrappers for docker compose run delan ..., dataset sync, and cleaning volumes.
- payload_estimation/Makefile — shortcuts for build, train, eval

### Preprocess service

- payload_estimation/services/preprocess/Dockerfile
- payload_estimation/services/preprocess/requirements.txt
Dockerfile, requirements.txt (sklearn, numpy, pandas, pyyaml)
- payload_estimation/services/preprocess/src/ with loaders/transforms; write shared/data/processed/ and shared/config/stats.yaml
- payload_estimation/services/preprocess/scripts/run_preprocess.py
- payload_estimation/services/preprocess/tests/

### Wiring:

- preprocess mounts shared/data/raw → produces shared/data/processed + stats.
- delan mounts shared/data/raw
    - delan consumes processed data/stats
    - writes shared/data/processed and shared/models/delan
- lstm mounts shared/data/processed, shared/models/delan
    - lstm consumes processed data + DeLaN outputs
    - writes shared/models/lstm
- evaluation mounts shared/data/raw (or processed), shared/models/delan, shared/models/lstm, and shared/config for combined evaluation
    - evaluation reads both models + processed data    
    - Logs go to shared/logs.