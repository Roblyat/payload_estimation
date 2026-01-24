#!/usr/bin/env bash
set -euo pipefail

H="${H:-50}"
SEED="${SEED:-4}"

NPZ_IN="${NPZ_IN:-/workspace/shared/data/preprocessed/delan_ur5_dataset/delan_ur5_dataset.npz}"
CKPT="${CKPT:-/workspace/shared/models/delan/delan_ur5_struct_seed${SEED}.jax}"
RES_OUT="${RES_OUT:-/workspace/shared/data/processed/ur5_residual_traj.npz}"
WIN_OUT="${WIN_OUT:-/workspace/shared/data/processed/ur5_lstm_windows_H${H}.npz}"
LSTM_OUTDIR="${LSTM_OUTDIR:-/workspace/shared/models/lstm/residual_lstm_H${H}_scaled}"

echo "== Stage 1: Train DeLaN =="
docker compose exec -T delan_jax bash -lc "
python3 -m deep_lagrangian_networks.train_ur5_jax \
  --npz ${NPZ_IN} -t structured -r 1 -m 1 \
  --save_path ${CKPT}
"

echo "== Stage 1: Export residual trajectories =="
docker compose exec -T delan_jax bash -lc "
python3 -m deep_lagrangian_networks.export_ur5_residuals_jax \
  --npz_in ${NPZ_IN} \
  --ckpt ${CKPT} \
  --out ${RES_OUT}
"

echo "== Stage 2: Build LSTM windows (H=${H}) =="
docker compose exec -T preprocess bash -lc "
python3 scripts/build_lstm_windows.py \
  --in_npz ${RES_OUT} \
  --out_npz ${WIN_OUT} \
  --H ${H}
"

echo "== Stage 2: Train LSTM =="
docker compose exec -T lstm bash -lc "
python3 scripts/train_residual_lstm.py \
  --npz ${WIN_OUT} \
  --out_dir ${LSTM_OUTDIR} \
  --epochs 60 --batch 64
"

echo "== Stage 2: Evaluate + combine =="
docker compose exec -T lstm bash -lc "
python3 scripts/evaluate_and_combine.py \
  --residual_npz ${RES_OUT} \
  --model ${LSTM_OUTDIR}/best.keras \
  --scalers ${LSTM_OUTDIR}/scalers_H${H}.npz \
  --out_dir ${LSTM_OUTDIR}/eval_combined \
  --H ${H} --split test --save_pred_npz
"

echo "Done."
