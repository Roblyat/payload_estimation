from __future__ import annotations

from sweep_base import (
    SVC_DELAN,
    MODELS_DELAN_DIR,
    MODELS_DELAN_DIR_HOST,
    DATASET_NAME,
    RUN_TAG,
    DELAN_MODEL_TYPE,
    DELAN_HP_PRESET,
    DELAN_HP_FLAGS,
    DELAN_EPOCHS,
    DELAN_SEEDS,
    DELAN_EVAL_EVERY,
    DELAN_LOG_EVERY,
    DELAN_EARLY_STOP,
    DELAN_EARLY_STOP_PATIENCE,
    DELAN_EARLY_STOP_MIN_DELTA,
    DELAN_EARLY_STOP_WARMUP_EVALS,
    SCRIPT_TRAIN_DELAN_JAX,
)
from sweep_helper import compose_exec, run_cmd, banner
from .sweep_delan_helper import hp_suffix_from_preset, read_delan_rmse_pair


def comp_delan(*, npz_in: str, K: int, seed: int, log_file):
    hp_suffix = hp_suffix_from_preset(DELAN_HP_PRESET, DELAN_HP_FLAGS)
    delan_candidates = []
    for delan_seed in DELAN_SEEDS:
        model_short = "struct"
        delan_tag = f"delan_jax_{model_short}_s{delan_seed}_ep{DELAN_EPOCHS}_{hp_suffix}"
        delan_id = f"{DATASET_NAME}__{RUN_TAG}__K{K}__seed{seed}__{delan_tag}"
        delan_run_dir = f"{MODELS_DELAN_DIR}/{delan_id}"
        ckpt = f"{delan_run_dir}/{delan_id}.jax"

        log_file.write(banner([
            f"DELAN VARIANT: delan_seed={delan_seed}",
            f"ckpt={ckpt}",
        ], char="#") + "\n")

        log_file.write("\n" + banner(["2) DELAN TRAIN"], char="#") + "\n")
        hp_flags = (DELAN_HP_FLAGS + " ") if DELAN_HP_FLAGS.strip() else ""
        cmd = compose_exec(
            SVC_DELAN,
            f"python3 {SCRIPT_TRAIN_DELAN_JAX} "
            f"--npz {npz_in} "
            f"-t {DELAN_MODEL_TYPE} "
            f"-s {delan_seed} "
            f"-r 0 "
            f"--hp_preset {DELAN_HP_PRESET} "
            f"--epochs {DELAN_EPOCHS} "
            f"--eval_every {DELAN_EVAL_EVERY} "
            f"--log_every {DELAN_LOG_EVERY} "
            f"{'--early_stop ' if DELAN_EARLY_STOP else ''}"
            f"--early_stop_patience {DELAN_EARLY_STOP_PATIENCE} "
            f"--early_stop_min_delta {DELAN_EARLY_STOP_MIN_DELTA} "
            f"--early_stop_warmup_evals {DELAN_EARLY_STOP_WARMUP_EVALS} "
            f"{hp_flags}"
            f"--save_path {ckpt}"
        )
        run_cmd(cmd, log_file)

        metrics_json = f"{MODELS_DELAN_DIR_HOST}/{delan_id}/metrics.json"
        metrics_json_container = f"{MODELS_DELAN_DIR}/{delan_id}/metrics.json"
        val_rmse, test_rmse = read_delan_rmse_pair(metrics_json)
        delan_candidates.append({
            "delan_seed": delan_seed,
            "delan_tag": delan_tag,
            "delan_id": delan_id,
            "ckpt": ckpt,
            "val_rmse": val_rmse,
            "test_rmse": test_rmse,
            "metrics_json": metrics_json,
            "metrics_json_container": metrics_json_container,
        })

    return delan_candidates


def select_best(delan_candidates: list[dict], log_file):
    delan_candidates.sort(key=lambda d: d["val_rmse"])
    best = delan_candidates[0]
    log_file.write("\n" + banner([
        "3) SELECT BEST DELAN (by val RMSE)",
        f"best_seed={best['delan_seed']} val_rmse={best['val_rmse']} test_rmse={best['test_rmse']}",
        f"metrics_json={best['metrics_json']}",
    ], char="#") + "\n")
    return best


# Backwards-friendly aliases
def compDelan(*args, **kwargs):
    return comp_delan(*args, **kwargs)


def selectBest(*args, **kwargs):
    return select_best(*args, **kwargs)
