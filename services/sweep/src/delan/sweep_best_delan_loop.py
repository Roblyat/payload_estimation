from __future__ import annotations

from sweep_base import (
    SVC_DELAN,
    MODELS_DELAN_DIR,
    MODELS_DELAN_DIR_HOST,
    DATASET_NAME,
    RUN_TAG,
    DELAN_MODEL_TYPE,
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
from sweep_helper import compose_exec, run_cmd_allow_fail, banner, safe_tag
from .sweep_delan_helper import hp_suffix_from_preset, read_delan_metrics_safe


def train_delan_seed(
    *,
    npz_in: str,
    K: int,
    dataset_seed: int,
    delan_seed: int,
    hp_preset: str,
    log_file,
) -> dict:
    hp_suffix = hp_suffix_from_preset(hp_preset, DELAN_HP_FLAGS)
    model_short = "struct"
    hp_tag = f"hp_{safe_tag(hp_preset)}"
    delan_tag = f"delan_jax_{model_short}_s{delan_seed}_ep{DELAN_EPOCHS}_{hp_tag}_{hp_suffix}"
    delan_id = f"{DATASET_NAME}__{RUN_TAG}__K{K}__seed{dataset_seed}__{delan_tag}"
    delan_run_dir = f"{MODELS_DELAN_DIR}/{delan_id}"
    ckpt = f"{delan_run_dir}/{delan_id}.jax"

    log_file.write(banner([
        f"DELAN VARIANT: hp_preset={hp_preset} delan_seed={delan_seed}",
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
        f"--hp_preset {hp_preset} "
        f"--epochs {DELAN_EPOCHS} "
        f"--eval_every {DELAN_EVAL_EVERY} "
        f"--log_every {DELAN_LOG_EVERY} "
        f"--early_stop {DELAN_EARLY_STOP} "
        f"--early_stop_patience {DELAN_EARLY_STOP_PATIENCE} "
        f"--early_stop_min_delta {DELAN_EARLY_STOP_MIN_DELTA} "
        f"--early_stop_warmup_evals {DELAN_EARLY_STOP_WARMUP_EVALS} "
        f"{hp_flags}"
        f"--save_path {ckpt}"
    )
    ok, _ = run_cmd_allow_fail(cmd, log_file)

    metrics_json = f"{MODELS_DELAN_DIR_HOST}/{delan_id}/metrics.json"
    metrics_json_container = f"{MODELS_DELAN_DIR}/{delan_id}/metrics.json"
    metrics = read_delan_metrics_safe(metrics_json)
    diverged = (not ok) or (not metrics.get("exists")) or (not metrics.get("finite"))

    return {
        "delan_seed": delan_seed,
        "delan_tag": delan_tag,
        "delan_id": delan_id,
        "ckpt": ckpt,
        "hp_preset": hp_preset,
        "metrics_json": metrics_json,
        "metrics_json_container": metrics_json_container,
        "val_rmse": metrics.get("val_rmse", float("inf")),
        "val_mse": metrics.get("val_mse", float("inf")),
        "test_rmse": metrics.get("test_rmse", float("inf")),
        "test_mse": metrics.get("test_mse", float("inf")),
        "best_epoch": metrics.get("best_epoch"),
        "metrics_exists": bool(metrics.get("exists")),
        "diverged": bool(diverged),
    }


def run_delan_seeds(
    *,
    npz_in: str,
    K: int,
    dataset_seed: int,
    hp_preset: str,
    log_file,
) -> list[dict]:
    candidates: list[dict] = []
    for delan_seed in DELAN_SEEDS:
        cand = train_delan_seed(
            npz_in=npz_in,
            K=K,
            dataset_seed=dataset_seed,
            delan_seed=delan_seed,
            hp_preset=hp_preset,
            log_file=log_file,
        )
        candidates.append(cand)
    return candidates


def select_best_candidate(candidates: list[dict], log_file):
    non_diverged = [c for c in candidates if not c.get("diverged")]
    if not non_diverged:
        log_file.write("\n" + banner([
            "3) SELECT BEST DELAN",
            "No non-diverged runs found for this fold.",
        ], char="!") + "\n")
        return None
    non_diverged.sort(key=lambda d: d.get("val_rmse", float("inf")))
    best = non_diverged[0]
    log_file.write("\n" + banner([
        "3) SELECT BEST DELAN (by val RMSE)",
        f"best_seed={best['delan_seed']} val_rmse={best['val_rmse']} test_rmse={best['test_rmse']}",
        f"metrics_json={best['metrics_json']}",
    ], char="#") + "\n")
    return best
