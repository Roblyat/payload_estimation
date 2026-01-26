from __future__ import annotations

import json
import os
import shlex


def lstm_train_cmd_patched(npz: str, out_dir: str, model_name: str,
                           epochs: int, batch: int, val_split: float, seed: int,
                           units: int, dropout: float, eps: float, no_plots: bool,
                           early_stop: bool, early_stop_patience: int,
                           early_stop_min_delta: float, early_stop_warmup_evals: int) -> str:
    no_plots_flag = "--no_plots" if no_plots else ""

    py = (
        "import runpy\n"
        "import keras.callbacks as cb\n"
        "_Orig = cb.ModelCheckpoint\n"
        "\n"
        "class PatchedModelCheckpoint(_Orig):\n"
        "    def __init__(self, filepath, *args, **kwargs):\n"
        "        if isinstance(filepath, str) and (not filepath.endswith('.keras')) and (not filepath.endswith('.h5')):\n"
        "            filepath = filepath + '.keras'\n"
        "        super().__init__(filepath, *args, **kwargs)\n"
        "\n"
        "cb.ModelCheckpoint = PatchedModelCheckpoint\n"
        "runpy.run_path('scripts/train_residual_lstm.py', run_name='__main__')\n"
    )

    return (
        f"python3 -c {shlex.quote(py)} "
        f"--npz {npz} "
        f"--out_dir {out_dir} "
        f"--model_name {model_name} "
        f"--epochs {epochs} "
        f"--batch {batch} "
        f"--val_split {val_split} "
        f"--seed {seed} "
        f"--units {units} "
        f"--dropout {dropout} "
        f"--eps {eps} "
        f"--early_stop {str(early_stop)} "
        f"--early_stop_patience {early_stop_patience} "
        f"--early_stop_min_delta {early_stop_min_delta} "
        f"--early_stop_warmup_evals {early_stop_warmup_evals} "
        f"{no_plots_flag}"
    )


def read_lstm_metrics(metrics_json_path: str) -> dict:
    if not os.path.exists(metrics_json_path):
        return {"exists": False}
    try:
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        train = d.get("train", {}) if isinstance(d, dict) else {}
        eval_test = d.get("eval_test", {}) if isinstance(d, dict) else {}
        return {
            "exists": True,
            "epochs_ran": train.get("epochs_ran"),
            "best_epoch": train.get("best_epoch"),
            "best_val_loss": train.get("best_val_loss"),
            "final_train_loss": train.get("final_train_loss"),
            "final_val_loss": train.get("final_val_loss"),
            "rmse_total": eval_test.get("rmse_total"),
            "mse_total": eval_test.get("mse_total"),
            "best_model_path": eval_test.get("best_model_path"),
        }
    except Exception:
        return {"exists": False}
