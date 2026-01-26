from __future__ import annotations

import json
import os
import shlex
import shutil
from pathlib import Path

from sweep_base import MODELS_DELAN_DIR_HOST, CLEANUP_NON_BEST_PLOTS


def fmt_hp(x: float) -> str:
    if x == 0:
        return "0"
    ax = abs(x)
    if ax < 1e-2 or ax >= 1e2:
        s = f"{x:.0e}"
    else:
        s = f"{x:g}"
    s = s.replace("+", "")
    s = s.replace("e-0", "e-").replace("e+0", "e")
    return s.replace(".", "p")


def parse_hp_flags(flag_str: str) -> dict:
    if not flag_str.strip():
        return {}
    toks = shlex.split(flag_str)
    out = {}
    for i, t in enumerate(toks):
        if t == "--n_width" and i + 1 < len(toks):
            out["n_width"] = int(toks[i + 1])
        if t == "--n_depth" and i + 1 < len(toks):
            out["n_depth"] = int(toks[i + 1])
        if t == "--batch" and i + 1 < len(toks):
            out["n_minibatch"] = int(toks[i + 1])
        if t == "--lr" and i + 1 < len(toks):
            out["learning_rate"] = float(toks[i + 1])
        if t == "--wd" and i + 1 < len(toks):
            out["weight_decay"] = float(toks[i + 1])
        if t == "--activation" and i + 1 < len(toks):
            out["activation"] = str(toks[i + 1])
    return out


def hp_suffix_from_preset(preset: str, hp_flags: str) -> str:
    base = {
        "n_width": 64,
        "n_depth": 2,
        "n_minibatch": 512,
        "activation": "tanh",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
    }
    presets = {
        "default": {},
        "fast_debug": {"n_minibatch": 256, "n_width": 64, "n_depth": 2, "learning_rate": 3e-4},
        "long_train": {"n_minibatch": 512, "n_width": 128, "n_depth": 3, "learning_rate": 1e-4, "weight_decay": 1e-5},
        "lutter_like": {"activation": "softplus", "n_minibatch": 1024, "n_width": 128, "n_depth": 2, "learning_rate": 1e-4, "weight_decay": 1e-5},
        "lutter_like_256": {"activation": "softplus", "n_minibatch": 1024, "n_width": 256, "n_depth": 2, "learning_rate": 1e-4, "weight_decay": 1e-5},
    }
    hp = dict(base)
    hp.update(presets.get(preset, {}))
    hp.update(parse_hp_flags(hp_flags))
    act = str(hp["activation"])
    return (
        f"act{act}_b{hp['n_minibatch']}_lr{fmt_hp(hp['learning_rate'])}"
        f"_wd{fmt_hp(hp['weight_decay'])}_w{hp['n_width']}_d{hp['n_depth']}"
    )


def read_delan_rmse_pair(metrics_json_path: str) -> tuple[float, float]:
    if not os.path.exists(metrics_json_path):
        return float("inf"), float("inf")
    try:
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        val_rmse = float(d.get("eval_val", {}).get("torque_rmse", float("inf")))
        test_rmse = float(d.get("eval_test", {}).get("torque_rmse", float("inf")))
        return val_rmse, test_rmse
    except Exception:
        return float("inf"), float("inf")


def read_delan_metrics(metrics_json_path: str) -> dict:
    if not os.path.exists(metrics_json_path):
        return {"exists": False}
    try:
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return {
            "exists": True,
            "val_rmse": float(d.get("eval_val", {}).get("torque_rmse", float("inf"))),
            "val_mse": float(d.get("eval_val", {}).get("torque_mse", float("inf"))),
            "test_rmse": float(d.get("eval_test", {}).get("torque_rmse", float("inf"))),
            "test_mse": float(d.get("eval_test", {}).get("torque_mse", float("inf"))),
        }
    except Exception:
        return {"exists": False}


def cleanup_non_best_plots(best_delan_id: str, delan_candidates: list[dict]) -> None:
    if not CLEANUP_NON_BEST_PLOTS:
        return
    suffixes = [
        "__DeLaN_Torque.png",
        "__DeLaN_Torque_RMSE_per_joint.png",
        "__DeLaN_Torque_RMSE_Time.png",
        "__elbow_train_vs_test.png",
        "__loss_components.png",
        "__loss_curve.png",
    ]
    for cand in delan_candidates:
        if cand.get("delan_id") == best_delan_id:
            continue
        run_dir = Path(MODELS_DELAN_DIR_HOST) / cand["delan_id"]
        for suffix in suffixes:
            p = run_dir / f"{cand['delan_id']}{suffix}"
            if p.exists():
                try:
                    p.unlink()
                except PermissionError:
                    print(f"[warn] no permission to delete {p}")


def copy_candidate_metrics_to_best(best_dir_host: str, delan_candidates: list[dict]) -> None:
    try:
        os.makedirs(best_dir_host, exist_ok=True)
        index_path = os.path.join(best_dir_host, "candidate_metrics_index.jsonl")
        with open(index_path, "w", encoding="utf-8") as idx:
            for cand in delan_candidates:
                src = cand.get("metrics_json")
                if not src or not os.path.exists(src):
                    continue
                dest_name = f"{cand['delan_id']}__metrics.json"
                dest_path = os.path.join(best_dir_host, dest_name)
                shutil.copyfile(src, dest_path)
                idx.write(json.dumps({
                    "delan_id": cand["delan_id"],
                    "delan_seed": cand.get("delan_seed"),
                    "metrics_json": dest_path,
                }) + "\n")
    except PermissionError:
        print(f"[warn] no permission to write in {best_dir_host}; skipping candidate metrics copy")
