#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _paths() -> tuple[Path, Path, Path]:
    script = Path(__file__).resolve()
    sweep_root = script.parents[1]          # payload_estimation/services/sweep
    payload_root = script.parents[3]        # payload_estimation
    repo_root = payload_root.parent         # algorithmic_payload_estimation
    return sweep_root, payload_root, repo_root


def _log_line(log_file, msg: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line)
    log_file.write(line + "\n")
    log_file.flush()


def _run_py(script_path: Path, env: dict, log_file, label: str) -> int:
    cmd = [sys.executable, str(script_path)]
    _log_line(log_file, f"START {label}: {' '.join(cmd)}")
    p = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_file.write(p.stdout + "\n")
    log_file.flush()
    _log_line(log_file, f"END {label}: exit_code={p.returncode}")
    return int(p.returncode)


def _handshake_paths(run_tag: str) -> tuple[str, str, str]:
    """
    Paths exposed to LSTM after the DeLaN sweep finishes.
    We point to the run_tag aliases (no timestamp) because the DeLaN sweep now
    writes runtag_timestamp primaries and copies them to these stable names.
    """
    base = "/workspace/shared/evaluation"
    hypers = f"{base}/summary_delan_best_hypers_{run_tag}.jsonl"
    folds = f"{base}/summary_delan_best_folds_{run_tag}.jsonl"
    model = f"{base}/delan_best_model_{run_tag}.json"
    return hypers, folds, model


def _load_best_dataset_seed(model_json_path: Path) -> int | None:
    try:
        import json
        data = json.loads(model_json_path.read_text(encoding="utf-8"))
        ds = data.get("dataset_seed")
        return int(ds) if ds is not None else None
    except Exception:
        return None


def main() -> int:
    sweep_root, payload_root, _ = _paths()
    scripts_dir = sweep_root / "scripts"

    log_dir = payload_root / "shared" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"kBestModel2_{ts}.log"

    env_base = os.environ.copy()

    overall_ok = True

    with log_path.open("w", encoding="utf-8") as log_file:
        _log_line(log_file, f"kBestModel^2 started ts={ts}")

        # 1) run_sweep.py (independent; failure does not stop pipeline)
        # env = env_base.copy()
        # env["SWEEP_DATASET_NAME"] = "UR3_Load0_cc"
        # env["SWEEP_RUN_TAG"] = "kStoryTest"
        # rc = _run_py(scripts_dir / "run_sweep.py", env, log_file, "run_sweep (kStoryTest)")
        # if rc != 0:
        #     _log_line(log_file, "run_sweep failed; continuing to best-Delan/LSTM")

        # 2) best DeLaN + best LSTM for UR3_Load0_5x10^4_under
        env = env_base.copy()
        env["SWEEP_DATASET_NAME"] = "UR3_Load0_5x10^4_under"
        env["SWEEP_RUN_TAG"] = "testHSK_2" #"best5x10L0"
        rc = _run_py(scripts_dir / "run_sweep_delan.py", env, log_file, "best_delan (5x10^4_under)")
        if rc == 0:
            env = env_base.copy()
            env["SWEEP_DATASET_NAME"] = "UR3_Load0_5x10^4_under"
            env["SWEEP_RUN_TAG"] = "testHSK_2"
            h, f, m = _handshake_paths("testHSK_2")
            env["LSTM_BEST_DELAN_HYPERS_JSONL"] = h
            env["LSTM_BEST_DELAN_FOLDS_JSONL"] = f
            env["LSTM_BEST_DELAN_MODEL_JSON"] = m
            best_seed = _load_best_dataset_seed(payload_root / "shared" / "evaluation" / f"delan_best_model_best5x10L0.json")
            if best_seed is not None:
                env["LSTM_BEST_DATASET_SEEDS"] = f"[{best_seed}]"
                _log_line(log_file, f"LSTM_BEST_DATASET_SEEDS=[{best_seed}] (from best DeLaN)")
            _log_line(log_file, "LSTM handshake paths (5x10^4_under):")
            _log_line(log_file, f"  LSTM_BEST_DELAN_HYPERS_JSONL={h}")
            _log_line(log_file, f"  LSTM_BEST_DELAN_FOLDS_JSONL={f}")
            _log_line(log_file, f"  LSTM_BEST_DELAN_MODEL_JSON={m}")
            rc_lstm = _run_py(scripts_dir / "run_sweep_lstm.py", env, log_file, "best_lstm (5x10^4_under)")
            if rc_lstm != 0:
                overall_ok = False
        else:
            _log_line(log_file, "best_delan failed for 5x10^4_under; skipping best_lstm")
            overall_ok = False

        # 3) best DeLaN + best LSTM for UR3_Load0_K86_uniform
        # env = env_base.copy()
        # env["SWEEP_DATASET_NAME"] = "UR3_Load0_K86_uniform"
        # env["SWEEP_RUN_TAG"] = "best86uL0"
        # rc = _run_py(scripts_dir / "run_sweep_delan.py", env, log_file, "best_delan (K86_uniform)")
        # if rc == 0:
        #     env = env_base.copy()
        #     env["SWEEP_DATASET_NAME"] = "UR3_Load0_K86_uniform"
        #     env["SWEEP_RUN_TAG"] = "best86uL0"
        #     h, f, m = _handshake_paths("best86uL0")
        #     env["LSTM_BEST_DELAN_HYPERS_JSONL"] = h
        #     env["LSTM_BEST_DELAN_FOLDS_JSONL"] = f
        #     env["LSTM_BEST_DELAN_MODEL_JSON"] = m
        #     best_seed = _load_best_dataset_seed(payload_root / "shared" / "evaluation" / f"delan_best_model_best86uL0.json")
        #     if best_seed is not None:
        #         env["LSTM_BEST_DATASET_SEEDS"] = f"[{best_seed}]"
        #         _log_line(log_file, f"LSTM_BEST_DATASET_SEEDS=[{best_seed}] (from best DeLaN)")
        #     _log_line(log_file, "LSTM handshake paths (K86_uniform):")
        #     _log_line(log_file, f"  LSTM_BEST_DELAN_HYPERS_JSONL={h}")
        #     _log_line(log_file, f"  LSTM_BEST_DELAN_FOLDS_JSONL={f}")
        #     _log_line(log_file, f"  LSTM_BEST_DELAN_MODEL_JSON={m}")
        #     rc_lstm = _run_py(scripts_dir / "run_sweep_lstm.py", env, log_file, "best_lstm (K86_uniform)")
        #     if rc_lstm != 0:
        #         overall_ok = False
        # else:
        #     _log_line(log_file, "best_delan failed for K86_uniform; skipping best_lstm")
        #     overall_ok = False

        _log_line(log_file, f"kBestModel^2 finished ok={overall_ok}")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
