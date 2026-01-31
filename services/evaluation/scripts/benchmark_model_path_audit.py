#!/usr/bin/env python3
"""
Audit benchmark model metric files and report missing/mismatched paths.

Checks:
  - residual NPZ (and optional unseen swap)
  - LSTM model + scalers
  - DeLaN model directory (inferred from residual naming)
  - Preprocessed dataset NPZ (delan_<dataset>_dataset.npz)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple


def resolve_local_path(p: str, project_root: Optional[Path]) -> Path:
    p_expanded = Path(os.path.expanduser(p))
    if p_expanded.exists():
        return p_expanded

    if p.startswith("/workspace/shared"):
        rel = Path(p).relative_to("/workspace/shared")
        if project_root:
            candidate = project_root / "shared" / rel
            if candidate.exists():
                return candidate
        candidate = Path("/workspace/shared") / rel
        if candidate.exists():
            return candidate

    if p.startswith("/workspace"):
        rel = Path(p).relative_to("/workspace")
        if project_root:
            candidate = project_root / rel
            if candidate.exists():
                return candidate
        candidate = Path("/workspace") / rel
        if candidate.exists():
            return candidate

    return p_expanded


def maybe_swap_to_unseen(residual_npz: Path, unseen_tag: Optional[str]) -> Path:
    if not unseen_tag:
        return residual_npz
    path_str = str(residual_npz)
    if "5x10^4_under" in path_str:
        candidate = Path(path_str.replace("5x10^4_under", unseen_tag))
        if candidate.exists():
            return candidate
    return residual_npz


def parse_residual_base(residual_npz: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract dataset/run_tag/delan_tag from:
      <dataset>__<run_tag>__residual__<delan_tag>.npz
    """
    stem = residual_npz.stem
    if "__residual__" not in stem:
        return None, None, None
    prefix, delan_tag = stem.split("__residual__", 1)
    if "__" not in prefix:
        return None, None, None
    dataset, run_tag = prefix.split("__", 1)
    return dataset, run_tag, delan_tag.lstrip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", type=str, default=None, help="evaluation root")
    ap.add_argument("--use-eval-root", action=argparse.BooleanOptionalAction, default=True,
                    help="Default to /shared/evaluation instead of /shared/evaluation/benchmark_models.")
    ap.add_argument("--split", type=str, default="test", choices=["test", "train"])
    ap.add_argument("--include", action="append", default=[],
                    help="Only process model dirs whose name contains this substring (repeatable).")
    ap.add_argument("--unseen-tag", type=str, default="5x10^3_under",
                    help="Dataset tag used for unseen residuals (e.g., 5x10^3_under).")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    bench_root = Path(args.bench_root) if args.bench_root else None
    if bench_root is None:
        if args.use_eval_root:
            for anc in script_dir.parents:
                candidate = anc / "shared" / "evaluation"
                if candidate.exists():
                    bench_root = candidate
                    break
            if bench_root is None:
                candidate = Path("/workspace/shared/evaluation")
                if candidate.exists():
                    bench_root = candidate
        else:
            for anc in script_dir.parents:
                candidate = anc / "shared" / "evaluation" / "benchmark_models"
                if candidate.exists():
                    bench_root = candidate
                    break
            if bench_root is None:
                candidate = Path("/workspace/shared/evaluation/benchmark_models")
                if candidate.exists():
                    bench_root = candidate
    if bench_root is None:
        raise FileNotFoundError("Could not locate benchmark_models; pass --bench-root explicitly.")

    metrics_files = list(bench_root.rglob(f"metrics_{args.split}_H*.json"))
    if args.include:
        metrics_files = [mf for mf in metrics_files if any(tok in str(mf) for tok in args.include)]

    if not metrics_files:
        raise RuntimeError(f"No metrics_{args.split}_H*.json found under {bench_root}")

    project_root = bench_root.parents[2] if len(bench_root.parents) >= 3 else None
    models_lstm_root = (project_root / "shared/models/lstm") if project_root else Path("/workspace/shared/models/lstm")
    models_delan_root = (project_root / "shared/models/delan") if project_root else Path("/workspace/shared/models/delan")
    preprocessed_root = (project_root / "shared/data/preprocessed") if project_root else Path("/workspace/shared/data/preprocessed")

    for mf in sorted(metrics_files):
        with open(mf, "r") as f:
            meta = json.load(f)

        dataset = meta.get("paths", {}).get("residual_npz", "")
        feature_mode = meta.get("feature_mode", "?")
        H = meta.get("H", "?")

        print("\n===", mf)
        paths = meta.get("paths", {})
        residual_npz = resolve_local_path(paths.get("residual_npz", ""), project_root)
        lstm_model = resolve_local_path(paths.get("lstm_model", ""), project_root)
        lstm_scalers = resolve_local_path(paths.get("lstm_scalers", ""), project_root)

        print("residual_npz:", residual_npz, "exists" if residual_npz.exists() else "MISSING")
        unseen_residual = maybe_swap_to_unseen(residual_npz, args.unseen_tag)
        if unseen_residual != residual_npz:
            print("  unseen_npz :", unseen_residual, "exists" if unseen_residual.exists() else "MISSING")

        print("lstm_model  :", lstm_model, "exists" if lstm_model.exists() else "MISSING")
        print("lstm_scalers:", lstm_scalers, "exists" if lstm_scalers.exists() else "MISSING")

        # Suggest LSTM candidates if missing
        if not lstm_model.exists() and models_lstm_root.exists():
            ds_tag = Path(lstm_model).parent.name.split("__feat_")[0] if str(lstm_model) else ""
            pattern = f"*{ds_tag}*__feat_{feature_mode}__lstm_*_H{H}_*"
            candidates = sorted(models_lstm_root.glob(pattern))[:5]
            if candidates:
                print("  lstm candidates:")
                for c in candidates:
                    print("   -", c)

        # Infer delan model dir from residual NPZ name
        ds, run_tag, delan_tag = parse_residual_base(residual_npz)
        if ds and run_tag and delan_tag:
            delan_dir = models_delan_root / f"{ds}__{run_tag}__{delan_tag}"
            ckpt = delan_dir / f"{ds}__{run_tag}__{delan_tag}.jax"
            print("delan_dir  :", delan_dir, "exists" if delan_dir.exists() else "MISSING")
            print("delan_ckpt :", ckpt, "exists" if ckpt.exists() else "MISSING")

        # Preprocessed dataset NPZ
        if ds:
            pre_npz = preprocessed_root / f"delan_{ds}_dataset" / f"delan_{ds}_dataset.npz"
            print("preprocessed:", pre_npz, "exists" if pre_npz.exists() else "MISSING")
            if args.unseen_tag and "5x10^4_under" in ds:
                ds_unseen = ds.replace("5x10^4_under", args.unseen_tag)
                pre_npz_u = preprocessed_root / f"delan_{ds_unseen}_dataset" / f"delan_{ds_unseen}_dataset.npz"
                print("preproc unseen:", pre_npz_u, "exists" if pre_npz_u.exists() else "MISSING")

        print("feature_mode:", feature_mode, "H:", H)


if __name__ == "__main__":
    main()
