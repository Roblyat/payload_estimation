import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from .types import Trajectory
from .filtering import DelanSignalFilter

class TrajectoryDatasetBuilder:
    def __init__(self, cfg, pivot_builder):
        self.cfg = cfg
        self.pivot_builder = pivot_builder
        self.filter = DelanSignalFilter(cfg)

    def build(self, df) -> list[Trajectory]:
        trajs: list[Trajectory] = []
        for traj_id, df_traj in df.groupby("trajectory_id"):
            t, q, qd, qdd, tau = self.pivot_builder.build_for_trajectory(df_traj)
            t, q, qd, qdd, tau = self.filter.process_trajectory(t, q, qd, qdd, tau)
            label = f"traj_{int(traj_id):04d}"
            trajs.append(Trajectory(label=label, t=t, q=q, qd=qd, qdd=qdd, tau=tau))
        return trajs

class NPZDatasetWriter:
    """Store trajectories (variable length) as object arrays in an .npz."""
    def write(self, path: str, train: list[Trajectory], val: list[Trajectory], test: list[Trajectory]) -> None:
        def pack(trajs: list[Trajectory]):
            labels = [tr.label for tr in trajs]
            t = [tr.t for tr in trajs]
            q = [tr.q for tr in trajs]
            qd = [tr.qd for tr in trajs]
            qdd = [tr.qdd for tr in trajs]
            tau = [tr.tau for tr in trajs]
            return labels, t, q, qd, qdd, tau

        tr_labels, tr_t, tr_q, tr_qd, tr_qdd, tr_tau = pack(train)
        va_labels, va_t, va_q, va_qd, va_qdd, va_tau = pack(val)
        te_labels, te_t, te_q, te_qd, te_qdd, te_tau = pack(test)

        np.savez_compressed(
            path,
            train_labels=np.array(tr_labels, dtype=object),
            train_t=np.array(tr_t, dtype=object),
            train_q=np.array(tr_q, dtype=object),
            train_qd=np.array(tr_qd, dtype=object),
            train_qdd=np.array(tr_qdd, dtype=object),
            train_tau=np.array(tr_tau, dtype=object),
            val_labels=np.array(va_labels, dtype=object),
            val_t=np.array(va_t, dtype=object),
            val_q=np.array(va_q, dtype=object),
            val_qd=np.array(va_qd, dtype=object),
            val_qdd=np.array(va_qdd, dtype=object),
            val_tau=np.array(va_tau, dtype=object),
            test_labels=np.array(te_labels, dtype=object),
            test_t=np.array(te_t, dtype=object),
            test_q=np.array(te_q, dtype=object),
            test_qd=np.array(te_qd, dtype=object),
            test_qdd=np.array(te_qdd, dtype=object),
            test_tau=np.array(te_tau, dtype=object),
        )


def normalize_out_npz_path(path: str) -> str:
    """
    Normalize an output dataset path to the folder-style convention:
      /root/stem.npz          -> /root/stem/stem.npz
      /root/stem/stem.npz     -> (unchanged)
      /root/stem (no suffix)  -> /root/stem/stem.npz
    """
    p = os.path.expanduser(str(path))

    if not p.endswith(".npz"):
        p = p + ".npz"

    stem = Path(p).stem
    root = os.path.dirname(p)

    # Already /root/stem/stem.npz
    if os.path.basename(root) == stem:
        return p

    return os.path.join(root, stem, f"{stem}.npz")


def dataset_json_path(out_npz_path: str) -> str:
    npz_path = normalize_out_npz_path(out_npz_path)
    stem = Path(npz_path).stem
    return os.path.join(os.path.dirname(npz_path), f"{stem}.json")


def _trajectory_duration(t: np.ndarray) -> Optional[float]:
    tt = np.asarray(t, dtype=float).reshape(-1)
    if tt.size == 0:
        return None
    dur = float(tt[-1] - tt[0])
    if not np.isfinite(dur):
        return None
    return dur


def _trajectory_dts(t: np.ndarray) -> np.ndarray:
    tt = np.asarray(t, dtype=float).reshape(-1)
    if tt.size < 2:
        return np.array([], dtype=float)
    dts = np.diff(tt)
    dts = dts[np.isfinite(dts)]
    dts = dts[dts > 0]
    return dts


def _trajectory_dt_median(t: np.ndarray) -> Optional[float]:
    dts = _trajectory_dts(t)
    if dts.size == 0:
        return None
    dt = float(np.median(dts))
    if not np.isfinite(dt) or dt <= 0:
        return None
    return dt


def _dataset_dt_mean(train: list[Trajectory], val: list[Trajectory], test: list[Trajectory]) -> Optional[float]:
    traj_dts: list[float] = []
    for tr in (train + val + test):
        dt = _trajectory_dt_median(tr.t)
        if dt is None:
            return None
        traj_dts.append(dt)
    if not traj_dts:
        return None
    dt_mean = float(np.mean(traj_dts))
    if not np.isfinite(dt_mean) or dt_mean <= 0:
        return None
    return dt_mean


def write_dataset_json(
    out_npz_path: str,
    train: list[Trajectory],
    val: list[Trajectory],
    test: list[Trajectory],
) -> str:
    """
    Write per-dataset metadata JSON next to the dataset NPZ.

    Always includes counts at the top-level:
      {"train": N, "val": M, "test": K, ...}

    Adds per-trajectory durations if they can be computed for all trajectories:
      "durations": {"train": {"traj_0000": 0.83, ...}, "val": {...}, "test": {...}}

    Adds per-trajectory sampling time (median dt within each trajectory), if it can be
    computed for all trajectories:
      "dts": {"train": {"traj_0000": 0.002, ...}, "val": {...}, "test": {...}}
    """
    json_path = dataset_json_path(out_npz_path)

    # Keep these keys first (top of JSON).
    meta: dict[str, Any] = {
        "train": int(len(train)),
        "val": int(len(val)),
        "test": int(len(test)),
    }

    splits: dict[str, list[Trajectory]] = {"train": train, "val": val, "test": test}
    durations: dict[str, dict[str, float]] = {}
    dts: dict[str, dict[str, float]] = {}

    include_durations = (len(train) + len(val) + len(test)) > 0
    if include_durations:
        for split_name, trajs in splits.items():
            split_durs: dict[str, float] = {}
            for tr in trajs:
                dur = _trajectory_duration(tr.t)
                if dur is None:
                    include_durations = False
                    break
                split_durs[str(tr.label)] = dur
            if not include_durations:
                break
            durations[split_name] = split_durs

    if include_durations:
        meta["durations"] = durations

    include_dts = (len(train) + len(val) + len(test)) > 0
    if include_dts:
        for split_name, trajs in splits.items():
            split_dts: dict[str, float] = {}
            for tr in trajs:
                dt = _trajectory_dt_median(tr.t)
                if dt is None:
                    include_dts = False
                    break
                split_dts[str(tr.label)] = dt
            if not include_dts:
                break
            dts[split_name] = split_dts

    if include_dts:
        meta["dts"] = dts
        dt_mean = _dataset_dt_mean(train, val, test)
        if dt_mean is not None:
            meta["dt"] = dt_mean

    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    return json_path
