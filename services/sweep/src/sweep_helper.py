from __future__ import annotations

import csv
import json
import shlex
import subprocess
from pathlib import Path

import numpy as np

from sweep_base import COMPOSE


def banner(lines, char="#"):
    width = max(len(s) for s in lines) if lines else 0
    bar = char * (width + 8)
    out = [bar]
    for s in lines:
        out.append(f"{char*3} {s.ljust(width)} {char*3}")
    out.append(bar)
    return "\n".join(out)


def run_cmd(cmd, log_file, also_print=True):
    if also_print:
        print(cmd)
    log_file.write("\n" + banner([cmd], char="=") + "\n")
    log_file.flush()

    p = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_file.write(p.stdout + "\n")
    log_file.flush()
    if also_print:
        print(p.stdout)

    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}")


def run_cmd_allow_fail(cmd, log_file, also_print=True):
    if also_print:
        print(cmd)
    log_file.write("\n" + banner([cmd], char="=") + "\n")
    log_file.flush()

    p = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_file.write(p.stdout + "\n")
    log_file.flush()
    if also_print:
        print(p.stdout)

    return p.returncode == 0, p.stdout


def compose_exec(service, inner_cmd):
    return f"{COMPOSE} exec -T {service} bash -lc {shlex.quote(inner_cmd)}"


def append_csv_row(path: str, fieldnames: list[str], row: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    exists = Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl(path: str, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_tag(x):
    return str(x).replace(".", "p")


def pad_curve_by_epoch(epochs: list[int], values: list[float], emax: int) -> np.ndarray:
    if emax <= 0:
        return np.zeros((0,), dtype=np.float32)
    arr = np.full((emax,), np.nan, dtype=np.float32)
    for e, v in zip(epochs, values):
        try:
            e_i = int(e)
        except Exception:
            continue
        if 1 <= e_i <= emax:
            arr[e_i - 1] = float(v)
    # forward fill
    last = np.nan
    for i in range(emax):
        if np.isfinite(arr[i]):
            last = arr[i]
        elif np.isfinite(last):
            arr[i] = last
    # backfill leading NaNs
    if emax > 0 and np.isnan(arr[0]):
        finite_idx = np.where(np.isfinite(arr))[0]
        if finite_idx.size > 0:
            arr[: finite_idx[0]] = arr[finite_idx[0]]
    return arr


def median_iqr_curves(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.stack(curves, axis=0)
    median = np.nanmedian(stack, axis=0)
    q25 = np.nanpercentile(stack, 25, axis=0)
    q75 = np.nanpercentile(stack, 75, axis=0)
    return median, q25, q75


def median_iqr_scalar(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("inf"), float("inf")
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.nanmedian(arr))
    q25 = float(np.nanpercentile(arr, 25))
    q75 = float(np.nanpercentile(arr, 75))
    return median, q75 - q25
