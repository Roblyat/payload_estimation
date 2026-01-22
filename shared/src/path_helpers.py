# shared/src/path_helpers.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class BasePaths:
    raw: str
    preprocessed: str
    processed: str
    models_delan: str
    models_lstm: str
    evaluation: str

def base_id(dataset: str, run_tag: str) -> str:
    return f"{dataset}__{run_tag}"

def delan_tag(structured: bool, seed: int, epochs: int) -> str:
    model_short = "struct" if structured else "black"
    return f"delan_{model_short}_s{seed}_ep{epochs}"

def delan_id(dataset: str, run_tag: str, delan_tag: str) -> str:
    return f"delan_{dataset}_{run_tag}__{delan_tag}"

def artifact_folder(root: str, stem: str) -> str:
    return os.path.join(root, stem)

def artifact_file(root: str, stem: str, ext: str) -> str:
    # new convention: root/stem/stem.ext
    return os.path.join(root, stem, f"{stem}.{ext.lstrip('.')}")

def artifact_file_flat(root: str, stem: str, ext: str) -> str:
    # old convention: root/stem.ext
    return os.path.join(root, f"{stem}.{ext.lstrip('.')}")

def resolve_npz_path(p: str) -> str:
    """
    Accepts:
      - /root/stem/stem.npz   (new)
      - /root/stem            (dir -> /root/stem/stem.npz)
      - /root/stem.npz        (old flat; if not found, try /root/stem/stem.npz)
    """
    p = os.path.expanduser(p)

    if os.path.isdir(p):
        stem = os.path.basename(os.path.normpath(p))
        return os.path.join(p, f"{stem}.npz")

    if p.endswith(".npz") and (not os.path.exists(p)):
        stem = Path(p).stem
        root = os.path.dirname(p)
        candidate = artifact_file(root, stem, "npz")  # root/stem/stem.npz
        if os.path.exists(candidate):
            return candidate

    return p

def normalize_out_npz(p: str) -> str:
    """
    If user passes old flat /root/stem.npz, we write to new /root/stem/stem.npz.
    If user already passes /root/stem/stem.npz, keep it.
    """
    p = os.path.expanduser(p)

    if p.endswith(".npz"):
        stem = Path(p).stem
        root = os.path.dirname(p)

        # If it's already root/stem/stem.npz, keep.
        if os.path.basename(root) == stem:
            return p

        # Otherwise treat as old flat and switch to folder style.
        return artifact_file(root, stem, "npz")

    # If they pass a directory, write stem/stem.npz
    if os.path.isdir(p):
        stem = os.path.basename(os.path.normpath(p))
        return os.path.join(p, f"{stem}.npz")

    return p