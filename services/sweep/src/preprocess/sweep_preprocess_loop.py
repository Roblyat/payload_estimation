from __future__ import annotations

from sweep_base import (
    SVC_PREPROCESS,
    SCRIPT_BUILD_DELAN_DATASET,
    COL_FORMAT,
    DERIVE_QDD,
    LOWPASS_SIGNALS,
    LOWPASS_CUTOFF_HZ,
    LOWPASS_ORDER,
    VAL_FRACTION,
)
from sweep_helper import compose_exec, run_cmd, banner


def comp_prep(*, npz_in: str, K: int, tf: float, seed: int, lowpass_qdd: bool, raw_csv: str, log_file):
    log_file.write("\n" + banner(["1) PREPROCESS"], char="#") + "\n")
    cmd = compose_exec(
        SVC_PREPROCESS,
        f"python3 {SCRIPT_BUILD_DELAN_DATASET} "
        f"--derive_qdd_from_qd {str(DERIVE_QDD)} "
        f"--col_format {COL_FORMAT} "
        f"--trajectory_amount {K} "
        f"--test_fraction {tf} "
        f"--val_fraction {VAL_FRACTION} "
        f"--seed {seed} "
        f"--lowpass_signals {LOWPASS_SIGNALS} "
        f"--lowpass_cutoff_hz {LOWPASS_CUTOFF_HZ} "
        f"--lowpass_order {LOWPASS_ORDER} "
        f"--lowpass_qdd {lowpass_qdd} "
        f"--raw_csv {raw_csv} "
        f"--out_npz {npz_in}"
    )
    run_cmd(cmd, log_file)


# Backwards-friendly alias
def compPrep(*args, **kwargs):
    return comp_prep(*args, **kwargs)
