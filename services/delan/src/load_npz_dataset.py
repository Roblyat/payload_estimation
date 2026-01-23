from __future__ import annotations
import numpy as np

def _maybe_list(data, key: str):
    if key in data:
        return list(data[key])
    return []

def load_npz_trajectory_dataset(filename: str):
    """
    Load trajectory dataset written by your preprocess pipeline (object arrays in .npz)
    and return flattened (N, dof) arrays for training/testing.

    Expected keys:
      train_labels, train_t, train_q, train_qd, train_qdd, train_tau
      val_labels,   val_t,   val_q,   val_qd,   val_qdd,   val_tau  (optional)
      test_labels,  test_t,  test_q,  test_qd,  test_qdd,  test_tau
    """
    data = np.load(filename, allow_pickle=True)

    train_labels = list(data["train_labels"])
    val_labels = _maybe_list(data, "val_labels")
    test_labels  = list(data["test_labels"])

    train_t   = list(data["train_t"])
    train_q   = list(data["train_q"])
    train_qd  = list(data["train_qd"])
    train_qdd = list(data["train_qdd"])
    train_tau = list(data["train_tau"])

    val_t   = _maybe_list(data, "val_t")
    val_q   = _maybe_list(data, "val_q")
    val_qd  = _maybe_list(data, "val_qd")
    val_qdd = _maybe_list(data, "val_qdd")
    val_tau = _maybe_list(data, "val_tau")

    test_t   = list(data["test_t"])
    test_q   = list(data["test_q"])
    test_qd  = list(data["test_qd"])
    test_qdd = list(data["test_qdd"])
    test_tau = list(data["test_tau"])

    if len(train_q) == 0 or len(test_q) == 0 or len(val_q) == 0:
        raise ValueError(
            "Empty split in dataset NPZ. "
            f"train={len(train_q)} val={len(val_q)} test={len(test_q)}. "
            "Adjust val/test fractions or trajectory amount."
        )

    # Flatten trajectories into sample matrices (N, dof)
    train_q   = np.asarray(np.vstack(train_q),   dtype=np.float32)
    train_qd  = np.asarray(np.vstack(train_qd),  dtype=np.float32)
    train_qdd = np.asarray(np.vstack(train_qdd), dtype=np.float32)
    train_tau = np.asarray(np.vstack(train_tau), dtype=np.float32)

    test_q   = np.asarray(np.vstack(test_q),   dtype=np.float32)
    test_qd  = np.asarray(np.vstack(test_qd),  dtype=np.float32)
    test_qdd = np.asarray(np.vstack(test_qdd), dtype=np.float32)
    test_tau = np.asarray(np.vstack(test_tau), dtype=np.float32)

    if len(val_q) > 0:
        val_q   = np.asarray(np.vstack(val_q),   dtype=np.float32)
        val_qd  = np.asarray(np.vstack(val_qd),  dtype=np.float32)
        val_qdd = np.asarray(np.vstack(val_qdd), dtype=np.float32)
        val_tau = np.asarray(np.vstack(val_tau), dtype=np.float32)
    else:
        val_q = np.zeros((0, train_q.shape[1]), dtype=np.float32)
        val_qd = np.zeros((0, train_q.shape[1]), dtype=np.float32)
        val_qdd = np.zeros((0, train_q.shape[1]), dtype=np.float32)
        val_tau = np.zeros((0, train_q.shape[1]), dtype=np.float32)

    # divider for plotting (boundaries between test trajectories)
    divider = [0]
    count = 0
    for q_traj in list(data["test_q"]):
        count += q_traj.shape[0]
        divider.append(count)

    # dt estimate (no hard assert; your resampling step comes later)
    dt_all = []
    for t_traj in (train_t + val_t + test_t):
        if len(t_traj) >= 2:
            dt_all.append(np.diff(t_traj))
    dt_all = np.concatenate(dt_all) if len(dt_all) else np.array([np.nan])
    dt_mean = float(np.nanmean(dt_all))

    return (train_labels, train_q, train_qd, train_qdd, train_tau), \
           (val_labels, val_q, val_qd, val_qdd, val_tau), \
           (test_labels, test_q, test_qd, test_qdd, test_tau), \
           divider, dt_mean
