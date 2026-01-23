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
