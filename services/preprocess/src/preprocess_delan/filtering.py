import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt
except Exception:  # pragma: no cover - handled at runtime if filtering enabled
    butter = None
    sosfiltfilt = None


class DelanSignalFilter:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def _design_sos(self, dt: float):
        if butter is None:
            raise RuntimeError("scipy is required for filtering but is not installed.")
        if not np.isfinite(dt) or dt <= 0:
            return None
        fs = 1.0 / float(dt)
        nyq = 0.5 * fs
        cutoff = float(self.cfg.filter_cutoff_hz)
        if cutoff <= 0 or cutoff >= nyq:
            return None
        wn = cutoff / nyq
        order = int(self.cfg.filter_order)
        return butter(order, wn, btype="low", output="sos")

    @staticmethod
    def _apply_sos(sos, x: np.ndarray) -> np.ndarray:
        if sos is None or sosfiltfilt is None:
            return x
        try:
            return sosfiltfilt(sos, x, axis=0)
        except ValueError:
            # Too short for filtfilt padding; return unfiltered
            return x

    @staticmethod
    def derive_qdd(dq: np.ndarray, t: np.ndarray) -> np.ndarray:
        if dq.size == 0 or len(t) < 2:
            return np.zeros_like(dq)
        if np.all(np.diff(t) > 0):
            return np.stack([np.gradient(dq[:, j], t) for j in range(dq.shape[1])], axis=1)
        return np.stack([np.gradient(dq[:, j]) for j in range(dq.shape[1])], axis=1)

    def process_trajectory(
        self,
        t: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        tau: np.ndarray,
    ):
        t = np.asarray(t, dtype=float)
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        qdd = np.asarray(qdd, dtype=float)
        tau = np.asarray(tau, dtype=float)

        use_filter = bool(getattr(self.cfg, "filter_accel", False))
        sos = None
        if use_filter:
            if len(t) >= 2:
                dt = float(np.nanmedian(np.diff(t)))
            else:
                dt = float("nan")
            sos = self._design_sos(dt)
            q = self._apply_sos(sos, q)
            qd = self._apply_sos(sos, qd)
            tau = self._apply_sos(sos, tau)

        if bool(getattr(self.cfg, "derive_qdd_from_dq", True)):
            qdd = self.derive_qdd(qd, t)

        if use_filter and bool(getattr(self.cfg, "filter_qdd", True)):
            qdd = self._apply_sos(sos, qdd)

        return t, q, qd, qdd, tau
