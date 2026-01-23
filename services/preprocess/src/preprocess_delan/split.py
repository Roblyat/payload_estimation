import random
from typing import Optional

class TrajectorySplitter:
    def __init__(
        self,
        test_fraction: float,
        val_fraction: float,
        seed: int,
        trajectory_amount: Optional[int] = None,
    ):
        self.test_fraction = float(test_fraction)
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)
        self.trajectory_amount = trajectory_amount

    def split(self, trajs: list):
        n_total = len(trajs)
        idx = list(range(n_total))
        rng = random.Random(self.seed)
        rng.shuffle(idx)

        # Optional subsampling of trajectories (K trajectories total)
        K = self.trajectory_amount
        if K is not None and int(K) > 0:
            K = int(K)
            K = min(K, n_total)
            idx = idx[:K]  # this is the ONLY set of trajectories we keep

        # Split within the selected subset
        n_subset = len(idx)
        if n_subset == 0:
            return [], [], []

        n_test = int(round(n_subset * self.test_fraction))
        n_val = int(round(n_subset * self.val_fraction))

        # Ensure at least 1 test if possible
        if n_subset >= 2:
            n_test = max(1, n_test)

        # Ensure at least 1 val if possible and requested
        if n_subset >= 3 and self.val_fraction > 0:
            n_val = max(1, n_val)

        # Keep at least 1 train if possible
        max_nontrain = n_subset - 1 if n_subset >= 2 else 0
        if n_test + n_val > max_nontrain:
            overflow = (n_test + n_val) - max_nontrain
            if n_val >= overflow:
                n_val -= overflow
            else:
                overflow -= n_val
                n_val = 0
                n_test = max(0, n_test - overflow)

        test_idx = set(idx[:n_test])
        val_idx = set(idx[n_test:n_test + n_val])
        train_idx = idx[n_test + n_val:]

        train = [trajs[i] for i in train_idx]
        val = [trajs[i] for i in val_idx]
        test = [trajs[i] for i in test_idx]
        return train, val, test
