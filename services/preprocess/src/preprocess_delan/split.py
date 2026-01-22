import random
from typing import Optional

class TrajectorySplitter:
    def __init__(self, test_fraction: float, seed: int, trajectory_amount: Optional[int] = None):
        self.test_fraction = float(test_fraction)
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
        n_test = max(1, int(round(n_subset * self.test_fraction)))
        n_test = min(n_test, n_subset - 1)  # ensure at least 1 train if possible

        test_idx = set(idx[:n_test])
        train_idx = idx[n_test:]

        train = [trajs[i] for i in train_idx]
        test = [trajs[i] for i in test_idx]
        return train, test