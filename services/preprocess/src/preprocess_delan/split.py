import random

class TrajectorySplitter:
    def __init__(self, test_fraction: float, seed: int):
        self.test_fraction = test_fraction
        self.seed = seed

    def split(self, trajs: list):
        rng = random.Random(self.seed)
        idx = list(range(len(trajs)))
        rng.shuffle(idx)

        n_test = max(1, int(len(trajs) * self.test_fraction))
        test_idx = set(idx[:n_test])

        train = [trajs[i] for i in range(len(trajs)) if i not in test_idx]
        test = [trajs[i] for i in range(len(trajs)) if i in test_idx]
        return train, test
