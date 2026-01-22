from .io_csv import RawCSVLoader
from .joints import JointSelector
from .segment import TimeGapSegmenter, FixedLengthSegmenter
from .pivot import WidePivotBuilder
from .dataset import TrajectoryDatasetBuilder, NPZDatasetWriter
from .split import TrajectorySplitter

import pandas as pd

class DelanPreprocessPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loader = RawCSVLoader()
        self.selector = JointSelector(cfg.dof_joints, cfg.col_joint)

        if cfg.segment_mode == "fixed_length":
            self.segmenter = FixedLengthSegmenter(cfg.col_time, cfg.frames_per_trajectory)
        else:
            self.segmenter = TimeGapSegmenter(cfg.col_time, cfg.time_gap_seconds)
            
        self.pivot = WidePivotBuilder(cfg)
        self.builder = TrajectoryDatasetBuilder(cfg, self.pivot)
        self.splitter = TrajectorySplitter(cfg.test_fraction, cfg.random_seed,
            trajectory_amount=getattr(cfg, "trajectory_amount", None),
        )
        self.writer = NPZDatasetWriter()

    def run(self, raw_csv_path: str, out_npz_path: str):
        df = self.loader.load(raw_csv_path, self.cfg)
        df = self.selector.filter(df)
        # If loader already provided trajectory_id (e.g., wide dataset with ID column),
        # do NOT overwrite it with heuristic segmentation.
        if "trajectory_id" not in df.columns:
            df = self.segmenter.add_trajectory_id(df)
        trajs = self.builder.build(df)
        train, test = self.splitter.split(trajs)
        self.writer.write(out_npz_path, train, test)
        return train, test
