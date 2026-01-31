from .io_csv import RawCSVLoader
from .joints import JointSelector
from .segment import TimeGapSegmenter, FixedLengthSegmenter
from .pivot import WidePivotBuilder
from .dataset import (
    TrajectoryDatasetBuilder,
    NPZDatasetWriter,
    normalize_out_npz_path,
    write_dataset_json,
)
from .split import TrajectorySplitter

import os
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
        self.splitter = TrajectorySplitter(
            cfg.test_fraction,
            cfg.val_fraction,
            cfg.random_seed,
            trajectory_amount=getattr(cfg, "trajectory_amount", None),
            allow_empty_splits=getattr(cfg, "allow_empty_splits", False),
        )
        self.writer = NPZDatasetWriter()

    def run(self, raw_csv_path: str, out_npz_path: str):
        out_npz_path = normalize_out_npz_path(out_npz_path)
        os.makedirs(os.path.dirname(out_npz_path) or ".", exist_ok=True)

        df = self.loader.load(raw_csv_path, self.cfg)
        df = self.selector.filter(df)
        # If loader already provided trajectory_id (e.g., wide dataset with ID column),
        # do NOT overwrite it with heuristic segmentation.
        if "trajectory_id" not in df.columns:
            df = self.segmenter.add_trajectory_id(df)
        trajs = self.builder.build(df)
        train, val, test = self.splitter.split(trajs)
        self.writer.write(out_npz_path, train, val, test)
        write_dataset_json(out_npz_path, train, val, test)
        return train, val, test
