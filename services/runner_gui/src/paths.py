# services/runner_gui/src/paths.py
from __future__ import annotations
import sys
from dataclasses import dataclass

if "/workspace/shared/src" not in sys.path:
    sys.path.insert(0, "/workspace/shared/src")

from path_helpers import BasePaths

@dataclass(frozen=True)
class Paths:
    base: BasePaths

    # passthroughs so sections can do paths.preprocessed, paths.processed, ...
    @property
    def preprocessed(self) -> str:
        return self.base.preprocessed

    @property
    def processed(self) -> str:
        return self.base.processed

    @property
    def models_delan(self) -> str:
        return self.base.models_delan

    @property
    def models_lstm(self) -> str:
        return self.base.models_lstm

    @property
    def evaluation(self) -> str:
        return self.base.evaluation


def make_paths(cfg) -> Paths:
    bp = BasePaths(
        preprocessed=cfg.BASE_PREPROCESSED,
        processed=cfg.BASE_PROCESSED,
        models_delan=cfg.BASE_MODELS_DELAN,
        models_lstm=cfg.BASE_MODELS_LSTM,
        evaluation=cfg.BASE_EVALUATION,
    )
    return Paths(base=bp)