from __future__ import annotations

from pathlib import Path
import sys


def _add_sweep_src() -> None:
    sweep_root = Path(__file__).resolve().parents[1]
    sweep_src = sweep_root / "src"
    sys.path.insert(0, str(sweep_root))
    sys.path.insert(0, str(sweep_src))


def main() -> None:
    _add_sweep_src()
    from sweep_best_lstm_tplvl import main as _main

    _main()


if __name__ == "__main__":
    main()
