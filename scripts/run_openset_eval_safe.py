"""Wrapper que mockea torch.utils.tensorboard para evitar el conflicto
tensorflow/scipy/cblas que rompe la importacion de OpenGait.

Uso identico a openset_eval.py, solo agrega el mock antes de cargar.

Ejemplo:
    python scripts/run_openset_eval_safe.py \
        --checkpoint checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt \
        --tag pso_baseline --class-num 13
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType


def _install_tensorboard_mock() -> None:
    """Reemplaza torch.utils.tensorboard con un stub que no toca tensorflow."""
    mock_tb = ModuleType("torch.utils.tensorboard")

    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

        def close(self):
            pass

        def flush(self):
            pass

    mock_tb.SummaryWriter = SummaryWriter
    sys.modules["torch.utils.tensorboard"] = mock_tb


def main() -> int:
    _install_tensorboard_mock()

    import runpy

    repo = Path(__file__).resolve().parents[1]
    target = repo / "scripts" / "openset_eval.py"

    sys.argv = [str(target)] + sys.argv[1:]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
