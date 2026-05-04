"""FIFO de siluetas. Emite secuencias de tamaño `window` cada `stride` frames."""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class SequenceBuffer:
    def __init__(self, window: int = 60, stride: int = 30) -> None:
        if stride <= 0 or stride > window:
            raise ValueError(f"stride debe estar en (0, window]; recibí stride={stride} window={window}")
        self.window = window
        self.stride = stride
        self._buf: deque[np.ndarray] = deque(maxlen=window)
        self._frames_since_emit = 0
        self._has_emitted_once = False

    def push(self, sil: np.ndarray) -> Optional[np.ndarray]:
        """Inserta una silueta. Devuelve (window, 64, 44) si toca emitir."""
        self._buf.append(sil)
        self._frames_since_emit += 1
        if not self._has_emitted_once:
            if len(self._buf) >= self.window:
                self._has_emitted_once = True
                self._frames_since_emit = 0
                return np.stack(list(self._buf), axis=0)
            return None
        if self._frames_since_emit >= self.stride:
            self._frames_since_emit = 0
            return np.stack(list(self._buf), axis=0)
        return None

    def reset(self) -> None:
        self._buf.clear()
        self._frames_since_emit = 0
        self._has_emitted_once = False

    def __len__(self) -> int:
        return len(self._buf)
