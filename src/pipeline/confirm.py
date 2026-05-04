"""Acumulador de confirmación: N secuencias consecutivas con mismo top-1 no-unknown."""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

from .types import MatchResult

log = logging.getLogger("pipeline.confirm")


class ConfirmationAccumulator:
    def __init__(self, n_consecutive: int = 2) -> None:
        if n_consecutive < 1:
            raise ValueError("n_consecutive debe ser >= 1")
        self.n = n_consecutive
        self._recent: deque[MatchResult] = deque(maxlen=n_consecutive)
        self._already_confirmed: bool = False
        self._confirmed_subject: Optional[str] = None

    def push(self, m: MatchResult) -> Optional[str]:
        """Devuelve el nombre confirmado en este push, o None.
        Una vez confirmado para esta sesión, no vuelve a emitir hasta reset()."""
        self._recent.append(m)
        if self._already_confirmed:
            return None
        if len(self._recent) < self.n:
            return None
        if any(r.unknown for r in self._recent):
            return None
        names = {r.subject for r in self._recent}
        if len(names) != 1:
            return None
        name = next(iter(names))
        self._already_confirmed = True
        self._confirmed_subject = name
        return name

    @property
    def confirmed_subject(self) -> Optional[str]:
        return self._confirmed_subject

    def mean_sim(self) -> float:
        if not self._recent:
            return 0.0
        return sum(r.sim for r in self._recent) / len(self._recent)

    def reset(self) -> None:
        self._recent.clear()
        self._already_confirmed = False
        self._confirmed_subject = None
