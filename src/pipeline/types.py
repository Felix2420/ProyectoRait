"""Tipos compartidos del pipeline de inferencia (Fase 6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float

    @property
    def area(self) -> int:
        return max(0, (self.x2 - self.x1) * (self.y2 - self.y1))

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


@dataclass
class MatchResult:
    subject: str | None
    sim: float
    unknown: bool
    top5: List[Tuple[str, float]]
