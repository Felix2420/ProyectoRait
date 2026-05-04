"""Log de asistencia en CSV append-only."""

from __future__ import annotations

import csv
import datetime as dt
import logging
import uuid
from pathlib import Path
from typing import Optional

log = logging.getLogger("pipeline.log")

COLS = ["timestamp_iso", "subject", "mean_sim", "n_sequences", "frames_used", "session_id"]


class AttendanceLogger:
    def __init__(self, csv_dir: Path) -> None:
        self.csv_dir = Path(csv_dir)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        today = dt.date.today().isoformat()
        self.csv_path = self.csv_dir / f"attendance_{today}.csv"
        is_new = not self.csv_path.exists()
        if is_new:
            with self.csv_path.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(COLS)
        log.info("attendance csv -> %s", self.csv_path)

    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex[:8]

    def log_confirmation(self, subject: str, mean_sim: float,
                         n_sequences: int, frames_used: int,
                         session_id: str, ts: Optional[dt.datetime] = None) -> None:
        ts = ts or dt.datetime.now()
        row = [ts.isoformat(timespec="seconds"), subject,
               f"{mean_sim:.4f}", n_sequences, frames_used, session_id]
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row)
        log.info("CONFIRMADO subject=%s sim=%.4f session=%s", subject, mean_sim, session_id)
