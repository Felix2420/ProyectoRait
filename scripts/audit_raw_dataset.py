"""
Fase 1 — Auditoría del dataset crudo.

Recorre Dataset Crudo/<sujeto>/{normal,rapido}_*.mp4 y para cada video registra:
  - metadatos por header (cv2 + ffmpeg)
  - conteo real de frames decodificando completo
  - hash corto (md5 de los primeros 4 MB)
  - 3 frames de muestra (inicial, medio, final) → JPEG en reports/audit_frames/

Salida:
  - reports/audit_dataset_raw.csv
  - reports/audit_frames/<sujeto>/<condicion>_{ini,mid,fin}.jpg

Uso:
  python scripts/audit_raw_dataset.py \
      --raw "C:/Proyecto3/Proyecto-Reconocedor-IA/Dataset Crudo" \
      --out  "C:/Proyecto3/ProyectoChino/reports"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("audit")

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
COND_RE = re.compile(r"^(normal|rapido)_(\d+)\.mp4$", re.IGNORECASE)


@dataclass
class VideoRecord:
    subject: str
    condition: str
    file: str
    size_mb: float
    md5_4mb: str
    cv_open_ok: bool
    cv_width: int
    cv_height: int
    cv_fps: float
    cv_nframes_header: int
    cv_nframes_real: int
    cv_duration_s: float
    ff_codec: str
    ff_fps: float
    ff_duration_s: float
    ff_width: int
    ff_height: int
    fps_disagreement: bool
    nframes_disagreement: bool
    notes: str


def md5_first_chunk(path: Path, n_bytes: int = 4 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        h.update(fh.read(n_bytes))
    return h.hexdigest()[:12]


def probe_with_ffmpeg(path: Path) -> dict:
    """Lee metadatos del header parseando stderr de `ffmpeg -i`."""
    out = {"codec": "?", "fps": 0.0, "duration_s": 0.0, "width": 0, "height": 0}
    try:
        res = subprocess.run(
            [FFMPEG, "-hide_banner", "-i", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        text = res.stderr  # ffmpeg imprime info en stderr
    except Exception as e:
        out["codec"] = f"ffmpeg_error:{e.__class__.__name__}"
        return out

    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", text)
    if m:
        h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        out["duration_s"] = round(h * 3600 + mn * 60 + s, 3)

    m = re.search(r"Stream #\d+:\d+.*?Video:\s*([^\s,]+).*?(\d{2,5})x(\d{2,5})", text, re.S)
    if m:
        out["codec"] = m.group(1)
        out["width"] = int(m.group(2))
        out["height"] = int(m.group(3))

    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", text)
    if m:
        out["fps"] = float(m.group(1))

    return out


def decode_frame_count(cap: cv2.VideoCapture) -> int:
    """Cuenta frames realmente decodificables (puede diferir del header)."""
    count = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        count += 1
    return count


def save_sample_frames(video_path: Path, dst_dir: Path, name_prefix: str) -> list[str]:
    """Extrae frame inicial/medio/final como JPEG. Devuelve los nombres guardados."""
    saved = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return saved
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return saved

    targets = {"ini": 0, "mid": n // 2, "fin": max(0, n - 2)}
    dst_dir.mkdir(parents=True, exist_ok=True)
    for tag, idx in targets.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        out_name = f"{name_prefix}_{tag}.jpg"
        cv2.imwrite(str(dst_dir / out_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        saved.append(out_name)
    cap.release()
    return saved


def audit_video(video_path: Path, frames_root: Path) -> VideoRecord:
    subject = video_path.parent.name
    fname = video_path.name
    m = COND_RE.match(fname)
    condition = m.group(1).lower() if m else "unknown"

    notes_list: list[str] = []
    if not m:
        notes_list.append(f"nombre fuera de convención: {fname}")

    size_mb = round(video_path.stat().st_size / (1024 * 1024), 2)
    md5 = md5_first_chunk(video_path)

    cap = cv2.VideoCapture(str(video_path))
    cv_open = cap.isOpened()
    if not cv_open:
        notes_list.append("OpenCV no pudo abrir el archivo")
        rec = VideoRecord(
            subject=subject, condition=condition, file=str(video_path),
            size_mb=size_mb, md5_4mb=md5, cv_open_ok=False,
            cv_width=0, cv_height=0, cv_fps=0.0,
            cv_nframes_header=0, cv_nframes_real=0, cv_duration_s=0.0,
            ff_codec="?", ff_fps=0.0, ff_duration_s=0.0, ff_width=0, ff_height=0,
            fps_disagreement=False, nframes_disagreement=False,
            notes=" | ".join(notes_list),
        )
        return rec

    cv_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cv_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cv_n_header = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cv_n_real = decode_frame_count(cap)
    cv_dur = round(cv_n_real / cv_fps, 3) if cv_fps > 0 else 0.0
    cap.release()

    ff = probe_with_ffmpeg(video_path)

    fps_disagree = (
        cv_fps > 0 and ff["fps"] > 0 and abs(cv_fps - ff["fps"]) > 0.5
    )
    nframes_disagree = abs(cv_n_header - cv_n_real) > 1

    if fps_disagree:
        notes_list.append(f"fps cv={cv_fps:.2f} vs ffmpeg={ff['fps']:.2f}")
    if nframes_disagree:
        notes_list.append(f"nframes header={cv_n_header} vs real={cv_n_real}")
    if cv_n_real < 60:
        notes_list.append(f"video MUY corto: {cv_n_real} frames")
    if (cv_w, cv_h) != (ff["width"], ff["height"]) and ff["width"] > 0:
        notes_list.append(f"resolución cv={cv_w}x{cv_h} vs ffmpeg={ff['width']}x{ff['height']}")

    save_sample_frames(
        video_path,
        dst_dir=frames_root / subject,
        name_prefix=Path(fname).stem,
    )

    return VideoRecord(
        subject=subject, condition=condition, file=str(video_path),
        size_mb=size_mb, md5_4mb=md5, cv_open_ok=True,
        cv_width=cv_w, cv_height=cv_h, cv_fps=round(cv_fps, 3),
        cv_nframes_header=cv_n_header, cv_nframes_real=cv_n_real,
        cv_duration_s=cv_dur,
        ff_codec=ff["codec"], ff_fps=round(ff["fps"], 3),
        ff_duration_s=ff["duration_s"], ff_width=ff["width"], ff_height=ff["height"],
        fps_disagreement=fps_disagree, nframes_disagreement=nframes_disagree,
        notes=" | ".join(notes_list),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Ruta a Dataset Crudo")
    ap.add_argument("--out", required=True, help="Carpeta de reports/")
    args = ap.parse_args()

    raw_root = Path(args.raw)
    out_root = Path(args.out)
    if not raw_root.is_dir():
        log.error("No existe %s", raw_root)
        sys.exit(2)
    out_root.mkdir(parents=True, exist_ok=True)
    frames_root = out_root / "audit_frames"

    videos = sorted(raw_root.glob("*/*.mp4"))
    if not videos:
        log.error("No se encontraron .mp4 en %s", raw_root)
        sys.exit(2)
    log.info("Videos detectados: %d (sujetos: %d)",
             len(videos), len({v.parent.name for v in videos}))

    records: list[VideoRecord] = []
    t0 = time.time()
    for i, v in enumerate(videos, 1):
        log.info("[%d/%d] %s/%s", i, len(videos), v.parent.name, v.name)
        try:
            rec = audit_video(v, frames_root)
        except Exception as e:
            log.exception("Error procesando %s", v)
            rec = VideoRecord(
                subject=v.parent.name, condition="error", file=str(v),
                size_mb=0.0, md5_4mb="", cv_open_ok=False,
                cv_width=0, cv_height=0, cv_fps=0.0,
                cv_nframes_header=0, cv_nframes_real=0, cv_duration_s=0.0,
                ff_codec="?", ff_fps=0.0, ff_duration_s=0.0,
                ff_width=0, ff_height=0,
                fps_disagreement=False, nframes_disagreement=False,
                notes=f"excepción: {e.__class__.__name__}: {e}",
            )
        records.append(rec)

    csv_path = out_root / "audit_dataset_raw.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(records[0]).keys()))
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))

    log.info("Auditoría completa en %.1fs", time.time() - t0)
    log.info("CSV  → %s", csv_path)
    log.info("JPEG → %s", frames_root)
    bad = [r for r in records if r.notes]
    log.info("Videos con notas/anomalías: %d/%d", len(bad), len(records))


if __name__ == "__main__":
    main()
