"""Pipeline offline: procesa un video .mp4 → log de asistencia.

Conecta los módulos de `src.pipeline` end-to-end. Sirve para validar el
pipeline completo contra videos del dataset (cuya GT conocemos) antes de
enchufar la cámara en vivo (Fase 6.4).

Uso:
    python scripts/infer_video.py \\
        --video path/al/video.mp4 \\
        --config configs/pipeline.yaml \\
        [--expected-subject jesusvalenzuela]   # opcional, para evaluar TPR

Salidas:
    logs/attendance_YYYY-MM-DD.csv  (append)
    stdout: trazas por evento (estado FSM, secuencia emitida, match, confirm).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.pipeline.capture import Capture          # noqa: E402
from src.pipeline.detect import PersonDetector    # noqa: E402
from src.pipeline.track import TrackerFSM         # noqa: E402
from src.pipeline.segment import Segmenter        # noqa: E402
from src.pipeline.seq_buffer import SequenceBuffer  # noqa: E402
from src.pipeline.embed import GaitEmbedder       # noqa: E402
from src.pipeline.match import GalleryMatcher     # noqa: E402
from src.pipeline.confirm import ConfirmationAccumulator  # noqa: E402
from src.pipeline.log import AttendanceLogger     # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("infer_video")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO / pp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Ruta al .mp4 a procesar")
    ap.add_argument("--config", default=str(REPO / "configs" / "pipeline.yaml"))
    ap.add_argument("--expected-subject", default=None,
                    help="Sujeto esperado (GT). Si se da, se imprime TPR booleano al final.")
    ap.add_argument("--n-consecutive", type=int, default=None,
                    help="Override de confirmation.n_consecutive (útil para videos cortos).")
    ap.add_argument("--quiet", action="store_true",
                    help="Solo loguea WARNING+ (útil para corridas batch).")
    args = ap.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    cfg = load_config(Path(args.config))
    if args.n_consecutive is not None:
        cfg["confirmation"]["n_consecutive"] = int(args.n_consecutive)

    # --- Inicializar módulos ---
    cap = Capture(args.video, target_fps=cfg["camera"]["target_fps"])
    det = PersonDetector(weights=resolve(cfg["detector"]["weights"]),
                         conf=cfg["detector"]["conf"],
                         iou=cfg["detector"]["iou"])
    fsm = TrackerFSM(warmup_frames=cfg["tracker"]["warmup_frames"],
                     reset_frames=cfg["tracker"]["reset_frames"],
                     iou_min=cfg["tracker"]["iou_min"])
    seg = Segmenter(downsample_ratio=cfg["segmenter"]["downsample_ratio"])
    buf = SequenceBuffer(window=cfg["sequence"]["window"],
                         stride=cfg["sequence"]["stride"])
    emb = GaitEmbedder(ckpt=resolve(cfg["embedder"]["ckpt"]),
                       class_num=cfg["embedder"]["class_num"])
    mch = GalleryMatcher(gallery_npy=resolve(cfg["matcher"]["gallery_npy"]),
                         valid_mask_npy=resolve(cfg["matcher"]["valid_mask_npy"]),
                         index_json=resolve(cfg["matcher"]["index_json"]),
                         tau=cfg["matcher"]["tau"])
    conf = ConfirmationAccumulator(n_consecutive=cfg["confirmation"]["n_consecutive"])
    logger = AttendanceLogger(csv_dir=resolve(cfg["logging"]["csv_dir"]))

    session_id = AttendanceLogger.new_session_id()
    n_sequences_emitted = 0
    n_frames_processed = 0
    n_matches = 0
    confirmed: str | None = None
    expected = args.expected_subject

    log.info("=== iniciando inferencia: %s ===", args.video)
    t0 = time.time()

    while True:
        item = cap.read()
        if item is None:
            break
        frame, ts = item
        n_frames_processed += 1

        bbox = det.detect(frame)
        state = fsm.step(bbox)

        if state == "RESET":
            log.info("[%05d] RESET → limpio buffer y empiezo nueva sesión", n_frames_processed)
            buf.reset()
            seg.reset_state()
            conf.reset()
            session_id = AttendanceLogger.new_session_id()
            continue

        if state != "ACTIVE":
            # En IDLE/WARMING aún no acumulamos al buffer (RVM sí avanza para
            # mantener consistencia del estado recurrente, pero la silueta
            # no se persiste).
            _ = seg.segment(frame, fsm.last_bbox)
            continue

        # ACTIVE: alimentar buffer
        sil = seg.segment(frame, fsm.last_bbox)
        if sil is None:
            continue
        seq = buf.push(sil)
        if seq is None:
            continue

        # Tenemos secuencia (window, 64, 44) → embed → match
        n_sequences_emitted += 1
        e = emb.embed(seq)
        m = mch.match(e)
        n_matches += 1
        verdict = "UNK" if m.unknown else f"{m.subject}"
        log.info("[%05d] seq#%d → top1=%-25s sim=%.4f %s",
                 n_frames_processed, n_sequences_emitted, verdict, m.sim,
                 ("(unknown)" if m.unknown else ""))

        confirmed_now = conf.push(m)
        if confirmed_now is not None and confirmed is None:
            confirmed = confirmed_now
            logger.log_confirmation(
                subject=confirmed, mean_sim=conf.mean_sim(),
                n_sequences=cfg["confirmation"]["n_consecutive"],
                frames_used=cfg["sequence"]["window"]
                + (cfg["confirmation"]["n_consecutive"] - 1) * cfg["sequence"]["stride"],
                session_id=session_id,
            )

    cap.release()
    dt = time.time() - t0
    log.info("=== fin ===")
    log.info("frames procesados=%d | secuencias=%d | matches=%d | tiempo=%.1fs (%.1f fps)",
             n_frames_processed, n_sequences_emitted, n_matches, dt,
             n_frames_processed / dt if dt > 0 else 0.0)
    log.info("confirmado=%s", confirmed or "—")
    if expected is not None:
        ok = (confirmed == expected)
        log.info("expected=%s confirmed=%s -> %s",
                 expected, confirmed, "TP" if ok else "FN/FP")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
