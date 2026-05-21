"""Test rápido: verifica que infer_live.py inicializa sin errores."""

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src.pipeline.detect import PersonDetector
from src.pipeline.segment import Segmenter
from src.pipeline.embed import GaitEmbedder
from src.pipeline.match import GalleryMatcher
from src.pipeline.track import TrackerFSM
from src.pipeline.seq_buffer import SequenceBuffer
from src.pipeline.log import AttendanceLogger


def resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO / pp)


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    print("[*] Cargando config...")
    cfg = load_config(REPO / "configs" / "pipeline.yaml")

    print("[*] Inicializando YOLO detector (imgsz=640)...")
    det = PersonDetector(
        weights=resolve(cfg["detector"]["weights"]),
        conf=cfg["detector"]["conf"],
        iou=cfg["detector"]["iou"],
        imgsz=cfg["detector"].get("imgsz", 640)
    )

    print("[*] Inicializando RVM segmenter...")
    seg = Segmenter(downsample_ratio=cfg["segmenter"]["downsample_ratio"])

    print("[*] Inicializando GaitBase embedder (compile=false)...")
    emb = GaitEmbedder(
        ckpt=resolve(cfg["embedder"]["ckpt"]),
        class_num=cfg["embedder"]["class_num"],
        compile=cfg["embedder"].get("compile", False)
    )

    print("[*] Inicializando GalleryMatcher...")
    mch = GalleryMatcher(
        gallery_npy=resolve(cfg["matcher"]["gallery_npy"]),
        valid_mask_npy=resolve(cfg["matcher"]["valid_mask_npy"]),
        index_json=resolve(cfg["matcher"]["index_json"]),
        tau=cfg["matcher"]["tau"]
    )

    print("[*] Inicializando TrackerFSM...")
    fsm = TrackerFSM(
        warmup_frames=cfg["tracker"]["warmup_frames"],
        reset_frames=cfg["tracker"]["reset_frames"],
        iou_min=cfg["tracker"]["iou_min"]
    )

    print("[*] Inicializando SequenceBuffer...")
    buf = SequenceBuffer(
        window=cfg["sequence"]["window"],
        stride=cfg["sequence"]["stride"]
    )

    print("[*] Inicializando AttendanceLogger...")
    logger = AttendanceLogger(csv_dir=resolve(cfg["logging"]["csv_dir"]))

    print("\n[OK] Todos los módulos inicializados sin errores")
    print(f"[OK] Config activa:")
    print(f"     - Detector: imgsz={cfg['detector'].get('imgsz', 640)}")
    print(f"     - Embedder: compile={cfg['embedder'].get('compile', False)}")
    print(f"     - Matcher: tau={cfg['matcher']['tau']}")
    print(f"[OK] infer_live.py está listo para ejecutar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
