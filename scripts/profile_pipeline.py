"""Profiling per-etapa del pipeline de Fase 6.

Mide YOLO, RVM, GaitBase embed, matcher con torch.cuda.Event.
Genera tabla de tiempos promedio y FPS estimado end-to-end.

Usa dataset2/ si existe, sino crea frames dummy.

Uso:
    python scripts/profile_pipeline.py
    python scripts/profile_pipeline.py --video dataset2/subject_name/normal_1.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.pipeline.capture import Capture
from src.pipeline.detect import PersonDetector
from src.pipeline.track import TrackerFSM
from src.pipeline.segment import Segmenter
from src.pipeline.seq_buffer import SequenceBuffer
from src.pipeline.embed import GaitEmbedder
from src.pipeline.match import GalleryMatcher


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO / pp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "pipeline.yaml"))
    ap.add_argument("--video", default=None,
                    help="Ruta de video. Si no se pasa, busca dataset2/ o crea frames dummy.")
    ap.add_argument("--n-frames", type=int, default=100,
                    help="Número de frames a procesar.")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    # Determinar fuente de video
    if args.video:
        video_path = Path(args.video)
    else:
        dataset2 = REPO / "dataset2"
        if dataset2.exists():
            for subj_dir in sorted(dataset2.iterdir()):
                if subj_dir.is_dir():
                    normal_mp4 = subj_dir / "normal_1.mp4"
                    if normal_mp4.exists():
                        video_path = normal_mp4
                        break
        else:
            video_path = None

    # Inicializar pipeline
    cap = (Capture(str(video_path), target_fps=cfg["camera"]["target_fps"])
           if video_path and Path(video_path).exists()
           else None)

    det = PersonDetector(
        weights=resolve(cfg["detector"]["weights"]),
        conf=cfg["detector"]["conf"],
        iou=cfg["detector"]["iou"],
        imgsz=cfg["detector"].get("imgsz", 640)
    )
    fsm = TrackerFSM(
        warmup_frames=cfg["tracker"]["warmup_frames"],
        reset_frames=cfg["tracker"]["reset_frames"],
        iou_min=cfg["tracker"]["iou_min"]
    )
    seg = Segmenter(downsample_ratio=cfg["segmenter"]["downsample_ratio"])
    buf = SequenceBuffer(
        window=cfg["sequence"]["window"],
        stride=cfg["sequence"]["stride"]
    )
    emb = GaitEmbedder(
        ckpt=resolve(cfg["embedder"]["ckpt"]),
        class_num=cfg["embedder"]["class_num"],
        compile=cfg["embedder"].get("compile", False)
    )
    mch = GalleryMatcher(
        gallery_npy=resolve(cfg["matcher"]["gallery_npy"]),
        valid_mask_npy=resolve(cfg["matcher"]["valid_mask_npy"]),
        index_json=resolve(cfg["matcher"]["index_json"]),
        tau=cfg["matcher"]["tau"]
    )

    # Profiling
    times = {
        "yolo": [],
        "rvm": [],
        "embed": [],
        "match": [],
    }

    n_frames = 0
    n_embeds = 0
    t_start = time.time()

    print(f"Procesando {args.n_frames} frames...")

    while n_frames < args.n_frames:
        if cap:
            item = cap.read()
            if item is None:
                break
            frame, _ = item
        else:
            # Dummy frame
            frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

        n_frames += 1

        # YOLO
        torch.cuda.reset_peak_memory_stats()
        e_start = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)
        e_start.record()
        bbox = det.detect(frame)
        e_end.record()
        torch.cuda.synchronize()
        times["yolo"].append(e_start.elapsed_time(e_end))

        # FSM
        state = fsm.step(bbox)

        # RVM
        e_start.record()
        sil = seg.segment(frame, fsm.last_bbox)
        e_end.record()
        torch.cuda.synchronize()
        times["rvm"].append(e_start.elapsed_time(e_end))

        # Buffer + Embed + Match
        if state == "ACTIVE" and sil is not None:
            seq = buf.push(sil)
            if seq is not None:
                n_embeds += 1
                e_start.record()
                e_vec = emb.embed(seq)
                e_end.record()
                torch.cuda.synchronize()
                times["embed"].append(e_start.elapsed_time(e_end))

                e_start.record()
                m = mch.match(e_vec)
                e_end.record()
                torch.cuda.synchronize()
                times["match"].append(e_start.elapsed_time(e_end))

    dt = time.time() - t_start

    # Resumen
    print("\n" + "=" * 70)
    print(f"Profiling: {n_frames} frames en {dt:.2f} s")
    print("=" * 70)
    print(f"{'Etapa':<15} {'Count':<8} {'Min (ms)':<12} {'Mean (ms)':<12} {'Max (ms)':<12}")
    print("-" * 70)

    for name, times_list in times.items():
        if times_list:
            print(f"{name:<15} {len(times_list):<8} {min(times_list):<12.2f} "
                  f"{np.mean(times_list):<12.2f} {max(times_list):<12.2f}")

    fps_estimated = n_frames / dt if dt > 0 else 0
    print("-" * 70)
    print(f"FPS efectivos: {fps_estimated:.1f} (baseline)")
    print(f"Embeddings procesados: {n_embeds}")
    print("=" * 70)

    if cap:
        cap.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())
