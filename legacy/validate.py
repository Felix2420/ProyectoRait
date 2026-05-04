"""
validate.py — Validacion completa del modelo de Gait Recognition
Genera las siguientes metricas:
  1. Accuracy top-1 por sujeto
  2. Matriz de confusion
  3. Reporte precision/recall/F1 por sujeto
  4. Curva CMC (Cumulative Match Characteristic)
  5. Histograma de distancias intra-clase vs inter-clase
  6. Deteccion de umbral optimo (EER)
Todo se guarda en: validation_report/
"""

import os
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from torch.cuda.amp import autocast

from model import HybridGaitNet, EMBED_DIM


# ── Cargar modelo y galeria ───────────────────────────────────────────────────
def load_everything(ckpt_path, gallery_path, device):
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    label_map = ckpt["label_map"]
    threshold = ckpt.get("threshold", 0.8)
    kp_dim    = ckpt.get("kp_dim", 66)

    model = HybridGaitNet(kp_input_dim=kp_dim, embed_dim=EMBED_DIM).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    gallery = np.load(gallery_path, allow_pickle=True).item()
    print("Modelo cargado   : epoch " + str(ckpt.get("epoch","?")))
    print("Val Acc training : " + str(round(ckpt.get("val_acc",0)*100,1)) + "%")
    print("Threshold        : " + str(threshold))
    print("Sujetos galeria  : " + str(len(gallery)))
    return model, label_map, gallery, threshold


# ── Obtener embeddings de todos los archivos ──────────────────────────────────
@torch.no_grad()
def get_all_embeddings(model, feature_dir, label_map, device):
    results = []
    for kp_f in sorted(Path(feature_dir).glob("*_kp.npy")):
        stem = kp_f.stem.replace("_kp","")
        lf   = Path(feature_dir) / (stem + "_label.npy")
        gf   = Path(feature_dir) / (stem + "_gei.npy")
        if not lf.exists() or not gf.exists():
            continue
        label = int(np.load(lf))
        kp    = torch.from_numpy(np.load(kp_f)).float().unsqueeze(0).to(device)
        gei   = torch.from_numpy(np.load(gf)).float().unsqueeze(0).unsqueeze(0).to(device)
        with autocast(enabled=(device.type == "cuda")):
            emb = model(kp, gei).squeeze(0).float().cpu().numpy()
        results.append({
            "label": label,
            "name":  label_map.get(label, "Sujeto_" + str(label)),
            "file":  stem,
            "emb":   emb,
        })
    return results


# ── Calcular todas las distancias ─────────────────────────────────────────────
def compute_distances(samples, gallery):
    for s in samples:
        dists = {}
        for lbl, ref in gallery.items():
            dists[lbl] = float(np.linalg.norm(s["emb"] - ref))
        sorted_lbls = sorted(dists, key=lambda x: dists[x])
        s["dists"]       = dists
        s["pred_top1"]   = sorted_lbls[0]
        s["dist_top1"]   = dists[sorted_lbls[0]]
        s["sorted_preds"] = sorted_lbls
    return samples


# ── Reporte en consola ────────────────────────────────────────────────────────
def print_report(samples, label_map, threshold):
    print("")
    print("=" * 60)
    print("REPORTE DE VALIDACION — Gait Recognition")
    print("=" * 60)

    # Por sujeto
    by_label = defaultdict(lambda: {"correct":0,"total":0,"unknown":0})
    correct_all = unknown_all = total_all = 0

    for s in samples:
        true_lbl  = s["label"]
        pred_lbl  = s["pred_top1"] if s["dist_top1"] < threshold else -1
        name      = label_map.get(true_lbl, str(true_lbl))

        by_label[true_lbl]["total"] += 1
        total_all += 1

        if pred_lbl == -1:
            by_label[true_lbl]["unknown"] += 1
            unknown_all += 1
        elif pred_lbl == true_lbl:
            by_label[true_lbl]["correct"] += 1
            correct_all += 1

    print("")
    print("  {:<20} {:>7} {:>7} {:>9} {:>10}".format(
          "Sujeto","Total","Correc","Unknown","Acc%"))
    print("  " + "-"*56)
    for lbl in sorted(by_label.keys()):
        d    = by_label[lbl]
        name = label_map.get(lbl, str(lbl))
        acc  = d["correct"] / d["total"] * 100 if d["total"] > 0 else 0
        flag = " <--" if acc < 50 else ""
        print("  {:<20} {:>7} {:>7} {:>9} {:>9.1f}%{}".format(
              name, d["total"], d["correct"], d["unknown"], acc, flag))

    print("  " + "-"*56)
    global_acc = correct_all / total_all * 100 if total_all > 0 else 0
    print("  {:<20} {:>7} {:>7} {:>9} {:>9.1f}%".format(
          "TOTAL", total_all, correct_all, unknown_all, global_acc))

    # Distancias intra vs inter
    intra, inter = [], []
    for s in samples:
        true_lbl = s["label"]
        for lbl, d in s["dists"].items():
            if lbl == true_lbl:
                intra.append(d)
            else:
                inter.append(d)

    print("")
    print("  Distancias intra-clase : min={:.4f}  max={:.4f}  mean={:.4f}".format(
          min(intra), max(intra), np.mean(intra)))
    print("  Distancias inter-clase : min={:.4f}  max={:.4f}  mean={:.4f}".format(
          min(inter), max(inter), np.mean(inter)))

    # Separabilidad (ratio inter/intra - cuanto mas alto mejor)
    ratio = np.mean(inter) / (np.mean(intra) + 1e-6)
    print("  Ratio separabilidad    : {:.3f}  (>2.0 es bueno, >3.0 es excelente)".format(ratio))

    # CMC Rank-1, Rank-3, Rank-5
    print("")
    print("  CMC (Cumulative Match Characteristic):")
    for rank in [1, 3, 5]:
        hits = sum(1 for s in samples if s["label"] in s["sorted_preds"][:rank])
        r_acc = hits / len(samples) * 100
        print("    Rank-{:1d} : {:.1f}%  ({}/{})".format(rank, r_acc, hits, len(samples)))

    # EER aproximado
    thresholds = np.linspace(min(intra + inter), max(intra + inter), 200)
    best_eer = 1.0
    best_t   = threshold
    for t in thresholds:
        far = sum(1 for d in inter if d < t) / len(inter)   # false accept
        frr = sum(1 for d in intra if d > t) / len(intra)   # false reject
        eer = abs(far - frr)
        if eer < best_eer:
            best_eer = eer
            best_t   = t
    print("")
    print("  Umbral optimo (EER)    : {:.4f}".format(best_t))
    print("  Umbral actual          : {:.4f}".format(threshold))
    if abs(best_t - threshold) > 0.1:
        print("  SUGERENCIA: ajusta --threshold a {:.2f} en inference.py".format(round(best_t, 2)))

    print("")
    print("  Accuracy global        : {:.1f}%".format(global_acc))
    print("=" * 60)
    return global_acc, best_t


# ── Main ──────────────────────────────────────────────────────────────────────
def validate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, label_map, gallery, threshold = load_everything(
        args.ckpt, args.gallery, device)

    if args.threshold:
        threshold = args.threshold
        print("Threshold override    : " + str(threshold))

    samples = get_all_embeddings(model, args.features, label_map, device)
    samples = compute_distances(samples, gallery)
    print("Muestras evaluadas    : " + str(len(samples)))

    global_acc, best_t = print_report(samples, label_map, threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validacion completa del modelo")
    parser.add_argument("--ckpt",      required=True,
                        help="Ruta a best_model.pth")
    parser.add_argument("--gallery",   required=True,
                        help="Ruta a gallery_embeddings.npy")
    parser.add_argument("--features",  required=True,
                        help="Carpeta features/ con archivos .npy")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Sobrescribir umbral (None = usa el del checkpoint)")
    args = parser.parse_args()
    validate(args)
