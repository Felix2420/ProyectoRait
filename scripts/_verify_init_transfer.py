"""Dry-run: instancia SkeletonGait++, transfiere pesos compatibles desde
GaitBase iter1200, e imprime estadisticas de cobertura. NO entrena."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
OPENGAIT = REPO / "third_party" / "OpenGait"
sys.path.insert(0, str(OPENGAIT / "opengait"))
sys.path.insert(0, str(OPENGAIT))

GB_CKPT = REPO / "checkpoints" / "finetune" / "gaitbase_ft_multisession_best_iter1200.pt"

MODEL_CFG = {
    "Backbone": {"in_channels": 3, "blocks": [1, 4, 4, 1], "C": 2},
    "SeparateBNNecks": {"class_num": 13},
    "use_emb2": False,
}


def _load_sgpp_class():
    import importlib, importlib.util
    importlib.import_module("modeling")
    importlib.import_module("modeling.models")
    mod_name = "modeling.models.skeletongaitpp"
    if mod_name not in sys.modules:
        f = OPENGAIT / "opengait" / "modeling" / "models" / "skeletongait++.py"
        spec = importlib.util.spec_from_file_location(mod_name, str(f))
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "modeling.models"
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[mod_name].SkeletonGaitPP


def build_mapping():
    """(src_key_GB, dst_key_SGPP) — sin shortcut3d/sbn ni stems heatmap/fusion."""
    pairs = []

    # Stem silueta (in_channels=1)
    pairs += [
        ("Backbone.forward_block.conv1.conv.weight", "sil_layer0.forward_block.0.weight"),
    ]
    for k in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
        pairs.append((f"Backbone.forward_block.bn1.{k}", f"sil_layer0.forward_block.1.{k}"))

    # layer1 (1 bloque) -> sil_layer1.forward_block.0
    for sub_conv, sub_bn in (("conv1", "bn1"), ("conv2", "bn2")):
        pairs.append((
            f"Backbone.forward_block.layer1.0.{sub_conv}.weight",
            f"sil_layer1.forward_block.0.{sub_conv}.weight",
        ))
        for k in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            pairs.append((
                f"Backbone.forward_block.layer1.0.{sub_bn}.{k}",
                f"sil_layer1.forward_block.0.{sub_bn}.{k}",
            ))

    # layer{2,3,4} bloque [0] (sin downsample 5D, sin shortcut3d/sbn)
    for L in (2, 3, 4):
        for sub_conv, sub_bn in (("conv1", "bn1"), ("conv2", "bn2")):
            # conv: GB layer{L}.0.{sub_conv}.weight -> SGPP layer{L}.0.{sub_conv}.forward_block.0.weight
            pairs.append((
                f"Backbone.forward_block.layer{L}.0.{sub_conv}.weight",
                f"layer{L}.0.{sub_conv}.forward_block.0.weight",
            ))
            # bn: GB layer{L}.0.{sub_bn}.* -> SGPP layer{L}.0.{sub_conv}.forward_block.1.*
            for k in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                pairs.append((
                    f"Backbone.forward_block.layer{L}.0.{sub_bn}.{k}",
                    f"layer{L}.0.{sub_conv}.forward_block.1.{k}",
                ))

    # FCs y BNNecks: mismo nombre
    for k in ("FCs.fc_bin", "BNNecks.fc_bin",
              "BNNecks.bn1d.weight", "BNNecks.bn1d.bias",
              "BNNecks.bn1d.running_mean", "BNNecks.bn1d.running_var",
              "BNNecks.bn1d.num_batches_tracked"):
        pairs.append((k, k))

    return pairs


def transfer(gb_state, sgpp_state):
    pairs = build_mapping()
    new_state = {k: v.clone() for k, v in sgpp_state.items()}
    transferred = []
    miss_src = []
    miss_dst = []
    shape_bad = []
    for sk, dk in pairs:
        if sk not in gb_state:
            miss_src.append(sk)
            continue
        if dk not in new_state:
            miss_dst.append(dk)
            continue
        if gb_state[sk].shape != new_state[dk].shape:
            shape_bad.append((sk, dk, tuple(gb_state[sk].shape), tuple(new_state[dk].shape)))
            continue
        new_state[dk] = gb_state[sk].clone()
        transferred.append(dk)
    return new_state, transferred, miss_src, miss_dst, shape_bad


def main():
    print("[info] cargando GaitBase iter1200...")
    gb = torch.load(GB_CKPT, map_location="cpu", weights_only=False)
    gb_state = gb.get("model", gb)
    print(f"[info] GaitBase: {len(gb_state)} keys")

    print("[info] instanciando SkeletonGait++ (random init)...")
    SGPP = _load_sgpp_class()
    model = SGPP.__new__(SGPP)
    nn.Module.__init__(model)
    model.training = True
    model.build_network(MODEL_CFG)
    sgpp_state = model.state_dict()
    print(f"[info] SGPP: {len(sgpp_state)} keys")

    print("[info] transfiriendo pesos compatibles...")
    new_state, transferred, miss_src, miss_dst, shape_bad = transfer(gb_state, sgpp_state)

    print(f"\n[mapping] transferidos = {len(transferred)} keys")
    print(f"[mapping] no en GB     = {len(miss_src)}")
    print(f"[mapping] no en SGPP   = {len(miss_dst)}")
    print(f"[mapping] shape bad    = {len(shape_bad)}")

    if miss_src:
        print(f"\n[!] keys de GB que no se encontraron (primeras 10):")
        for k in miss_src[:10]: print(f"    {k}")
    if miss_dst:
        print(f"\n[!] keys de SGPP target que no existen (primeras 10):")
        for k in miss_dst[:10]: print(f"    {k}")
    if shape_bad:
        print(f"\n[!] shape mismatches (primeros 5):")
        for sk, dk, ss, ds in shape_bad[:5]:
            print(f"    {sk} {ss} -> {dk} {ds}")

    # Cobertura por param
    total_params = sum(v.numel() for v in sgpp_state.values())
    transferred_params = sum(sgpp_state[k].numel() for k in transferred)
    print(f"\n[cov] params totales SGPP   = {total_params:,}")
    print(f"[cov] params transferidos   = {transferred_params:,} ({transferred_params/total_params*100:.1f}%)")

    # Verificar que el load funciona
    res = model.load_state_dict(new_state, strict=True)
    print(f"\n[load] strict=True OK")

    # Lista resumida de keys NO transferidas (random) por componente
    not_transferred = sorted(set(sgpp_state.keys()) - set(transferred))
    by_prefix = {}
    for k in not_transferred:
        # primer segmento como bucket
        prefix = k.split(".")[0]
        by_prefix.setdefault(prefix, 0)
        by_prefix[prefix] += 1
    print(f"\n[random-init buckets] (prefijo -> #keys)")
    for p, n in sorted(by_prefix.items(), key=lambda x: -x[1]):
        print(f"    {p:30s} {n}")

    print(f"\n[summary]")
    print(f"  - rama silueta (sil_layer0/1) inicializada desde GaitBase")
    print(f"  - rama heatmap (map_layer0/1) random (in_channels=2 != 1, sin contraparte)")
    print(f"  - layer2/3/4 conv+bn principales: inicializados desde GaitBase")
    print(f"  - layer2/3/4 shortcut3d/sbn/downsample 3D: random (no existen en GB)")
    print(f"  - fusion: random")
    print(f"  - FCs y BNNecks: copiados desde GaitBase")


if __name__ == "__main__":
    main()
