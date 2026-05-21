"""PSO para optimizar hiperparametros del matcher (Fase 6.5 + extension).

Optimiza simultaneamente 19 parametros del matcher (sin tocar el pipeline
de extraccion de embeddings):

  [0] tau_NN        threshold para protocolo Normal-Normal       [0.70, 0.99]
  [1] tau_NR        threshold para protocolo Normal-Rapido       [0.70, 0.99]
  [2] alpha_norm    interpola L2 global vs L2 por-parte           [0.0, 1.0]
  [3:19] w_parts    peso por cada una de las 16 partes corporales [0.0, 2.0]

Funcion objetivo (estandar biometria):
    si FAR_avg <= 0.05:  fitness = TAR_avg
    si FAR_avg >  0.05:  fitness = -(FAR_avg - 0.05) * 10   (penalizacion fuerte)

Maximizar TAR (aceptar enrolados) sujeto a FAR <= 5%. Permite aceptar
algo de falsos positivos a cambio de mas verdaderos positivos.

Estrategia rapida: extrae los embeddings UNA VEZ (25-30s) y luego cada
evaluacion del PSO es ~5ms (solo algebra lineal sobre arrays cacheados).

Salidas:
    logs/pso/run_<tag>/best_params.json
    logs/pso/run_<tag>/history.json
    logs/pso/run_<tag>/convergence.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Tuple


# === Mock tensorboard ANTES de importar OpenGait ===
def _install_tensorboard_mock() -> None:
    mock_tb = ModuleType("torch.utils.tensorboard")

    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

        def close(self):
            pass

        def flush(self):
            pass

    mock_tb.SummaryWriter = SummaryWriter
    sys.modules["torch.utils.tensorboard"] = mock_tb


_install_tensorboard_mock()

import numpy as np
import torch
import torch.nn as nn
import pyswarms as ps

REPO = Path(__file__).resolve().parents[1]
OPENGAIT = REPO / "third_party" / "OpenGait"
sys.path.insert(0, str(OPENGAIT / "opengait"))
sys.path.insert(0, str(OPENGAIT))

from modeling.backbones.resnet import ResNet9  # noqa: E402
from modeling.modules import (  # noqa: E402
    SetBlockWrapper,
    HorizontalPoolingPyramid,
    PackSequenceWrapper,
    SeparateFCs,
    SeparateBNNecks,
)


class GaitBaseInfer(nn.Module):
    def __init__(self, class_num: int = 3000) -> None:
        super().__init__()
        backbone = ResNet9(
            block="BasicBlock",
            channels=[64, 128, 256, 512],
            in_channel=1,
            layers=[1, 1, 1, 1],
            strides=[1, 2, 2, 1],
            maxpool=False,
        )
        self.Backbone = SetBlockWrapper(backbone)
        self.FCs = SeparateFCs(in_channels=512, out_channels=256, parts_num=16)
        self.BNNecks = SeparateBNNecks(class_num=class_num, in_channels=256, parts_num=16)
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

    def forward(self, sils: torch.Tensor, seqL: List[torch.Tensor]) -> torch.Tensor:
        if sils.dim() == 4:
            sils = sils.unsqueeze(1)
        outs = self.Backbone(sils)
        outs = self.TP(outs, seqL, options={"dim": 2})[0]
        feat = self.HPP(outs)
        return self.FCs(feat)


def _load_silhouettes(pkl_path: Path) -> np.ndarray:
    with pkl_path.open("rb") as f:
        arr = pickle.load(f)
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{pkl_path}: esperaba ndarray, obtuve {type(arr)}")
    return arr


@torch.no_grad()
def _extract_embedding(model: GaitBaseInfer, sils_np: np.ndarray, device: torch.device) -> np.ndarray:
    sils = torch.from_numpy(sils_np.astype(np.float32) / 255.0).unsqueeze(0).to(device)
    seqL = [torch.tensor([sils.shape[1]], device=device)]
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        emb = model(sils, seqL)
    return emb.float().squeeze(0).cpu().numpy().reshape(-1)


def _load_pkls(root: Path, conditions: Tuple[str, ...]) -> Dict[Tuple[str, str], Path]:
    out: Dict[Tuple[str, str], Path] = {}
    for subj_dir in sorted(root.iterdir()):
        if not subj_dir.is_dir():
            continue
        for cond in conditions:
            seq_dir = subj_dir / cond / "090"
            if not seq_dir.is_dir():
                continue
            pkls = sorted(seq_dir.glob("*.pkl"))
            if not pkls:
                continue
            out[(subj_dir.name, cond)] = pkls[0]
    return out


def extract_all_embeddings(
    checkpoint: Path,
    class_num: int,
    gallery_root: Path,
    probe_root: Path,
) -> Dict:
    """Extrae todos los embeddings UNA VEZ y los devuelve en arrays (n, 16, 256)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device} | checkpoint={checkpoint.name}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model = GaitBaseInfer(class_num=class_num).to(device).eval()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[info] pesos: missing={len(missing)} unexpected={len(unexpected)}")

    g_pkls = _load_pkls(gallery_root, ("normal",))
    p_pkls = _load_pkls(probe_root, ("normal", "rapido"))

    t0 = time.time()
    gallery_emb: Dict[str, np.ndarray] = {}
    probes_n: Dict[str, np.ndarray] = {}
    probes_r: Dict[str, np.ndarray] = {}
    for (subj, _cond), path in g_pkls.items():
        gallery_emb[subj] = _extract_embedding(model, _load_silhouettes(path), device)
    for (subj, cond), path in p_pkls.items():
        emb = _extract_embedding(model, _load_silhouettes(path), device)
        if cond == "normal":
            probes_n[subj] = emb
        elif cond == "rapido":
            probes_r[subj] = emb
    dt = time.time() - t0
    print(f"[info] embeddings extraidos en {dt:.1f}s "
          f"(gallery={len(gallery_emb)}, probes_n={len(probes_n)}, probes_r={len(probes_r)})")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[info] VRAM pico: {peak:.0f} MB")

    g_subjects = sorted(gallery_emb.keys())
    G = np.stack([gallery_emb[s] for s in g_subjects]).reshape(-1, 16, 256)

    n_subjects = sorted(probes_n.keys())
    Pn = np.stack([probes_n[s] for s in n_subjects]).reshape(-1, 16, 256)

    r_subjects = sorted(probes_r.keys())
    Pr = np.stack([probes_r[s] for s in r_subjects]).reshape(-1, 16, 256)

    return {
        "G": G, "g_subjects": g_subjects,
        "Pn": Pn, "n_subjects": n_subjects,
        "Pr": Pr, "r_subjects": r_subjects,
    }


def apply_matcher(
    G: np.ndarray,
    P: np.ndarray,
    part_weights: np.ndarray,
    alpha_norm: float,
) -> np.ndarray:
    """Aplica pesos por parte + normalizacion mixta. Devuelve sims en [-1, 1].

    G: (n_g, 16, 256)
    P: (n_p, 16, 256)
    part_weights: (16,) pesos por parte
    alpha_norm: 0 -> cosine global (pesos escalan features antes de norma)
                1 -> promedio ponderado de cosine por parte

    Ambos caminos producen similitudes en [-1, 1] para que tau funcione igual.
    """
    # Camino 1: cosine global con pesos aplicados a features
    Gw = G * part_weights[None, :, None]
    Pw = P * part_weights[None, :, None]
    Gw_flat = Gw.reshape(Gw.shape[0], -1)
    Pw_flat = Pw.reshape(Pw.shape[0], -1)
    Gn_g = Gw_flat / (np.linalg.norm(Gw_flat, axis=1, keepdims=True) + 1e-12)
    Pn_g = Pw_flat / (np.linalg.norm(Pw_flat, axis=1, keepdims=True) + 1e-12)
    sim_g = Pn_g @ Gn_g.T  # (n_p, n_g) en [-1, 1]

    # Camino 2: cosine por parte, promedio ponderado por part_weights
    G_parts = G / (np.linalg.norm(G, axis=2, keepdims=True) + 1e-12)  # cada parte unit-norm
    P_parts = P / (np.linalg.norm(P, axis=2, keepdims=True) + 1e-12)
    # cos_per_part[p, g, k] = <P_parts[p, k], G_parts[g, k]>
    cos_per_part = np.einsum('pkd,gkd->pgk', P_parts, G_parts)  # (n_p, n_g, 16) en [-1, 1]
    w_sum = part_weights.sum() + 1e-12
    w_norm = part_weights / w_sum  # promedio ponderado, suma a 1
    sim_p = cos_per_part @ w_norm  # (n_p, n_g) en [-1, 1]

    return (1.0 - alpha_norm) * sim_g + alpha_norm * sim_p


def compute_metrics_protocol(
    sims: np.ndarray,
    p_subjects: List[str],
    g_subjects: List[str],
    tau: float,
) -> Dict:
    """Calcula TAR, FAR, F1 para un protocolo (LOSO)."""
    g_set = set(g_subjects)
    g_idx = {s: i for i, s in enumerate(g_subjects)}

    TP = FP = FN = TN = 0

    for pi, pname in enumerate(p_subjects):
        if pname in g_set:
            row = sims[pi]
            top1_idx = int(np.argmax(row))
            top1_name = g_subjects[top1_idx]
            top1_sim = float(row[top1_idx])
            if top1_sim >= tau and top1_name == pname:
                TP += 1
            else:
                FN += 1

            # Impostor LOSO: gallery sin self
            row_imp = row.copy()
            row_imp[g_idx[pname]] = -np.inf
            if float(np.max(row_imp)) >= tau:
                FP += 1
            else:
                TN += 1
        else:
            if float(np.max(sims[pi])) >= tau:
                FP += 1
            else:
                TN += 1

    TAR = TP / max(1, TP + FN)
    FAR = FP / max(1, FP + TN)
    precision = TP / max(1, TP + FP)
    f1 = 2 * precision * TAR / max(1e-9, precision + TAR)

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "TAR": TAR, "FAR": FAR, "precision": precision, "F1": f1,
    }


def evaluate_particle(params: np.ndarray, data: Dict) -> Tuple[float, Dict]:
    """Evalua una particula (vector de 19 dims). Retorna (fitness, detalles)."""
    tau_NN = float(params[0])
    tau_NR = float(params[1])
    alpha_norm = float(params[2])
    part_weights = params[3:19].astype(np.float64)

    sims_NN = apply_matcher(data["G"], data["Pn"], part_weights, alpha_norm)
    sims_NR = apply_matcher(data["G"], data["Pr"], part_weights, alpha_norm)

    m_NN = compute_metrics_protocol(sims_NN, data["n_subjects"], data["g_subjects"], tau_NN)
    m_NR = compute_metrics_protocol(sims_NR, data["r_subjects"], data["g_subjects"], tau_NR)

    F1_avg = (m_NN["F1"] + m_NR["F1"]) / 2.0
    FAR_avg = (m_NN["FAR"] + m_NR["FAR"]) / 2.0
    TAR_avg = (m_NN["TAR"] + m_NR["TAR"]) / 2.0

    # Funcion objetivo: maximizar TAR sujeto a FAR <= 5% (estandar biometria)
    FAR_LIMIT = 0.05
    if FAR_avg <= FAR_LIMIT:
        fitness = TAR_avg  # premia TAR cuando FAR es aceptable
    else:
        fitness = -(FAR_avg - FAR_LIMIT) * 10.0  # penaliza exceso de FAR

    return fitness, {
        "fitness": fitness,
        "F1_avg": F1_avg, "FAR_avg": FAR_avg, "TAR_avg": TAR_avg,
        "F1_NN": m_NN["F1"], "FAR_NN": m_NN["FAR"], "TAR_NN": m_NN["TAR"],
        "F1_NR": m_NR["F1"], "FAR_NR": m_NR["FAR"], "TAR_NR": m_NR["TAR"],
        "tau_NN": tau_NN, "tau_NR": tau_NR, "alpha_norm": alpha_norm,
    }


def baseline_eval(data: Dict) -> Dict:
    """Evalua la configuracion baseline (sin pesos, L2 global, tau optimo barrido)."""
    part_weights = np.ones(16, dtype=np.float64)
    alpha_norm = 0.0

    # Barrer tau para encontrar el optimo
    sims_NN = apply_matcher(data["G"], data["Pn"], part_weights, alpha_norm)
    sims_NR = apply_matcher(data["G"], data["Pr"], part_weights, alpha_norm)

    taus = np.linspace(0.70, 0.99, 2901)
    best_fitness = -1.0
    best_tau_NN = best_tau_NR = 0.85
    best_details = {}
    for t in taus:
        m_NN = compute_metrics_protocol(sims_NN, data["n_subjects"], data["g_subjects"], float(t))
        m_NR = compute_metrics_protocol(sims_NR, data["r_subjects"], data["g_subjects"], float(t))
        F1_avg = (m_NN["F1"] + m_NR["F1"]) / 2.0
        FAR_avg = (m_NN["FAR"] + m_NR["FAR"]) / 2.0
        # Misma funcion objetivo que evaluate_particle
        FAR_LIMIT = 0.05
        if FAR_avg <= FAR_LIMIT:
            fit = (m_NN["TAR"] + m_NR["TAR"]) / 2.0
        else:
            fit = -(FAR_avg - FAR_LIMIT) * 10.0
        if fit > best_fitness:
            best_fitness = fit
            best_tau_NN = best_tau_NR = float(t)
            best_details = {
                "fitness": fit, "F1_avg": F1_avg, "FAR_avg": FAR_avg,
                "TAR_avg": (m_NN["TAR"] + m_NR["TAR"]) / 2.0,
                "F1_NN": m_NN["F1"], "FAR_NN": m_NN["FAR"], "TAR_NN": m_NN["TAR"],
                "F1_NR": m_NR["F1"], "FAR_NR": m_NR["FAR"], "TAR_NR": m_NR["TAR"],
                "tau_NN": best_tau_NN, "tau_NR": best_tau_NR, "alpha_norm": 0.0,
            }
    return best_details


def run_pso(
    data: Dict,
    n_particles: int,
    iterations: int,
    output_dir: Path,
    seed_baseline: Dict = None,
) -> Dict:
    """Ejecuta PSO con PySwarms. Maximiza fitness (PySwarms minimiza, asi que negamos).

    Si seed_baseline se da, una particula se siembra con esa solucion para
    garantizar que PSO nunca sea peor que el baseline.
    """
    lower = np.array([0.70, 0.70, 0.0] + [0.0] * 16, dtype=np.float64)
    upper = np.array([0.99, 0.99, 1.0] + [2.0] * 16, dtype=np.float64)
    bounds = (lower, upper)

    history: List[Dict] = []

    def objective(particles: np.ndarray) -> np.ndarray:
        out = np.zeros(particles.shape[0])
        for i, params in enumerate(particles):
            fitness, details = evaluate_particle(params, data)
            out[i] = -fitness  # PySwarms minimiza
            history.append(details)
        return out

    # Inicializacion: particula 0 = baseline, resto = jitter gaussiano alrededor
    init_pos = None
    if seed_baseline is not None:
        rng = np.random.default_rng(42)
        baseline_vec = np.zeros(19, dtype=np.float64)
        baseline_vec[0] = seed_baseline["tau_NN"]
        baseline_vec[1] = seed_baseline["tau_NR"]
        baseline_vec[2] = seed_baseline.get("alpha_norm", 0.0)
        baseline_vec[3:] = 1.0  # pesos uniformes

        init_pos = np.zeros((n_particles, 19), dtype=np.float64)
        init_pos[0] = baseline_vec  # semilla exacta del baseline
        # Resto: jitter gaussiano respetando bounds
        for i in range(1, n_particles):
            jitter = rng.normal(0, 0.05, size=19)
            jitter[3:] = rng.normal(0, 0.3, size=16)  # mas variacion en pesos
            init_pos[i] = np.clip(baseline_vec + jitter, lower, upper)

    options = {"c1": 0.5, "c2": 0.5, "w": 0.5}  # w=0.5 baja inercia (converge mas rapido)
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=19,
        options=options,
        bounds=bounds,
        init_pos=init_pos,
    )

    print(f"\n[pso] {n_particles} particulas x {iterations} iter = {n_particles*iterations} evals")
    t0 = time.time()
    cost, pos = optimizer.optimize(objective, iters=iterations, verbose=True)
    dt = time.time() - t0
    print(f"[pso] terminado en {dt:.1f}s")

    best_fitness = -float(cost)
    _, best_details = evaluate_particle(pos, data)

    print("\n=== MEJOR SOLUCION PSO ===")
    print(f"  fitness = {best_fitness:.4f}")
    print(f"  tau_NN = {pos[0]:.4f}   tau_NR = {pos[1]:.4f}   alpha_norm = {pos[2]:.4f}")
    print(f"  weights min/max/mean = {pos[3:].min():.3f}/{pos[3:].max():.3f}/{pos[3:].mean():.3f}")
    print(f"  F1_avg = {best_details['F1_avg']:.4f}   FAR_avg = {best_details['FAR_avg']:.4f}   TAR_avg = {best_details['TAR_avg']:.4f}")
    print(f"  NN: F1={best_details['F1_NN']:.4f} FAR={best_details['FAR_NN']:.4f} TAR={best_details['TAR_NN']:.4f}")
    print(f"  NR: F1={best_details['F1_NR']:.4f} FAR={best_details['FAR_NR']:.4f} TAR={best_details['TAR_NR']:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "best_params.json").open("w") as f:
        json.dump({
            "tau_NN": float(pos[0]),
            "tau_NR": float(pos[1]),
            "alpha_norm": float(pos[2]),
            "part_weights": pos[3:].tolist(),
            "fitness": float(best_fitness),
            "metrics": {k: float(v) for k, v in best_details.items()},
            "elapsed_seconds": float(dt),
            "n_particles": n_particles,
            "iterations": iterations,
            "n_evals": len(history),
        }, f, indent=2)

    with (output_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)

    cost_history = optimizer.cost_history
    with (output_dir / "convergence.json").open("w") as f:
        json.dump({
            "iteration_cost": [float(c) for c in cost_history],
            "iteration_fitness": [float(-c) for c in cost_history],
        }, f, indent=2)

    print(f"\n-> {output_dir}/best_params.json")
    print(f"-> {output_dir}/history.json ({len(history)} evals)")
    print(f"-> {output_dir}/convergence.json")

    return best_details


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt")
    ap.add_argument("--class-num", type=int, default=13)
    ap.add_argument("--gallery-root", default="data/pkl")
    ap.add_argument("--probe-root", default="data/pkl_s2")
    ap.add_argument("--n-particles", type=int, default=20)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = REPO / "logs" / "pso" / f"run_{tag}"

    data = extract_all_embeddings(
        checkpoint=REPO / args.checkpoint,
        class_num=args.class_num,
        gallery_root=REPO / args.gallery_root,
        probe_root=REPO / args.probe_root,
    )

    print("\n=== BASELINE (sin pesos, L2 global, tau barrido) ===")
    baseline = baseline_eval(data)
    print(f"  tau_optimo = {baseline['tau_NN']:.4f}")
    print(f"  fitness = {baseline['fitness']:.4f}")
    print(f"  F1_avg = {baseline['F1_avg']:.4f}   FAR_avg = {baseline['FAR_avg']:.4f}   TAR_avg = {baseline['TAR_avg']:.4f}")
    print(f"  NN: F1={baseline['F1_NN']:.4f} FAR={baseline['FAR_NN']:.4f} TAR={baseline['TAR_NN']:.4f}")
    print(f"  NR: F1={baseline['F1_NR']:.4f} FAR={baseline['FAR_NR']:.4f} TAR={baseline['TAR_NR']:.4f}")

    pso_result = run_pso(data, args.n_particles, args.iterations, output_dir, seed_baseline=baseline)

    print("\n=== MEJORA vs BASELINE ===")
    delta_fit = pso_result["fitness"] - baseline["fitness"]
    delta_f1 = pso_result["F1_avg"] - baseline["F1_avg"]
    delta_far = pso_result["FAR_avg"] - baseline["FAR_avg"]
    print(f"  delta fitness = {delta_fit:+.4f}")
    print(f"  delta F1 avg  = {delta_f1:+.4f}")
    print(f"  delta FAR avg = {delta_far:+.4f}")

    # Guardar resumen baseline tambien
    with (output_dir / "baseline.json").open("w") as f:
        json.dump({k: float(v) for k, v in baseline.items()}, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
