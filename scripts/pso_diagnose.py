"""Diagnostico rapido: explora el espacio de busqueda con grid + random
para verificar si EXISTEN configuraciones mejores que el baseline.

Si encontramos alguna mejor manualmente, PSO esta mal configurado.
Si NO encontramos ninguna, el baseline es realmente cerca del optimo.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Reusar el setup de pso_matcher (mock tensorboard + extract embeddings)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pso_matcher import (  # noqa: E402
    extract_all_embeddings,
    evaluate_particle,
    REPO,
)

import numpy as np


def main():
    data = extract_all_embeddings(
        checkpoint=REPO / "checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt",
        class_num=13,
        gallery_root=REPO / "data/pkl",
        probe_root=REPO / "data/pkl_s2",
    )

    # Punto de referencia: baseline (tau=0.98, alpha=0, weights=1)
    baseline_params = np.array([0.9808, 0.9808, 0.0] + [1.0] * 16)
    fit_base, det_base = evaluate_particle(baseline_params, data)
    print(f"\n[baseline] fitness={fit_base:.4f}  F1={det_base['F1_avg']:.4f}  FAR={det_base['FAR_avg']:.4f}  TAR={det_base['TAR_avg']:.4f}")

    print("\n=== EXPLORACION 1: variar alpha_norm con weights=1, tau barrido AMPLIO ===")
    best_alpha = 0.0
    best_fit = fit_base
    best_det = det_base
    # Rango tau ampliado a [0.85, 0.9999]
    tau_range = np.concatenate([np.linspace(0.85, 0.99, 50), np.linspace(0.99, 0.9999, 100)])
    for alpha in np.linspace(0, 1, 11):
        local_best_fit = -1
        local_best_det = None
        local_best_tau = 0.85
        for tau in tau_range:
            params = np.array([tau, tau, alpha] + [1.0] * 16)
            f, d = evaluate_particle(params, data)
            if f > local_best_fit:
                local_best_fit = f
                local_best_det = d
                local_best_tau = tau
        marker = " <- MEJOR" if local_best_fit > best_fit else ""
        print(f"  alpha={alpha:.2f}  tau_opt={local_best_tau:.6f}  fit={local_best_fit:.4f}  F1={local_best_det['F1_avg']:.4f}  FAR={local_best_det['FAR_avg']:.4f}  TAR={local_best_det['TAR_avg']:.4f}{marker}")
        if local_best_fit > best_fit:
            best_fit = local_best_fit
            best_alpha = alpha
            best_det = local_best_det

    print(f"\n[mejor alpha] alpha={best_alpha:.2f}  fitness={best_fit:.4f}")

    print("\n=== EXPLORACION 2: random search con pesos por parte (1000 muestras) ===")
    rng = np.random.default_rng(7)
    n_samples = 1000
    t0 = time.time()
    best_random_fit = best_fit
    best_random_params = None
    best_random_det = best_det
    improvements = 0
    for i in range(n_samples):
        # Random sample en bounds (tau ampliado)
        tau_NN = rng.uniform(0.85, 0.9999)
        tau_NR = rng.uniform(0.85, 0.9999)
        alpha = rng.uniform(0, 1)
        weights = rng.uniform(0, 2, size=16)
        params = np.concatenate([[tau_NN, tau_NR, alpha], weights])
        f, d = evaluate_particle(params, data)
        if f > best_random_fit:
            best_random_fit = f
            best_random_params = params
            best_random_det = d
            improvements += 1
    dt = time.time() - t0

    print(f"  {n_samples} muestras en {dt:.1f}s ({n_samples/dt:.0f} evals/s)")
    print(f"  mejoras encontradas: {improvements}")
    print(f"  mejor random fitness: {best_random_fit:.4f} (vs baseline {fit_base:.4f}, vs grid alpha {best_fit:.4f})")
    if best_random_params is not None:
        print(f"  best params: tau={best_random_params[0]:.4f}  alpha={best_random_params[2]:.4f}")
        print(f"  weights min/max/mean = {best_random_params[3:].min():.3f}/{best_random_params[3:].max():.3f}/{best_random_params[3:].mean():.3f}")
        print(f"  F1={best_random_det['F1_avg']:.4f}  FAR={best_random_det['FAR_avg']:.4f}  TAR={best_random_det['TAR_avg']:.4f}")
        print(f"  NN: F1={best_random_det['F1_NN']:.4f} FAR={best_random_det['FAR_NN']:.4f} TAR={best_random_det['TAR_NN']:.4f}")
        print(f"  NR: F1={best_random_det['F1_NR']:.4f} FAR={best_random_det['FAR_NR']:.4f} TAR={best_random_det['TAR_NR']:.4f}")

    print("\n=== CONCLUSION ===")
    if best_random_fit > fit_base + 1e-4:
        print(f"  HAY MARGEN DE MEJORA: +{(best_random_fit - fit_base):.4f}")
        print("  -> PSO debe poder encontrar esto. Necesita mas exploracion.")
    else:
        print(f"  BASELINE ES OPTIMO LOCAL FUERTE.")
        print(f"  Random search no encontro mejora en {n_samples} muestras.")
        print("  -> Probablemente PSO ya esta en optimo. Necesitamos otra estrategia.")


if __name__ == "__main__":
    main()
