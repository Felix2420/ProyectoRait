# Sesion PSO: Optimizacion del Matcher + Sensitivity Analysis del Pipeline

**Fecha:** 2026-05-20
**Hardware:** Laptop (GTX 1650 4GB, i5-12450H, 16GB RAM)
**Objetivo:** Aplicar PSO (Particle Swarm Optimization) para optimizar hiperparametros del sistema de gait recognition.

---

## TL;DR

1. **PSO sobre el matcher (Ruta A)**: 3 funciones objetivo distintas, 18,000 evaluaciones totales. Resultado: **el baseline ya es el optimo local fuerte**. PSO confirma que el matcher actual (tau=0.98, L2 global, sin pesos por parte) no se puede mejorar sin re-entrenar el embedder.
2. **Bug encontrado y corregido** en la normalizacion por parte de `apply_matcher`.
3. **Pipeline completo (baseline)**: 97.3% rank-1 accuracy en 37 videos (Fase 6.5 sin necesidad de Fase 6.5 todavia).
4. **Sensitivity Analysis** lanzado para identificar parametros impactantes en FPS antes de aplicar PSO al pipeline completo (Ruta B) en la PC con RTX 2060.

---

## 1. Ruta A: PSO sobre el matcher

### Setup
- Script: `scripts/pso_matcher.py`
- Embeddings extraidos UNA vez (25s) y cacheados (~5ms por evaluacion)
- Libreria: PySwarms 1.3.0
- Validacion: LOSO sobre 19 sujetos cross-session (s1_normal -> s2_normal y s2_rapido)
- Espacio de busqueda: 19 dimensiones
  - tau_NN, tau_NR  [0.70, 0.99]
  - alpha_norm      [0.0, 1.0]   (mezcla L2 global vs cosine por parte)
  - 16 pesos por parte  [0.0, 2.0]

### Funciones objetivo probadas

| # | Formula | Baseline | Mejor PSO | Mejora |
|---|---------|----------|-----------|--------|
| 1 | 0.5 * F1 + 0.5 * (1 - FAR) | 0.9639 | 0.9639 | 0 |
| 2 | 0.8 * F1 + 0.2 * (1 - FAR) | 0.9422 | 0.9422 | 0 |
| 3 | TAR @ FAR <= 5% (estandar biometria) | 0.8655 | 0.8655 | 0 |

**Conclusion:** 18,000 evaluaciones PSO + 1,000 random samples = 0 mejoras sobre baseline.

### Bug encontrado durante el proceso

En `apply_matcher`, la normalizacion por parte producia vectores con norma sqrt(16) ~ 4, asi que las similitudes salian en rango [-16, 16] en lugar de [-1, 1]. Cualquier `tau < 16` aceptaba todo (FAR=100%).

**Fix aplicado:** cambio a "promedio ponderado de cosine por parte" usando einsum. Ahora `sim_p in [-1, 1]` correctamente.

```python
# Antes (incorrecto):
Gn_p_parts = Gw / (np.linalg.norm(Gw, axis=2, keepdims=True) + 1e-12)
Gn_p = Gn_p_parts.reshape(Gn_p_parts.shape[0], -1)
# producto punto -> rango ~[-16, 16]

# Despues (correcto):
G_parts = G / (np.linalg.norm(G, axis=2, keepdims=True) + 1e-12)
P_parts = P / (np.linalg.norm(P, axis=2, keepdims=True) + 1e-12)
cos_per_part = np.einsum('pkd,gkd->pgk', P_parts, G_parts)  # (n_p, n_g, 16)
w_norm = part_weights / (part_weights.sum() + 1e-12)
sim_p = cos_per_part @ w_norm  # (n_p, n_g) en [-1, 1]
```

### Diagnostico final (Ruta A)
- El matcher actual `tau=0.98 + L2 global + sin pesos` **es el optimo local fuerte**.
- Los enrolados perdidos (4 de 56 pruebas) son casos duros estructurales:
  - `ricardomora` confundido con `hectorsanchez` (sim=0.9769 vs propia menor)
  - Bajar tau no los rescata (top-1 ya esta incorrecto).
  - Pesos por parte tampoco los rescatan (PSO probo 12,000 configs sin exito).
- **Para mejorar:** re-entrenar embedder, cambiar arquitectura del matcher, o usar mas datos.

### Resultados baseline matcher (Fase 6.5 offline, LOSO)

```
NN  EER= 5.56%  TAR@FAR<=1%=88.89% (tau=0.9795)  gap_means=+0.0225
NR  EER=10.53%  TAR@FAR<=1%=84.21% (tau=0.9807)  gap_means=+0.0208
ALL EER= 8.11%  TAR@FAR<=1%=86.49% (tau=0.9807)
```

---

## 2. Pipeline completo (baseline para Ruta B)

### Setup
- Script: `scripts/eval_pipeline_offline.py`
- 37 videos (~5 segundos cada uno) en `dataset2/<subj>/{normal_1,rapido_1}.mp4`
- Config baseline: `configs/pipeline.yaml` (tau=0.85, imgsz=640, downsample_ratio=0.25)

### Resultados
| Metrica | Valor |
|---------|-------|
| Videos procesados | 37 / 37 |
| Top-1 correcto | **36 / 37 = 97.3%** |
| Aceptados (sim >= tau) | **36 / 37 = 97.3%** |
| FPS promedio | ~22 |
| Tiempo total | 285 segundos |

**Observacion:** El pipeline completo es mas robusto que el matcher offline (LOSO) porque:
- Procesa secuencia completa (no template unico)
- Permite tau mas laxo (0.85 vs 0.98 offline) porque las similitudes en vivo son ~0.95 para enrolados

---

## 3. Sensitivity Analysis (Ruta B, parte 1)

### Objetivo
Identificar que parametros del pipeline mas impactan FPS antes de aplicar PSO completo en la PC con RTX 2060.

### Parametros evaluados
| Parametro | Baseline | Valores probados |
|-----------|----------|------------------|
| `detector.imgsz` | 640 | 320, 480, 832 |
| `detector.conf` | 0.35 | 0.25, 0.50, 0.70 |
| `detector.iou` | 0.5 | 0.30, 0.70 |
| `segmenter.downsample_ratio` | 0.25 | 0.125, 0.50 |
| `sequence.window` | 60 | 30, 90, 120 |
| `sequence.stride` | 30 | 15, 50 |

Total: 1 baseline + 15 variaciones = ~80-90 min en la GTX 1650.

### Resultados (85 min en GTX 1650)

**Tabla completa de variaciones:**

| Parametro | Valor | rank1 | delta_acc | fps_mean | delta_fps |
|-----------|-------|-------|-----------|----------|-----------|
| baseline | - | 0.973 | - | 10.32 | - |
| detector.imgsz | 480 | 0.973 | +0.000 | 13.26 | +2.94 |
| detector.imgsz | 832 | 0.973 | +0.000 | 10.09 | -0.23 |
| detector.conf | 0.25 | 0.973 | +0.000 | 10.45 | +0.13 |
| detector.conf | 0.50 | 0.973 | +0.000 | 17.70 | +7.38 |
| detector.conf | 0.70 | 0.973 | +0.000 | 23.27 | +12.95 |
| detector.iou | 0.30 | 0.973 | +0.000 | 23.18 | +12.86 |
| detector.iou | 0.70 | 0.973 | +0.000 | 23.29 | +12.97 |
| segmenter.downsample_ratio | 0.125 | 0.946 | **-0.027** | 26.53 | +16.21 |
| segmenter.downsample_ratio | 0.50 | 0.973 | +0.000 | 13.26 | +2.94 |
| sequence.window | 30 | **0.784** | **-0.189** | 22.95 | +12.63 |
| sequence.window | 90 | **1.000** | **+0.027** | 23.13 | +12.81 |
| sequence.window | 120 | **1.000** | **+0.027** | 23.13 | +12.81 |
| sequence.stride | 15 | 0.973 | +0.000 | 23.09 | +12.77 |
| sequence.stride | 50 | 0.973 | +0.000 | 23.12 | +12.80 |

### Conclusiones del sensitivity

**Hallazgos solidos (accuracy):**
1. **`window=90` rescata 1 video adicional** -> rank1 sube de 0.973 a 1.000. Sin costo en FPS. **Cambio recomendado inmediatamente.**
2. **`window=120` da el mismo resultado** (1.000). Equivalente a 90.
3. **`window=30` es desastroso** (rank1 cae a 0.784). No usar.
4. **`downsample_ratio=0.125` reduce accuracy 2.7%** y sube FPS pero NO es free lunch.
5. **Otros parametros** (conf, iou, stride, imgsz) no afectan accuracy en este dataset.

**Hallazgos de FPS (con sesgo de cache de filesystem):**
- Los evals 1-5 tuvieron fps ~10-17, los evals 6-15 fps ~23. El cambio brusco entre eval 5 y 6 sugiere caching de videos en RAM, no efecto real del parametro.
- Para validar FPS de forma confiable habria que correr cada eval multiples veces o en orden aleatorio.

**Recomendacion inmediata:** Cambiar `sequence.window: 60 -> 90` en `configs/pipeline.yaml`. Mejora gratuita.

**Para PSO en RTX 2060**, los parametros prometedores (orden de impacto):
1. `segmenter.downsample_ratio` (alto impacto en FPS, requiere cuidado con accuracy)
2. `detector.conf` (FPS sin tocar accuracy)
3. `detector.imgsz` (trade-off FPS/accuracy)
4. `sequence.window` (critico para accuracy, ya identificado optimo en 90-120)

Espacio de busqueda reducido a 4 parametros -> PSO 20x40 = 800 evals podria caber en ~15-20h overnight.

---

## 4. Proximos pasos

### En esta laptop (GTX 1650)
- [x] Sensitivity analysis del pipeline (en curso)
- [ ] Documentar resultados del sensitivity
- [ ] Export del proyecto (`scripts/export_project.py`)
- [ ] Commit y push al repo `Felix2420/ProyectoRait`

### En la PC con RTX 2060
- [ ] Importar proyecto (zips de `C:/Proyecto3/export/`)
- [ ] Re-crear venv con `requirements.lock.txt`
- [ ] Verificar que `python scripts/eval_pipeline_offline.py` corre
- [ ] PSO enfocado solo en parametros impactantes (5-8h overnight)
- [ ] Validar mejores params en pipeline completo
- [ ] Actualizar `configs/pipeline.yaml` con resultados

### Despues
- [ ] Validacion end-to-end en vivo (Fase 6.5)
- [ ] Considerar re-entrenamiento del embedder si quedan casos duros
- [ ] Documentacion final del proyecto

---

## 5. Archivos generados en esta sesion

### Codigo nuevo
- `scripts/run_openset_eval_safe.py` — wrapper que mockea torch.utils.tensorboard
- `scripts/pso_matcher.py` — PSO sobre matcher (Ruta A)
- `scripts/pso_diagnose.py` — grid + random search para verificar espacio de busqueda
- `scripts/sensitivity_analysis.py` — barrido de parametros del pipeline

### Logs
- `logs/pso/run_fase1_20x30/` — PSO inicial 20x30 (sin seed)
- `logs/pso/run_fase1_seeded_30x200/` — PSO 30x200 con seed del baseline
- `logs/pso/run_fase2_obj_080_020/` — PSO con funcion objetivo 0.8/0.2
- `logs/pso/run_fase3_tar_at_far05/` — PSO con TAR@FAR<=5%
- `logs/sensitivity/sensitivity_fps_focus_2026-05-20.{json,csv}` — sensitivity (pendiente)

### Reportes nuevos
- `reports/06_openset_pso_baseline.{json,csv,roc.png}` — baseline open-set
- `reports/SESION_PSO_2026-05-20.md` — este documento

### Cambios en codigo existente
Ninguno — todo el trabajo de PSO va en scripts nuevos sin tocar el pipeline en produccion.

---

## 6. Notas tecnicas

### Por que el bug de tensorboard

Al importar OpenGait, `modeling.modules` carga `utils.msg_manager` que hace `from torch.utils.tensorboard import SummaryWriter`. Esto a su vez intenta cargar tensorflow via `tensorboard.compat`, lo cual choca con scipy/cblas en este venv. El workaround es mockear `torch.utils.tensorboard` con un stub antes de importar OpenGait.

### Por que tau in vivo (0.85) != tau offline (0.98)

Las similitudes en `openset_eval.py` (LOSO con un template por sujeto) estan en rango ~0.95-0.99 porque comparan embeddings ya buenos contra embeddings buenos. En vivo (`infer_live.py`), el matcher usa la gallery multi-template y compara contra secuencias frescas que tienen mas variacion. Por eso tau requiere recalibracion segun el escenario.

### PSO no es la herramienta correcta para este matcher

Cuando el baseline ya esta cerca del optimo, PSO no aporta. PSO brilla cuando hay multiples maximos locales o el espacio es muy alto-dimensional. Aqui, con 19 dims y un baseline ya bien calibrado, el optimo es practicamente unico y PSO solo lo confirma. Para Ruta B (pipeline completo) la situacion puede ser distinta porque imgsz, downsample_ratio, etc., tienen mas interacciones no obvias.
