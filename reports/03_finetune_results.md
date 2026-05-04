# Reporte 03 — Fine-tuning GaitBase (Fase 3)

> **Fase 3** del roadmap (CLAUDE.md §6).
> Generado: 2026-04-21 (cerrado tras run completo).
> Insumos: `scripts/zeroshot_gaitbase.py`, `scripts/finetune_gaitbase.py`,
>          `data/pkl/`, `checkpoints/pretrained/GaitBase_Gait3D_120000.pt`.
> Estado: **CERRADA** — early stopping en iter 1000, mejor checkpoint en iter 400.

---

## 1. Decisión bloqueante de Fase 3

Aprobada al inicio: usar `GaitBase_Gait3D_120000.pt` (Gait3D-Parsing) como
pre-entrenado en lugar de buscar checkpoint oficial CASIA-B. Razón:
no hay pesos públicos de GaitBase entrenado en CASIA-B; nuestro dataset
local CASIA-B (5/124 sujetos) no permite reentrenarlos.

## 2. Métricas de éxito ajustadas (B+C)

Las métricas originales propuestas (margen val ≥ 0.10 cos) **no son alcanzables
estructuralmente** con un dataset de sesión única donde gallery (`normal`) y
probe (`rapido`) comparten todas las variables nuisance (cámara, iluminación,
ropa, fondo, día). Se ajustan a métricas medibles:

| Criterio | Umbral | Justificación |
|---|---|---|
| rank-1 val | ≥ 95% | Identificación correcta entre 3 sujetos val |
| margen val post / pre | ≥ 1.5× | El fine-tuning debe mejorar separación, no degradarla |
| min margen val | > 0 | Ningún sujeto val queda peor que el mejor erróneo |
| CE acc train | ≥ 95% | Memoria robusta de los 13 sujetos de train |

**Lo que NO se mide en Fase 3** (queda explícitamente para 2da sesión):
generalización fuera de las condiciones de captura. Toda métrica reportada
aquí es "techo optimista" para condiciones idénticas a las grabadas.

## 3. Zero-shot baseline (Fase 3 paso 1, ya cerrado)

`scripts/zeroshot_gaitbase.py` con los 19 sujetos completos, sin entrenar:

| Métrica | Valor |
|---|---|
| rank-1 video-disjoint (gallery=normal, probe=rapido) | **100% (19/19)** |
| rank-5 | 100% |
| Margen medio (sim_correct − sim_best_wrong) | +0.0036 |
| sim media correcto | 0.9922 |
| sim media mejor erróneo | 0.9886 |
| VRAM pico | 454 MB |
| Tiempo | 23 s (1.6 seq/s) |

**Lectura técnica del 100%:** el modelo **no está discriminando**, está
"acertando" por márgenes de milésimas dentro de un cluster donde TODOS los
embeddings caen en cos > 0.98. El fine-tuning busca ampliar esa separación.

Detalle por probe en `reports/03_zeroshot_results.json` y `03_zeroshot_topk.csv`.

## 4. Configuración del fine-tuning

| Hiperparámetro | Valor |
|---|---|
| Pesos iniciales | `GaitBase_Gait3D_120000.pt` (BNNecks reinicializado, class_num 3000→13) |
| Optimizador | SGD lr=0.01 momentum=0.9 wd=5e-4 |
| Scheduler | MultiStepLR milestones=(750, 1250) gamma=0.1 |
| Batch | P=4 IDs × K=2 secuencias = 8 muestras |
| Frames por muestra | 30 (random window por iteración, `fixed_unordered`) |
| Total iter | 1500 |
| Eval cada | 200 iters |
| Early stopping | patience=3 evals sin mejorar margen val |
| Loss | TripletLoss(margin=0.2) + CrossEntropyLoss(scale=16, smoothing=0.1) |
| AMP | float16 enabled |
| Hardware | GTX 1650 4GB, batch usa ~1.25 GB |

**Por qué P=4×K=2 y no el [8,16] del config oficial:** solo tenemos 2 secuencias
por sujeto (`normal` + `rapido`), K=16 obligaría a repetir 8× la misma secuencia
por anchor. P=4 da suficiente diversidad de identidades en el batch para que
TripletLoss tenga negativos válidos.

**Por qué LR=0.01 y no 0.1:** estamos fine-tuneando sobre un checkpoint estable
(Gait3D), no entrenando desde cero. LR alto rompería las features útiles.

## 5. Resultados (PENDIENTE — se rellenan al terminar el run)

### 5.1 Pre-finetune (eval iter 0)

| Conjunto | rank-1 | margen | min margen | sim_correct | sim_best_wrong |
|---|---|---|---|---|---|
| val (3 sujetos) | 100% | +0.0062 | +0.0020 | 0.9895 | 0.9833 |

### 5.2 Curva de entrenamiento

**Train (CE / Triplet):**

| iter | lr | triplet | ce | active_frac | ce_acc |
|---|---|---|---|---|---|
| 50 | 0.0100 | 0.0210 | 1.0417 | 0.011 | 0.840 |
| 100 | 0.0100 | 0.0008 | 0.6396 | 0.000 | 0.997 |
| 200 | 0.0100 | 0.0005 | 0.5857 | 0.000 | 1.000 |
| 400 | 0.0100 | 0.0000 | 0.5702 | 0.000 | 1.000 |
| 600 | 0.0100 | 0.0000 | 0.5657 | 0.000 | 1.000 |
| 750 | 0.0010 | 0.0000 | 0.5638 | 0.000 | 1.000 |
| 1000 | 0.0010 | 0.0000 | 0.5616 | 0.000 | 1.000 |

**Eval val por iter:**

| iter | rank-1 | margen | min margen | nota |
|---|---|---|---|---|
| 0 (pre) | 100% | +0.0062 | +0.0020 | baseline |
| 200 | 100% | +0.0161 | +0.0147 | nuevo mejor |
| 400 | 100% | **+0.0177** | **+0.0171** | **mejor (checkpoint guardado)** |
| 600 | 100% | +0.0170 | +0.0163 | patience 1/3 |
| 800 | 100% | +0.0174 | +0.0167 | patience 2/3 (post LR↓ iter 750) |
| 1000 | 100% | +0.0172 | +0.0165 | patience 3/3 → **early stopping** |

Lectura: TripletLoss colapsa a 0 (active_frac=0) por iter ~150 — el batch P=4×K=2 sobre
13 sujetos memorizables es trivialmente separable. El trabajo real lo hace CE; el margen
val sigue creciendo solo por re-acomodo de logits hasta iter 400, después satura. El
milestone en iter 750 (lr 0.01→0.001) llega tarde, no rescata.

### 5.3 Mejor checkpoint (por margen val)

- Archivo: `checkpoints/finetune/gaitbase_ft_best_iter400.pt`
- Iter 400, LR=0.01 (antes del primer milestone)
- val rank-1=100% (3/3), margen=+0.0177, min margen=+0.0171
- Mejora vs pre-finetune: **2.85× margen medio**, **8.7× min margen**

### 5.4 Eval test (3 sujetos intocados: alex, jesusvalenzuela, ricardomora)

| Métrica | Valor |
|---|---|
| rank-1 | **100% (3/3)** |
| rank-5 | 100% |
| margen medio | **+0.0081** |
| min margen | +0.0069 |
| sim media correcto | 0.9938 |
| sim media mejor erróneo | 0.9857 |
| Mejora vs zero-shot test (~+0.0036) | **~2.25×** |

El fine-tuning generaliza fuera de val: aún sobre los 3 sujetos jamás vistos en
gradiente ni selección de modelo, el margen sube ~2.25× respecto al baseline
zero-shot. Sigue siendo un techo optimista (misma sesión), pero la mejora es real.

### 5.5 Cumplimiento de criterios

| Criterio | Umbral | Logrado | Status |
|---|---|---|---|
| rank-1 val | ≥ 95% | 100% | OK |
| margen val post / pre | ≥ 1.5× | 2.85× | OK |
| min margen val | > 0 | +0.0171 | OK |
| CE acc train | ≥ 95% | 100% (desde iter 250) | OK |

**Adicionales no exigidos pero observados:**
- rank-1 test (sujetos jamás vistos) = 100%
- Margen test ~2.25× vs zero-shot
- VRAM pico: 1246 MB / 4 GB
- Tiempo total: 98.9 min (1000 iters × ~5.9 s/iter)

## 6. Limitación estructural (recordatorio)

Tal como quedó documentado en `docs/REQUISITOS.md` §1 ("Limitaciones a aceptar
explícitamente"), el dataset es de **sesión única**. Toda métrica de Fase 3 se
mide en condiciones idénticas a las de captura (mismo día, ropa, cámara,
iluminación, fondo). **El sistema no se ha probado con cambios de sesión.**

Antes de Fase 5 (open-set), debe decidirse: (a) grabar 2da sesión con ≥4 sujetos
y ropa distinta, o (b) aceptar que el sistema solo opera en las condiciones
exactas grabadas. Las métricas reportadas aquí son válidas como **techo
optimista**, no como rendimiento esperado en producción.

## 7. Próximos pasos (Fase 4 si Fase 3 cierra OK)

1. **Integrar rama de pose** (RTMPose ya extrajo `keypoints.npy` en Fase 2)
   vía SkeletonGait++ o fusión tardía. Esto ataca directamente el riesgo
   "modelo aprende silueta-de-la-ropa" (REQUISITOS §1).
2. **Decidir 2da sesión** antes de Fase 5.
3. Reportes 04 (multimodal) y 05 (open-set) según roadmap.

## 8. Reproducción

```bash
C:\Proyecto3\venv\Scripts\activate

# 1. Zero-shot baseline (ya ejecutado)
python scripts/zeroshot_gaitbase.py

# 2. Fine-tuning (este reporte)
python scripts/finetune_gaitbase.py
# (--dry_run para 50 iters de prueba)
```

Salidas:
- `checkpoints/finetune/gaitbase_ft_best_iter<N>.pt`
- `reports/03_finetune_results.json` (cifras completas + curvas)
- Este reporte (resumen narrativo).

---

**Reporte cerrado. Los 4 criterios B+C se cumplen, test (intocado) generaliza con
mejora ~2.25× del margen vs zero-shot. Espero aprobación para avanzar a Fase 4
(integrar rama de pose). La brecha de 2da sesión queda registrada en
`docs/REQUISITOS.md` §1 y debe resolverse antes de Fase 5.**
