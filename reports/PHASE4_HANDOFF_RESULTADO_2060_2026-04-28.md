# Phase 4 Handoff — Resultado Ruta A — RTX 2060 — 2026-04-28

## Resumen ejecutivo

**Caso C — Ruta A no superó Fase 3.5.**

El entrenamiento de SkeletonGait++ con init desde GaitBase convergió bien y
mejoró las métricas de val significativamente respecto al baseline random-init.
Sin embargo, en open-set (protocolo de producción) el α óptimo es 1.0 (solo
GaitBase) — la fusión no aporta mejora sobre el modelo de Fase 3.5.

**Producción sigue siendo:** `gaitbase_ft_multisession_best_iter1200.pt`
**Recomendación:** arrancar Fase 6 (pipeline en vivo) con ese checkpoint.

---

## 1. Parámetros del entrenamiento

| Parámetro | Valor |
|---|---|
| Script | `scripts/finetune_skeletongaitpp.py` (Ruta A) |
| Init | `gaitbase_ft_multisession_best_iter1200.pt` (76 keys, 51.7% params) |
| Warmup | 200 iter, sil_layer0 + sil_layer1 congelados (9 tensores) |
| early_stop_patience | 6 |
| early_stop_min_iter | 1250 |
| LR | 0.01 → 0.001 (iter 750) → 0.0001 (iter 1250) |
| Batch | P=4 K=2 |
| Iter máx | 1500 |
| GPU | RTX 2060, VRAM pico 5218 MB |
| Tiempo total | **23.8 min** |

## 2. Iter ganador y métricas val

**Iter ganador: 1000** — margen NR val = +0.0050

| iter | NR rank1 val | NR margen (media) | NR margen (mín) |
|---|---|---|---|
| pre (0) | 33.3% | +0.0006 | — |
| 200 (warmup) | 33.3% | +0.0006 | — |
| 400 | 100% | +0.0026 | +0.0016 |
| 600 | 66.7% | +0.0033 | -0.0015 |
| 800 | 100% | +0.0047 | +0.0045 |
| **1000** | **100%** | **+0.0050** | **+0.0048** |
| 1200 | 100% | +0.0048 | +0.0043 |
| 1400 | 100% | +0.0049 | +0.0047 |

Patience al terminar: 5/6 (no hubo early stop, terminó en iter 1500).

## 3. Métricas test cross-session (3 sujetos intocados)

| Proto | rank1 | rank5 | margen (media) | margen (mín) |
|---|---|---|---|---|
| NN | 66.7% | 100% | +0.0005 | -0.0094 |
| NR | 66.7% | 100% | +0.0026 | -0.0038 |

1/3 sujetos falla rank1 en test (probablemente el caso difícil del split test).

## 4. Open-set SGPP solo (protocolo LOSO, 19 sujetos)

| Proto | EER | TAR@FAR=0% |
|---|---|---|
| NN | 16.67% | 77.8% |
| NR | 15.79% | 10.5% |
| ALL | 18.92% | 18.9% |

SGPP corrige `ricardomora` (genuino sim=0.990 OK en ambos protocolos) pero
introduce nuevos errores en otros sujetos. En global, peor que GaitBase solo.

## 5. Barrido α — fusión tardía GaitBase + SGPP Ruta A

| α (GB) | EER_NN | EER_NR | EER_ALL | TAR@0%_NN | TAR@0%_NR | TAR@0%_ALL |
|---|---|---|---|---|---|---|
| 0.0 | 16.67% | 15.79% | 18.92% | 77.8% | 10.5% | 18.9% |
| 0.1 | 16.67% | 15.79% | 18.92% | 83.3% | 42.1% | 51.3% |
| 0.2 | 16.67% | 15.79% | 13.51% | 83.3% | 63.2% | 67.6% |
| 0.3 | 11.11% | 15.79% | 13.51% | 83.3% | 73.7% | 75.7% |
| 0.4 | 11.11% | 15.79% | 16.22% | 83.3% | 73.7% | 75.7% |
| 0.5 | 11.11% | 15.79% | 16.22% | 83.3% | 73.7% | 75.7% |
| 0.6 | 11.11% | 15.79% | 16.22% | 83.3% | 79.0% | 78.4% |
| 0.7 | 5.56% | 15.79% | 10.81% | 88.9% | 84.2% | 83.8% |
| 0.8 | 5.56% | 15.79% | 10.81% | 88.9% | 84.2% | 86.5% |
| 0.9 | 5.56% | 15.79% | 10.81% | 88.9% | 84.2% | 86.5% |
| **1.0** | **5.56%** | **10.53%** | **8.11%** | **88.9%** | **84.2%** | **86.5%** |

**α* = 1.0** — solo GaitBase es el óptimo. La fusión degrada para todo α < 1.0.

## 6. Clasificación del resultado

**Caso C — Sin mejora sobre Fase 3.5**

Condición: α* = 1.0 (solo silueta) y EER_ALL > 8% para cualquier fusión.

## 7. Top-5 errores (genuinos sim baja e impostores sim alta) — α=1.0

### Impostores sim más alta
| Proto | Probe | Top1 (impostor) | sim |
|---|---|---|---|
| NR | jesusvalenzuela | alexbojorquez | 0.9807 |
| NR | luislopez | robertpereira | 0.9801 |
| NN | alexbojorquez | carlostorres | 0.9795 |
| NR | alexbojorquez | carlostorres | 0.9788 |
| NN | luislopez | robertpereira | 0.9784 |

### Genuinos sim más baja
| Proto | Probe | Top1 | sim | Estado |
|---|---|---|---|---|
| NR | carlostorres | carlostorres | 0.9740 | OK |
| NN | ricardomora | hectorsanchez | 0.9767 | ERR |
| NR | ricardomora | hectorsanchez | 0.9769 | ERR |
| NR | hectorsanchez | hectorsanchez | 0.9789 | OK |
| NN | carlostorres | carlostorres | 0.9792 | OK |

El caso `ricardomora → hectorsanchez` persiste con GaitBase. SGPP lo resuelve
pero a costa de empeorar otros casos.

## 8. Análisis — por qué no mejoró

1. **Techo de datos:** 19 sujetos, ángulo 90° lateral fijo, 2 sesiones.
   El heatmap de RTMPose en vista lateral tiene keypoints muy similares entre
   sujetos (oclusión de articulaciones contralaterales). La rama de pose no
   aporta discriminación adicional sobre la silueta en estas condiciones.

2. **SGPP resuelve ricardomora pero introduce ruido:** La rama de heatmap
   discrimina parcialmente casos difíciles de silueta pero confunde otros
   sujetos (jesusvalenzuela ↔ juliouriarte, carloscarrillo ↔ jesusvalenzuela).

3. **Dataset no es suficientemente grande** para que SGPP aprenda a generalizar
   la rama de pose. Con 51 secuencias de entrenamiento, el overfitting a
   heatmaps específicos es inevitable.

## 9. Recomendación

**Adoptar Fase 3.5 como producción final:**
- Checkpoint: `gaitbase_ft_multisession_best_iter1200.pt`
- EER=8.11%, TAR@FAR=0%=86.49%

**Siguiente paso: Fase 6 — Pipeline en vivo**
- Integrar captura → YOLO → tracking → segmentación → secuencia → GaitBase → matching
- Gallery: 19 sujetos, umbral τ=0.9807 (FAR=0% en val)
- Log de asistencia CSV/SQLite

## 10. Archivos a copiar de vuelta a la GTX 1650

```
checkpoints/finetune/skeletongaitpp_best_iter1000.pt      (53.5 MB)
reports/09_finetune_skeletongaitpp_results.json            (15 KB)
reports/09_train_routeA_log.txt                            (7 KB)
reports/10_openset_skeletongaitpp_routeA.json              (3 KB)
reports/10_openset_skeletongaitpp_routeA_pairs.csv         (4 KB)
reports/11_openset_fusion.json                             (11 KB)
reports/PHASE4_HANDOFF_RESULTADO_2060_2026-04-28.md        (este archivo)
```

Total: **7 archivos**.
