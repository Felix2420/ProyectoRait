# Fase 4 — Cierre formal — 2026-04-28

> Documento de cierre técnico de Fase 4 (modelo multimodal). Se decide no
> incorporar la rama de pose al score de producción y avanzar a Fase 6 con
> GaitBase iter1200 como modelo único. Este documento congela el estado y
> deja la deuda técnica explícita.

---

## 1. Resumen ejecutivo

- **Fase 4 cerrada como Caso C** (sin mejora multimodal sobre Fase 3.5).
- **Producción:** `checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt`,
  τ=0.9807, EER ALL=8.11 %, TAR@FAR=0% = 86.49 %.
- **Requisito multimodal del CLAUDE.md** se reinterpreta: pose se **calcula** en el
  pipeline (RTMPose corre en runtime) pero **no se fusiona** en el score; queda
  reservada para *quality gating* y/o *tracking robusto* en Fase 6/7.
- Causa raíz documentada: dataset acotado (19 sujetos, 90° lateral fijo, 2
  sesiones, 51 secs train) — pose no aporta señal discriminativa marginal sobre
  silueta a este ángulo y volumen.

## 2. Recorrido de intentos

| Intento | Script | Init silueta | Resultado open-set | Veredicto |
|---|---|---|---|---|
| SGPP iter800 (otra PC) | `finetune_skeletongaitpp.py` | random | EER 18.92 %, TAR@FAR=0% 54.0 % | Embedding colapsado por init incorrecto |
| Fusión tardía iter800 | `openset_eval_fusion.py` | — | α*=0.8 → EER 8.11 % (= silueta sola) | Pose no aporta |
| ST-GCN lite (Ruta B) | `train_stgcn.py` | — | α*=1.0 → EER 8.11 % | Pose no aporta |
| **SGPP iter1000 Ruta A (2060)** | `finetune_skeletongaitpp.py` con init GaitBase iter1200 | GaitBase iter1200 (76 keys, 51.7 % params) | α*=1.0 → EER 8.11 % | Pose no aporta |

Todos los caminos de fusión convergen al mismo techo: el modelo de silueta sola
de Fase 3.5.

## 3. Análisis — por qué pose no aporta

1. **Oclusión lateral**: a 90° las articulaciones contralaterales están
   solapadas. Los keypoints de RTMPose son muy similares entre sujetos en este
   ángulo; la silueta ya captura casi toda la información de marcha disponible.
2. **Volumen de datos**: 51 secuencias de train son insuficientes para que la
   rama de pose generalice. SGPP corrige `ricardomora` pero introduce nuevos
   errores (`jesusvalenzuela ↔ alexbojorquez`, `luislopez ↔ robertpereira`).
3. **Sesiones=2**: variación intra-sujeto baja, lo que hace que la red aprenda
   ruido de heatmap como firma de identidad.

Conclusión: el problema no son los hiperparámetros ni el modelo, sino el
dataset y la geometría de captura. No se puede arreglar con más entrenamiento.

## 4. Deuda técnica documentada

Tres palancas identificadas pero **no ejecutadas**, archivadas para retomar si
en el futuro se amplía dataset o se agrega un segundo ángulo:

1. **Limpieza de keypoints** — filtro de confianza, suavizado temporal,
   normalización por longitud de tronco.
2. **Representación enriquecida** — bone vectors + velocidades + aceleraciones
   en lugar de keypoints crudos (T,17,3).
3. **Fusión a nivel embedding** — concat 256+128 → MLP con loss conjunta, en
   lugar de score-level averaging.

Si se abre una tercera sesión o un ángulo distinto, retomar Fase 4 con estas
palancas + PoseConv3D o GaitGraph2 entrenado de verdad.

## 5. Caso conocido en producción

`ricardomora ↔ hectorsanchez` (build corporal similar): falla en NN y NR con
GaitBase. SGPP lo resuelve aislado pero rompe otros pares. **Se acepta como
limitación documentada de producción.**

## 6. Artefactos archivados

| Ruta | Contenido |
|---|---|
| `checkpoints/finetune/skeletongaitpp_best_iter1000.pt` | Mejor SGPP Ruta A (53.5 MB) — referencia, no se usa en producción |
| `checkpoints/finetune/skeletongaitpp_best_iter800.pt` | SGPP intento previo (otra PC) — sin valor operativo |
| `checkpoints/finetune/stgcn_lite_best.pt` | ST-GCN Ruta B — sin valor operativo |
| `reports/09_finetune_skeletongaitpp_results.json` | Métricas val Ruta A |
| `reports/09_train_routeA_log.txt` | Log entrenamiento 2060 |
| `reports/10_openset_skeletongaitpp_routeA.json` + `_pairs.csv` | Open-set SGPP solo |
| `reports/11_openset_fusion.json` | Barrido α GaitBase + SGPP Ruta A |
| `reports/12_train_stgcn_lite.json` | Métricas Ruta B |
| `reports/13_openset_fusion_pose.json` | Barrido α GaitBase + ST-GCN |
| `reports/PHASE4_HANDOFF_2026-04-27.md` | Handoff intermedio (intento iter800) |
| `reports/PHASE4_HANDOFF_RUTA_A_2026-04-27.md` | Plan Ruta A (input para 2060) |
| `reports/PHASE4_HANDOFF_RESULTADO_2060_2026-04-28.md` | Resultado Ruta A (output 2060) |
| `reports/PHASE4_CIERRE_2026-04-28.md` | **Este documento** |

## 7. Decisión y siguiente paso

**Decisión del usuario (2026-04-28):** producir con silueta sola, abrir Fase 6.

**Siguiente paso: Fase 6 — Pipeline en vivo.** Diseño detallado en
`reports/14_pipeline_design.md` (siguiente documento).

Parámetros ya cerrados con el usuario:

- FPS de procesamiento: **15** (cámara captura 30, se procesa 1 de cada 2).
- Personas en cuadro: **una a la vez**; bbox de mayor área = sujeto activo.
- Política de identificación: **N=2 secuencias consecutivas con sim>τ y mismo
  top-1**.
- Gallery: **fija (19 sujetos)**, sin enrolamiento en vivo.
- Ventana de secuencia: **60 frames @ 15 fps = 4 s**.
- Stride: **30 frames** (solapamiento 50 %).
- Trigger: bbox válido durante ≥15 frames antes de acumular.
- Reset: bbox ausente >15 frames descarta el buffer.
- Umbral: **τ=0.9807** (Fase 5, sin recalibrar).
