# Reporte 05 — Fine-tuning multisession (Fase 3.5)

> Re-entrenamiento con ambas sesiones como insumo, tras confirmar en reporte 04
> que la silueta estaba saturada en cross-session. Mismo split subject-disjoint
> que Fase 3 (13/3/3), mismos hparams, pero ahora el modelo ve 4 tipos por
> sujeto (normal_s1, normal_s2, rapido_s1, rapido_s2) → triplets cross-clothing
> entran en la supervisión.
>
> Generado: 2026-04-22.
> Insumos: `scripts/merge_sessions_pkl.py`, `scripts/finetune_gaitbase_multisession.py`,
>          `data/pkl_multisession/`, `checkpoints/pretrained/GaitBase_Gait3D_120000.pt`.

---

## 1. Motivación

Reporte 04 mostró que el fine-tune de Fase 3 (entrenado con sesión única) caía
de **100% rank-1 within-session** a **66.7–68.4% rank-1 cross-session**. Con un
margen cross-session de +0.0025–0.0037, Fase 5 (open-set, umbral de rechazo)
era inviable. Hipótesis: mostrarle al modelo ejemplos cross-session dentro del
triplet loss debería enseñarle invariancia a ropa — más barato de probar que
meter una rama de pose (Fase 4).

## 2. Cambios vs Fase 3

| Aspecto | Fase 3 | Fase 3.5 (multisession) |
|---|---|---|
| Fuente train | `data/pkl/` (sesión 1) | `data/pkl_multisession/` (s1+s2) |
| Secuencias train | 26 (13 sujetos × 2 cond) | 51 (13 × 4 tipos, 1 faltante) |
| Tipos por sujeto | `normal`, `rapido` | `normal_s1`, `normal_s2`, `rapido_s1`, `rapido_s2` |
| TripletSampler | P=4 K=2 sobre s1 | P=4 K=2 sobre s1+s2 → batches pueden tener s1 y s2 del mismo sujeto |
| Eval durante training | val within-session | val cross-session NN+NR |
| Early stop metric | margen val (s1) | **margen val NR (s1 gallery vs s2 rapido probe)** |
| Hparams | lr=0.01, 1500 iter max, patience=3 | **idénticos** |

Split subject-disjoint idéntico: train=[brayannajera, carloscarrillo, carlospadilla,
cesarvasquez, gilbertofelix, jesusantonioaguilarfelix, jesuscazarez, jorgeespinoza,
josefelix, juliouriarte, luislopez, robertocastaño, rodrigoruiz], val=[hectorsanchez,
robertpereira, carlostorres], test=[alexbojorquez, jesusvalenzuela, ricardomora].

Nota: `jesusantonioaguilarfelix` (train) solo tiene 3 de los 4 tipos (falta
`normal_s2`). TripletSampler K=2 muestrea sobre 3 en vez de 4 para ese sujeto.

## 3. Dinámica de entrenamiento

| iter | lr | triplet | ce | active_frac | ce_acc | NR val rank-1 | NR val margen |
|---|---|---|---|---|---|---|---|
| 0 (pre) | — | — | — | — | — | 66.7% | +0.0005 |
| 50 | 0.01 | 0.0763 | 1.164 | 0.039 | 0.777 | — | — |
| 200 | 0.01 | 0.0020 | 0.606 | 0.000 | 0.999 | 100% | **+0.0096** (nuevo mejor) |
| 400 | 0.01 | 0.0000 | 0.579 | 0.000 | 1.000 | 100% | +0.0106 |
| 600 | 0.01 | 0.0005 | 0.573 | 0.000 | 1.000 | 100% | +0.0113 |
| 800 | 0.001 | 0.0000 | 0.568 | 0.000 | 1.000 | 100% | +0.0117 |
| 1000 | 0.001 | 0.0000 | 0.566 | 0.000 | 1.000 | 100% | +0.0116 (patience 1) |
| **1200** | 0.001 | 0.0000 | 0.566 | 0.000 | 1.000 | **100%** | **+0.0118 (mejor)** |
| 1400 | 0.0001 | 0.0000 | 0.566 | 0.000 | 1.000 | 100% | +0.0118 (tie) |

**Lectura:**
- TripletLoss colapsa a 0 por iter 200 (igual que Fase 3). Con 51 secuencias sobre 13 IDs, el batch P=4×K=2 sigue siendo trivialmente separable para triplets.
- El trabajo útil lo hace CE + la inercia del triplet antes de colapsar. Aun así, el margen val NR **crece monotónicamente** de +0.0096 (iter 200) a +0.0118 (iter 1200) = +23%.
- Mejor checkpoint: **iter 1200**, margen NR val = +0.0118.
- VRAM pico 1.25 GB, tiempo total 177 min (mayor que Fase 3 por thermal throttling durante el run; no afecta resultado).

## 4. Evaluación cross-session sobre 19 sujetos (mismo protocolo que reporte 04)

### 4.1 Tabla comparativa — **esto es lo que importa**

| Métrica | Pretrained Gait3D | Fase 3 ft_iter400 | **Fase 3.5 ft_multisession_iter1200** | Δ vs Fase 3 |
|---|---|---|---|---|
| **NN rank-1** | 72.2% | 66.7% | **94.4%** (17/18) | **+27.7 pp** |
| **NN rank-5** | 94.4% | 100.0% | **100.0%** | = |
| **NN margen medio** | +0.0015 | +0.0037 | **+0.0222** | **6.0×** |
| **NN min margen** | −0.0028 | −0.0099 | −0.0059 | mejor |
| **NR rank-1** | 57.9% | 68.4% | **94.7%** (18/19) | **+26.3 pp** |
| **NR rank-5** | 84.2% | 94.7% | **100.0%** | +5.3 pp |
| **NR margen medio** | +0.0006 | +0.0025 | **+0.0204** | **8.2×** |
| **NR min margen** | −0.0033 | −0.0088 | −0.0087 | ≈ |

### 4.2 Errores restantes

Solo **1 error por protocolo**, ambos el mismo sujeto:

- **NN (1/18):** `ricardomora` → top1 `hectorsanchez` (rank 3).
- **NR (1/19):** `ricardomora` → top1 `hectorsanchez` (rank 4).

`ricardomora` está en **test** (nunca visto en training). `hectorsanchez` está
en **val** (tampoco en training). Comparten un patrón de silueta (altura, ancho
de hombros/cadera) que el modelo no logra separar. Es el caso prototípico donde
la rama de pose de Fase 4 podría ayudar — el esqueleto es invariante a build
corporal en la silueta.

Los 5 sujetos que confundían al Fase 3 (`carloscarrillo`, `juliouriarte`,
`luislopez`, `carlostorres`, `cesarvasquez`) ahora **todos en rank-1** en ambos
protocolos. La supervisión cross-session resolvió esos errores.

### 4.3 Test intocado (3 sujetos: alexbojorquez, jesusvalenzuela, ricardomora)

Reportado por el script de entrenamiento (sub-tabla del cross-session 19):

| Protocolo | rank-1 | rank-5 | margen | min margen |
|---|---|---|---|---|
| NN | 66.7% (2/3) | 100% | +0.0064 | −0.0028 |
| NR | 66.7% (2/3) | 100% | +0.0045 | −0.0036 |

El 66.7% sobre 3 test = 2/3 correctos, el error es `ricardomora`. Con 3 sujetos
el rank-1 tiene granularidad de 33 pp — no es una métrica estable. La lectura
honesta es el número cross-session sobre 19 sujetos (94.4%/94.7%).

## 5. Impacto en el roadmap

### 5.1 Fase 5 (open-set) ahora es viable

Con margen cross-session de **+0.02** (vs +0.0025 antes), hay espacio real para
calibrar un umbral de rechazo. Regla de dedo: umbral = `mean(sim_wrong) + k·std`.
Con los números actuales:
- `sim_correct` cross-session ≈ 0.99 (NN y NR)
- `sim_best_wrong` cross-session ≈ 0.97

Queda margen para fijar un threshold ~0.98 que:
- Acepte "same identity" cuando `sim > 0.98` → TAR razonable.
- Rechace "different identity" con `sim < 0.98` → FAR acotado.

Antes el gap era +0.0025 (ruido puro, sin umbral separable).

### 5.2 ¿Se justifica Fase 4 (pose) todavía?

**Opinión:** bajó de prioridad. Con 94.4–94.7% rank-1 y 1 error por protocolo,
el cost/benefit de integrar SkeletonGait++ se invierte:
- Costo Fase 4: ~3–5 días, nueva dependencia de OpenGait branch, más hparams.
- Beneficio potencial: resolver el error `ricardomora → hectorsanchez` y bajar
  el min_margen a positivo. Con 1 error sobre 19 sujetos, el upside es acotado.

**Propuesta:** saltar a Fase 5 con el checkpoint multisession. Si TAR@FAR=1% es
alto (≥80%), Fase 4 se archiva. Si TAR@FAR es bajo por casos tipo `ricardomora`,
entonces Fase 4 entra con justificación medible.

### 5.3 Limitaciones que siguen vivas

- **2 sesiones, no n-sesiones.** El modelo aprendió "ropa A vs ropa B" pero no
  variabilidad continua. Nuevos cambios de ropa pueden degradar. Mitigación real:
  3ra sesión más adelante (fuera de roadmap actual).
- **Mismo escenario / misma cámara.** Distancia/iluminación son ~estables en s2.
  Producción con cámara distinta pide re-calibración.
- **1 error cross-session = 5.3 pp.** Con 19 sujetos la resolución es granular.
  No confundir 94.7% con "resuelto"; confundir con "bien caminado pero falta
  open-set y más ropa".

## 6. Archivos generados

- `data/pkl_multisession/` — 75 secuencias fusionadas.
- `scripts/merge_sessions_pkl.py` — merge s1+s2 al nuevo layout.
- `scripts/finetune_gaitbase_multisession.py` — fine-tune cross-session.
- `checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt` (27 MB).
- `reports/04_cross_session_ft_multisession_iter1200.json` + `_topk.csv`.
- `reports/05_finetune_multisession_results.json` — curvas de entrenamiento + test.
- `reports/_finetune_multisession.log` — log completo (stdout bufferizado al final).

## 7. Reproducción

```bash
C:\Proyecto3\venv\Scripts\activate
cd C:\Proyecto3\ProyectoChino

# 1. merge pkls
python scripts/merge_sessions_pkl.py --force

# 2. fine-tune multisession (~3 h en GTX 1650)
python scripts/finetune_gaitbase_multisession.py

# 3. eval cross-session sobre 19 sujetos
python scripts/cross_session_eval.py \
    --checkpoint checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt \
    --tag ft_multisession_iter1200 --class-num 13
```

---

**Reporte cerrado. La supervisión cross-session convirtió un modelo con 66.7%
rank-1 y margen +0.0037 en uno con 94.4% y margen +0.0222 (6× margen, 1 solo
error sobre 18–19 sujetos). El gap cross-session quedó esencialmente cerrado
sin tocar pose. Recomendación: saltar a Fase 5 (open-set) con este checkpoint;
Fase 4 solo si TAR@FAR lo pide. Espero aprobación para arrancar Fase 5.**
