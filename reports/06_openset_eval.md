# Reporte 06 — Open-set detection (Fase 5)

> Evaluación open-set sobre el checkpoint de Fase 3.5 (multisession, iter 1200).
> Mide la capacidad del sistema de **aceptar** legítimos y **rechazar** desconocidos
> con un único umbral τ sobre similitud coseno.
>
> Generado: 2026-04-22.
> Insumos: `scripts/openset_eval.py`, `data/pkl/`, `data/pkl_s2/`,
>          `checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt`,
>          baseline `checkpoints/pretrained/GaitBase_Gait3D_120000.pt`.

---

## 1. Motivación

Hasta Fase 3.5 el sistema operaba **closed-set**: dado un probe, devolvía el
sujeto más parecido de la galería, asumiendo que siempre había match. En el
escenario real de pase de lista puede entrar una persona **no enrolada**
(visita, intruso). El sistema debe poder decir "no lo conozco" en vez de
forzar una identidad.

Fase 5 define y evalúa un umbral τ tal que:
- `sim_top1 ≥ τ AND top1 == identidad correcta` → **aceptar** (TAR).
- `sim_top1 < τ` → **rechazar** (correcto si el probe es desconocido, incorrecto
  si era legítimo).

Criterio de salida que fijé en reporte 05: **TAR@FAR≤1% ≥ 80%**.

## 2. Protocolo — Leave-One-Subject-Out (LOSO)

19 sujetos. Para cada probe con identidad `i` del set cross-session:

| | Gallery | Etiqueta esperada |
|---|---|---|
| **Genuine** | 19 sujetos s1_normal | top1=`i` AND sim_top1 ≥ τ → aceptar |
| **Impostor** | 19 sujetos **sin `i`** | sim_top1 < τ → rechazar (correcto) |

El mismo probe se usa dos veces: como legítimo (contra gallery completa) y
como impostor sintético (contra gallery sin su identidad). Este diseño
maximiza el uso de los 19 sujetos sin inventar datos.

Falta `jesusantonioaguilarfelix/normal_s2` (ya reportado). Totales:
- Genuines: 18 (NN) + 19 (NR) = **37**.
- Impostores: 18 (NN) + 19 (NR) = **37**.

Sweep de τ con 2001 puntos. Operating points reportados: EER, τ@FAR=0%,
τ@FAR≤1%, τ@FAR≤5%.

## 3. Resultados — Comparativa

### 3.1 Tabla principal (protocolo ALL = NN + NR combinados)

| Métrica | Pretrained Gait3D | **ft_multisession_iter1200** |
|---|---|---|
| **EER** | 48.65% | **8.11%** |
| **TAR @ FAR=0%** | 16.22% | **86.49%** |
| **TAR @ FAR≤1%** | 16.22% | **86.49%** |
| **TAR @ FAR≤5%** | 18.92% | **86.49%** |
| gap_means (genuine - impostor) | +0.0016 | **+0.0217** |
| genuine sim mean | 0.9907 | 0.9883 |
| impostor sim mean | 0.9890 | 0.9666 |
| genuine std | 0.0024 | 0.0054 |
| impostor std | 0.0024 | 0.0113 |

### 3.2 Tabla por protocolo (multisession)

| Protocolo | n_gen | n_imp | EER | τ_EER | TAR @ FAR=0% | τ @ FAR=0% | Rank-1 top1 correcto |
|---|---|---|---|---|---|---|---|
| **NN** (normal s2) | 18 | 18 | 5.56% | 0.9784 | 88.89% | 0.9795 | 94.44% |
| **NR** (rápido s2) | 19 | 19 | 10.53% | 0.9789 | 84.21% | 0.9807 | 94.74% |
| **ALL** | 37 | 37 | 8.11% | 0.9789 | 86.49% | 0.9807 | 94.59% |

**Lectura:**
- El pretrained es inservible en open-set: distribución genuina e impostor
  casi solapadas (std 0.0024, gap 0.0016 — ruido).
- La supervisión multisession de Fase 3.5 separó las distribuciones: gap
  creció **13×** (de 0.0016 a 0.0217).
- EER bajó de 48.65% a 8.11% (6×), y el TAR@FAR≤1% saltó de 16% a **86.49%**.
- **Cumple el criterio de salida ≥80%.**

## 4. Casos críticos

### 4.1 Impostores más peligrosos (sim_top1 más alta del multisession)

| Probe | Tipo | top1 match | sim | Comentario |
|---|---|---|---|---|
| jesusvalenzuela | NR | alexbojorquez | 0.9807 | ambos en test, build similar |
| luislopez | NR | robertpereira | 0.9801 | train vs val |
| alexbojorquez | NN | carlostorres | 0.9795 | test vs val |
| alexbojorquez | NR | carlostorres | 0.9788 | test vs val (mismo patrón) |
| luislopez | NN | robertpereira | 0.9784 | train vs val |

El par `alexbojorquez ↔ carlostorres` vuelve a aparecer (ya había salido en
reporte 04 como confusión del pretrained). El modelo multisession los
distingue en closed-set pero apenas — sim 0.979 es alta.

### 4.2 Genuinos con sim_top1 más baja

| Probe | Tipo | top1 match | sim | Estado |
|---|---|---|---|---|
| carlostorres | NR | carlostorres | 0.9740 | **OK pero rechazado** (τ=0.9807) |
| ricardomora | NN | hectorsanchez | 0.9767 | **ERR** (ya conocido) |
| ricardomora | NR | hectorsanchez | 0.9769 | **ERR** (ya conocido) |
| hectorsanchez | NR | hectorsanchez | 0.9789 | **OK pero rechazado** (τ=0.9807) |
| carlostorres | NN | carlostorres | 0.9792 | OK pero rechazado en NR τ; aceptado en NN τ=0.9795 |

### 4.3 Desglose de los 5 "no aceptados" en τ@FAR=0% (protocolo ALL, 37 genuines)

- **2 errores reales**: `ricardomora` (NN y NR) → mal clasificado como
  `hectorsanchez`. El modelo se equivoca y además con sim alta; no lo salva
  ningún τ.
- **3 rechazos de legítimos**: `carlostorres` (NN y NR) y `hectorsanchez` (NR).
  El top1 era correcto pero sim < τ=0.9807.

O sea: **los 2 errores en closed-set de Fase 3.5 siguen vivos aquí**. El
costo adicional de open-set son 3 rechazos legítimos por un τ conservador.

## 5. Recomendación de τ para producción

### 5.1 Dos operating points viables

| Perfil | τ recomendado | TAR | FAR esperado | Cuándo usar |
|---|---|---|---|---|
| **Conservador (FAR=0%)** | **0.9807** | 86.49% | 0% en este set | Alta seguridad, baja tolerancia a intrusos |
| Balanceado (EER) | 0.9789 | ≈92% | ≈8% | Protoproducción, ajustar con más datos |

Con 19 sujetos la granularidad de FAR es 1/37 ≈ 2.7% — el número de 0% es
frágil, cambiaría con más impostores. En producción con 50+ personas habrá
que re-calibrar.

### 5.2 Política sugerida para pase de lista

```
sim_top1 >= 0.9807  ->  presente (identidad = top1)
sim_top1 <  0.9807  ->  desconocido (alerta al operador + re-intento)
```

Con re-intento a los 5s (otra pasada frente a la cámara) se puede promediar
2 medidas y subir TAR sin bajar τ. Eso se define en Fase 6.

## 6. Impacto en el roadmap

### 6.1 Fase 5 cerrada ✅

- TAR@FAR≤1% = 86.49% > 80% objetivo.
- EER = 8.11% aceptable para 19 sujetos con sólo 2 sesiones.
- Umbral separable: 0.9807 con gap real (+0.022 entre medias).

### 6.2 ¿Fase 4 (multimodal pose)?

**Propuesta:** **archivar Fase 4** salvo que Fase 6 (integración) muestre
que `ricardomora → hectorsanchez` degrada la experiencia real. Razones:
- El error restante es el mismo de Fase 3.5 (1 sujeto sobre 19). Fase 4
  costaría 3–5 días para atacar un caso puntual.
- TAR@FAR=0% de 86.49% cumple criterio sin pose.
- Si en producción aparecen más pares `build similar + ropa similar`, Fase 4
  entra con justificación medible (métrica de producción, no supuesto).

### 6.3 Siguiente: Fase 6 — Integración del pipeline

Lo que falta para un pase de lista usable:
1. **Captura → inferencia end-to-end**: video → YOLO11n + RVM + RTMPose →
   pkl 64×44 → embedding → match contra gallery → decisión con τ=0.9807.
2. **Galería persistente**: archivo (pickle o SQLite) con {subject →
   embedding s1_normal}. Refresco periódico.
3. **UI mínima**: mostrar lista de asistentes con estado
   (presente/ausente/desconocido) y confianza. Puede ser Streamlit o
   script CLI con salida a CSV.
4. **Logging**: cada decisión con timestamp, subject, sim, aceptado/rechazado,
   para auditoría y re-calibración.

Estimado: 3–5 días para MVP funcional.

### 6.4 Limitaciones que siguen vivas

- **19 sujetos**: granularidad FAR = 2.7%. Números estables requieren 50+.
- **2 sesiones**: el modelo conoce 2 "universos de ropa". Persona nueva con
  ropa muy distinta puede degradar hasta 3ra sesión.
- **Misma cámara/escenario**: cambio de cámara pide recalibración de τ.
- **τ es frágil**: los impostores peores (0.9807) están justo en el borde.
  Un outlier nuevo podría empujarlo.

## 7. Archivos generados

- `scripts/openset_eval.py` — eval parametrizable.
- `reports/06_openset_ft_multisession_iter1200.json` — resultados principales.
- `reports/06_openset_ft_multisession_iter1200_pairs.csv` — 74 pares
  (37 genuine + 37 impostor).
- `reports/06_openset_ft_multisession_iter1200_roc.png` — curva ROC.
- `reports/06_openset_pretrained.json` + `_pairs.csv` + `_roc.png` —
  baseline Gait3D.

## 8. Reproducción

```bash
C:\Proyecto3\venv\Scripts\activate
cd C:\Proyecto3\ProyectoChino

# Eval con checkpoint multisession (recomendado)
python scripts/openset_eval.py \
    --checkpoint checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt \
    --tag ft_multisession_iter1200 --class-num 13

# Baseline (sanity check)
python scripts/openset_eval.py \
    --checkpoint checkpoints/pretrained/GaitBase_Gait3D_120000.pt \
    --tag pretrained --class-num 3000
```

---

**Reporte cerrado. El fine-tune multisession no sólo cerró el gap cross-session
(Fase 3.5) — también generó separación suficiente para operar en open-set con
τ=0.9807: TAR=86.49% @ FAR=0% en este set de 37 probes, EER=8.11%. Criterio
de salida superado (objetivo ≥80%). Fase 4 (pose) se archiva por ahora;
el siguiente paso recomendado es Fase 6 (integración del pipeline de pase
de lista). Espero aprobación.**
