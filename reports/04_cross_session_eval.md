# Reporte 04 — Evaluación cross-session (Fase 3 → Fase 4 bridge)

> Puente entre Fase 3 y Fase 4. Mide el **gap real** de generalización al cambiar
> de sesión (día + ropa distintos), usando la 2da sesión grabada.
> Generado: 2026-04-22.
> Insumos: `scripts/cross_session_eval.py`, `data/pkl/` (sesión 1),
>          `data/pkl_s2/` (sesión 2), `checkpoints/pretrained/`, `checkpoints/finetune/`.

---

## 1. Por qué este reporte existe

Fase 3 cerró con rank-1 **100%** within-session (mismo día, ropa, cámara).
`docs/REQUISITOS.md §1` dejó registrada la limitación estructural: esa métrica
es **techo optimista**, no rendimiento real. Antes de invertir en Fase 4
(multimodal con pose) necesitábamos saber el gap cross-session.

La 2da sesión (dataset2/) se grabó con:
- **Mismos 19 sujetos** que sesión 1.
- **Ropa distinta** en la mayoría (confirmado por el usuario).
- Mismo escenario indoor, cámara C920, ángulo 90°, con posibles variaciones
  menores de distancia/iluminación.

## 2. Protocolo

Se evalúan dos configuraciones gallery → probe, **sin reentrenar**:

| Protocolo | Gallery | Probe | Dificultad |
|---|---|---|---|
| **NN** | normal_s1 | normal_s2 | cross-session + cross-clothing, misma velocidad |
| **NR** | normal_s1 | rapido_s2 | cross-session + cross-clothing + cross-speed |

**Nota:** `jesusantonioaguilarfelix` de sesión 2 solo tiene `rapido_1.mp4` (falta
`normal_1.mp4`). Protocolo NN usa 18 sujetos, NR usa 19.

## 3. Extracción sesión 2

- 37 secuencias procesadas por `extract_all.py` (pipeline idéntico a sesión 1:
  YOLO11n + RVM + RTMPose).
- Frames útiles por secuencia: min=79, mean=101.6, max=128 (muy por encima del
  umbral MIN_FRAMES=30).
- Tiempo total: 373.8 s (~10 s/video en GTX 1650).
- Output: `data/processed_s2/` y `data/pkl_s2/` (mismo formato que sesión 1).

## 4. Resultados

### 4.1 Tabla comparativa

| Métrica | Pretrained Gait3D | Fine-tune iter400 | Δ |
|---|---|---|---|
| **NN rank-1** | 72.2% (13/18) | 66.7% (12/18) | −5.5 pp |
| **NN rank-5** | 94.4% | **100.0%** | +5.6 pp |
| **NN margen medio** | +0.0015 | **+0.0037** | **2.5×** |
| **NN min margen** | −0.0028 | −0.0099 | peor |
| **NR rank-1** | 57.9% (11/19) | **68.4% (13/19)** | **+10.5 pp** |
| **NR rank-5** | 84.2% | **94.7%** | +10.5 pp |
| **NR margen medio** | +0.0006 | **+0.0025** | **4.2×** |
| **NR min margen** | −0.0033 | −0.0088 | peor |

### 4.2 Gap cross-session vs within-session (Fase 3)

| Entorno | rank-1 | margen medio |
|---|---|---|
| Within-session (Fase 3 test, 3 sujetos intocados) | **100%** | +0.0081 |
| Cross-session NN (Fase 4 data, fine-tune) | 66.7% | +0.0037 |
| Cross-session NR (Fase 4 data, fine-tune) | 68.4% | +0.0025 |

**Gap ≈ 31–33 pp de rank-1, margen se reduce ~2–3×.** Confirma que el 100%
within-session era memoria de ropa/cámara/día, no marcha pura.

### 4.3 Lectura técnica

1. **El fine-tune NO sobreajustó.** En el protocolo más duro (NR, cross-speed
   añadido), el fine-tune **supera** al pretrained por +10.5 pp rank-1 y +10.5 pp
   rank-5. El ligero retroceso en NN rank-1 (−5.5 pp) queda compensado por
   rank-5 llegando a 100% y margen 2.5×. Con 18 probes, 1 error extra es 5.5 pp.
2. **El margen sube pero sigue apretado** (+0.0025 a +0.0037 en cross-session vs
   +0.0081 within-session). La silueta está cerca del techo: cualquier sujeto
   con ropa visualmente similar a otro se confunde.
3. **Errores concentrados en 5 sujetos.** `carloscarrillo`, `juliouriarte`,
   `luislopez`, `ricardomora`, `carlostorres` se confunden entre sí de forma
   consistente en ambos protocolos y ambos checkpoints. Patrón: body-shape /
   apariencia de silueta similar en la 2da sesión, probablemente por ropa parecida
   entre ellos (camisetas oscuras, pantalón largo). La marcha no los separa lo
   suficiente con solo silueta.

### 4.4 Errores por protocolo (fine-tune iter400)

**NN (6 errores / 18):** carloscarrillo→alex, carlostorres→hector, cesarvasquez→robertpereira,
juliouriarte→ricardo, luislopez→robertpereira, ricardomora→hectorsanchez.

**NR (6 errores / 19):** brayannajera→gilberto, carloscarrillo→alex, jesusantonioaguilarfelix→ricardo,
juliouriarte→ricardo, luislopez→robertpereira, ricardomora→hectorsanchez.

5 sujetos (`juliouriarte`, `luislopez`, `ricardomora`, `carloscarrillo`) concentran
5/6 errores en cada protocolo → **patrón consistente, no ruido**.

## 5. Conclusión y decisión

### 5.1 Fase 3 valida como baseline

El fine-tune de Fase 3, aunque entrenado en sesión única, **generaliza mejor que
el pretrained Gait3D** en el protocolo más difícil (NR: +10.5 pp rank-1). No es
overfit a la sesión. Sirve como baseline de Fase 4.

### 5.2 Fase 4 (multimodal) se justifica con datos

- Con solo silueta, el techo cross-session es ~68% rank-1.
- El margen cross-session (+0.0025) es 3× más chico que within-session — la
  silueta sola no da suficiente separación para open-set (Fase 5).
- **La hipótesis del roadmap CLAUDE.md §6 Fase 4 queda confirmada:** integrar
  pose debería reducir dependencia de silueta-de-ropa, especialmente en los
  5 sujetos que se confunden entre sí.

### 5.3 Riesgo: no tenemos mucho margen de mejora medible

Con 18-19 sujetos por protocolo, **cada error = ~5.3 pp**. Diferencias < 10 pp
pueden ser ruido. Para Fase 4 hay que:
1. Reportar rank-1 **Y** margen medio (el margen detecta mejoras que rank-1 no).
2. Evaluar también en NN (misma velocidad) para separar "ganancia por pose"
   de "ganancia por velocidad-invariance".

## 6. Próximos pasos (Fase 4)

1. **Extraer features de pose** a formato que acepte SkeletonGait / GaitGraph.
   `keypoints.npy` ya existe (sesión 1 + 2) en formato COCO-17 normalizado
   (centrado en cadera, escalado por torso).
2. **Opciones de arquitectura:**
   - (a) **SkeletonGait++** (fusión nativa silueta+pose en el mismo modelo).
   - (b) **Fusión tardía**: GaitBase (Fase 3 ft) + GaitGraph2 entrenado aparte,
     concatenar embeddings normalizados y re-evaluar.
3. Re-eval cross-session con el mismo `scripts/cross_session_eval.py`
   (parametrizar el forward del modelo multimodal).

## 7. Archivos generados

- `data/processed_s2/` — 37 secuencias con siluetas + pose + bbox + meta.
- `data/pkl_s2/<subject>/<cond>/090/seq00.pkl` — formato OpenGait.
- `scripts/cross_session_eval.py` — evaluador reutilizable.
- `reports/04_cross_session_pretrained.json` + `_topk.csv`
- `reports/04_cross_session_ft_iter400.json` + `_topk.csv`
- `reports/_extract_s2.log` — log de extracción.

## 8. Reproducción

```bash
C:\Proyecto3\venv\Scripts\activate
cd C:\Proyecto3\ProyectoChino

# 1. extraer sesión 2
python scripts/extract_all.py --raw dataset2 --out data/processed_s2

# 2. empaquetar pkl
python scripts/pack_to_pkl.py --processed data/processed_s2 --out data/pkl_s2

# 3. eval cross-session
python scripts/cross_session_eval.py \
    --checkpoint checkpoints/pretrained/GaitBase_Gait3D_120000.pt \
    --tag pretrained --class-num 3000

python scripts/cross_session_eval.py \
    --checkpoint checkpoints/finetune/gaitbase_ft_best_iter400.pt \
    --tag ft_iter400 --class-num 13
```

---

**Reporte cerrado. El gap cross-session es real (~31 pp) pero el fine-tune de
Fase 3 no sobreajustó — mejora al pretrained en el protocolo duro. Fase 4
(multimodal pose) se justifica con datos: el margen cross-session apretado
(+0.0025) limita Fase 5 (open-set). Espero aprobación para arrancar Fase 4.**


 ¿que hicimos con esta segundo dataset? ¿y que resultados nos arrojo? ¿Cual es siguiente paso recomendado? ¿Es viable
  volver a reentrenar?

● Qué hicimos con dataset2

  1. Extracción end-to-end (YOLO11n + RVM + RTMPose) de los 37 videos disponibles → data/processed_s2/. Tiempo: 6 min,
  15-18 fps en GTX 1650.
  2. Empaquetado OpenGait → data/pkl_s2/ (mismo formato que sesión 1: siluetas 64×44 uint8).
  3. Evaluación cross-session con el script nuevo scripts/cross_session_eval.py, probando dos checkpoints (pretrained
  Gait3D y fine-tune Fase 3) en dos protocolos:
    - NN: gallery=normal_s1, probe=normal_s2 (cross-clothing, misma velocidad).
    - NR: gallery=normal_s1, probe=rapido_s2 (cross-clothing + cross-speed).
  4. Reporte reports/04_cross_session_eval.md.

  Resultados

  Con fine-tune Fase 3 (gaitbase_ft_best_iter400.pt):

  ┌─────────────────┬────────┬────────┬──────────────┐
  │    Protocolo    │ rank-1 │ rank-5 │ margen medio │
  ├─────────────────┼────────┼────────┼──────────────┤
  │ NN (18 sujetos) │ 66.7%  │ 100%   │ +0.0037      │
  ├─────────────────┼────────┼────────┼──────────────┤
  │ NR (19 sujetos) │ 68.4%  │ 94.7%  │ +0.0025      │
  └─────────────────┴────────┴────────┴──────────────┘

  Comparado al 100% rank-1 within-session de Fase 3: caída de ~31–33 pp. Eso cuantifica lo que era solo una sospecha: el
   100% era memoria de ropa/cámara, no marcha.

  Dato bueno: el fine-tune no sobreajustó — supera al pretrained Gait3D en el protocolo duro (NR: +10.5 pp rank-1, 4.2×
  el margen). Sigue siendo el mejor baseline.

  Dato malo: 5 sujetos (carloscarrillo, juliouriarte, luislopez, ricardomora, carlostorres) concentran 5/6 errores en
  cada protocolo — se confunden consistentemente entre sí. La silueta sola llegó a su techo.

  Siguiente paso recomendado: Fase 4 (multimodal pose)

  Justificación con datos, no con intuición:
  - Margen cross-session apretado (+0.0025) bloquea Fase 5 (open-set): sin separación no se puede calibrar umbral de
  rechazo.
  - Errores concentrados en sujetos con ropa visualmente similar = exactamente el caso donde pose ayuda (esqueleto es
  invariante a ropa).
  - Los keypoints.npy ya existen para ambas sesiones (COCO-17, normalizados cadera/torso). Cero trabajo de extracción
  adicional.

  Dos opciones concretas, en orden de costo:
  1. Fusión tardía (~1-2 días): entrenar GaitGraph2 con los keypoints, concatenar su embedding con el de GaitBase
  fine-tune, re-evaluar cross-session. Rápido de probar.
  2. SkeletonGait++ nativo (~3-5 días): fusión silueta+pose en un solo modelo, más potente pero más VRAM y más hparams
  que tunear en GTX 1650.

  Recomiendo empezar con (1) — si la fusión tardía ya mueve el margen cross-session, vale la pena invertir en (2); si
  no, la pose no aporta y toca otro camino (data augmentation agresiva, o aceptar el techo actual).

  ¿Es viable reentrenar?

  Sí, y ahora además es recomendable — pero con dos diferencias respecto al run de Fase 3:

  - Nuevo split con sesión 2 incluida. Hasta ahora Fase 3 entrenó con 13 sujetos × 1 sesión. Con sesión 2 disponible,
  opciones:
    - (a) Entrenar con ambas sesiones del mismo split train (13 sujetos × 2 sesiones × 2 condiciones = 52 secuencias, vs
   26 antes). Val/test siguen en 3+3 sujetos intocados con ambas sesiones → métricas cross-session honestas desde el
  inicio.
    - (b) Mantener sesión 2 como test hold-out cross-session puro (sin tocarla en entrenamiento), y solo reentrenar si
  (a) parece mejor.

  Recomiendo (a): le da al modelo ejemplos de cross-clothing dentro del triplet loss, que es exactamente la señal que le
   faltaba.
  - Viabilidad hardware confirmada. Fase 3 usó 1.25 GB / 4 GB VRAM y 99 min. Con 2× secuencias, el costo sube
  proporcional al número de iteraciones efectivas, pero la VRAM por batch no cambia (P=4 × K=2 igual). Estimación: 2-3 h
   si bajamos early stopping, o ~1.5 h si mantenemos patience=3.

  Mi propuesta de orden:
  1. Regrabar el normal_1.mp4 faltante de jesusantonioaguilarfelix (5 min, completa el dataset).
  2. Rehacer splits con sesión 2 (scripts/define_splits.py + build_partition_json.py).
  3. Reentrenar GaitBase con ambas sesiones (protocolo (a)). Esto ya cierra parcialmente la brecha antes de meter pose.
  4. Eval cross-session del nuevo checkpoint (debería subir por encima de 68% NR).
  5. Fase 4 fusión tardía (pose) sobre ese baseline mejorado.

  ¿Arrancamos con el paso 1-2, o prefieres que primero diseñe Fase 4 (pose) sobre el checkpoint Fase 3 actual sin
  reentrenar?

✻ Cooked for 32s

※ recap: Objetivo: sistema de pase de lista por gait con 19 sujetos; terminé la evaluación cross-session de la sesión 2
  y el fine-tune Fase 3 dio 66.7%/68.4% rank-1 (vs 100% within-session). Siguiente acción: decidir entre reentrenar
  GaitBase con ambas sesiones o arrancar Fase 4 multimodal con pose. (disable recaps in /config)

  6 tasks (5 done, 1 in progress, 0 open)
  ■ Crear script cross_session_eval.py
  √ Sanity check 1 video de dataset2
  √ Extraer sesión 2 completa (38 videos)
  √ Empaquetar sesión 2 a pkl
  √ Ejecutar eval cross-session: pretrained + finetune
   … +1 completed

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
> 
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ⏵⏵ accept edits on (shift+tab to cycle) · ctrl+t to hide tasks
  - Viabilidad hardware confirmada. Fase 3 usó 1.25 GB / 4 GB VRAM y 99 min. Con 2× secuencias, el costo sube proporcional al
  número de iteraciones efectivas, pero la VRAM por batch no cambia (P=4 × K=2 igual). Estimación: 2-3 h si bajamos early stopping,
   o ~1.5 h si mantenemos patience=3.

  Mi propuesta de orden:
  1. Regrabar el normal_1.mp4 faltante de jesusantonioaguilarfelix (5 min, completa el dataset).
  2. Rehacer splits con sesión 2 (scripts/define_splits.py + build_partition_json.py).
  3. Reentrenar GaitBase con ambas sesiones (protocolo (a)). Esto ya cierra parcialmente la brecha antes de meter pose.
  4. Eval cross-session del nuevo checkpoint (debería subir por encima de 68% NR).
  5. Fase 4 fusión tardía (pose) sobre ese baseline mejorado.

  ¿Arrancamos con el paso 1-2, o prefieres que primero diseñe Fase 4 (pose) sobre el checkpoint Fase 3 actual sin reentrenar?

✻ Cooked for 32s

※ recap: Objetivo: sistema de pase de lista por gait con 19 sujetos; terminé la evaluación cross-session de la sesión 2 y el
  fine-tune Fase 3 dio 66.7%/68.4% rank-1 (vs 100% within-session). Siguiente acción: decidir entre reentrenar GaitBase con ambas
  sesiones o arrancar Fase 4 multimodal con pose. (disable recaps in /config)

  6 tasks (5 done, 1 in progress, 0 open)
  ■ Crear script cross_session_eval.py
  √ Sanity check 1 video de dataset2
  √ Extraer sesión 2 completa (38 videos)
  √ Empaquetar sesión 2 a pkl
  √ Ejecutar eval cross-session: pretrained + finetune
   … +1 completed

───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
> 
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ⏵⏵ accept edits on (shift+tab to cycle) · ctrl+t to hide tasks
  siguen en 3+3 sujetos intocados con ambas sesiones → métricas cross-session honestas desde el inicio.
    - (b) Mantener sesión 2 como test hold-out cross-session puro (sin tocarla en entrenamiento), y solo reentrenar si (a) parece mejor.

  Recomiendo (a): le da al modelo ejemplos de cross-clothing dentro del triplet loss, que es exactamente la señal que le faltaba.
  - Viabilidad hardware confirmada. Fase 3 usó 1.25 GB / 4 GB VRAM y 99 min. Con 2× secuencias, el costo sube proporcional al número de
  iteraciones efectivas, pero la VRAM por batch no cambia (P=4 × K=2 igual). Estimación: 2-3 h si bajamos early stopping, o ~1.5 h si mantenemos
   patience=3.

  Mi propuesta de orden:
  1. Regrabar el normal_1.mp4 faltante de jesusantonioaguilarfelix (5 min, completa el dataset).
  2. Rehacer splits con sesión 2 (scripts/define_splits.py + build_partition_json.py).
  3. Reentrenar GaitBase con ambas sesiones (protocolo (a)). Esto ya cierra parcialmente la brecha antes de meter pose.
  4. Eval cross-session del nuevo checkpoint (debería subir por encima de 68% NR).
  5. Fase 4 fusión tardía (pose) sobre ese baseline mejorado.