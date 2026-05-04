# Reporte 01 — Auditoría del Dataset Crudo

> **Fase 1** del roadmap (CLAUDE.md §6).
> Generado: 2026-04-21.
> Insumos: `scripts/audit_raw_dataset.py`, `reports/audit_dataset_raw.csv`, `reports/audit_frames/`.
> Estado: **A LA ESPERA DE APROBACIÓN** antes de avanzar a Fase 2.

---

## 1. Resumen ejecutivo

| Métrica | Valor |
|---|---|
| Sujetos | 19 |
| Videos | 38 (2 por sujeto: `normal` + `rapido`) |
| Resolución | 1920×1080, **uniforme** |
| FPS | 30.0, **uniforme** |
| Codec | `mpeg4` (MPEG-4 Part 2 / DivX-like), **NO H.264** |
| Duración por video | **5.000 s exactos**, **uniforme** |
| Frames por video | **150 exactos**, **uniforme** |
| Frames totales del dataset | **5,700** |
| Hashes únicos | 38/38 (sin duplicados) |
| Videos corruptos / no abren | 0 |
| Discrepancias FPS (cv vs ffmpeg) | 0 |
| Discrepancias nframes (header vs decodificado real) | 0 |

**Veredicto técnico:** dataset **íntegro a nivel de archivo** pero **insuficiente y con sesgos sistemáticos** que limitan severamente lo que un modelo puede aprender. Avanzar a Fase 2 es viable únicamente bajo el alcance "pase de lista para *este* aula" ya acotado en `docs/REQUISITOS.md §1`.

---

## 2. Anomalías y observaciones

### 2.1 Re-encoding previo (no es la salida directa de la C920)

- La C920 en su modo nativo entrega **H.264 a duración variable**.
- Lo que está en `Dataset Crudo/` es **`mpeg4` part 2 con duración exacta de 5.000 s y exactamente 150 frames en los 38 videos**.
- **Conclusión:** alguien recortó/transcodificó los videos antes de depositarlos. Hay al menos un paso de pérdida visual previo del que no tenemos control.
- **Implicación:** la calidad efectiva (especialmente para extracción de siluetas finas) no es la mejor posible. Para grabaciones futuras, registrar directamente el archivo original de la C920.

### 2.2 Convención de nombres con excepciones

Tres archivos no siguen `<condicion>_1.mp4` sino `_2.mp4` (toma 2):

- `cesarvasquez/normal_2.mp4`
- `jesusantonioaguilarfelix/rapido_2.mp4`
- `jorgeespinoza/rapido_2.mp4`

**Implicación:** ninguna funcional — el regex acepta cualquier sufijo numérico. Solo dejar constancia de que la primera toma fue descartada (no está el `_1` en el directorio).

### 2.3 Frames "útiles" reales < 150

Inspección visual de `audit_frames/<sujeto>/<condicion>_ini.jpg` muestra que en varios videos el sujeto **está parado al inicio** y empieza a caminar después (~1–1.5 s). El frame final, en algunos casos, lo captura saliendo del cuadro.

**Estimación honesta de frames con marcha activa por video:** ~90–110 (no 150).
**Frames con marcha activa totales:** ~3,500–4,200 (no 5,700).

Hay que medir esto con precisión en Fase 2 al correr la detección+tracking — recortar la secuencia útil con base en cuándo el bbox del sujeto entra y sale del área central del cuadro.

### 2.4 Vestimenta — corrección a `REQUISITOS.md §1`

`REQUISITOS.md §1` dice "Vestimenta: Misma para todas las grabaciones". **Esto NO es correcto.** La inspección de muestras (alex, héctor, jesús valenzuela, rodrigo) muestra **ropa distinta entre sujetos**: distintas camisetas, jeans, calzado. Lo uniforme es solo entre los 2 videos del mismo sujeto.

**Implicación crítica para gait recognition:**
- **Bueno:** evita que el modelo aprenda "el sujeto que viste rojo es Juan" en general (más realista que CASIA-B condición NM).
- **Malo:** dado que cada sujeto solo tiene 2 videos y los 2 con la misma ropa, **un modelo basado solo en silueta puede aprender la silueta-de-la-ropa como atajo de identidad**, no la dinámica de marcha. Riesgo concreto de fuga apariencia → identidad incluso con split subject-disjoint, porque la ropa lateral del sujeto domina la silueta.
- **Mitigación:** la rama de **pose/esqueleto** (Fase 4, SkeletonGait++ o GaitGraph2) es **mucho más necesaria** de lo que se anticipó en Fase 0. Recomiendo no quedarnos en GaitBase puro como sistema final.

**Acción sugerida:** corregir la celda "Vestimenta" en `REQUISITOS.md §1` a *"Misma vestimenta entre los 2 videos del mismo sujeto; distinta entre sujetos"* y agregar la implicación arriba a las "Limitaciones a aceptar explícitamente".

### 2.5 Una sola dirección de marcha

Los 4 sujetos inspeccionados caminan **siempre izq → der** (entran por la izquierda, salen por la derecha). No hay diversidad direccional.

**Implicación:** para gait simétrico, izq→der vs der→izq son equivalentes solo si el modelo es invariante a la reflexión horizontal. **OpenGait usa horizontal flip como augmentation por defecto**, lo cual cubre este caso. No es bloqueante, pero conviene explicitar el `flip_aug=True` en el config de Fase 3.

### 2.6 Tarima elevada y fondo idéntico

Los 4 sujetos caminan sobre **la misma tarima** con **el mismo fondo** (pizarra blanca + pantalla de proyección + pared crema). Esto:
- **Para silueta binaria:** no afecta si el segmentador (RVM/YOLO-seg en Fase 2) limpia bien el fondo. La silueta resultante no contiene contexto del aula.
- **Para keypoints (RTMPose):** no afecta — pose es invariante al fondo.
- **Para cualquier modelo que opere sobre RGB crudo:** sería catastrófico (memorización del aula). **No vamos a entrenar modelos RGB**, así que descartado.

### 2.7 Blur de movimiento

Visible en los frames `_mid` (sujeto en plena marcha). Causa probable: shutter speed bajo en la C920 con poca luz. Esto **degradará la calidad de las siluetas extraídas** (contornos imprecisos, especialmente extremidades). En Fase 1.5 / Fase 2 al validar el primer batch de siluetas con RVM mediremos el impacto.

---

## 3. Estadísticas por video

Todas las filas del CSV son idénticas en lo siguiente:

```
cv_open_ok           = True
cv_width × cv_height = 1920 × 1080
cv_fps               = 30.000
cv_nframes_header    = 150
cv_nframes_real      = 150
cv_duration_s        = 5.000
ff_codec             = mpeg4
ff_fps               = 30.000
ff_duration_s        = 5.000
ff_width × ff_height = 1920 × 1080
fps_disagreement     = False
nframes_disagreement = False
```

**No hay outliers.** Tamaños en disco oscilan entre **2.39 MB** (`jorgeespinoza/normal_1.mp4`) y **2.78 MB** (`robertpereira/normal_1.mp4`), variación atribuible al contenido (cantidad de movimiento → bitrate del codec). Ningún video sospechosamente pequeño (todos ≥ 2.39 MB).

Listado completo: `reports/audit_dataset_raw.csv`.

---

## 4. Inspección visual

114 JPEGs (3 por video × 38 videos) en `reports/audit_frames/<sujeto>/<stem>_{ini,mid,fin}.jpg`.

Muestras revisadas detalladamente: `alexbojorquez`, `hectorsanchez`, `jesusvalenzuela`, `rodrigoruiz` (cubren 4 de 19 sujetos = 21%). Recomiendo al usuario ojear las restantes para detectar:
- Sujetos que NO caminen sobre la tarima.
- Sujetos parcialmente fuera del cuadro.
- Oclusiones (otra persona, mochila grande, etc.).
- Vestimenta excepcional (vestido largo, abrigo, etc. que cambie radicalmente la silueta).

---

## 5. Frames útiles efectivos — proyección para Fase 2/3

Suponiendo **~100 frames útiles por video** (conservador, descontando warmup y salida del cuadro):

| Concepto | Cantidad |
|---|---|
| Frames útiles por video | ~100 |
| Frames útiles por sujeto (2 videos) | ~200 |
| Frames útiles totales | ~3,800 |
| Ventanas no superpuestas de 30 frames por video | ~3 |
| Ventanas con stride 15 (overlap 50%) | ~5 |
| Ventanas totales con stride 15 | ~190 |

Esto es **muy escaso** para entrenar gait recognition desde cero (lo cual ya descartamos en CLAUDE.md §3) y **marginal** para fine-tuning. **Refuerza la decisión** de partir de pesos preentrenados (`GaitBase_Gait3D_120000.pt`) y aplicar augmentations agresivas (flip, crop temporal, jitter de brillo).

---

## 6. Propuesta de splits

Con 19 sujetos y 2 videos por sujeto, hay dos splits **simultáneos** que necesitamos según qué evaluemos:

### 6.1 Split A — **subject-disjoint** (para validar el MODELO)

Mide capacidad del modelo de generalizar a identidades nunca vistas. Único split metodológicamente honesto para reportar rank-1, mAP, TAR@FAR.

| Partición | # sujetos | Sujetos (ejemplo, fijar con seed) |
|---|---|---|
| train | 13 | (a definir con seed=42 estratificado por nada — son todos iguales en condiciones) |
| val   | 3  | (a definir) |
| test  | 3  | (a definir) |

- **Por qué 13/3/3 y no 14/2/3:** con solo 3 sujetos en val es ya muy ruidoso (cada sujeto bien/mal mueve mucho la métrica). Con 2 sería peor. 13/3/3 es el mínimo razonable.
- **Implicación brutal:** **rank-1 con 3 identidades en gallery es trivial**. Para tener un número comparable con la literatura habría que reportar también **rank-1 dentro de N=3** y aceptar que es un upper bound optimista.
- **Recomendación honesta:** este split es **insuficiente para creerse números absolutos**. Sirve como *señal direccional* (¿GaitBase fine-tuneado mejora vs. zero-shot?), no como prueba de calidad del sistema.

### 6.2 Split B — **video-disjoint** (para validar el SISTEMA de pase de lista)

Para el caso de uso real: los 19 sujetos están en la lista, queremos saber si el sistema los reconoce.

| Rol | Asignación |
|---|---|
| gallery | 1 video por sujeto (p. ej. el `normal_*`) — 19 secuencias |
| probe   | el otro video por sujeto (p. ej. el `rapido_*`) — 19 secuencias |

- Esto es **exactamente** el escenario de uso: enrolar a la persona con un video y reconocerla con otro.
- **Limitación:** reportar accuracy aquí mide identificación cerrada con N=19 con la misma ropa entre gallery y probe (sesgo apariencia). Es la métrica más relevante para el deliverable, pero no se debe vender como gait recognition real.
- **Para open-set (Fase 5):** se necesitan probes "no enrolados". Como no tenemos sujetos extra en el dataset, hay 2 opciones: (a) hacer hold-out de 3 sujetos como "desconocidos" y usar los 16 restantes como pool, o (b) grabar 2-3 sujetos adicionales solo para probar rechazo. Decisión a tomar al inicio de Fase 5.

### 6.3 Decisión propuesta

**Mantener ambos splits.** Reportar siempre las dos métricas con etiqueta clara:
- `rank1_subj_disjoint_N3` — capacidad de generalización del modelo (limitado por N=3).
- `rank1_video_disjoint_N19` — funcionalidad real del sistema.

Asignación concreta de qué sujeto cae en train/val/test la haré al inicio de Fase 2 con seed fijo, usando hash determinístico del nombre del sujeto para que sea reproducible y no dependa del orden alfabético.

---

## 7. Acciones sugeridas antes de Fase 2

1. **[Documental]** Corregir `REQUISITOS.md §1` (vestimenta NO uniforme entre sujetos; sí entre videos del mismo sujeto).
2. **[Documental]** Añadir a `REQUISITOS.md §1` la observación del re-encoding previo (videos no son salida directa de la C920).
3. **[Recomendado, no bloqueante]** Considerar grabar 1–2 sujetos adicionales con **vestimenta distinta** a futuro, para usar en Fase 5 como pool de "desconocidos" del open-set.
4. **[Recomendado, no bloqueante]** Para cualquier grabación futura: usar la C920 con **shutter más rápido** (más luz o ISO más alto) para reducir blur, y grabar **bidireccional** (izq→der y der→izq).
5. **[Bloqueante para Fase 2]** Confirmar plan de extracción: detector (YOLO11n) + tracker (ByteTrack) + segmentador (RVM o YOLO11-seg) + pose (RTMPose). Esto ya está en `REQUISITOS.md §6` pero conviene revalidar antes de gastar tiempo de instalación/ajuste.

---

## 8. Riesgos identificados (para tener presentes en fases siguientes)

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Modelo aprende silueta-de-la-ropa, no marcha | **Alta** | **Alto** | Sumar rama de pose (Fase 4) sí o sí. Comparar silueta-only vs silueta+pose. |
| Frames útiles reales <100/video → ventanas insuficientes | Media | Alto | Tracking con bbox para recortar warmup; usar stride 15; grabar más videos en futuro. |
| Blur de movimiento degrada siluetas | Media | Medio | Validar con primeros 5 sujetos de RVM antes de procesar todo. |
| N=3 en test subject-disjoint da métricas ruidosas | Cierta | Medio | Reportar IC; complementar con K-fold de 6 folds (deja 3-3-13 rotando). |
| Open-set Fase 5 sin "desconocidos" reales | Cierta | Alto | Grabar 2-3 sujetos extra antes de Fase 5 o usar leave-one-subject-out como proxy. |

---

## 9. Próximos pasos (Fase 2 si apruebas)

1. Configurar pipeline de extracción: YOLO11n (detección) → ByteTrack (tracking) → RVM (silueta) → RTMPose (keypoints).
2. Recortar secuencias por bbox del sujeto (descartar frames sin sujeto en cuadro o donde aún esté parado).
3. Normalizar siluetas: centrado por bbox, resize 64×44 (formato OpenGait).
4. Normalizar keypoints: centrado en cadera, escala por altura.
5. Definir splits con seed fijo (subject-disjoint 13/3/3 + video-disjoint 19/19).
6. Empacar al formato pkl que espera OpenGait (`<sujeto>/<condicion>/<view>/silhouette.pkl`).
7. Validar pipeline end-to-end con 1 sujeto antes de procesar los 19.
8. Entregable: `reports/02_dataset_preparation.md` con estadísticas finales y ejemplos de siluetas/poses por sujeto.

---

**Espero tu aprobación o ajustes antes de avanzar a Fase 2.**
