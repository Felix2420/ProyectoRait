# Presentación de Clase — Sistema de Reconocimiento de Marcha
**Equipo:** 4 personas | **Tiempo estimado:** ~20 min + Q&A  
**Fecha:** 2026-05-06

---

## DISTRIBUCIÓN POR PERSONA

| Persona | Sección | Tiempo |
|---|---|---|
| **P1** | Problema, motivación, dataset y cómo dividimos el problema | ~5 min |
| **P2** | Estado del arte, modelos evaluados, arquitecturas, por qué descartamos keypoints | ~6 min |
| **P3** | Entrenamiento, hiperparámetros, resultados, calibración del umbral | ~5 min |
| **P4** | Pipeline en vivo, componentes, demo, limitaciones y trabajo futuro | ~5 min |

---

---

# PERSONA 1 — Problema, Dataset y División del Problema

## Slides sugeridos

### Slide 1: ¿Qué problema resolvemos?
- Pase de lista manual → lento, propenso a errores, suplantación posible
- **Propuesta:** identificar personas por su forma de caminar (marcha), sin interacción, con cámara fija
- Biometría de marcha: **no intrusiva**, funciona a distancia, difícil de falsificar
- Aplicación concreta: sala de clases, entrada a edificio

### Slide 2: ¿Por qué marcha y no cara?
- Reconocimiento facial requiere buena iluminación y que la persona mire a la cámara
- La marcha funciona de lejos, de espaldas, con cubrebocas, con mochila
- Desventajas honestas: sensible al ángulo de cámara, a la ropa, a lesiones temporales
- **Decisión de diseño:** cámara lateral fija a 90°, un sujeto a la vez

### Slide 3: Nuestro dataset
- **19 sujetos** grabados con cámara Logitech C920 (1080p, ~30 fps)
- **2 sesiones por sujeto:** s1 (paso normal), s2 (paso rápido)
- Vista: **90° lateral**, escenario indoor controlado
- Resultado: ~120 secuencias en total (~4 tipos × 2 sesiones × varios sujetos)
- **Restricción importante:** no podemos re-grabar, dataset fijo

### Slide 4: Cómo dividimos el problema (las 7 fases)
```
Fase 0: Auditoría del dataset (¿qué tenemos exactamente?)
Fase 1: Extracción de siluetas + keypoints
Fase 2: Empaquetado en formato OpenGait (PKL)
Fase 3: Fine-tuning del modelo base (GaitBase)
Fase 4: Intento multimodal (silueta + pose) → descartado
Fase 5: Evaluación open-set y calibración del umbral τ
Fase 6: Pipeline en vivo con cámara C920
```
- Cada fase produce entregables concretos (scripts, checkpoints, reportes)
- Split inmutable desde el inicio: 13 train / 3 val / 3 test (seed=42, por sujeto)

### Slide 5: Dataset split (subject-disjoint)
- **Train (13):** jorgeespinoza, gilbertofelix, jesusantonioaguilarfelix, josefelix, carloscarrillo, juliouriarte, jesuscazarez, robertocastaño, luislopez, brayannajera, cesarvasquez, carlospadilla, rodrigoruiz
- **Val (3):** hectorsanchez, robertpereira, carlostorres
- **Test (3):** jesusvalenzuela, alexbojorquez, ricardomora
- **Subject-disjoint** = el modelo nunca ve en entrenamiento a las personas de prueba
- Garantiza que no estamos midiendo memorización

## Guión (lo que dices)

> "El problema que atacamos es automatizar el pase de lista usando reconocimiento de marcha. En lugar de que alguien pase lista manualmente, una cámara fija identifica a las personas por cómo caminan, sin que tengan que hacer nada.
>
> ¿Por qué marcha y no cara? Porque funciona de lejos, sin cooperación del sujeto, con cubrebocas o de espaldas. La desventaja es que es sensible al ángulo de cámara y a la ropa — eso lo vamos a ver cuando discutamos las limitaciones.
>
> Nuestro dataset lo grabamos nosotros: 19 compañeros, cámara a 90° lateral, dos sesiones cada uno — una caminando normal y otra más rápido. Esto nos da variabilidad de ropa y velocidad. No podemos re-grabarlo, así que trabajamos con lo que hay.
>
> Para no abordar todo de golpe, dividimos el proyecto en 7 fases. Cada fase tiene entregables concretos. Les voy a pasar a mis compañeros para que expliquen las fases técnicas más profundas."

---

---

# PERSONA 2 — Estado del Arte, Modelos y Por Qué Descartamos Keypoints

## Slides sugeridos

### Slide 6: Estado del arte — Reconocimiento de marcha

**Dos enfoques principales en la literatura:**

| Enfoque | Entrada | Ejemplo | Ventaja | Desventaja |
|---|---|---|---|---|
| **Apariencia** (silhouette-based) | Silueta binaria | GaitSet, GaitPart, GaitGL, GaitBase | Simple, robusto, rápido | Sensible a ropa, oclusión |
| **Modelo** (pose-based) | Keypoints del esqueleto | GaitGraph, SkeletonGait, GPGait | Invariante a ropa/fondo | Keypoints pueden fallar |

- **Tendencia 2023-2025:** métodos multimodales que combinan ambos (SkeletonGait++, ZipGait, PSGait)
- OpenGait (CVPR 2023, Highlight Paper): framework de benchmarking y **GaitBase** como baseline sólido

### Slide 7: Benchmarks públicos de referencia

| Dataset | Sujetos | Condiciones | Mejor rank-1 |
|---|---|---|---|
| CASIA-B | 124 | 3 tipos de ropa, 11 ángulos | >95% |
| Gait3D | 4,000 | In-the-wild, múltiples cámaras | ~75% (GaitBase) |
| GREW | 26,345 | Exterior, no controlado | ~60-70% |
| **Nuestro dataset** | **19** | **1 ángulo, 2 sesiones** | **94.4% rank-1** |

- Nuestro dataset es pequeño pero controlado → rank-1 alto no es sorpresa
- El reto real es el escenario **open-set** (rechazar desconocidos)

### Slide 8: ¿Qué es GaitBase y por qué lo elegimos?

**GaitBase (Fan et al., CVPR 2023 — OpenGait):**
- Baseline "simple pero poderoso" del framework OpenGait
- Preentrenado en **Gait3D** (4,000 sujetos, in-the-wild)
- Arquitectura probada, código público, bien documentada

**Arquitectura GaitBase:**
```
Entrada: secuencia de siluetas (T × 64 × 44, binarias)
  ↓ SetBlockWrapper (aplica ResNet9 frame a frame)
  ↓ ResNet9: [64→128→256→512], strides [1,2,2,1]
  ↓ TemporalPooling: max sobre tiempo → (512, H', W')
  ↓ HorizontalPoolingPyramid: divide en 16 bandas horizontales
  ↓ SeparateFCs: 512 → 256 por cada banda
  ↓ BNNeck: BatchNorm bottleneck (class_num=13)
Salida: embedding 256×16 = 4096 dims, L2-normalizado → cosine similarity
```

**¿Por qué ResNet9 y no ResNet50?**
- 4 capas: [1,1,1,1] — suficiente para siluetas 64×44 simples
- VRAM: cabe en GTX 1650 (4 GB) sin sacrificar calidad

### Slide 9: Modelos alternativos evaluados (y por qué los descartamos)

| Modelo | Tipo | Resultado en nuestro dataset | Decisión |
|---|---|---|---|
| GaitBase pretrained (sin fine-tune) | Silueta | EER 48.65%, TAR 16% | ❌ No generaliza |
| GaitBase fine-tune s1 solo | Silueta | Rank-1 OK, cross-session degrada | ❌ Overfitting a sesión 1 |
| **GaitBase fine-tune s1+s2** | Silueta | EER 8.11%, TAR@FAR=0% 86.49% | ✅ **PRODUCCIÓN** |
| SkeletonGait++ (SGPP) | Silueta + Pose | α*=1.0 (silueta sola gana) | ❌ Pose no aporta |
| ST-GCN Lite | Solo Pose | α*=1.0 (silueta sola gana) | ❌ Pose no aporta |

### Slide 10: Por qué los keypoints (pose) NO funcionaron

**Cuatro razones técnicas con evidencia:**

1. **Ángulo lateral 90° limita la información de pose**
   - A 90°, brazos y piernas contralaterales se solapan en la imagen
   - Los keypoints de RTMPose ven muy poca diferencia entre sujetos a este ángulo
   - La silueta binaria ya captura prácticamente toda la dinámica visible

2. **Dataset demasiado pequeño para entrenar una rama de pose**
   - Solo ~51 secuencias de entrenamiento (13 sujetos × ~4 tipos)
   - La red de pose no tiene suficientes ejemplos para generalizar
   - Aprende "ruido de heatmap" como firma de identidad → sobreajuste

3. **Solo 2 sesiones no dan variabilidad suficiente**
   - El modelo de pose aprende la diferencia s1 vs s2 (velocidad)
   - No aprende variabilidad real de marcha de la persona
   - Multimodal fijó un caso (ricardomora) pero rompió otros → trade-off negativo

4. **Evidencia cuantitativa directa**
   - Al optimizar α en la fusión tardía (silueta × α + pose × (1−α))
   - Resultado: α*=1.0 en todos los experimentos
   - Interpretación: el optimizador descarta pose completamente

**Conclusión:** Pose se sigue calculando en runtime, pero solo para quality gating (descartar siluetas mal extraídas). No entra en el score de identidad.

## Guión (lo que dices)

> "En la literatura, hay dos grandes familias de métodos para reconocimiento de marcha: los basados en silueta, que usan la silueta binaria de la persona, y los basados en pose, que usan los keypoints del esqueleto. La tendencia reciente es combinar ambos en modelos multimodales.
>
> Nosotros elegimos GaitBase como modelo base. Es el baseline del framework OpenGait, publicado en CVPR 2023. No lo elegimos al azar — está preentrenado en 4,000 sujetos en entornos no controlados, tiene código público y cabe en nuestra GPU de 4 GB.
>
> Probamos tres variantes. La versión pretrained sin fine-tune falló completamente — EER de 48%, prácticamente random. El fine-tune con solo sesión 1 funcionó en cerrado pero se degradaba cross-session. El fine-tune con s1+s2 juntas fue la que funcionó: EER 8.11%, 86.49% de identificación sin falsos positivos.
>
> También probamos dos modelos basados en pose: SkeletonGait++ y ST-GCN Lite. Los dos fallaron — cuando optimizamos el peso α de la fusión, el algoritmo encontró α=1.0, es decir, silueta sola gana. No es que la implementación esté mal. Es que a 90° lateral, los keypoints no aportan dimensionalidad nueva que la silueta no tenga, y con solo 51 secuencias de entrenamiento, la red de pose no tiene datos para generalizar."

---

---

# PERSONA 3 — Entrenamiento, Hiperparámetros y Métricas

## Slides sugeridos

### Slide 11: Estrategia de entrenamiento

**Problema a resolver:** El modelo base (Gait3D) no reconoce a nuestros 19 sujetos. Hay que adaptarlo.

**Fine-tuning strategy:**
- Inicializar desde `GaitBase_Gait3D_120000.pt` (pretrained, ~27 MB descargado)
- Congelar capas bajas, afinar capas altas + BNNeck
- Cambiar `class_num` de 4000 → 13 (nuestros sujetos de train)
- Entrenar con **s1 + s2** combinadas (cross-session desde el inicio)

**¿Por qué cross-session desde el inicio?**
- Si entrenamos solo con s1 y validamos en s2 → el modelo aprende "ropa A"
- Al mezclar sesiones en train → el modelo aprende la dinámica de marcha, no la ropa

### Slide 12: Hiperparámetros de entrenamiento

| Parámetro | Valor | Razón |
|---|---|---|
| Optimizador | SGD | Estándar en OpenGait, más estable que Adam con triplet |
| Learning Rate | 0.01 | Mismo que configuración original GaitBase |
| Momentum | 0.9 | Estándar |
| Weight Decay | 5×10⁻⁴ | Regularización, dataset pequeño |
| Batch (P×K) | P=4, K=2 | 4 identidades × 2 secuencias = 8 muestras/iter |
| Frames por clip | 30 | Ventana aleatoria en cada iteración |
| Total iteraciones | 1500 (early stop a 1200) | Mejor val sin sobreajuste |
| LR Milestones | [750, 1250] | Decay ×0.1 en cada milestone |
| Triplet Margin | 0.2 | Margen para TripletLoss |
| CE Scale / Smooth | 16 / 0.1 | Label smoothing para dataset pequeño |
| AMP (float16) | Activado | Necesario para GTX 1650 4 GB |

**Función de pérdida combinada:**
```
Loss = TripletLoss(margin=0.2) + CrossEntropyLoss(scale=16, smooth=0.1)
```

### Slide 13: Resultados — Closed-set (Fase 3.5)

**Protocolo: Leave-One-Session-Out (LOSO)**
- Train: s1+s2 de 13 sujetos
- Test: evaluación en 19 sujetos (NN = normal→normal, NR = normal→rápido)

| Condición | Rank-1 | Rank-5 |
|---|---|---|
| NN (normal → normal) | **94.4%** (17/18) | 100% |
| NR (normal → rápido) | **94.7%** (18/19) | 100% |
| Promedio | **94.5%** | 100% |

- Error único consistente: `ricardomora` ↔ `hectorsanchez` (mismo build corporal, ambas condiciones)
- Checkpoint final: **iter 1200** (best val, early stop activado a patience=3)
- VRAM pico: 1.25 GB | Tiempo total de entrenamiento: ~177 min en GTX 1650

### Slide 14: Problema open-set y calibración de τ

**Closed-set vs Open-set:**
- Closed-set: siempre hay una respuesta (¿quién de los N es más parecido?)
- **Open-set:** puede que la persona NO esté en el sistema → hay que rechazarla

**Protocolo de evaluación:**
- Pares genuinos: misma persona, distinta sesión (37 pares de test)
- Pares impostores: personas distintas (todos vs todos, test set)
- Métrica: curva DET → encontrar τ donde FAR=0% (sin falsos positivos)

**Resultados open-set:**

| Métrica | Valor |
|---|---|
| EER (Equal Error Rate) | **8.11%** |
| TAR @ FAR=0% | **86.49%** (32/37 genuinos aceptados) |
| Umbral τ | **0.9807** |
| Similitud media genuina | 0.9883 |
| Similitud media impostor | 0.9666 |
| Gap entre medias | +0.0217 |

**¿Por qué FAR=0% como restricción?**
- Preferimos no identificar a alguien (falso negativo) que identificar a un desconocido como alguien que no es (falso positivo)
- Un sistema de asistencia con FP identificaría a personas ausentes como presentes → inaceptable

### Slide 15: Cómo funciona el embedding y la galería

**Galería multi-template:**
- Para cada sujeto: K≤4 secuencias de referencia → K embeddings (256×16)
- Al consultar: calcular similitud coseno con cada template y tomar el máximo
- Ventaja sobre promedio: captura variabilidad intra-sujeto (s1 vs s2)

```
gallery/embeddings.npy  → (19, K_max, 256, 16)
gallery/valid_mask.npy  → (19, K_max) bool
gallery/index.json      → metadata (subject_id, session, etc.)
```

## Guión (lo que dices)

> "El fine-tuning parte del modelo preentrenado en Gait3D — 4,000 sujetos en condiciones no controladas. Eso nos da una base mucho mejor que entrenar desde cero con solo 13 sujetos.
>
> La decisión más importante en el entrenamiento fue mezclar las dos sesiones desde el inicio. Si entrenamos con s1 solo, el modelo aprende a distinguir 'ropa A de s1' de 'ropa B de s1', no la marcha real. Al mezclar s1 y s2, forzamos al modelo a aprender lo que no cambia entre sesiones: la dinámica de la marcha.
>
> Los hiperparámetros más relevantes: SGD con lr=0.01, batch de 4 identidades × 2 secuencias, early stop en iteración 1200. La pérdida combina TripletLoss para que los embeddings de la misma persona estén cerca, y CrossEntropy para que el modelo distinga clases.
>
> En closed-set obtuvimos 94.4% rank-1. El error consistente es ricardomora contra hectorsanchez — mismo build corporal, prácticamente indistinguibles a 90° lateral con silueta sola.
>
> Para open-set, calibramos el umbral τ=0.9807 con la restricción de FAR=0%. Preferimos no reconocer a alguien que reconocer a un impostor — en un sistema de asistencia, un falso positivo significa marcar presente a alguien que no vino."

---

---

# PERSONA 4 — Pipeline en Vivo, Demo y Limitaciones

## Slides sugeridos

### Slide 16: Componentes del pipeline en vivo

**Cadena completa:**
```
Cámara C920 (15 fps)
  → [1] Detector: YOLOv11n
  → [2] Tracker FSM
  → [3] Segmentador: RVM MobileNetV3
  → [4] Buffer de secuencias: FIFO(60, stride=30)
  → [5] Embedder: GaitBase iter1200
  → [6] Matcher: max-sim contra galería (τ=0.9847)
  → [7] Confirmador: N=2 secuencias consecutivas
  → [8] Log: CSV attendance_YYYY-MM-DD.csv
```

**¿Por qué cada componente?**

| Componente | Elegido | Razón |
|---|---|---|
| Detector | YOLOv11n (6 MB, conf=0.35) | Rápido, ligero, funciona en GPU 4 GB |
| Segmentador | RVM MobileNetV3 | Mejor que GrabCut/BackSub, robusto a fondo |
| Tracker | FSM propio | Control explícito: 1 persona a la vez |
| Embedder | GaitBase | Ya entrenado, checkpoint de producción |

### Slide 17: El Finite State Machine (FSM) Tracker

**4 estados:**
```
IDLE ──(bbox detectado)──→ WARMING
WARMING ──(≥15 frames)──→ ACTIVE
ACTIVE ──(bbox perdido >15 frames)──→ RESET
RESET ──────────────────────────────→ IDLE
```

**¿Por qué un warmup de 15 frames?**
- Sin warmup, una detección fugaz (persona pasando rápido) dispara una inferencia incompleta
- 15 frames = 1 segundo a 15 fps → confirmamos que la persona está caminando frente a la cámara

**¿Por qué N=2 confirmaciones consecutivas?**
- Una sola secuencia puede tener ruido de silueta → falsa identificación
- Dos secuencias consecutivas por encima de τ reducen drásticamente falsos positivos

### Slide 18: Latencias por componente

| Componente | Latencia típica |
|---|---|
| Captura de frame | ~67 ms (1/15 fps) |
| Detección YOLOv11n | ~10-20 ms |
| Segmentación RVM | ~30-40 ms |
| Embedding GaitBase | ~80-100 ms |
| Matching (galería 19 sujetos) | <1 ms |
| **Total por frame** | **~130-160 ms** |
| **Tiempo total para identificar** | **~4 s** (2 secuencias × 2 s con stride 30) |

- Pipeline corre a ~7-8 fps teórico, display a 15 fps (buffering de frames)
- VRAM pico total: ~3.5 GB (YOLOv11n + RVM + GaitBase)

### Slide 19: Demo / Resultados en vivo
*(aquí mostrar video grabado o demo en tiempo real)*

**Qué muestra el sistema:**
- Bounding box con estado FSM (IDLE / WARMING / ACTIVE)
- Contador de frames en buffer
- Similitud actual con mejor match
- Banner de confirmación: `✓ IDENTIFICADO: [nombre] (sim=0.9851)`
- FPS en tiempo real
- Hotkeys: `q` = salir, `r` = reset manual, `f` = pantalla completa

**Log de salida:**
```
timestamp,subject,mean_sim,n_sequences,frames_used,session_id
2026-05-03 10:15:23,jesusvalenzuela,0.9834,2,120,sess_abc123
```

### Slide 20: Limitaciones conocidas y trabajo futuro

**Limitaciones honestas:**

| Limitación | Causa | Impacto |
|---|---|---|
| Par ricardomora ↔ hectorsanchez | Mismo build corporal, indistinguibles a 90° | 1 error persistente |
| Dataset pequeño (19 sujetos) | Recursos limitados | Generalización restringida |
| 1 solo ángulo (90°) | Cámara fija | Sensible a posición del sujeto |
| Cambio de ropa extremo | Solo 2 sesiones de entrenamiento | Puede degradar similitud |
| Cambio de cámara | τ depende de la cámara específica | Requiere recalibración |

**Trabajo futuro (Fase 7):**
- Quantización INT8 / TensorRT para reducir latencia
- SQLite en lugar de CSV (log persistente y consultable)
- Recalibración automática de τ con nuevas sesiones
- Soporte multi-cámara / multi-ángulo

## Guión (lo que dices)

> "El pipeline en vivo está compuesto por 8 módulos encadenados. La entrada es la cámara C920 a 15 fps y la salida es un CSV de asistencia con timestamp, nombre y nivel de confianza.
>
> Cada módulo tiene una razón de diseño. YOLOv11n pesa 6 MB y corre en ~15ms — ideal para no saturar la GPU. RVM MobileNetV3 nos da siluetas limpias incluso con fondos complejos. El FSM tracker nos permite ignorar detecciones fugaces gracias al warmup de 15 frames.
>
> La identificación tarda ~4 segundos desde que la persona entra al campo de visión. Ese es el tiempo para acumular dos secuencias de 60 frames con stride de 30, lo que da dos votaciones independientes. Si ambas superan τ=0.9847, el sistema confirma la identidad.
>
> En cuanto a limitaciones: hay un par — ricardomora y hectorsanchez — que el sistema confunde consistentemente. Lo documentamos desde la Fase 3. Es un límite real del approach de silueta a 90° cuando dos personas tienen el mismo build corporal. No es un bug, es una limitación del enfoque.
>
> Lo que viene: optimizar latencia con TensorRT, mejorar el logging con SQLite, y si se consiguen más sesiones de grabación, reentrenar con mayor variabilidad."

---

---

# PREGUNTAS DIFÍCILES DEL PROFESOR + RESPUESTAS

## Sobre el dataset

**P: ¿Por qué solo 19 sujetos? ¿No es muy poco para un sistema de reconocimiento biométrico?**
> R: Es el dataset que pudimos grabar con los recursos disponibles (tiempo, sujetos voluntarios, una cámara). Somos conscientes de que es pequeño — por eso usamos fine-tuning desde un modelo preentrenado en 4,000 sujetos en lugar de entrenar desde cero. Las métricas (EER 8.11%) son coherentes con datasets similares de tamaño reducido en condiciones controladas.

**P: ¿Qué pasa si una persona no está en la galería?**
> R: El sistema tiene un mecanismo open-set calibrado. Si la similitud máxima con cualquier sujeto de la galería está por debajo de τ=0.9847, el sistema devuelve "UNKNOWN" y no registra asistencia. Calibramos τ para FAR=0% — preferimos no identificar a nadie que identificar a alguien incorrecto.

## Sobre el modelo

**P: ¿Por qué fine-tuning y no entrenar desde cero?**
> R: Con 13 sujetos de entrenamiento y ~51 secuencias, entrenar desde cero llevaría a overfitting severo. El modelo preentrenado en Gait3D ya aprendió representaciones de marcha de 4,000 personas — nosotros solo ajustamos esas representaciones para discriminar entre 13. Eso es transfer learning.

**P: ¿Qué es el HorizontalPoolingPyramid y por qué se usa?**
> R: Divide la silueta horizontalmente en 16 bandas (cabeza, torso, piernas, pies). Cada banda produce un embedding de 256 dimensiones independiente. Así el modelo puede decir "esta persona tiene una dinámica de piernas característica" sin que el movimiento de brazos interfiera. Es una técnica estándar en gait recognition.

**P: ¿Por qué cosine similarity y no distancia euclidiana?**
> R: Los embeddings están L2-normalizados, por lo que cosine similarity y distancia euclidiana son equivalentes matemáticamente. La elección es por convención en gait recognition y facilita la interpretación del umbral τ (entre 0 y 1).

## Sobre los keypoints (pose)

**P: El requisito original decía multimodal (silueta + pose). ¿Por qué lo descartaron?**
> R: Lo intentamos con dos arquitecturas: SkeletonGait++ y ST-GCN Lite. En ambos casos, al optimizar el peso α de fusión, el algoritmo encontró α=1.0 — silueta sola gana. Esto tiene sentido técnico: a 90° lateral, los keypoints a esa vista ya están codificados en la silueta. La pose no añade información discriminativa adicional con este ángulo y este volumen de datos.

**P: ¿No podría funcionar la pose con más datos?**
> R: Posiblemente sí. Con una tercera sesión o un segundo ángulo de cámara, la pose podría aportar. Lo documentamos como trabajo futuro. Con el dataset actual, los experimentos mostraron que no valía la pena.

## Sobre el pipeline

**P: ¿Por qué 15 fps si la cámara graba a 30 fps?**
> R: El pipeline completo (YOLO + RVM + GaitBase) toma ~130-160 ms por frame. Procesar a 30 fps requeriría <33 ms/frame — no alcanzable con la GPU disponible. A 15 fps el pipeline es fluido y 4 segundos de identificación sigue siendo razonable para uso en tiempo real.

**P: ¿Cómo manejan el caso de dos personas en el campo de visión?**
> R: El sistema toma el bounding box de mayor área — la persona más cercana a la cámara. Es una decisión de diseño consciente: el sistema está pensado para uso en pasillo/entrada donde las personas pasan de a una. Si pasan dos simultáneamente, el sistema puede identificar incorrectamente o no identificar. Está documentado como limitación.

**P: ¿Cómo saben que el umbral τ=0.9847 es el correcto?**
> R: Lo calibramos con curva DET (Detection Error Tradeoff) sobre el set de validación. τ=0.9807 fue el punto de EER. Lo subimos levemente a 0.9847 en la integración al pipeline para reducir aún más los falsos positivos a expensas de más falsos negativos — decisión acorde a la aplicación de asistencia.

---

---

# RESUMEN DE MÉTRICAS PARA REFERENCIA RÁPIDA

| Métrica | Valor |
|---|---|
| Sujetos totales | 19 |
| Sujetos train / val / test | 13 / 3 / 3 |
| Sesiones por sujeto | 2 |
| Rank-1 closed-set (NN) | 94.4% |
| Rank-1 closed-set (NR) | 94.7% |
| EER open-set | **8.11%** |
| TAR @ FAR=0% | **86.49%** |
| Umbral τ | 0.9847 |
| FPS procesamiento | ~7-8 fps teórico |
| Tiempo identificación | ~4 segundos |
| Modelo en producción | `gaitbase_ft_multisession_best_iter1200.pt` |
| VRAM pico pipeline | ~3.5 GB |

---

*Generado: 2026-05-06 | Proyecto: ProyectoChino — Gait Recognition*

Sources:
- [OpenGait: Revisiting Gait Recognition Toward Better Practicality (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Fan_OpenGait_Revisiting_Gait_Recognition_Towards_Better_Practicality_CVPR_2023_paper.pdf)
- [SkeletonGait: Gait Recognition Using Skeleton Maps (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/27933)
- [Emerging trends in gait recognition based on deep learning: a survey (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323174/)
- [OpenGait: A Comprehensive Benchmark Study (IEEE TPAMI 2025)](https://arxiv.org/html/2405.09138v2)
- [Real-time multiple people gait recognition in the edge (2025)](https://www.nature.com/articles/s41598-025-02351-x)
