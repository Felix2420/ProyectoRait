# Guión de Presentación — Sistema de Reconocimiento de Marcha
**18 slides | 4 personas | ~20 min estimados**

---

## DISTRIBUCIÓN

| Persona | Slides | Contenido |
|---|---|---|
| **P1** | 1 – 6 | Portada, problema, marcha vs. facial, dataset, fases, split |
| **P2** | 7 – 10 | Estado del arte, benchmarks, GaitBase, por qué descartamos pose |
| **P3** | 11 – 14 | Transfer learning, hiperparámetros, resultados closed-set, open-set |
| **P4** | 15 – 18 | Pipeline en vivo, FSM, latencia, cierre |

---
---

# PERSONA 1 — Slides 1 al 6

---

## Slide 1 — Portada
> *Título: "Reconocimiento de Marcha para Asistencia Automatizada"*

"Buenos días. Vamos a presentarles nuestro proyecto del semestre: un sistema que identifica personas por su forma de caminar y registra la asistencia de forma automática.

La idea central es que una cámara fija observe la entrada del salón y, mientras los alumnos caminan hacia su lugar, el sistema los reconozca sin que ellos tengan que hacer nada. Sin firmas, sin lista, sin interrupciones."

---

## Slide 2 — El Problema de la Asistencia Manual
> *Bullets: ineficiencia, error humano, suplantación, intrusión*

"¿Por qué atacar este problema? El pase de lista manual tiene cuatro problemas concretos.

Primero, tiempo: se pierden entre 5 y 10 minutos por sesión — en un semestre eso se acumula. Segundo, precisión: el error humano genera registros incorrectos. Tercero, y el más importante para nosotros: la suplantación. Es trivial firmar por alguien que no vino. Y cuarto, intrusión: interrumpe el flujo natural de la clase.

El reto que nos planteamos fue: ¿podemos identificar a las personas de forma pasiva, mientras caminan hacia su lugar, sin que ellos hagan nada?"

---

## Slide 3 — Marcha vs. Reconocimiento Facial
> *Comparativa: ventajas de marcha, desventajas de facial, seguridad*

"La primera pregunta obvia es: ¿por qué no usar reconocimiento facial? Tenemos tres razones.

La marcha funciona a larga distancia, de espaldas, con cubrebocas, con mochila — no requiere cooperación activa del sujeto. El reconocimiento facial, en cambio, necesita alta resolución, buena iluminación y que la persona mire directamente a la cámara. En un pasillo de universidad eso rara vez se cumple.

Y en términos de seguridad: falsificar la dinámica de marcha de alguien es mucho más difícil que mostrar una fotografía."

---

## Slide 4 — Especificaciones del Dataset
> *Bullets: 19 sujetos, C920, 90° lateral, 2 sesiones, ~120 secuencias, no re-grabable*

"Nuestro dataset lo grabamos nosotros mismos. 19 compañeros, con la cámara Logitech C920 a 1080p, posicionada a 90 grados lateral en un escenario indoor controlado.

Cada persona tiene dos sesiones: s1 con paso normal y s2 con paso rápido. Eso nos da aproximadamente 120 secuencias en total.

La restricción más importante: este dataset es estático e inmutable. No podemos re-grabarlo. Todas las decisiones técnicas que van a ver tienen esta restricción como denominador común."

---

## Slide 5 — División del Problema en 7 Fases
> *Diagrama de fases F0 a F5 (F6 implícita en el pipeline)*

"Para no abordar todo de golpe, dividimos el problema en 7 fases secuenciales.

Empezamos con una auditoría del dataset — verificar que lo que teníamos era de calidad suficiente. Luego extracción de siluetas y keypoints, estandarización al formato PKL de OpenGait, fine-tuning del modelo base, un intento de fusión multimodal que les va a explicar mi compañero y que terminamos descartando, calibración del umbral open-set, y finalmente el pipeline en vivo.

Cada fase produce entregables concretos: scripts, checkpoints, reportes técnicos. Así podíamos medir avance real."

---

## Slide 6 — Subject-Disjoint Dataset Split
> *Regla de oro: el modelo nunca ve sujetos de prueba en entrenamiento*

"Antes de entrenar cualquier cosa, definimos los splits del dataset de forma que las personas de prueba nunca aparecen en entrenamiento. Esto se llama subject-disjoint split.

13 sujetos para entrenar, 3 para validación y 3 para prueba final. La semilla es fija — seed 42 — para que sea reproducible.

Esto es la 'regla de oro' del proyecto. Si no respetamos este split, cualquier métrica que reportemos estaría inflada artificialmente. El modelo no aprende caras ni nombres — aprende patrones de marcha de personas que nunca ha visto durante la evaluación.

Le paso la palabra a [Persona 2] para que les explique el estado del arte y cómo elegimos el modelo."

---
---

# PERSONA 2 — Slides 7 al 10

---

## Slide 7 — Silueta vs. Pose
> *Dos columnas: apariencia (silueta binaria) vs. modelo (keypoints de esqueleto)*

"En la literatura de reconocimiento de marcha hay dos grandes familias de métodos.

Los métodos basados en silueta usan una máscara binaria del cuerpo — blanco sobre negro. Son el estándar actual, robustos en entornos controlados y son el estado del arte en los benchmarks principales como CASIA-B y Gait3D.

Los métodos basados en pose usan las coordenadas del esqueleto — los keypoints de las articulaciones. En teoría son invariantes a la ropa porque no dependen de la apariencia. En práctica, son sensibles a oclusiones y al ruido del detector de poses.

Nosotros exploramos ambos enfoques. Ahora les explico qué encontramos."

---

## Slide 8 — Benchmarks Globales de Referencia
> *Tabla de benchmarks: CASIA-B, Gait3D, GREW, nuestro dataset*

"Para contextualizar nuestros resultados, aquí están los benchmarks públicos más usados en la comunidad.

CASIA-B tiene 124 sujetos en condiciones controladas — los mejores modelos superan 95% de rank-1 ahí. Gait3D tiene 4,000 sujetos en entornos no controlados, y el mejor modelo alcanza alrededor de 75%. GREW es el más difícil: 26,000 sujetos en exteriores, rank-1 en torno a 60-70%.

Nuestro dataset es pequeño — 19 sujetos — pero está controlado. Un rank-1 alto en nuestro caso no es sorprendente. El verdadero reto es el escenario open-set: rechazar a personas que no están en el sistema."

---

## Slide 9 — Modelo Base: GaitBase
> *Bullets: OpenGait CVPR 2023, ResNet9, Temporal Pooling, HPP*

"El modelo que usamos se llama GaitBase. Es el baseline del framework OpenGait, publicado como Highlight Paper en CVPR 2023 por Fan et al.

No lo elegimos arbitrariamente. Está preentrenado en Gait3D — 4,000 sujetos — tiene código público, documentación sólida, y cabe en nuestra GPU de 4 GB.

La arquitectura en resumen: un ResNet9 modificado que procesa cada frame de silueta de forma independiente. Luego un Temporal Pooling que condensa toda la secuencia en un solo descriptor usando el máximo temporal. Y finalmente el HPP — Horizontal Pooling Pyramid — que divide la silueta en 16 bandas horizontales: cabeza, torso, piernas, pies. Cada banda produce un embedding de 256 dimensiones. El resultado final es un vector de 4,096 dimensiones normalizado, sobre el que calculamos similitud coseno."

---

## Slide 10 — ¿Por Qué Descartamos la Pose?
> *Tres bullets: ángulo lateral, dataset pequeño, resultados empíricos α=0*

"El objetivo original del proyecto incluía fusión multimodal — silueta más pose. Lo intentamos con dos arquitecturas distintas: SkeletonGait++ y ST-GCN Lite.

El resultado fue el mismo en ambos casos: al optimizar el peso de fusión entre silueta y pose, el modelo asignó peso cero a la rama de pose. Silueta sola gana siempre.

Hay tres razones técnicas para eso.

Primera: el ángulo. A 90° lateral, los brazos y piernas contralaterales se solapan en la imagen. Los keypoints pierden dimensionalidad crítica — la silueta ya captura casi toda la información visible de la marcha a ese ángulo.

Segunda: el tamaño del dataset. Las redes de pose necesitan miles de ejemplos para generalizar. Nosotros tenemos 51 secuencias de entrenamiento. Con eso, la red aprende el ruido del detector de poses como si fuera una firma de identidad — sobreajuste severo.

Tercera: la evidencia empírica. El optimizador nos dijo directamente que la pose no aporta. No es una suposición, es un resultado.

La pose sigue corriendo en el pipeline, pero solo para descartar siluetas de mala calidad. No entra en el score de identidad.

Le paso la palabra a [Persona 3]."

---
---

# PERSONA 3 — Slides 11 al 14

---

## Slide 11 — Estrategia de Transfer Learning
> *Bullets: pesos iniciales Gait3D, entrenamiento multi-sesión, función de pérdida*

"La pregunta central del entrenamiento era: ¿cómo adaptar un modelo entrenado en 4,000 sujetos para que distinga a nuestros 19?

La respuesta es transfer learning. Partimos de los pesos del GaitBase preentrenado en Gait3D. No entrenamos desde cero — con 13 sujetos de entrenamiento eso sería sobreajuste garantizado.

La decisión más importante fue mezclar las dos sesiones desde el inicio del entrenamiento: s1 normal y s2 rápido juntas. Si entrenamos solo con s1, el modelo aprende a distinguir 'la ropa de s1 de cada persona'. Al mezclar sesiones, lo forzamos a aprender lo que no cambia entre sesiones — la dinámica real de marcha.

La función de pérdida combina dos términos: TripletLoss, que jala los embeddings de la misma persona para que estén cerca en el espacio vectorial, y CrossEntropyLoss con label smoothing, para que el modelo distinga entre las 13 clases de entrenamiento."

---

## Slide 12 — Configuración del Experimento
> *Tabla de hiperparámetros*

"Los hiperparámetros más relevantes.

Optimizador SGD con learning rate de 0.01, momentum 0.9 y weight decay de 5×10⁻⁴. El batch tiene una composición específica: P=4 identidades por K=2 secuencias, que da 8 muestras por iteración — el mínimo para que TripletLoss funcione bien.

Cada clip usa 30 frames, seleccionados aleatoriamente de la secuencia completa. Esto actúa como data augmentation: el modelo nunca ve exactamente los mismos 30 frames dos veces.

Entrenamos hasta 1,500 iteraciones máximo con early stopping de paciencia 3. El mejor checkpoint fue en la iteración 1,200. El decay de learning rate ocurre en las iteraciones 750 y 1,250.

Habilitamos Automatic Mixed Precision en float16 — fue necesario para que todo cupiera en los 4 GB de la GTX 1650. El entrenamiento tomó aproximadamente 177 minutos."

---

## Slide 13 — Resultados Closed-Set (Rank-1)
> *Tabla de resultados, nota sobre Ricardo Mora vs Hector Sanchez*

"Los resultados en closed-set.

Rank-1 de 94.4% en la condición NN — normal contra normal — y 94.7% en NR — normal contra rápido. Rank-5 de 100% en ambas condiciones, lo que significa que la persona correcta siempre está en el top-5.

El único error recurrente es entre Ricardo Mora y Hector Sanchez. El modelo los confunde en ambas condiciones. La razón es que tienen una complexión física casi idéntica — mismo build corporal, misma estatura aproximada. A 90° lateral con silueta binaria, son prácticamente indistinguibles. No es un bug, es un límite del enfoque que documentamos y aceptamos."

---

## Slide 14 — Open-Set: El Desafío del Desconocido
> *EER 8.11%, TAR@FAR=0% 86.49%, umbral τ=0.9847*

"Cerrado funciona bien, pero el problema real es open-set: ¿qué pasa cuando se presenta alguien que no está en el sistema?

En open-set calibramos un umbral τ. Si la similitud máxima con cualquier sujeto de la galería está por debajo de τ, el sistema devuelve 'desconocido' y no registra nada.

El EER — Equal Error Rate — es 8.11%. Ese es el punto donde falsos positivos y falsos negativos se igualan.

El TAR a FAR=0% es 86.49% — 32 de 37 pares genuinos se identificaron correctamente sin ningún falso positivo. El umbral τ=0.9847 fue calibrado con esa restricción: FAR cero antes que cualquier otra cosa.

¿Por qué esa restricción? Porque en un sistema de asistencia, identificar a alguien que no vino como presente es inaceptable. Preferimos no registrar a alguien que registrar a un impostor.

Le paso a [Persona 4]."

---
---

# PERSONA 4 — Slides 15 al 18

---

## Slide 15 — Arquitectura del Pipeline en Vivo
> *Diagrama de flujo, tiempo total ~4 segundos*

"El pipeline en vivo conecta ocho módulos encadenados.

La cámara C920 captura a 15 fps. Un detector YOLOv11n localiza a la persona. Un tracker FSM controla el estado del sistema. El segmentador RVM extrae la silueta binaria limpia. Un buffer FIFO acumula frames hasta completar una secuencia de 60 frames. El embedder GaitBase convierte esa secuencia en un vector. Un matcher compara ese vector contra la galería de 19 sujetos. Y un confirmador exige 2 secuencias consecutivas por encima del umbral antes de registrar.

El tiempo total desde que la persona entra al campo de visión hasta que se confirma la identidad es aproximadamente 4 segundos. Eso es el tiempo para acumular dos ventanas de 60 frames con un stride de 30 — dos votaciones independientes."

---

## Slide 16 — Lógica del FSM Tracker
> *4 estados: IDLE, WARMING, ACTIVE, RESET*

"El tracker usa una máquina de estados finita con 4 estados.

IDLE: el sistema espera, no hay nadie en el campo de visión.

WARMING: se detectó una persona pero esperamos 15 frames consecutivos — 1 segundo — antes de hacer nada. Esto evita que una detección fugaz dispare una inferencia con datos incompletos.

ACTIVE: confirmada la presencia, el sistema empieza a acumular siluetas y procesar secuencias.

RESET: si la persona desaparece del campo de visión por más de 15 frames, el sistema limpia todos los buffers y vuelve a IDLE.

El warmup de 15 frames y el reset de 15 frames son parámetros ajustables en el archivo de configuración — no están hardcodeados."

---

## Slide 17 — Análisis de Latencia y Recursos
> *Tabla de latencias por componente*

"En cuanto a tiempos.

La detección con YOLOv11n toma entre 10 y 20 milisegundos. La segmentación RVM entre 30 y 40. El embedding de GaitBase es el paso más costoso: 80 a 100 milisegundos. El matching contra la galería de 19 sujetos es prácticamente instantáneo, menos de 1 milisegundo.

El total por frame es entre 130 y 160 milisegundos, lo que da 7-8 fps teórico de procesamiento. El display corre a 15 fps con buffering de frames.

El uso de VRAM en pico es de aproximadamente 3.5 GB para los tres modelos combinados — YOLO, RVM y GaitBase — lo que cabe justo en la GTX 1650 de 4 GB."

---

## Slide 18 — Resumen de Impacto y Trabajo Futuro
> *Resultados clave: rank-1 94.5%, TAR 86.5%, no intrusivo. Próximos pasos: TensorRT, SQL, multicámara*

"Para cerrar, los tres resultados más importantes.

Rank-1 de 94.5% en closed-set. TAR a FAR=0% de 86.5% en open-set. Y un sistema funcional, no intrusivo, que corre en hardware de consumo — una GTX 1650.

Los tres pasos más concretos hacia una versión de producción: primero, optimización con TensorRT INT8 para reducir la latencia del embedding de 100 ms a alrededor de 30 ms. Segundo, reemplazar el log CSV por una base de datos SQL con persistencia y capacidad de consulta. Tercero, soporte multicámara para cubrir múltiples ángulos y eliminar la dependencia del punto de vista lateral fijo.

Gracias. Estamos disponibles para preguntas."

---
---

# PREGUNTAS FRECUENTES — Respuestas cortas para Q&A

**¿Por qué 15 fps y no 30?**
> El pipeline completo tarda ~150 ms por frame. A 30 fps necesitaríamos menos de 33 ms — no alcanzable con la GPU disponible. A 15 fps el sistema es fluido.

**¿Qué pasa si dos personas entran al mismo tiempo?**
> El sistema toma el bounding box de mayor área — la persona más cercana. Es una limitación de diseño documentada. El sistema está pensado para un sujeto a la vez.

**¿Pueden agregar más personas sin reentrenar?**
> Sí, solo hay que grabar secuencias de referencia para la persona nueva y reconstruir la galería con `build_gallery.py`. No es necesario reentrenar el modelo.

**¿Por qué el umbral es tan cercano a 1?**
> Los embeddings están L2-normalizados, entonces la similitud coseno entre dos secuencias de la misma persona a 90° lateral controlado es naturalmente alta — en torno a 0.98. El rango operativo es estrecho pero bien separado: genuinos en ~0.9883, impostores en ~0.9666.

**¿Qué significan EER y TAR@FAR?**
> EER es el punto donde el porcentaje de falsos positivos y falsos negativos son iguales — mide el balance del sistema. TAR@FAR=0% es cuántos genuinos identificamos correctamente cuando nos exigimos cero falsos positivos — mide el rendimiento bajo la restricción más estricta.

**¿Por qué descartaron la pose si el requisito original la incluía?**
> La ejecutamos, la evaluamos con dos arquitecturas distintas y los resultados empíricos mostraron que el optimizador le asigna peso cero. La pose sigue corriendo en el pipeline para quality gating, pero no entra en el score de identidad. El requisito se reinterpretó basado en evidencia.

**¿Cómo calibraron τ=0.9847?**
> Calculamos la curva DET sobre el set de validación. El EER ocurre en τ=0.9807. Subimos levemente a 0.9847 para garantizar FAR=0% absoluto en validación. Es un tradeoff: aceptamos perder algunos genuinos adicionales a cambio de cero falsos positivos.

---

*Generado: 2026-05-06 | Basado en las 18 slides de "Sistema de Reconocimiento de Marcha.pptx"*
