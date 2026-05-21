"""
Reporte de avance del proyecto Gait Recognition - 2026-05-17.

Contiene:
- Portada con integrantes
- Resumen ejecutivo
- Estado actual (fases completadas)
- Pendientes por terminar
- Propuesta: algoritmos genéticos / estrategias evolutivas
  para optimización de hiperparámetros
- Hardware / material a comprar
- Limitaciones conocidas
- Próximos pasos
"""

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(r"C:\Proyecto3\ProyectoChino\reports\REPORTE_AVANCE_2026-05-17.pdf")

NAVY = colors.HexColor("#0B2545")
ACCENT = colors.HexColor("#1B4965")
LIGHT = colors.HexColor("#E8EEF7")
GREY = colors.HexColor("#5C6B73")
GREEN = colors.HexColor("#1B7F4B")
RED = colors.HexColor("#A0322F")
AMBER = colors.HexColor("#B5651D")

styles = getSampleStyleSheet()

TITLE = ParagraphStyle(
    "Title", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=24, textColor=NAVY, alignment=TA_CENTER,
    spaceAfter=14, leading=28,
)
SUBTITLE = ParagraphStyle(
    "Subtitle", parent=styles["Title"], fontName="Helvetica",
    fontSize=14, textColor=ACCENT, alignment=TA_CENTER,
    spaceAfter=10, leading=18,
)
COVER_LABEL = ParagraphStyle(
    "CoverLabel", parent=styles["BodyText"], fontName="Helvetica-Bold",
    fontSize=11, textColor=GREY, alignment=TA_CENTER, spaceAfter=4,
)
COVER_NAME = ParagraphStyle(
    "CoverName", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=12, textColor=NAVY, alignment=TA_CENTER, spaceAfter=2,
    leading=16,
)
H1 = ParagraphStyle(
    "H1", parent=styles["Heading1"], fontName="Helvetica-Bold",
    fontSize=17, textColor=NAVY, spaceAfter=10, spaceBefore=14, leading=22,
)
H2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=13, textColor=ACCENT, spaceAfter=6, spaceBefore=12, leading=16,
)
H3 = ParagraphStyle(
    "H3", parent=styles["Heading3"], fontName="Helvetica-Bold",
    fontSize=11, textColor=NAVY, spaceAfter=4, spaceBefore=8,
)
BODY = ParagraphStyle(
    "Body", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=10, leading=13.5, alignment=TA_JUSTIFY, spaceAfter=4,
)
BULLET = ParagraphStyle(
    "Bullet", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=2,
)
NOTE = ParagraphStyle(
    "Note", parent=BODY, fontSize=9, textColor=GREY,
    alignment=TA_LEFT, spaceAfter=2,
)


def bullet(items):
    return [Paragraph(f"&bull;&nbsp;&nbsp;{t}", BULLET) for t in items]


def section(title):
    return [Paragraph(title, H1)]


def subsection(title):
    return [Paragraph(title, H2)]


def kv_table(rows, col_widths=None):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, GREY),
        ("BOX", (0, 0), (-1, -1), 0.5, NAVY),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


# ---------------------------------------------------------------------------
# Contenido
# ---------------------------------------------------------------------------
story = []

# ---- Portada ----
story.append(Spacer(1, 3.5 * cm))
story.append(Paragraph(
    "Sistema de Pase de Lista Automático<br/>"
    "por Reconocimiento de Marcha",
    TITLE,
))
story.append(Spacer(1, 0.4 * cm))
story.append(Paragraph(
    "Reporte de Avance del Proyecto",
    SUBTITLE,
))
story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph(
    "Visión por Computadora &middot; Biometría por Marcha (Gait Recognition)",
    COVER_LABEL,
))

story.append(Spacer(1, 2.0 * cm))

# Tarjeta de integrantes
integrantes = [
    "Aguilar Felix, Jesús Antonio",
    "Cazarez Lara, Jesús Kevin",
    "Espinoza Mendivil, Jorge Alberto",
    "Vázquez Cárdenas, César Enrique",
]
team_rows = [[Paragraph("<b>Integrantes</b>", COVER_LABEL)]]
for nombre in integrantes:
    team_rows.append([Paragraph(nombre, COVER_NAME)])

team_table = Table(team_rows, colWidths=[12 * cm], hAlign="CENTER")
team_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
    ("BOX", (0, 0), (-1, -1), 1, NAVY),
    ("INNERGRID", (0, 0), (-1, -1), 0.3, GREY),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 8),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
]))
story.append(team_table)

story.append(Spacer(1, 1.6 * cm))
story.append(Paragraph(
    f"Fecha de elaboración: {datetime(2026, 5, 17).strftime('%d / %m / %Y')}",
    COVER_LABEL,
))
story.append(Paragraph(
    "Fase actual: 6.5 (validación) &middot; 7.0 (optimización completada)",
    COVER_LABEL,
))

story.append(PageBreak())

# ---- 1. Resumen ejecutivo ----
story += section("1. Resumen ejecutivo")
story.append(Paragraph(
    "El proyecto busca construir un sistema automático de pase de lista basado en "
    "<b>reconocimiento de marcha (gait recognition)</b>: identificar a cada persona "
    "que pasa frente a una cámara fija mediante el patrón temporal de su caminar, "
    "y detectar personas <i>fuera de la lista</i> (problema <b>open-set</b>).",
    BODY,
))
story.append(Paragraph(
    "A la fecha hemos completado el pipeline end-to-end: detección, segmentación, "
    "extracción de embedding y matching contra galería, todo corriendo en vivo a "
    "<b>20.7 FPS</b> sobre GTX 1650 (4&nbsp;GB). El modelo en producción es "
    "<b>GaitBase</b> fine-tuneado multisesión sobre nuestro dataset propio de "
    "<b>19 sujetos</b> a 90° lateral. La etapa abierta es la validación del nuevo "
    "umbral &tau;=0.85 en condiciones en vivo. Como bloque final se propone "
    "incorporar <b>algoritmos genéticos / estrategias evolutivas</b> para la "
    "optimización conjunta de hiperparámetros del pipeline.",
    BODY,
))

story.append(Spacer(1, 4 * mm))
story.append(kv_table([
    ["Indicador", "Valor"],
    ["Sujetos enrolados", "19"],
    ["FPS efectivos en vivo", "20.7 (+21 % vs baseline)"],
    ["Umbral activo (τ)", "0.85 (recalibrado para vivo)"],
    ["EER open-set (val)", "8.11 %"],
    ["TAR@FAR=0 % (val)", "86.49 %"],
    ["Hardware actual", "GTX 1650 4 GB · i5-12450H · 16 GB RAM"],
    ["Cámara", "Logitech C920 (USB)"],
], col_widths=[6 * cm, 9.5 * cm]))

# ---- 2. Estado actual ----
story.append(PageBreak())
story += section("2. Estado actual del proyecto")

story += subsection("2.1 Fases completadas")
story.append(kv_table([
    ["Fase", "Descripción", "Estado"],
    ["0", "Setup, descubrimiento y stack validado", "Cerrada"],
    ["1", "Auditoría del dataset (19 sujetos, 90° lateral, 2 sesiones)", "Cerrada"],
    ["2", "Preparación, normalización y splits subject-disjoint", "Cerrada"],
    ["3", "Baseline GaitBase + fine-tuning monosesión", "Cerrada"],
    ["3.5", "Fine-tuning multisesión (checkpoint en producción)", "Cerrada"],
    ["4", "Multimodal (silueta + pose) — cierre como Caso C", "Cerrada"],
    ["5", "Open-set: τ calibrada, EER y TAR@FAR reportados", "Cerrada"],
    ["6", "Pipeline de inferencia en vivo (script ejecutable)", "Cerrada"],
    ["6.5", "Validación end-to-end con τ recalibrada", "En curso"],
    ["7.0", "Optimización de FPS y fix CUBLAS", "Cerrada"],
    ["7.1", "SQLite + documentación + GA/CMA-ES", "Pendiente"],
], col_widths=[1.5 * cm, 10.5 * cm, 3.5 * cm]))

story += subsection("2.2 Logros técnicos recientes (mayo 2026)")
story += bullet([
    "<b>Fix CUBLAS</b>: resuelto <i>CUBLAS_STATUS_EXECUTION_FAILED</i> en GTX 1650 "
    "desactivando autocast FP16 (la GTX 1650 no tiene Tensor Cores). El embedder "
    "ahora corre estable en FP32.",
    "<b>Optimización de FPS</b>: imgsz YOLO reducido de 1280&times;720 a 640. "
    "Resultado: <b>+21 % de throughput</b> (de 17.1 a 20.7 FPS efectivos).",
    "<b>Recalibración de umbral</b>: &tau; bajado de 0.9847 a <b>0.85</b> tras "
    "observar que enrolados producían similitud ≈0.95 en vivo pero eran rechazados "
    "por el umbral derivado del set de validación.",
    "<b>Profiling per-etapa</b>: con <code>torch.cuda.Event</code>; ahora sabemos que "
    "RVM (segmentación) es el nuevo cuello de botella (~35 % del tiempo de frame).",
])

story += subsection("2.3 Métricas de rendimiento del pipeline (post-optimización)")
story.append(kv_table([
    ["Etapa", "Tiempo (ms)", "% del frame", "Comentario"],
    ["YOLO (detección)", "17.98", "23 %", "imgsz=640 aplicado"],
    ["RVM (segmentación)", "26.45", "35 %", "Cuello de botella actual"],
    ["GaitBase (embedding)", "59.48", "8 % amortizado", "Solo en cierre de ventana"],
    ["Matcher", "0.28", "<1 %", "CPU, negligible"],
    ["FPS efectivos", "20.7", "—", "+21 % vs baseline"],
], col_widths=[4.5 * cm, 2.8 * cm, 3.0 * cm, 5.2 * cm]))

# ---- 3. Pendientes ----
story.append(PageBreak())
story += section("3. Pendientes por terminar")

story += subsection("3.1 Validación end-to-end (Fase 6.5 — alta prioridad)")
story += bullet([
    "Correr <code>infer_live.py</code> con &tau;=0.85 sobre los 19 sujetos enrolados "
    "y confirmar que aparecen como <b>PRESENTE</b>, registrando similitud observada.",
    "Probar con personas <b>fuera de la galería</b> y confirmar que se rechazan como "
    "<b>UNK</b> (open-set).",
    "Pruebas de robustez en al menos 4 condiciones: cambio de ropa, velocidad de "
    "caminata distinta, iluminación variable, oclusión parcial.",
    "<b>Métrica de éxito:</b> &tau;=0.85 acepta a todos los enrolados y rechaza al "
    "100 % de los no enrolados, con margen de seguridad ≥ 0.05.",
])

story += subsection("3.2 Persistencia y trazabilidad (Fase 7.1)")
story += bullet([
    "Migrar el log de asistencia de <b>CSV → SQLite</b> para soportar consultas, "
    "audit trail y exportes filtrados (por sujeto, fecha, rango horario).",
    "Esquema mínimo: tabla <code>events</code> (timestamp, subject_id, similarity, "
    "decision, session_id) + tabla <code>gallery</code> (subject_id, enroll_date, "
    "template_path).",
    "Backups automáticos diarios.",
])

story += subsection("3.3 Optimización de hiperparámetros con GA / ES (Fase 7.2)")
story += bullet([
    "Implementar el bloque descrito en la sección 4 de este reporte.",
    "Definir presupuesto de evaluaciones (≈ 200 individuos × 30 generaciones).",
    "Validar mejora real sobre el conjunto de val antes de promover a producción.",
])

story += subsection("3.4 Documentación final y entrega")
story += bullet([
    "Manual de operación (cómo enrolar, cómo correr, cómo interpretar el log).",
    "Guía de calibración de &tau; para una cámara nueva.",
    "Reporte técnico final con métricas consolidadas.",
    "Defensa / presentación con demo en vivo.",
])

story += subsection("3.5 Mejoras opcionales (Fase 8 — si hay tiempo)")
story += bullet([
    "Reemplazar RVM por un segmentador más ligero (SegFormer-B0 o YOLOv8-seg) "
    "para subir a 30 FPS sostenidos.",
    "Exportar el embedder a <b>ONNX</b> o <b>TensorRT</b> para despliegue en hardware "
    "modesto (laptop sin GPU dedicada).",
    "Ampliar el dataset: capturar a 0°, 45° y 135° para mejorar la generalización.",
])

# ---- 4. Algoritmos genéticos / Estrategias evolutivas ----
story.append(PageBreak())
story += section("4. Optimización de hiperparámetros con Algoritmos Genéticos / Estrategias Evolutivas")

story += subsection("4.1 Justificación")
story.append(Paragraph(
    "El pipeline expone una decena de hiperparámetros cuya interacción <b>no es "
    "diferenciable</b> respecto a la métrica final (TAR@FAR=1 % en open-set). "
    "Búsqueda en malla (<i>grid search</i>) es prohibitiva por la combinatoria, y "
    "<i>random search</i> ignora dependencias entre parámetros. Por eso "
    "proponemos un esquema <b>basado en población</b>:",
    BODY,
))
story += bullet([
    "<b>Algoritmo Genético (GA)</b> con codificación mixta (real + entera), "
    "cruce <i>simulated binary crossover</i> (SBX) y mutación polinomial.",
    "<b>CMA-ES</b> (Covariance Matrix Adaptation Evolution Strategy) para los "
    "subespacios puramente continuos: aprende correlaciones entre parámetros y "
    "es el referente moderno en optimización <i>black-box</i>.",
    "Comparativa entre ambos: GA explora mejor espacios mixtos; CMA-ES converge "
    "más rápido en lo continuo. Reportaremos ambos y elegiremos el ganador en val.",
])

story += subsection("4.2 Hiperparámetros candidatos")
story.append(kv_table([
    ["Parámetro", "Tipo", "Rango propuesto", "Impacto esperado"],
    ["&tau; (umbral open-set)", "real", "[0.70, 0.95]", "TAR / FAR"],
    ["ventana (frames)", "entero", "[30, 90]", "Estabilidad embedding"],
    ["stride (frames)", "entero", "[10, 45]", "Latencia / cobertura"],
    ["N (confirmaciones)", "entero", "[1, 4]", "Robustez vs latencia"],
    ["imgsz YOLO", "discreto", "{416, 512, 640, 736}", "FPS / detección"],
    ["IoU NMS YOLO", "real", "[0.30, 0.70]", "Tracking estable"],
    ["conf YOLO", "real", "[0.25, 0.60]", "Precisión / recall"],
    ["bbox padding silueta", "real", "[1.00, 1.30]", "Calidad de silueta"],
    ["agregación embedding", "discreto", "{mean, max, attn}", "Robustez temporal"],
], col_widths=[4.0 * cm, 1.8 * cm, 4.0 * cm, 5.7 * cm]))

story += subsection("4.3 Función de fitness")
story.append(Paragraph(
    "El fitness combina <b>desempeño biométrico</b> y <b>velocidad</b>, evitando "
    "soluciones que sólo optimizan una dimensión:",
    BODY,
))
story.append(Paragraph(
    "<i>fitness</i> = w<sub>1</sub> &middot; TAR@FAR=1 % "
    "&minus; w<sub>2</sub> &middot; (1 &minus; rank-1) "
    "&minus; w<sub>3</sub> &middot; max(0, 15 &minus; FPS) / 15",
    BODY,
))
story += bullet([
    "<b>w<sub>1</sub>=0.6:</b> peso del open-set (objetivo principal).",
    "<b>w<sub>2</sub>=0.3:</b> peso del closed-set (rank-1).",
    "<b>w<sub>3</sub>=0.1:</b> penalización si FPS &lt; 15 (requisito de tiempo real).",
    "Evaluación sobre <b>val subject-disjoint</b>; el test queda intocable para "
    "el reporte final (evita sobreajuste por búsqueda).",
])

story += subsection("4.4 Configuración propuesta de la búsqueda")
story.append(kv_table([
    ["Parámetro de la búsqueda", "Valor"],
    ["Tamaño de población", "30 individuos"],
    ["Número de generaciones", "30"],
    ["Presupuesto total de evaluaciones", "≈ 900 (cacheable hasta 60–70 %)"],
    ["Operador de cruce (GA)", "SBX, η<sub>c</sub>=15"],
    ["Operador de mutación (GA)", "Polinomial, η<sub>m</sub>=20, p<sub>m</sub>=1/D"],
    ["Estrategia (ES)", "CMA-ES con σ<sub>0</sub>=0.3"],
    ["Selección", "Torneo binario (GA) / (μ,λ) (ES)"],
    ["Elitismo", "Sí, 2 mejores (GA)"],
    ["Semilla", "Fija para reproducibilidad (seed=42)"],
    ["Librería sugerida", "DEAP (GA) + pycma (CMA-ES)"],
], col_widths=[6.5 * cm, 9.0 * cm]))

story += subsection("4.5 Plan de implementación (estimado: 1–2 semanas)")
story += bullet([
    "<b>Día 1–2:</b> envolver el pipeline en una función "
    "<code>evaluate(individual) → fitness</code> determinista.",
    "<b>Día 3:</b> integrar DEAP y correr GA piloto (10 generaciones, 10 individuos).",
    "<b>Día 4:</b> integrar pycma y correr CMA-ES piloto en el subespacio continuo.",
    "<b>Día 5–7:</b> correr la búsqueda completa (usar la segunda PC RTX 2060).",
    "<b>Día 8–10:</b> validar el mejor individuo en test y documentar.",
])

story.append(Spacer(1, 4 * mm))
story.append(Paragraph(
    "<b>Nota técnica:</b> dado que el embedding es la operación más cara, "
    "cachearemos embeddings por <i>(checkpoint, frame_set)</i> para que las "
    "evaluaciones que sólo cambian &tau;, ventana o stride reutilicen vectores "
    "ya calculados. Esto reduce el costo efectivo de cada evaluación en ~70 %.",
    NOTE,
))

# ---- 5. Hardware / Material a comprar ----
story.append(PageBreak())
story += section("5. Material y hardware a comprar")

story.append(Paragraph(
    "Este apartado lista lo que recomendamos adquirir para cerrar el proyecto con "
    "calidad de entrega y soportar las fases pendientes (validación robusta, "
    "búsqueda evolutiva, demo final). Los precios son <b>orientativos en MXN</b> "
    "(mayo 2026) y deben confirmarse en cotización.",
    BODY,
))

story += subsection("5.1 Indispensable (sin esto, no se cierra el proyecto)")
story.append(kv_table([
    ["Ítem", "Justificación", "Costo aprox."],
    [
        "Trípode robusto 1.6 m con cabezal fluido",
        "Estabilidad de la C920 en captura y demos. Hoy se improvisa.",
        "$700 – $1,200",
    ],
    [
        "Cable USB activo 5 m (con repetidor)",
        "Permite colocar la cámara a 3–4 m del sujeto sin pérdida de señal.",
        "$300 – $500",
    ],
    [
        "SSD NVMe externo o interno 1 TB",
        "Dataset + checkpoints + caché de embeddings para GA/CMA-ES.",
        "$1,500 – $2,500",
    ],
    [
        "Panel LED de iluminación regulable",
        "Condiciones reproducibles para pruebas de robustez (Fase 6.5).",
        "$800 – $1,500",
    ],
], col_widths=[5.5 * cm, 7.5 * cm, 2.5 * cm]))

story += subsection("5.2 Recomendado (mejora notable de calidad)")
story.append(kv_table([
    ["Ítem", "Justificación", "Costo aprox."],
    [
        "UPS / No-break 600–900 VA",
        "Protege entrenamientos y corridas de GA largas (varias horas).",
        "$1,200 – $2,000",
    ],
    [
        "Marca de piso (cinta + conos)",
        "Define el corredor de captura, reduce varianza entre sesiones.",
        "$150 – $300",
    ],
    [
        "Fondo neutro (tela 2&times;3 m, gris claro)",
        "Mejora la calidad de la silueta (segmentación RVM).",
        "$400 – $800",
    ],
    [
        "Cámara IP / segunda C920 de respaldo",
        "Redundancia y posibilidad de capturar a 0° / 45° para Fase 8.",
        "$1,500 – $2,500",
    ],
], col_widths=[5.5 * cm, 7.5 * cm, 2.5 * cm]))

story += subsection("5.3 Opcional (si el presupuesto lo permite)")
story.append(kv_table([
    ["Ítem", "Justificación", "Costo aprox."],
    [
        "GPU RTX 3060 12 GB o RTX 4060 8 GB",
        "Acelera GA/CMA-ES y permite Tensor Cores (FP16 real, sin el bug CUBLAS).",
        "$6,500 – $9,500",
    ],
    [
        "Mini-PC / NUC para despliegue final",
        "Demo en aula sin depender de la laptop personal.",
        "$8,000 – $12,000",
    ],
    [
        "Monitor secundario 24''",
        "Productividad durante depuración del pipeline en vivo.",
        "$2,500 – $4,500",
    ],
], col_widths=[5.5 * cm, 7.5 * cm, 2.5 * cm]))

story.append(Spacer(1, 5 * mm))
story.append(Paragraph(
    "<b>Resumen presupuestal estimado:</b> indispensable ≈ $3,300 – $5,700 MXN; "
    "con recomendado, ≈ $6,500 – $11,300 MXN; con opcional completo, "
    "≈ $23,500 – $37,300 MXN.",
    BODY,
))

# ---- 6. Limitaciones conocidas ----
story.append(PageBreak())
story += section("6. Limitaciones conocidas y riesgos")
story += bullet([
    "<b>Dataset pequeño y monoangular:</b> 19 sujetos, todos a 90° lateral, "
    "en 2 sesiones. Generalización limitada a otras vistas o cámaras.",
    "<b>Par crítico <i>ricardomora &harr; hectorsanchez</i>:</b> mismo build "
    "corporal, falla en NN y NR. Documentado como caso aceptado.",
    "<b>Dependencia de calibración por cámara:</b> &tau; debe re-ajustarse si se "
    "cambia la cámara o el escenario (mitigable con GA reentrenando &tau;).",
    "<b>Sin Tensor Cores en la GTX 1650:</b> nos obliga a FP32; un upgrade a "
    "RTX 30/40 daría ~2&times; en throughput inmediato.",
    "<b>Dataset no es re-grabable a corto plazo:</b> evitar transformaciones "
    "destructivas; cualquier limpieza debe ser sobre copias.",
])

# ---- 7. Próximos pasos inmediatos ----
story += section("7. Próximos pasos inmediatos (próximas 2 semanas)")
story.append(kv_table([
    ["Semana", "Entregable", "Responsable principal"],
    ["1", "Validación en vivo con τ=0.85 sobre los 19 enrolados + 4 desconocidos", "Equipo completo"],
    ["1", "Migración del log a SQLite con esquema definido", "Aguilar / Cazarez"],
    ["2", "Implementación del wrapper evaluate(individual) → fitness", "Espinoza"],
    ["2", "Piloto GA (DEAP) + piloto CMA-ES en RTX 2060", "Vázquez / Aguilar"],
    ["2", "Manual de operación y guía de calibración", "Cazarez / Espinoza"],
], col_widths=[1.5 * cm, 11.0 * cm, 3.5 * cm]))

story.append(Spacer(1, 6 * mm))
story.append(Paragraph(
    "<b>Criterio de cierre del proyecto:</b> pipeline en vivo a ≥15 FPS, "
    "&tau; optimizado por GA/CMA-ES, log persistido en SQLite, manual entregado y "
    "demo funcional ante el grupo.",
    BODY,
))

# ---- 8. Apéndice ----
story.append(PageBreak())
story += section("Apéndice A. Configuración activa del pipeline")
story.append(Paragraph(
    "Extracto del archivo <code>configs/pipeline.yaml</code>:",
    BODY,
))
story.append(kv_table([
    ["Sección", "Parámetro", "Valor"],
    ["camera", "target_fps", "60 (pipeline limita a ~20)"],
    ["detector", "imgsz", "640"],
    ["embedder", "compile", "false (Triton no disponible en Windows)"],
    ["matcher", "tau", "0.85"],
    ["sequence", "window", "60 frames (~4 s @ 15 FPS)"],
    ["sequence", "stride", "30 frames (50 % de solapamiento)"],
], col_widths=[3.0 * cm, 4.0 * cm, 8.5 * cm]))

story.append(Spacer(1, 6 * mm))
story.append(Paragraph(
    "<b>Checkpoint del modelo en producción:</b> "
    "<code>checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt</code>",
    BODY,
))
story.append(Paragraph(
    "<b>Reportes técnicos previos:</b> ver carpeta <code>reports/</code> con la "
    "secuencia 01–07 (auditoría → fases), <code>PHASE4_CIERRE_2026-04-28.md</code>, "
    "y <code>CHECKPOINT_2026-05-12.md</code> (último checkpoint detallado).",
    BODY,
))

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, 1.2 * cm,
                      "Gait Recognition · Reporte de avance · 2026-05-17")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm,
                           f"Página {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=2 * cm, rightMargin=2 * cm,
    topMargin=2 * cm, bottomMargin=2 * cm,
    title="Reporte de avance - Gait Recognition - 2026-05-17",
    author="Equipo ProyectoChino",
)
doc.build(story, onFirstPage=_footer, onLaterPages=_footer)

print(f"[OK] PDF generado: {OUT}")
print(f"     Tamaño: {OUT.stat().st_size / 1024:.1f} KB")
