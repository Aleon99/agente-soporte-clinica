# Clasificación y priorización de incidencias de soporte técnico (IS302)

Trabajo final del curso **IS302 – Inteligencia Artificial** (UPC). Solución
híbrida para el área de soporte técnico de una clínica de Lima:

1. **Clasificación con Machine Learning**: una Regresión Logística sobre TF-IDF
   predice el `tipo_problema` del ticket a partir de su descripción, área y sede.
2. **Priorización por regla**: la matriz de la Tabla 1 del informe asigna la
   prioridad y la cola se ordena por prioridad y, dentro de cada nivel, por fecha.
3. **Agente de triaje** (Microsoft Agent Framework + Foundry Local, probado en
   Agent Inspector): el LLM solo extrae los campos, llama a las herramientas y
   redacta la justificación. **No clasifica ni prioriza**: eso lo hace código
   determinista.

Todo corre **en local**: el LLM es Qwen2.5-7B-Instruct en Foundry Local y no
se usa Azure ni ningún modelo en la nube.

## Resultados principales

Dataset sintético de **3 299 tickets** (holdout estratificado de 660, separado
antes de cualquier ajuste). Cifras completas en
[outputs/metricas_finales.json](outputs/metricas_finales.json).

**Validación cruzada estratificada (k = 5) sobre el 80 %**
([comparacion_modelos.csv](outputs/comparacion_modelos.csv)):

| Modelo | F1 macro | Recall `paneles_totems` | Accuracy |
|---|---|---|---|
| Regresión Logística (principal) | 0,871 ± 0,009 | 0,927 ± 0,051 | 0,873 |
| SVM lineal (alternativa) | 0,874 ± 0,010 | 0,918 ± 0,037 | 0,875 |
| Naive Bayes Multinomial (línea base) | 0,799 ± 0,028 | 0,583 ± 0,111 | 0,829 |

La SVM queda 0,003 por encima en F1 macro, una diferencia menor que la
desviación entre pliegues; la Regresión Logística iguala ese valor tras el
`GridSearchCV` (0,874; C = 2) y tiene mejor recall en `paneles_totems`.
Se mantiene como modelo final porque, además, da **probabilidades**, que el
umbral de confianza necesita (tal como plantea el informe).

**Modelo final en el holdout** ([reporte](outputs/reporte_clasificacion_test.txt)):

| Métrica | Valor | Meta | ¿Cumple? |
|---|---|---|---|
| F1 macro | **0,874** | ≥ 0,80 | ✅ |
| Recall `paneles_totems` | **0,923** (48/52) | ≥ 0,90 | ✅ |
| Accuracy | 0,870 | – | – |

La clase más difícil es `otros` (F1 0,77), que se confunde con `aplicativos`.
En los tickets ambiguos y con etiqueta ruidosa la accuracy cae a 0,11 y 0,13,
como es de esperar.

**Umbral de confianza** ([analisis_umbral.csv](outputs/analisis_umbral.csv)):
se recomienda **0,6**. Deriva al técnico el 21 % de los tickets y los
clasificados automáticamente tienen 94,8 % de precisión. Criterio: la mayor
precisión automática sin derivar más del 25 % (en las sedes pequeñas hay a
veces un solo técnico). Con 0,7 la precisión prácticamente no cambia
(94,83 % frente a 94,80 %) y la derivación sube al 30 %.

**Cola FIFO vs. priorizada** ([simulacion_cola.csv](outputs/simulacion_cola.csv),
[gráfico](outputs/simulacion_cola.png)), posición promedio (1 = primero):

| Escenario | Prioridad real | FIFO (actual) | Priorizada con el modelo | Priorizada con la etiqueta real |
|---|---|---|---|---|
| Sede grande (60 tickets) | Crítica | 28,0 | **4,7** | 3,2 |
| Sede grande (60 tickets) | Alta | 31,9 | **13,7** | 13,7 |
| Sede pequeña (25 tickets) | Crítica | 14,1 | **2,5** | 1,9 |
| Sede pequeña (25 tickets) | Alta | 12,8 | **6,2** | 6,2 |

Respuesta a la pregunta de investigación: con la clasificación automática y la
matriz, un ticket crítico pasa de la posición ~28 a la ~5 en una sede grande.
Los errores del modelo explican la diferencia con la cola ideal: un tótem mal
clasificado puede quedar hasta en la posición 39 (máximo).

**LLM zero-shot vs. Regresión Logística** ([comparacion_llm.csv](outputs/comparacion_llm.csv)),
mismo holdout:

| Modelo | F1 macro | Recall `paneles_totems` | Latencia / ticket | Respuestas mal formadas |
|---|---|---|---|---|
| Regresión Logística | **0,874** | **0,923** | **2,6 ms** | 0 % |
| Qwen2.5-7B zero-shot (GPU) | 0,686 | 0,442 | 308 ms | 0 % |

Qwen clasifica los televisores de la sala de espera como `otros` y los tótems
que no imprimen como `impresion`, aunque el prompt define la clase. Es la
evidencia empírica de por qué el LLM no es el clasificador principal.

## Entorno de ejecución de los resultados

Todos los resultados de este README se obtuvieron en un solo equipo, entre el
16 y el 17 de octubre de 2026.

| Componente | Detalle |
|---|---|
| Equipo | Dell Pro Max 16 Premium (portátil), Windows 11 Pro 10.0.26200 |
| CPU | Intel Core Ultra 9 285H (16 núcleos) |
| RAM | 32 GB |
| GPU | NVIDIA RTX PRO 2000 Blackwell Laptop, 8 GB de VRAM (driver 595.79) |
| Python | 3.13.11 |
| scikit-learn | 1.9.1 (pandas 3.0.6, numpy 2.5.4) |
| Microsoft Agent Framework | agent-framework-core 1.17.0, agent-framework-openai 1.14.2, agent-framework-foundry-hosting 1.0.0b260821 |
| Foundry Local | 0.11.0 |
| Modelo LLM | `qwen2.5-7b-instruct-cuda-gpu:4` (Qwen2.5-7B-Instruct, cuantizado, en la GPU) |

Las versiones exactas de Python están fijadas en `uv.lock`.

**Cómo se midió la latencia.** La Regresión Logística corre en la CPU y
clasifica un ticket a la vez (`predict_proba` sobre una fila, con limpieza y
TF-IDF incluidos). Qwen corre en la GPU: se mide el tiempo de cada llamada HTTP
al endpoint local, con `temperature = 0` y `max_tokens = 40`. Los
**2,6 ms frente a 308 ms** son promedios sobre los 660 tickets del holdout. La
corrida completa del zero-shot sumó 203 s de inferencia.

**Configuración del generador (Agente 1).** Parámetros guardados en
[data/prompts/generador_parametros.json](data/prompts/generador_parametros.json).

| Parámetro | Valor |
|---|---|
| Modelo | `qwen2.5-7b-instruct-cuda-gpu:4` |
| `temperature` / `top_p` | 0,9 / 0,95 |
| `max_tokens` por llamada | 1 400 |
| Tamaño del lote | 15 descripciones por llamada (JSON) |
| `seed` | 42 · 100 000 + n.º de llamada |
| Llamadas realizadas | 334 (1 respuesta mal formada, descartada) |
| Descripciones recibidas | 4 868 → 3 217 tras deduplicar y muestrear, + 82 ambiguas generadas por código = 3 299 |
| Latencia por llamada | 6,3 s en promedio (mín. 2,8 s; máx. 12,7 s) |
| Tiempo total de inferencia | 2 108 s (≈ 35 min) |

El tiempo total es la suma de las latencias de las llamadas. La duración de
reloj no se registró y fue algo mayor por el procesamiento entre llamadas.
Durante parte de la generación también corrió en la CPU una prueba del
pipeline, lo que pudo alargar algunas llamadas.

## Limitaciones

- **Dataset sintético.** Las métricas valen para este dataset; deben
  validarse con tickets reales de Jira. Los generó el mismo tipo de modelo que
  luego se compara en zero-shot.
- **Ruido no planificado del generador.** A veces Qwen se salió del tipo pedido
  (por ejemplo, "Impr 200 no imprime" en un lote de `aplicativos`), así que hay
  etiquetas incorrectas además del 2,5 % de ruido inyectado. Se ven en
  [errores_frecuentes.csv](outputs/errores_frecuentes.csv).
- **Orquestación del agente.** Qwen2.5-7B a veces escribe la llamada a la
  herramienta como texto en lugar de ejecutarla. En las conversaciones
  grabadas, 6 de 8 casos salen bien. La lógica determinista nunca guardó un
  dato incorrecto, pero el usuario debe repetir el mensaje (ver
  [outputs/agente/](outputs/agente/README.md)).
- `confirmar_tipo` guarda la corrección para reentrenar, pero **no recalcula**
  la prioridad del ticket.

## Matriz de priorización (Tabla 1)

Se evalúa de arriba hacia abajo:

| Nivel | Prioridad | Criterio |
|---|---|---|
| 1 | Crítica | tipo `paneles_totems`, en cualquier área |
| 2 | Alta | cualquier tipo en `consultorio` |
| 3 | Media | cualquier tipo en `triaje` o `counter` (admisión cuenta como counter) |
| 4 | Baja | cualquier tipo en `administrativo` |

Al registrar un ticket, el área se traduce por código a una de estas cuatro
(`admisión`, `caja`, `recepción` → counter; `contabilidad`, `RR. HH.`,
`logística` → administrativo); cualquier otra se rechaza y el agente pregunta.

Si la probabilidad máxima del modelo es menor que el umbral, el ticket se marca
`revision_tecnico = 1`, pero **igual se prioriza y entra a la cola**.

## Requisitos

- Windows, [uv](https://docs.astral.sh/uv/getting-started/installation/)
  (`winget install --id=astral-sh.uv -e`) y
  [Foundry Local](https://learn.microsoft.com/azure/foundry-local/get-started).
- VS Code con **Python Debugger** y **Foundry Toolkit** (F5 y Agent Inspector).
- Un solo entorno virtual para el agente y el ML (el de la plantilla). El
  modelo `.joblib` debe cargarse con la misma versión de scikit-learn con la
  que se entrenó; `uv.lock` la fija.

## Instalación

```powershell
Set-Location .\src\agent-framework-agent-basic-responses
uv sync --frozen
Copy-Item .env.example .env      # luego elige el modelo según tu equipo
```

| Equipo | `FOUNDRY_LOCAL_MODEL` en `.env` |
|---|---|
| GPU NVIDIA con 8 GB de VRAM o más | `qwen2.5-7b-instruct-cuda-gpu:4` |
| Solo CPU (16 GB de RAM o más) | `qwen2.5-7b-instruct-generic-cpu:4` |

Usa siempre el ID completo de la variante, no el alias (`qwen2.5-7b`); con el
alias la API local responde `404 Model not found`.

## Ejecución por fases

Todos los scripts del pipeline se ejecutan **desde la raíz del repositorio**
con el entorno del agente. Para abreviar:

```powershell
function ia { uv run --frozen --project src/agent-framework-agent-basic-responses python -m @args }
```

| Fase | Comando | Necesita LLM | Genera |
|---|---|---|---|
| 1. Datos (Agente 1) | `ia ml.generador` | Sí (~1 h en GPU) | `data/tickets_sinteticos.csv`, `outputs/dataset_resumen.csv`, `outputs/ejemplos_dataset.csv` |
| 1. Limpieza | `ia ml.limpieza` | No | `outputs/ejemplos_limpieza.csv` |
| 2. Entrenamiento | `ia ml.entrenamiento` | No | `models/modelo_rl.joblib`, `outputs/comparacion_modelos.csv`, `outputs/mejores_hiperparametros.json` |
| 2. Evaluación | `ia ml.evaluacion` | No | reporte, matriz de confusión, top términos, análisis del umbral, errores, métricas |
| 3. Priorización | `ia ml.priorizacion` | No | `outputs/simulacion_cola.csv` y `.png` |
| 4. Agente de triaje | F5 en VS Code (ver abajo) | Sí | `data/tickets.db` |
| 4. Conversaciones | `ia ml.conversaciones_demo` (con el agente corriendo) | Sí | `outputs/agente/conversacion_XX.md` |
| 4. Preparar la demo | `ia ml.preparar_demo` | No | `data/tickets.db` con 9 tickets precargados; respaldo en `data/respaldos/` |
| 5. Zero-shot (Agente 3) | `ia ml.comparacion_llm` | Solo la primera vez (~4 min en GPU); después reutiliza `data/generacion/zero_shot.jsonl` | `outputs/comparacion_llm.csv` |

El generador tiene dos fases: `--fase generar` (llama a Qwen y guarda cada lote
en `data/generacion/lotes.jsonl`; si se corta, el mismo comando **reanuda**) y
`--fase ensamblar` (determinista: deduplica, inyecta ruido y asigna sede y
fecha). Las fases 2, 3 y 5 se reproducen exactamente con el dataset incluido,
sin volver a llamar al LLM: la fase 5 reutiliza las respuestas de Qwen
guardadas.

**Para reproducir los resultados sin Foundry Local**, ejecuta en este orden:

```powershell
ia ml.limpieza; ia ml.entrenamiento; ia ml.evaluacion; ia ml.priorizacion; ia ml.comparacion_llm
```

`ml.entrenamiento` reinicia `outputs/metricas_finales.json`, y cada fase
siguiente agrega su sección; por eso hay que correr las cinco. Las métricas
salen idénticas; solo cambian los tiempos medidos (entrenamiento y latencia de
la RL) y la fecha de la versión del modelo.

### Agente de triaje en Agent Inspector

1. Entrena el modelo (fase 2) para que exista `models/modelo_rl.joblib`.
2. Abre la **carpeta raíz** en VS Code, pulsa **F5** y elige
   **"Debug Local Agent HTTP Server"**. Se abre Agent Inspector en el puerto `8088`.
3. Escribe, por ejemplo:
   - `Ticket: el tótem de admisión no deja sacar turno. Área: admisión. Sede: San Isidro.`
   - `¿Qué tickets pendientes hay en San Isidro?`
   - `El ticket 3 en realidad era de impresión.`

Alternativa por terminal: `uv run --frozen python main.py` desde
`src/agent-framework-agent-basic-responses` y luego
**Foundry Toolkit: Open Agent Inspector** (puerto 8088).

Herramientas del agente
([herramientas.py](src/agent-framework-agent-basic-responses/herramientas.py)):

| Herramienta | Qué hace (código determinista) |
|---|---|
| `registrar_ticket(descripcion, area, sede)` | Valida área y sede contra el catálogo → limpia → clasifica (RL) → aplica umbral → prioriza (Tabla 1) → guarda en SQLite. Devuelve id, tipo, probabilidad, prioridad, `revision_tecnico` y los 3 términos con más peso. |
| `ver_cola(sede)` | Tickets pendientes ordenados por (nivel, fecha), como tabla markdown. La posición se calcula al consultar. |
| `confirmar_tipo(id_ticket, tipo_real)` | Guarda la corrección del técnico en `tipo_real` (para reentrenar). |

El agente conserva el MCP de Microsoft Learn (solo lectura) para sugerir pasos
de solución de productos Microsoft. **No envíes datos de pacientes, contraseñas
ni información interna** en esas consultas: van a internet.

## Datos sintéticos (Agente 1)

- Qwen2.5-7B local redacta lotes de 15 descripciones en JSON por combinación
  (tipo, área) y estilo: formal, apurado, con abreviaturas, con errores, muy
  corto/ambiguo y narrativo. Cada prompt sortea 4 situaciones concretas del
  tipo para dar variedad. Prompts y parámetros en [data/prompts/](data/prompts/);
  `temperature = 0.9`, `top_p = 0.95`, `seed = 42·100000 + n.º de llamada`
  (Foundry Local puede no garantizar determinismo exacto con la misma semilla;
  por eso se guardan todas las respuestas crudas).
- Por código (semilla 42): errores de digitación en el 15 % de los textos,
  2,5 % de tickets genéricos ambiguos ("urgente no funciona") y 2,5 % de
  etiquetas ruidosas (confusión plausible del técnico). Las columnas
  `estilo`, `errores_inyectados`, `ambiguo` y `etiqueta_ruidosa` permiten
  analizar el efecto del ruido; **no se usan como variables del modelo**.
- Distribución desbalanceada: `paneles_totems` ≈ 8 %. Sedes: San Isidro y
  Miraflores (grandes, peso 60) y Surco, La Molina y San Miguel (pequeñas,
  peso 25). Fechas entre el 1 de abril y el 29 de setiembre de 2026, con más
  carga de lunes a viernes y en horas punta.
- Volumen: 2 sedes × 60 + 3 × 25 = 195 tickets/semana darían ≈ 5 000 en
  26 semanas. Se generaron ≈ 3 300, así que el
  dataset equivale a una **muestra de ≈ 65 %** de los tickets del semestre con
  la misma proporción entre sedes.

## Estructura

```text
.vscode/                           F5 y Agent Inspector (plantilla)
src/agent-framework-agent-basic-responses/
├── main.py                        Agente de triaje (instrucciones + servidor en 8088)
├── herramientas.py                registrar_ticket, ver_cola, confirmar_tipo
├── .env.example                   Modelo de Foundry Local (copiar a .env)
├── pyproject.toml, uv.lock        Entorno único: agente + ML
└── tests/                         Pruebas de la plantilla y de las herramientas
ml/
├── config.py                      Rutas, catálogos cerrados y semilla
├── limpieza.py                    Limpieza de texto y del dataset
├── generador.py                   Agente 1: datos sintéticos con Qwen local
├── entrenamiento.py               Holdout, CV (RL, SVM, NB), GridSearchCV, modelo final
├── evaluacion.py                  Métricas en el holdout, umbral, errores
├── clasificador.py                Predicción de un ticket + términos clave (para el agente)
├── priorizacion.py                Tabla 1 y simulación FIFO vs. priorizada
├── db.py                          SQLite con consultas parametrizadas
├── llm_local.py                   Cliente del endpoint loopback de Foundry Local
├── comparacion_llm.py             Agente 3: Qwen zero-shot vs. RL
├── conversaciones_demo.py         Graba conversaciones de ejemplo del agente
└── preparar_demo.py               Respaldo + cola precargada para la demo en vivo
data/        tickets_sinteticos.csv, holdout_ids.csv, prompts/, generacion/ (respuestas crudas)
models/      modelo_rl.joblib, version_modelo.json (versión, hiperparámetros, umbral)
outputs/     salidas para el informe (y outputs/agente/ con las conversaciones)
```

## Pruebas

```powershell
Set-Location .\src\agent-framework-agent-basic-responses
uv run --frozen python -m unittest discover -s tests -v
```

## Solución de problemas

| Síntoma | Solución |
|---|---|
| `Falta FOUNDRY_LOCAL_MODEL` | Copia `.env.example` a `.env` |
| `No existe models/modelo_rl.joblib` (al registrar un ticket) | Ejecuta `ia ml.entrenamiento` y `ia ml.evaluacion` |
| `did not find executable at 'C:\Users\<otra persona>\...'` | Borra `.venv` y ejecuta `uv sync --frozen` |
| `404 Model not found` | Usa el ID completo del modelo (`foundry model list --loaded -v`) |
| F5 se detiene en "Validate prerequisites" | Cierra otras instancias que usen los puertos 5679 u 8088 |
| Respuesta con texto como `<tool_call>` | El modelo generó una llamada mal formada: repite el mensaje |
| Cada respuesta tarda ~30 s (antes 5–10 s) | `OTEL_EXPORTER_OTLP_ENDPOINT` está activo en `.env` y el colector de Tracing está apagado: coméntalo o enciende el colector |
| Todo lento aunque el modelo está en GPU | VRAM llena (`nvidia-smi` cerca de 8 GB): `foundry server stop`, cierra apps que usen la GPU y vuelve a pulsar F5 |

## Equipo

Curso IS302 – Inteligencia Artificial, sección 2095.

- Carrillo Ormeño, Italo Marco Jesus
- Julca Gallardo, Yali Anderson
- Lázaro Lizano, Santos Adrián
- León Ramirez, Anthony Frank
- Quispe Ccaccya, Johanni Angie Stephani
