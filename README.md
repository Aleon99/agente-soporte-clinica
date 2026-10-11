# Agente de soporte técnico para una clínica (avance PC2)

Agente de IA que apoya al área de soporte técnico de una clínica de Lima a
**clasificar y priorizar incidencias** registradas en Jira. Es el avance del
trabajo final del curso **IS302 – Inteligencia Artificial** (UPC).

El agente se ejecuta **100 % en local** con Microsoft Agent Framework y
[Foundry Local](https://learn.microsoft.com/azure/foundry-local/get-started).
No requiere suscripción de Azure, claves de API ni modelos en la nube. Para
consultar documentación oficial se conecta al servidor MCP público de
Microsoft Learn.

> **Alcance del avance.** El agente es un prototipo de interfaz. Según el
> informe, la versión final clasificará los tickets con Regresión Logística
> sobre TF-IDF y aplicará la matriz de priorización de forma explícita. En esta
> etapa, la prioridad que sugiere el agente es orientativa y el técnico siempre
> tiene la decisión final.

## Qué hace el agente

- **Clasifica** cada ticket según su descripción, área y sede en: Impresión,
  Encendido, Bloqueo de usuarios, Aplicativos, Paneles o tótems, u Otros.
- **Sugiere la prioridad** con la matriz del informe:

  | Prioridad | Criterio | Impacto |
  |---|---|---|
  | Crítica | Paneles o tótems, en cualquier área | Desorden inmediato de la sala de espera |
  | Alta | Cualquier problema en consultorios médicos | Detención de la consulta médica |
  | Media | Cualquier problema en triaje o counters | Retraso en la atención al paciente |
  | Baja | Cualquier problema en áreas administrativas | Sin impacto asistencial directo |

- **Deriva al técnico** los tickets ambiguos o con baja confianza.
- **Consulta Microsoft Learn por MCP** para sugerir pasos de solución sobre
  productos Microsoft (Windows, Office, Teams, Microsoft Entra) y cita la URL.

Las instrucciones completas del agente están en
[main.py](src/agent-framework-agent-basic-responses/main.py).

## Modelo seleccionado

**Qwen2.5-7B-Instruct**, ejecutado con Foundry Local.

Se compararon tres modelos locales con los mismos casos de prueba:

| Modelo | Uso de herramientas (MCP) | Resultado |
|---|---|---|
| Phi-4-mini | ❌ Escribe la llamada como texto e inventa enlaces | Descartado |
| Qwen3-4B | ⚠️ Genera llamadas mal formadas | Descartado |
| **Qwen2.5-7B-Instruct** | ✅ Decide cuándo consultar Microsoft Learn y cita fuentes reales | **Elegido** |

Además responde bien en español, corre sin conexión a la nube y su licencia
(Apache 2.0) permite el uso académico.

### Variante según el equipo

| Equipo | Valor en `.env` | Tiempo por ticket |
|---|---|---|
| GPU NVIDIA con 8 GB de VRAM o más | `qwen2.5-7b-instruct-cuda-gpu:4` | ~3–5 s |
| Solo CPU (16 GB de RAM o más) | `qwen2.5-7b-instruct-generic-cpu:4` | ~20–35 s |

En CPU, una consulta al MCP puede tardar varios minutos porque el modelo debe
leer unos 15 000 tokens de documentación.

> Usa siempre el **ID completo** de la variante, no el alias (`qwen2.5-7b`).
> La API HTTP de Foundry Local responde `404 Model not found` con el alias.
> Para ver el ID de un modelo cargado: `foundry model list --loaded -v`.

## Requisitos

- Windows (las tareas de VS Code usan `.venv\Scripts\python.exe`).
- [Foundry Local](https://learn.microsoft.com/azure/foundry-local/get-started).
- [uv](https://docs.astral.sh/uv/getting-started/installation/):
  ```powershell
  winget install --id=astral-sh.uv -e
  ```
  Después de instalarlo, cierra y vuelve a abrir VS Code. Si falta Python 3.13,
  `uv` lo descarga solo.
- VS Code con las extensiones **Python Debugger** y **Foundry Toolkit**
  (necesarias para F5 y Agent Inspector).
- Conexión a internet para la primera descarga del modelo y para el MCP.

## Instalación

Desde la raíz del repositorio:

```powershell
Set-Location .\src\agent-framework-agent-basic-responses
uv sync --frozen
```

Configura el modelo en
[.env](src/agent-framework-agent-basic-responses/.env) según tu equipo (ver la
tabla anterior). Por ejemplo, para CPU:

```dotenv
FOUNDRY_LOCAL_MODEL=qwen2.5-7b-instruct-generic-cpu:4
```

La primera vez, descarga el modelo (unos 5–6 GB):

```powershell
foundry model download qwen2.5-7b-instruct-generic-cpu:4
```

## Ejecución

### Opción A: F5 con Agent Inspector (recomendada)

1. Abre en VS Code la **carpeta raíz** del repositorio.
2. Pulsa **F5** y elige **"Debug Local Agent HTTP Server"**.
3. Se abre Agent Inspector conectado al puerto `8088`. Escribe los mensajes en
   su chat.
4. Para detener todo, pulsa Shift+F5.

F5 valida que los puertos `5679` (depurador) y `8088` (agente) estén libres,
arranca el agente y abre el Inspector.

### Opción B: terminal

```powershell
Set-Location .\src\agent-framework-agent-basic-responses
uv run --frozen python main.py
```

Cuando aparezca `Running on http://0.0.0.0:8088`, abre
**Ctrl+Shift+P → Foundry Toolkit: Open Agent Inspector** y conéctalo al puerto
`8088`.

Al iniciar, `main.py` arranca Foundry Local, carga el modelo del `.env` y
detecta el puerto del servicio. Solo acepta endpoints locales (loopback).

## Ejemplos de uso

```text
Hola, ¿qué puedes hacer?
```

```text
Ticket: la pc del doctor no prende. Área: Consultorio de cardiología. Sede: Miraflores.
```
Esperado: Encendido / Alta.

```text
Ticket: no puedo imprimir las facturas. Área: Contabilidad (administrativa). Sede: San Isidro.
```
Esperado: Impresión / Baja.

```text
Consulta de documentación: ¿cómo reinicio el servicio de cola de impresión en Windows? Usa Microsoft Learn y cita la URL.
```
Esperado: una llamada a `microsoft_docs_search`, visible en la traza de Agent
Inspector, con resultados de `learn.microsoft.com`.

## Microsoft Learn MCP

El agente se conecta al endpoint público `https://learn.microsoft.com/api/mcp`
y solo expone herramientas de lectura:

- `microsoft_docs_search`: busca en la documentación de Microsoft.
- `microsoft_docs_fetch`: lee un artículo.
- `microsoft_code_sample_search`: busca ejemplos de código oficiales.

La inferencia del modelo se queda en el equipo, pero las búsquedas se envían a
Microsoft Learn. **No incluyas datos de pacientes, contraseñas ni información
interna** en las consultas.

## Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| `did not find executable at 'C:\Users\<otra persona>\...python.exe'` | El `.venv` se copió de otro equipo | Borra `.venv` y ejecuta `uv sync --frozen` |
| `uv` no se reconoce | uv no está instalado o VS Code no ve el nuevo PATH | Instala uv y reinicia VS Code |
| `404 Model not found` al enviar un mensaje | El `.env` usa el alias del modelo | Pon el ID completo (`foundry model list --loaded -v`) |
| F5 se detiene en "Validate prerequisites" | Los puertos 5679 u 8088 están ocupados | Cierra otras instancias de `main.py` |
| Respuestas lentas en GPU | La VRAM se llenó y Windows usa memoria compartida | Ejecuta `foundry server stop`, cierra apps que usen la GPU y vuelve a pulsar F5 |
| Todo se vuelve lento después de una consulta al MCP | El historial arrastra unos 15 000 tokens de documentación | Abre una conversación nueva en el Inspector |
| Respuesta con texto como `brtc` o `<tool_call>` | El modelo generó una llamada mal formada | Repite el mensaje |

## Limitaciones conocidas

- La prioridad que sugiere el modelo es orientativa. Por ejemplo, a veces
  asigna *Alta* a un tótem cuando la matriz indica *Crítica*. En la versión
  final la matriz se aplicará de forma explícita.
- El uso de herramientas MCP depende del modelo y no es 100 % estable.
- No incluyas la carpeta `.venv` al compartir el proyecto: depende del equipo
  donde se creó.

## Pruebas

Desde `src/agent-framework-agent-basic-responses`:

```powershell
uv run --frozen python -m unittest discover -s tests -v
```

## Estructura

```text
.vscode/                         Tareas y configuración de F5 (Agent Inspector)
src/agent-framework-agent-basic-responses/
├── main.py                      Cliente del modelo local, instrucciones y servidor HTTP
├── .env                         Modelo de Foundry Local (FOUNDRY_LOCAL_MODEL)
├── pyproject.toml, uv.lock      Dependencias reproducibles
└── tests/                       Pruebas de configuración
```

## Equipo

Curso IS302 – Inteligencia Artificial, sección 2095.

- Carrillo Ormeño, Italo Marco Jesus
- Julca Gallardo, Yali Anderson
- Lázaro Lizano, Santos Adrián
- León Ramirez, Anthony Frank
- Quispe Ccaccya, Johanni Angie Stephani
