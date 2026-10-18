# Conversaciones de ejemplo del agente de triaje

Grabadas con `python -m ml.conversaciones_demo` contra el agente local
(Qwen2.5-7B-Instruct CUDA en Foundry Local, `temperature = 0`), partiendo de
una base de datos vacía. Cada archivo muestra las llamadas a herramientas y su
resultado exacto, además de la respuesta del agente. Son transcripciones
reales, sin editar: se incluyen también los fallos.

| # | Caso | Esperado | Resultado |
|---|---|---|---|
| 1 | Tótem en admisión | `admisión` → counter; paneles_totems → **crítica** | ✅ Correcto |
| 2 | PC del médico no enciende | encendido en consultorio → **alta** | ✅ Correcto |
| 3 | Falta el área | El agente pregunta el área | ❌ Llamada mal formada: el modelo escribe la llamada como texto (`ronics: {...}`); no se registra nada |
| 4 | Área fuera del catálogo (laboratorio) | Error de validación y el agente pregunta | ⚠️ El agente pregunta el área, pero antes intenta `confirmar_tipo(12345)` (id inventado, rechazado por la herramienta) y en el segundo turno vuelve a escribir la llamada como texto (`brtc = {...}`) |
| 5 | Ticket ambiguo "urgente no funciona nada" | p < umbral → `revision_tecnico = 1`, igual entra a la cola | ✅ Correcto (p = 0,51, prioridad media, revisión: sí) |
| 6 | Dos tickets + cola de San Isidro | Cola ordenada por (nivel, fecha) | ✅ Correcto: los dos críticos primero, por orden de llegada |
| 7 | Corrección del técnico | `confirmar_tipo` guarda `tipo_real` | ✅ Correcto (la respuesta repite la probabilidad del modelo junto al tipo corregido, lo que puede confundir) |
| 8 | Documentación de Microsoft Learn | Búsqueda en Learn, sin registrar ticket | ✅ Correcto |

## Análisis

- La **lógica determinista nunca falló**: las 6 filas guardadas en
  `data/tickets.db` tienen el tipo, la prioridad y la marca de revisión que
  corresponden. La validación de las herramientas rechazó el id inventado del
  caso 4.
- Los fallos son del LLM orquestador: **llamadas a herramientas mal formadas**
  (escritas como texto) y, ocasionalmente, una llamada innecesaria. Es el
  mismo problema que se observó al elegir el modelo (ver el README
  principal). Que el LLM solo orqueste limita el impacto: cuando falla no se
  guarda nada incorrecto, pero el usuario debe repetir el mensaje.
- Ajustes del prompt cambian qué casos fallan (en una versión anterior el
  caso 4 funcionaba y el 5 y el 8 fallaban): el comportamiento con un modelo
  de 7B no es estable ante pequeños cambios de redacción.
