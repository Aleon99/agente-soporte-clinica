"""Herramientas del agente de triaje.

El LLM NO clasifica ni decide la prioridad: solo extrae los campos del mensaje,
llama a estas herramientas y redacta la justificación. Toda la lógica
(limpiar -> clasificar -> umbral -> priorizar -> guardar) es código determinista.
"""

import sys
from pathlib import Path
from typing import Annotated

from agent_framework import tool

# El pipeline de ML vive en la raíz del repositorio (carpeta ml/).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml import db  # noqa: E402
from ml.clasificador import clasificar  # noqa: E402
from ml.config import AREAS, CLASES, SEDES, SINONIMOS_AREA  # noqa: E402
from ml.limpieza import normalizar_categoria  # noqa: E402
from ml.priorizacion import priorizar  # noqa: E402

MOTIVO_PRIORIDAD = {
    1: "regla 1: paneles o tótems en cualquier área (desorden inmediato de la sala de espera)",
    2: "regla 2: consultorio (detención de la consulta médica)",
    3: "regla 3: triaje o counter (retraso en la atención al paciente)",
    4: "regla 4: área administrativa (sin impacto asistencial directo)",
}


def validar_area(area: str) -> str | None:
    valor = normalizar_categoria(area)
    valor = SINONIMOS_AREA.get(valor, valor)
    return valor if valor in AREAS else None


def validar_sede(sede: str) -> str | None:
    valor = normalizar_categoria(sede)
    return valor if valor in SEDES else None


@tool(name="registrar_ticket")
def registrar_ticket(
    descripcion: Annotated[str, "Descripción del problema tal como la escribió el usuario."],
    area: Annotated[str, "Área: consultorio, triaje, counter (incluye admisión) o administrativo."],
    sede: Annotated[str, "Sede: san_isidro, miraflores, surco, la_molina o san_miguel."],
) -> dict:
    """Registra un ticket nuevo: lo clasifica con el modelo de ML, aplica el umbral
    de confianza, le asigna la prioridad de la matriz y lo guarda en la cola.
    Devuelve id, tipo, probabilidad, prioridad, revision_tecnico y los 3 términos
    con más peso. Si devuelve 'error', pide al usuario el dato que falta."""
    if not descripcion or not descripcion.strip():
        return {"error": "Falta la descripción del problema."}
    area_valida = validar_area(area)
    if area_valida is None:
        return {"error": f"Área no válida: {area!r}.", "areas_validas": AREAS}
    sede_valida = validar_sede(sede)
    if sede_valida is None:
        return {"error": f"Sede no válida: {sede!r}.", "sedes_validas": list(SEDES)}

    try:
        resultado = clasificar(descripcion, area_valida, sede_valida)
    except RuntimeError as error:
        return {"error": str(error)}
    prioridad, nivel = priorizar(resultado["tipo_predicho"], area_valida)
    terminos = resultado["terminos_clave"]
    justificacion = (
        f"Tipo {resultado['tipo_predicho']} (p={resultado['probabilidad']:.2f}); "
        f"términos: {', '.join(terminos) or 'sin términos destacados'}. "
        f"Prioridad {prioridad} por {MOTIVO_PRIORIDAD[nivel]}."
    )
    id_ticket = db.insertar_ticket({
        "descripcion": descripcion.strip(),
        "area": area_valida,
        "sede": sede_valida,
        "tipo_predicho": resultado["tipo_predicho"],
        "probabilidad": resultado["probabilidad"],
        "prioridad": prioridad,
        "nivel_prioridad": nivel,
        "revision_tecnico": resultado["revision_tecnico"],
        "justificacion": justificacion,
        "version_modelo": resultado["version_modelo"],
    })
    revision = "sí" if resultado["revision_tecnico"] else "no"
    return {
        "resumen": (
            f"ID {id_ticket} | Tipo: {resultado['tipo_predicho']} | "
            f"Probabilidad: {resultado['probabilidad']:.2f} | Prioridad: {prioridad} | "
            f"Área: {area_valida} | Revisión del técnico: {revision}"
        ),
        "id": id_ticket,
        "tipo": resultado["tipo_predicho"],
        "probabilidad": resultado["probabilidad"],
        "prioridad": prioridad,
        "nivel_prioridad": nivel,
        "motivo_prioridad": MOTIVO_PRIORIDAD[nivel],
        "revision_tecnico": resultado["revision_tecnico"],
        "umbral": resultado["umbral"],
        "terminos_clave": terminos,
        "area": area_valida,
        "sede": sede_valida,
    }


@tool(name="ver_cola")
def ver_cola(
    sede: Annotated[str | None, "Sede a filtrar; omitir para ver todas las sedes."] = None,
) -> str:
    """Muestra los tickets pendientes en orden de atención (prioridad y luego fecha
    de registro) como una tabla en markdown."""
    sede_valida = None
    if sede:
        sede_valida = validar_sede(sede)
        if sede_valida is None:
            return f"Sede no válida: {sede!r}. Sedes válidas: {', '.join(SEDES)}."
    tickets = db.listar_cola(sede_valida)
    if not tickets:
        return "No hay tickets pendientes" + (f" en {sede_valida}." if sede_valida else ".")
    # Tabla compacta para que se lea bien en el chat: sin la columna Sede si se
    # filtró por sede y con la fecha corta (dd/mm hh:mm).
    columnas = ["Pos", "ID", "Prioridad", "Tipo", "Prob", "Área"]
    if not sede_valida:
        columnas.append("Sede")
    columnas += ["Registro", "Revisión"]
    filas = ["| " + " | ".join(columnas) + " |", "|" + "---|" * len(columnas)]
    for posicion, t in enumerate(tickets, start=1):
        fecha, hora = t["fecha_registro"].split("T")
        celdas = [str(posicion), str(t["id"]), t["prioridad"], t["tipo_predicho"],
                  f"{t['probabilidad']:.0%}", t["area"]]
        if not sede_valida:
            celdas.append(t["sede"])
        celdas += [f"{fecha[8:10]}/{fecha[5:7]} {hora[:5]}",
                   "sí" if t["revision_tecnico"] else "no"]
        filas.append("| " + " | ".join(celdas) + " |")
    return "\n".join(filas)


@tool(name="confirmar_tipo")
def confirmar_tipo(
    id_ticket: Annotated[int, "ID del ticket devuelto por registrar_ticket o ver_cola."],
    tipo_real: Annotated[str, "Tipo confirmado por el técnico: impresion, encendido, "
                              "bloqueo_usuario, aplicativos, paneles_totems u otros."],
) -> dict:
    """Guarda el tipo de problema confirmado por el técnico (se usa para reentrenar)."""
    tipo = normalizar_categoria(tipo_real)
    if tipo not in CLASES:
        return {"error": f"Tipo no válido: {tipo_real!r}.", "tipos_validos": CLASES}
    ticket = db.obtener_ticket(id_ticket)
    if ticket is None:
        return {"error": f"No existe el ticket {id_ticket}."}
    db.guardar_tipo_real(id_ticket, tipo)
    return {
        "id": id_ticket,
        "tipo_predicho": ticket["tipo_predicho"],
        "tipo_real": tipo,
        "coincide": ticket["tipo_predicho"] == tipo,
    }


HERRAMIENTAS = [registrar_ticket, ver_cola, confirmar_tipo]
