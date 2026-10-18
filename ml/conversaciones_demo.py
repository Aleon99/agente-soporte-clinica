"""Graba conversaciones de ejemplo con el agente de triaje para el informe.

Requiere el agente corriendo en http://localhost:8088 (F5 o `python main.py`).
Envía cada conversación por la API Responses local, encadenando los turnos con
previous_response_id, y guarda la transcripción (incluidas las llamadas a
herramientas y sus resultados) en outputs/agente/conversacion_XX.md.

Uso:  python -m ml.conversaciones_demo
"""

import argparse
import json
import time

from openai import OpenAI

from ml.config import DIR_OUTPUTS

DIR_SALIDA = DIR_OUTPUTS / "agente"

CONVERSACIONES = [
    ("Tótem caído en admisión (crítica)", [
        "Ticket: el tótem de autoatención de la entrada no deja sacar turno, los pacientes "
        "están haciendo cola. Área: admisión. Sede: San Isidro.",
    ]),
    ("PC del médico no enciende (alta)", [
        "La pc del doctor no prende y tiene paciente en consulta. Consultorio 4, sede Miraflores.",
    ]),
    ("Falta el área: el agente pregunta", [
        "No puedo imprimir las facturas, sale error en la cola de impresión. Sede Surco.",
        "Es en contabilidad, área administrativa.",
    ]),
    ("Área fuera del catálogo", [
        "Ticket: el sistema de resultados no carga. Área: laboratorio. Sede: La Molina.",
        "Ok, entonces regístralo en triaje.",
    ]),
    ("Ticket ambiguo: derivado al técnico", [
        "urgente no funciona nada. triaje, san miguel",
    ]),
    ("Cola de una sede", [
        "Registra: la cuenta de la enfermera se bloqueó por intentos fallidos. Triaje, San Isidro.",
        "Registra también: el panel de llamados de la sala de espera no muestra los turnos. "
        "Consultorio, San Isidro.",
        "¿Qué tickets pendientes hay en San Isidro y cuál atiendo primero?",
    ]),
    ("Corrección del técnico", [
        "Registra: Outlook no sincroniza los correos desde la mañana. Administrativo, Miraflores.",
        "Revisé el ticket que acabas de registrar y en realidad era bloqueo de usuario, la "
        "contraseña había vencido. Corrige el tipo.",
    ]),
    ("Documentación de Microsoft Learn", [
        "¿Cómo reinicio el servicio de cola de impresión en Windows? Usa Microsoft Learn y cita la URL.",
    ]),
]


def describir_salida(respuesta) -> list[str]:
    """Convierte los items de salida de la Responses API en líneas de markdown."""
    lineas = []
    for item in respuesta.output:
        tipo = getattr(item, "type", "")
        if tipo == "function_call":
            lineas.append(f"> 🔧 **Llamada** `{item.name}({item.arguments})`")
        elif tipo == "function_call_output":
            salida = item.output if isinstance(item.output, str) else json.dumps(item.output, ensure_ascii=False)
            lineas.append("> 📦 **Resultado**\n>\n> " + str(salida).replace("\n", "\n> "))
        elif tipo == "mcp_call":
            lineas.append(f"> 🔧 **MCP** `{item.name}({item.arguments})`")
        elif tipo == "message":
            texto = "".join(getattr(c, "text", "") for c in item.content)
            lineas.append(f"**Agente:** {texto}")
    return lineas


def main(url: str) -> None:
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    cliente = OpenAI(base_url=url, api_key="local")
    for n, (titulo, turnos) in enumerate(CONVERSACIONES, start=1):
        lineas = [f"# Conversación {n}: {titulo}", ""]
        anterior = None
        for mensaje in turnos:
            inicio = time.perf_counter()
            respuesta = cliente.responses.create(
                model="agente-triaje", input=mensaje, previous_response_id=anterior)
            anterior = respuesta.id
            lineas += [f"**Usuario:** {mensaje}", "", *describir_salida(respuesta),
                       "", f"<sub>{time.perf_counter() - inicio:.1f} s</sub>", ""]
        ruta = DIR_SALIDA / f"conversacion_{n:02d}.md"
        ruta.write_text("\n".join(lineas), encoding="utf-8")
        print(f"{ruta.name}: {titulo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8088",
                        help="URL del agente (si 8088 está ocupado: PORT=8089 python main.py)")
    main(parser.parse_args().url)
