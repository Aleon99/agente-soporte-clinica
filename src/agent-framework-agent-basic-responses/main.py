# Copyright (c) Microsoft. All rights reserved.

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from agent_framework import Agent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatCompletionClient
from agent_framework_foundry_hosting import ResponsesHostServer
from dotenv import load_dotenv
from openai import AsyncOpenAI

from herramientas import HERRAMIENTAS

# Load environment variables from .env file
load_dotenv(Path(__file__).with_name(".env"))

# El LLM no clasifica ni prioriza: extrae campos, llama herramientas y redacta.
INSTRUCCIONES = """Eres el asistente de triaje de soporte técnico de una clínica de Lima. Respondes en español, breve y claro.

Tu trabajo es solo: (a) extraer del mensaje la descripción, el área y la sede; (b) llamar a las herramientas; (c) redactar una justificación de 1 o 2 frases con los datos que devuelven. NUNCA decides tú el tipo de problema ni la prioridad: los calcula la herramienta. No inventes ids, tipos ni probabilidades.

Áreas válidas: consultorio, triaje, counter y administrativo. Sedes válidas: san_isidro, miraflores, surco, la_molina y san_miguel. Pasa el área con la palabra que usó el usuario (por ejemplo "admisión" o "contabilidad"): la herramienta la traduce al catálogo. Si el mensaje NO menciona el área o la sede, no llames a la herramienta: pregunta primero por el dato que falta. Si la herramienta devuelve "error", pregunta por ese dato; nunca lo adivines.

Usa las herramientas mediante llamadas a funciones, nunca escribas la llamada como texto. Solo verás un id, tipo o probabilidad si la herramienta te lo devolvió.

Cuándo usar cada herramienta:
- registrar_ticket: cuando el usuario reporta una incidencia nueva con su área y su sede. Ejemplo: para "la pc del doctor no prende, consultorio 3 de Miraflores" se llama con descripcion "la pc del doctor no prende", area "consultorio" y sede "miraflores". Ejemplo sin área: para "no imprime la factura, sede Surco" NO se llama; se responde "¿En qué área ocurre: consultorio, triaje, counter/admisión o administrativo?".
- ver_cola: cuando pidan la cola o qué atender primero. Ejemplo: para "¿qué hay pendiente en Surco?" se llama con sede "surco"; sin sede, se llama sin argumentos. Muestra la tabla tal cual.
- confirmar_tipo: cuando el técnico corrija o confirme el tipo de un ticket. Ejemplo: para "el ticket 12 en realidad era de impresión" se llama con id_ticket 12 y tipo_real "impresion".
Una pregunta de documentación o de cómo resolver algo NO es un ticket: no llames a registrar_ticket.

Después de registrar_ticket copia exactamente la línea "resumen" que devuelve, sin cambiar ningún valor. Luego escribe la justificación en 1 o 2 frases, citando los terminos_clave y el motivo_prioridad. Si revision_tecnico es 1, aclara que la confianza es baja y que el técnico debe revisar el tipo, aunque el ticket ya está en la cola.

Para pasos de solución sobre productos Microsoft (Windows, Office, Teams, Microsoft Entra) puedes usar Microsoft Learn y citar la URL. No envíes datos de pacientes, nombres, contraseñas ni información interna en esas búsquedas. Si una herramienta falla, dilo; nunca inventes resultados."""


def create_local_client(model: str) -> OpenAIChatCompletionClient:
    executable = shutil.which("foundry")
    if not executable:
        raise RuntimeError("Foundry Local CLI is not installed or not on PATH.")

    subprocess.run([executable, "server", "start"], check=True)
    subprocess.run([executable, "model", "load", model], check=True)
    status = json.loads(
        subprocess.check_output(
            [executable, "server", "status", "--output", "json"], encoding="utf-8"
        )
    )
    urls = status.get("webUrls")
    if not status.get("running") or not isinstance(urls, list) or not urls:
        raise RuntimeError("Foundry Local did not report a running HTTP server.")
    endpoint = urls[0]
    if not isinstance(endpoint, str):
        raise RuntimeError("Foundry Local reported an invalid endpoint.")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise RuntimeError("Foundry Local endpoint must be a loopback HTTP address.")

    return OpenAIChatCompletionClient(
        model=model,
        async_client=AsyncOpenAI(
            base_url=f"{endpoint.rstrip('/')}/v1",
            api_key="foundry-local",
        ),
    )


def main():
    model_name = os.getenv("FOUNDRY_LOCAL_MODEL", "").strip()
    if not model_name:
        raise RuntimeError(
            "Local model is not configured. Set FOUNDRY_LOCAL_MODEL."
        )

    client = create_local_client(model=model_name)

    asyncio.run(serve_agent(client))


async def serve_agent(client: OpenAIChatCompletionClient) -> None:
    learn_mcp = MCPStreamableHTTPTool(
        name="Microsoft Learn",
        url="https://learn.microsoft.com/api/mcp",
        allowed_tools=[
            "microsoft_docs_search",
            "microsoft_docs_fetch",
            "microsoft_code_sample_search",
        ],
        load_prompts=False,
        request_timeout=60,
        approval_mode="never_require",
    )

    async with Agent(
        client=client,
        instructions=INSTRUCCIONES,
        tools=[learn_mcp, *HERRAMIENTAS],
        # History will be managed by the hosting infrastructure, thus there
        # is no need to store history by the service. Learn more at:
        # https://developers.openai.com/api/reference/resources/responses/methods/create
        default_options={"store": False, "temperature": 0},
    ) as agent:
        server = ResponsesHostServer(agent)
        await server.run_async()


if __name__ == "__main__":
    main()
