"""Cliente OpenAI hacia el endpoint loopback de Foundry Local (sin nube).

Lo usan los scripts del Agente 1 (generador) y del Agente 3 (comparación
zero-shot). Reutiliza el mismo modelo configurado en el .env del agente.
"""

import json
import os
import shutil
import subprocess
from urllib.parse import urlsplit

from dotenv import load_dotenv
from openai import OpenAI

from ml.config import DIR_AGENTE


def nombre_modelo() -> str:
    load_dotenv(DIR_AGENTE / ".env")
    modelo = os.getenv("FOUNDRY_LOCAL_MODEL", "").strip()
    if not modelo:
        raise RuntimeError(
            "Falta FOUNDRY_LOCAL_MODEL. Copia .env.example a .env en "
            f"{DIR_AGENTE} y define el modelo."
        )
    return modelo


def endpoint_local() -> str:
    """Devuelve la URL del servidor de Foundry Local; exige que sea loopback."""
    ejecutable = shutil.which("foundry")
    if not ejecutable:
        raise RuntimeError("Foundry Local CLI no está instalado o no está en el PATH.")
    estado = json.loads(
        subprocess.check_output(
            [ejecutable, "server", "status", "--output", "json"], encoding="utf-8"
        )
    )
    if not estado.get("running"):
        subprocess.run([ejecutable, "server", "start"], check=True)
        estado = json.loads(
            subprocess.check_output(
                [ejecutable, "server", "status", "--output", "json"], encoding="utf-8"
            )
        )
    urls = estado.get("webUrls") or []
    if not urls or not isinstance(urls[0], str):
        raise RuntimeError("Foundry Local no reportó un servidor HTTP activo.")
    url = urls[0]
    partes = urlsplit(url)
    if partes.scheme != "http" or partes.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError("El endpoint de Foundry Local debe ser loopback HTTP.")
    return url.rstrip("/")


def crear_cliente(cargar_modelo: bool = True) -> tuple[OpenAI, str]:
    modelo = nombre_modelo()
    if cargar_modelo:
        subprocess.run([shutil.which("foundry"), "model", "load", modelo], check=True)
    cliente = OpenAI(base_url=f"{endpoint_local()}/v1", api_key="foundry-local")
    return cliente, modelo
