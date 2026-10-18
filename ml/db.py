"""Persistencia de tickets en SQLite (data/tickets.db).

Todas las consultas son parametrizadas; el LLM nunca genera SQL.
La posición en la cola NO se guarda: se calcula al consultar.
"""

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from ml.config import RUTA_DB

ESQUEMA = """
CREATE TABLE IF NOT EXISTS tickets (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  fecha_registro   TEXT NOT NULL,
  descripcion      TEXT NOT NULL,
  area             TEXT NOT NULL,
  sede             TEXT NOT NULL,
  tipo_predicho    TEXT NOT NULL,
  probabilidad     REAL NOT NULL,
  prioridad        TEXT NOT NULL,
  nivel_prioridad  INTEGER NOT NULL,
  revision_tecnico INTEGER NOT NULL DEFAULT 0,
  justificacion    TEXT,
  estado           TEXT NOT NULL DEFAULT 'pendiente',
  tipo_real        TEXT,
  version_modelo   TEXT NOT NULL
);
"""

ESTADOS = ("pendiente", "en_atencion", "resuelto")


def conectar(ruta: Path | None = None) -> sqlite3.Connection:
    ruta = ruta or RUTA_DB
    ruta.parent.mkdir(parents=True, exist_ok=True)
    conexion = sqlite3.connect(ruta)
    conexion.row_factory = sqlite3.Row
    conexion.executescript(ESQUEMA)
    return conexion


def insertar_ticket(datos: dict, ruta: Path | None = None) -> int:
    datos = {"fecha_registro": datetime.now().isoformat(timespec="seconds"), **datos}
    columnas = ["fecha_registro", "descripcion", "area", "sede", "tipo_predicho",
                "probabilidad", "prioridad", "nivel_prioridad", "revision_tecnico",
                "justificacion", "version_modelo"]
    with closing(conectar(ruta)) as conexion, conexion:
        cursor = conexion.execute(
            f"INSERT INTO tickets ({', '.join(columnas)}) VALUES ({', '.join('?' * len(columnas))})",
            [datos.get(c) for c in columnas],
        )
        return int(cursor.lastrowid)


def listar_cola(sede: str | None = None, ruta: Path | None = None) -> list[dict]:
    """Tickets pendientes en orden de atención (nivel, fecha)."""
    consulta = "SELECT * FROM tickets WHERE estado = 'pendiente'"
    parametros: list = []
    if sede:
        consulta += " AND sede = ?"
        parametros.append(sede)
    consulta += " ORDER BY nivel_prioridad, fecha_registro, id"
    with closing(conectar(ruta)) as conexion:
        return [dict(fila) for fila in conexion.execute(consulta, parametros)]


def obtener_ticket(id_ticket: int, ruta: Path | None = None) -> dict | None:
    with closing(conectar(ruta)) as conexion:
        fila = conexion.execute("SELECT * FROM tickets WHERE id = ?", (id_ticket,)).fetchone()
        return dict(fila) if fila else None


def guardar_tipo_real(id_ticket: int, tipo_real: str, ruta: Path | None = None) -> bool:
    with closing(conectar(ruta)) as conexion, conexion:
        cursor = conexion.execute(
            "UPDATE tickets SET tipo_real = ? WHERE id = ?", (tipo_real, id_ticket)
        )
        return cursor.rowcount == 1
