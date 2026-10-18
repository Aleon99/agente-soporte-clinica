"""Prepara la base de datos para la demo en vivo del agente de triaje.

1. Respalda data/tickets.db (si existe) en data/respaldos/ y deja la base vacía.
2. Precarga la cola con 9 tickets del holdout registrándolos con la MISMA
   función registrar_ticket del agente (sin LLM): limpiar -> clasificar ->
   umbral -> priorizar -> guardar. No toca el modelo ni el código del agente.
3. Escalona sus horas de registro durante las últimas 2 horas para que la cola
   parezca la de una mañana real (la hora es el único dato que se ajusta).

La selección es determinista: tickets del holdout sin ruido inyectado, cuya
predicción coincide con la etiqueta real y con probabilidad >= umbral, para que
el único ticket "en revisión" de la demo sea el que se registra en vivo.

Uso (desde la raíz):  python -m ml.preparar_demo
                      python -m ml.preparar_demo --restaurar   (vuelve al último respaldo)
"""

import argparse
import shutil
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timedelta

import pandas as pd

from ml.config import DIR_AGENTE, DIR_DATA, RUTA_DB, RUTA_HOLDOUT, RUTA_DATASET

DIR_RESPALDOS = DIR_DATA / "respaldos"

# (sede, área, cuántos) -> la matriz da: consultorio = alta, triaje/counter = media,
# administrativo = baja. El crítico se pide aparte (paneles_totems en counter).
CUPOS = [
    ("san_isidro", "counter", "paneles_totems", 1),   # 1 crítico (llegó temprano)
    ("san_isidro", "consultorio", None, 2),           # 2 altos
    ("san_isidro", "triaje", None, 2),                # 2 medios
    ("san_isidro", "counter", None, 1),               # 1 medio
    ("san_isidro", "administrativo", None, 1),        # 1 bajo
    ("miraflores", "consultorio", None, 1),           # otra sede
    ("miraflores", "administrativo", None, 1),
]


def respaldar_y_vaciar() -> None:
    if RUTA_DB.exists():
        DIR_RESPALDOS.mkdir(parents=True, exist_ok=True)
        destino = DIR_RESPALDOS / f"tickets_{datetime.now():%Y%m%d_%H%M%S}.db"
        shutil.copy2(RUTA_DB, destino)
        RUTA_DB.unlink()
        print(f"Respaldo: {destino}")


def restaurar() -> None:
    respaldos = sorted(DIR_RESPALDOS.glob("tickets_*.db"))
    if not respaldos:
        sys.exit("No hay respaldos en data/respaldos/.")
    shutil.copy2(respaldos[-1], RUTA_DB)
    print(f"Restaurado: {respaldos[-1].name} -> {RUTA_DB}")


def seleccionar_tickets() -> list[dict]:
    from ml.clasificador import clasificar

    ids = set(pd.read_csv(RUTA_HOLDOUT)["id_ticket"])
    df = pd.read_csv(RUTA_DATASET)
    df = df[df["id_ticket"].isin(ids) & (df["ambiguo"] == 0) & (df["etiqueta_ruidosa"] == 0)
            & (df["errores_inyectados"] == 0)]
    df = df.sample(frac=1, random_state=42)  # orden aleatorio pero reproducible

    elegidos, usados = [], set()
    for sede, area, tipo, cuantos in CUPOS:
        candidatos = df[(df["sede"] == sede) & (df["area"] == area) & ~df["id_ticket"].isin(usados)]
        if tipo:
            candidatos = candidatos[candidatos["tipo_problema"] == tipo]
        else:
            candidatos = candidatos[candidatos["tipo_problema"] != "paneles_totems"]
        tomados = 0
        for fila in candidatos.itertuples():
            r = clasificar(fila.descripcion, fila.area, fila.sede)
            if r["tipo_predicho"] == fila.tipo_problema and not r["revision_tecnico"]:
                elegidos.append({"descripcion": fila.descripcion, "area": fila.area, "sede": fila.sede})
                usados.add(fila.id_ticket)
                tomados += 1
                if tomados == cuantos:
                    break
    return elegidos


def main() -> None:
    sys.path.insert(0, str(DIR_AGENTE))
    import herramientas  # misma lógica que usa el agente

    respaldar_y_vaciar()
    tickets = seleccionar_tickets()
    # Orden de llegada mezclado (el crítico no llega primero) y horas escalonadas.
    orden = [2, 0, 5, 3, 7, 1, 4, 8, 6][: len(tickets)]
    ahora = datetime.now()
    registrados = []
    for k, i in enumerate(orden):
        r = herramientas.registrar_ticket.func(**tickets[i])
        if "error" in r:
            raise RuntimeError(r["error"])
        hora = ahora - timedelta(minutes=120 - k * 12)
        with closing(sqlite3.connect(RUTA_DB)) as conexion, conexion:
            conexion.execute("UPDATE tickets SET fecha_registro = ? WHERE id = ?",
                             (hora.isoformat(timespec="seconds"), r["id"]))
        registrados.append(r)
        print(f"  #{r['id']} {r['sede']:<11} {r['area']:<15} {r['tipo']:<16} "
              f"p={r['probabilidad']:.2f}  {r['prioridad']}")
    print(f"Cola precargada: {len(registrados)} tickets.\n")
    print(herramientas.ver_cola.func("san_isidro"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara data/tickets.db para la demo.")
    parser.add_argument("--restaurar", action="store_true", help="Restaura el último respaldo.")
    if parser.parse_args().restaurar:
        restaurar()
    else:
        main()
