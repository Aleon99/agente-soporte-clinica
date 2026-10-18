"""Priorización por regla (Tabla 1 del informe) y simulación de la cola.

La matriz se evalúa de arriba hacia abajo:
    1 Crítica  tipo = paneles_totems, en cualquier área
    2 Alta     cualquier tipo en consultorio
    3 Media    cualquier tipo en triaje o counter
    4 Baja     cualquier tipo en administrativo
La cola se ordena por nivel (1 -> 4) y, dentro de cada nivel, por fecha (FIFO).

Uso:  python -m ml.priorizacion   (genera outputs/simulacion_cola.csv y .png)
"""

import json

import pandas as pd

from ml.config import AREAS, DIR_OUTPUTS

NOMBRE_PRIORIDAD = {1: "critica", 2: "alta", 3: "media", 4: "baja"}
NIVEL_POR_AREA = {"consultorio": 2, "triaje": 3, "counter": 3, "administrativo": 4}


def priorizar(tipo: str, area: str) -> tuple[str, int]:
    """Devuelve (prioridad, nivel_prioridad) según la Tabla 1."""
    if area not in AREAS:
        raise ValueError(f"Área fuera del catálogo: {area!r}. Válidas: {AREAS}")
    nivel = 1 if tipo == "paneles_totems" else NIVEL_POR_AREA[area]
    return NOMBRE_PRIORIDAD[nivel], nivel


def ordenar_cola(df: pd.DataFrame, col_nivel: str = "nivel_prioridad",
                 col_fecha: str = "fecha") -> pd.DataFrame:
    """Orden de atención: nivel 1 -> 4 y, dentro de cada nivel, por fecha."""
    return df.sort_values([col_nivel, col_fecha], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Simulación: cola FIFO vs. cola priorizada (responde la pregunta de investigación)
# ---------------------------------------------------------------------------
ESCENARIOS = {"sede grande (60 tickets)": 60, "sede pequeña (25 tickets)": 25}


def simular_cola(holdout: pd.DataFrame) -> pd.DataFrame:
    """Compara la posición de los tickets críticos y altos en tres colas.

    `holdout` debe tener: fecha, area, tipo_problema (real) y tipo_predicho.
    Se parte el holdout, ordenado por fecha, en lotes consecutivos del tamaño de
    la cola semanal de cada tipo de sede. En cada lote se calcula la posición
    (1 = primero en ser atendido) en:
      - fifo:          orden de llegada (situación actual);
      - priorizada:    matriz aplicada al tipo PREDICHO por la Regresión Logística;
      - ideal:         matriz aplicada al tipo real (cota superior, sin errores del modelo).
    Las métricas se agrupan por la prioridad REAL del ticket.
    """
    datos = holdout.sort_values("fecha").reset_index(drop=True).copy()
    datos["nivel_real"] = [priorizar(t, a)[1] for t, a in zip(datos["tipo_problema"], datos["area"])]
    datos["nivel_predicho"] = [priorizar(t, a)[1] for t, a in zip(datos["tipo_predicho"], datos["area"])]

    filas = []
    for escenario, tam in ESCENARIOS.items():
        posiciones = []
        n_lotes = len(datos) // tam
        for k in range(n_lotes):
            lote = datos.iloc[k * tam:(k + 1) * tam].copy()
            lote["pos_fifo"] = range(1, len(lote) + 1)
            for cola, col_nivel in [("priorizada", "nivel_predicho"), ("ideal", "nivel_real")]:
                orden = lote.sort_values([col_nivel, "fecha"], kind="stable")
                lote.loc[orden.index, f"pos_{cola}"] = range(1, len(lote) + 1)
            posiciones.append(lote)
        todas = pd.concat(posiciones)
        for nivel in (1, 2):
            sub = todas[todas["nivel_real"] == nivel]
            for cola in ("fifo", "priorizada", "ideal"):
                filas.append({
                    "escenario": escenario,
                    "lotes": n_lotes,
                    "prioridad_real": NOMBRE_PRIORIDAD[nivel],
                    "cola": cola,
                    "tickets": len(sub),
                    "posicion_promedio": round(sub[f"pos_{cola}"].mean(), 2),
                    "posicion_maxima": int(sub[f"pos_{cola}"].max()),
                })
    return pd.DataFrame(filas)


def graficar_simulacion(resultado: pd.DataFrame, ruta) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colores = {"fifo": "#9aa0a6", "priorizada": "#1a73e8", "ideal": "#34a853"}
    etiquetas = {"fifo": "FIFO (actual)", "priorizada": "Priorizada (modelo RL)",
                 "ideal": "Priorizada (etiqueta real)"}
    escenarios = list(ESCENARIOS)
    fig, ejes = plt.subplots(1, len(escenarios), figsize=(11, 4.2), sharey=False)
    for eje, escenario in zip(ejes, escenarios):
        sub = resultado[resultado["escenario"] == escenario]
        grupos = ["critica", "alta"]
        ancho = 0.26
        for j, cola in enumerate(["fifo", "priorizada", "ideal"]):
            valores = [sub[(sub["prioridad_real"] == g) & (sub["cola"] == cola)]["posicion_promedio"].iloc[0]
                       for g in grupos]
            x = [i + (j - 1) * ancho for i in range(len(grupos))]
            barras = eje.bar(x, valores, ancho, color=colores[cola], label=etiquetas[cola])
            eje.bar_label(barras, fmt="%.1f", fontsize=8, padding=2)
        eje.set_xticks(range(len(grupos)), ["Crítica", "Alta"])
        eje.set_title(escenario.capitalize(), fontsize=10)
        eje.set_ylabel("Posición promedio en la cola (1 = primero)")
        eje.spines[["top", "right"]].set_visible(False)
    ejes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Cola FIFO vs. cola priorizada: posición de los tickets críticos y altos", fontsize=11)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def ejecutar_simulacion() -> pd.DataFrame:
    from ml.entrenamiento import cargar_holdout_con_predicciones
    from ml.evaluacion import actualizar_metricas

    holdout = cargar_holdout_con_predicciones()
    resultado = simular_cola(holdout)
    DIR_OUTPUTS.mkdir(exist_ok=True)
    resultado.to_csv(DIR_OUTPUTS / "simulacion_cola.csv", index=False, encoding="utf-8-sig")
    graficar_simulacion(resultado, DIR_OUTPUTS / "simulacion_cola.png")
    resumen = {
        f"{f.escenario} | {f.prioridad_real} | {f.cola}": {
            "posicion_promedio": f.posicion_promedio, "posicion_maxima": f.posicion_maxima}
        for f in resultado.itertuples()
    }
    actualizar_metricas({"simulacion_cola": resumen})
    return resultado


if __name__ == "__main__":
    print(ejecutar_simulacion().to_string(index=False))
