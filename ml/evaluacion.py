"""Evaluación del modelo final en el holdout y salidas para el informe.

Genera: reporte_clasificacion_test.txt, matriz_confusion.png,
top_terminos_por_clase.csv, analisis_umbral.csv, errores_frecuentes.csv y
metricas_finales.json. Además fija el umbral recomendado en
models/version_modelo.json (lo usa el agente de triaje).

Uso:  python -m ml.evaluacion
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from ml.config import CLASES, DIR_OUTPUTS, RUTA_MODELO, RUTA_VERSION_MODELO, SEMILLA, UMBRALES

RUTA_METRICAS = DIR_OUTPUTS / "metricas_finales.json"
META_F1 = 0.80
META_RECALL_PANELES = 0.90
# Criterio de recomendación: mayor precisión automática sin derivar más del
# 25 % de los tickets al técnico (las sedes pequeñas tienen a veces 1 técnico).
MAX_DERIVACION = 0.25


def actualizar_metricas(nuevas: dict, reiniciar: bool = False) -> None:
    DIR_OUTPUTS.mkdir(exist_ok=True)
    actuales = {}
    if RUTA_METRICAS.exists() and not reiniciar:
        actuales = json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))
    actuales.update(nuevas)
    RUTA_METRICAS.write_text(json.dumps(actuales, indent=2, ensure_ascii=False, default=str),
                             encoding="utf-8")


def guardar_matriz_confusion(y_real, y_pred) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, eje = plt.subplots(figsize=(7.5, 6.5))
    ConfusionMatrixDisplay(confusion_matrix(y_real, y_pred, labels=CLASES),
                           display_labels=CLASES).plot(ax=eje, cmap="Blues", colorbar=False)
    eje.set_xlabel("Clase predicha")
    eje.set_ylabel("Clase real")
    eje.set_title("Regresión Logística – matriz de confusión (holdout 20 %)")
    plt.setp(eje.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(DIR_OUTPUTS / "matriz_confusion.png", dpi=150)
    plt.close(fig)


def top_terminos(modelo, n: int = 10) -> pd.DataFrame:
    nombres = modelo.named_steps["preprocesador"].get_feature_names_out()
    clasificador = modelo.named_steps["clasificador"]
    filas = []
    for k, clase in enumerate(clasificador.classes_):
        orden = np.argsort(clasificador.coef_[k])[::-1][:n]
        for rango, i in enumerate(orden, start=1):
            tipo_rasgo, rasgo = nombres[i].split("__", 1)
            filas.append({"clase": clase, "rango": rango, "rasgo": rasgo,
                          "tipo_rasgo": tipo_rasgo,
                          "coeficiente": round(float(clasificador.coef_[k][i]), 4)})
    return pd.DataFrame(filas)


def analisis_umbral(prueba: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    filas = []
    for umbral in UMBRALES:
        automatico = prueba["probabilidad"] >= umbral
        auto = prueba[automatico]
        filas.append({
            "umbral": umbral,
            "pct_derivado_tecnico": round(100 * (1 - automatico.mean()), 2),
            "tickets_automaticos": int(automatico.sum()),
            "precision_automaticos": round(accuracy_score(auto["tipo_problema"], auto["tipo_predicho"]), 4)
            if len(auto) else None,
            "f1_macro_automaticos": round(f1_score(auto["tipo_problema"], auto["tipo_predicho"],
                                                   average="macro", zero_division=0), 4)
            if len(auto) else None,
        })
    tabla = pd.DataFrame(filas)
    viables = tabla[tabla["pct_derivado_tecnico"] <= 100 * MAX_DERIVACION]
    recomendado = float(viables["umbral"].max()) if not viables.empty else float(min(UMBRALES))
    tabla["recomendado"] = (tabla["umbral"] == recomendado).astype(int)
    return tabla, recomendado


def errores_frecuentes(prueba: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """15 errores, priorizando los pares de confusión más frecuentes (máx. 3 por par)."""
    errores = prueba[prueba["tipo_problema"] != prueba["tipo_predicho"]].copy()
    if errores.empty:
        return errores
    frecuencia = errores.groupby(["tipo_problema", "tipo_predicho"]).size().rename("frecuencia_par")
    errores = errores.join(frecuencia, on=["tipo_problema", "tipo_predicho"])
    errores = errores.sample(frac=1, random_state=SEMILLA).sort_values("frecuencia_par", ascending=False,
                                                                         kind="stable")
    seleccion = errores.groupby(["tipo_problema", "tipo_predicho"], sort=False).head(3).head(n)
    columnas = ["id_ticket", "descripcion", "area", "sede", "tipo_problema", "tipo_predicho",
                "probabilidad", "frecuencia_par", "estilo", "ambiguo", "etiqueta_ruidosa"]
    salida = seleccion[columnas].rename(columns={"tipo_problema": "clase_real",
                                                 "tipo_predicho": "clase_predicha"})
    salida["probabilidad"] = salida["probabilidad"].round(4)
    return salida


def main() -> None:
    from ml.entrenamiento import cargar_holdout_con_predicciones

    prueba = cargar_holdout_con_predicciones()
    y_real, y_pred = prueba["tipo_problema"], prueba["tipo_predicho"]

    reporte = classification_report(y_real, y_pred, labels=CLASES, digits=4, zero_division=0)
    (DIR_OUTPUTS / "reporte_clasificacion_test.txt").write_text(
        "Regresión Logística (modelo final) – holdout estratificado del 20 %\n\n" + reporte,
        encoding="utf-8")
    print(reporte)

    guardar_matriz_confusion(y_real, y_pred)
    modelo = joblib.load(RUTA_MODELO)["modelo"]
    top_terminos(modelo).to_csv(DIR_OUTPUTS / "top_terminos_por_clase.csv", index=False,
                                encoding="utf-8-sig")

    tabla_umbral, umbral = analisis_umbral(prueba)
    tabla_umbral.to_csv(DIR_OUTPUTS / "analisis_umbral.csv", index=False, encoding="utf-8-sig")
    print(tabla_umbral.to_string(index=False))
    version = json.loads(RUTA_VERSION_MODELO.read_text(encoding="utf-8"))
    version["umbral"] = umbral
    RUTA_VERSION_MODELO.write_text(json.dumps(version, indent=2, ensure_ascii=False), encoding="utf-8")

    errores_frecuentes(prueba).to_csv(DIR_OUTPUTS / "errores_frecuentes.csv", index=False,
                                      encoding="utf-8-sig")

    f1 = f1_score(y_real, y_pred, average="macro")
    recall_paneles = recall_score(y_real, y_pred, labels=["paneles_totems"], average="macro")
    holdout = {
        "f1_macro": round(float(f1), 4),
        "recall_paneles_totems": round(float(recall_paneles), 4),
        "accuracy": round(float(accuracy_score(y_real, y_pred)), 4),
        "tickets": len(prueba),
        "meta_f1_macro_0_80": bool(f1 >= META_F1),
        "meta_recall_paneles_0_90": bool(recall_paneles >= META_RECALL_PANELES),
    }
    # Desempeño separado según el tipo de ticket (efecto del ruido inyectado).
    for nombre, mascara in {"tickets_ambiguos": prueba["ambiguo"] == 1,
                            "etiqueta_ruidosa": prueba["etiqueta_ruidosa"] == 1,
                            "tickets_limpios": (prueba["ambiguo"] == 0) & (prueba["etiqueta_ruidosa"] == 0)}.items():
        holdout[f"accuracy_{nombre}"] = round(float(accuracy_score(y_real[mascara], y_pred[mascara])), 4)
    actualizar_metricas({
        "holdout_modelo_final": holdout,
        "umbral_recomendado": umbral,
        "criterio_umbral": f"mayor precisión automática con derivación <= {int(MAX_DERIVACION * 100)} %",
        "analisis_umbral": tabla_umbral.to_dict(orient="records"),
    })
    print(json.dumps(holdout, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
