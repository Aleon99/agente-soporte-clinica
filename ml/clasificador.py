"""Uso del modelo final para un ticket: clasificar, aplicar el umbral y explicar.

Lo usan las herramientas del agente de triaje. Toda la lógica es determinista.
"""

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from ml.config import RUTA_MODELO, RUTA_VERSION_MODELO
from ml.limpieza import limpiar_texto

UMBRAL_POR_DEFECTO = 0.6


@lru_cache(maxsize=1)
def cargar_modelo():
    """Devuelve (pipeline, versión, umbral). Se carga una sola vez por proceso."""
    if not RUTA_MODELO.exists():
        raise RuntimeError(
            f"No existe {RUTA_MODELO}. Entrena primero con: python -m ml.entrenamiento"
        )
    paquete = joblib.load(RUTA_MODELO)
    umbral = UMBRAL_POR_DEFECTO
    if RUTA_VERSION_MODELO.exists():
        umbral = json.loads(RUTA_VERSION_MODELO.read_text(encoding="utf-8")).get("umbral") or umbral
    return paquete["modelo"], paquete["version"], float(umbral)


def terminos_clave(modelo, fila: pd.DataFrame, clase: str, n: int = 3) -> list[str]:
    """Los n términos (n-gramas de palabras) que más aportan a la clase predicha.

    Aporte = valor TF-IDF del término en el ticket x coeficiente de la clase.
    Se usan los rasgos de palabras porque son legibles para el técnico; los
    n-gramas de caracteres y el One-Hot también influyen en la predicción.
    """
    preprocesador = modelo.named_steps["preprocesador"]
    clasificador = modelo.named_steps["clasificador"]
    x = preprocesador.transform(fila).tocsr()
    nombres = preprocesador.get_feature_names_out()
    coeficientes = clasificador.coef_[list(clasificador.classes_).index(clase)]
    indices = x.indices
    aportes = x.data * coeficientes[indices]
    candidatos = [
        (aporte, nombres[i].split("__", 1)[1])
        for aporte, i in zip(aportes, indices)
        if nombres[i].startswith("palabras__") and aporte > 0
    ]
    candidatos.sort(reverse=True)
    return [termino for _, termino in candidatos[:n]]


def clasificar(descripcion: str, area: str, sede: str) -> dict:
    modelo, version, umbral = cargar_modelo()
    fila = pd.DataFrame(
        [{"descripcion_limpia": limpiar_texto(descripcion), "area": area, "sede": sede}]
    )
    probabilidades = modelo.predict_proba(fila)[0]
    k = int(np.argmax(probabilidades))
    tipo = str(modelo.classes_[k])
    probabilidad = float(probabilidades[k])
    return {
        "tipo_predicho": tipo,
        "probabilidad": round(probabilidad, 4),
        "revision_tecnico": int(probabilidad < umbral),
        "umbral": umbral,
        "terminos_clave": terminos_clave(modelo, fila, tipo),
        "version_modelo": version,
    }
