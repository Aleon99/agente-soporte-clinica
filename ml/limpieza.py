"""Limpieza de texto y del dataset (sección de preprocesamiento del informe).

Reglas: minúsculas, quitar tildes (conservando la ñ), reemplazar IP, correos y
números por <ip>, <correo> y <num>, quitar símbolos y espacios repetidos.
NO se eliminan stopwords: "no prende" y "prende" deben ser distintos.

Uso:  python -m ml.limpieza   (genera outputs/ejemplos_limpieza.csv)
"""

import re
import unicodedata

import pandas as pd

from ml.config import DIR_OUTPUTS, RUTA_DATASET, SEDES, AREAS, CLASES

_RE_CORREO = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_RE_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_RE_NUM = re.compile(r"\d+")
_RE_SIMBOLOS = re.compile(r"[^a-zñ<>\s]")
_RE_ESPACIOS = re.compile(r"\s+")

# Patrón de tokens para TF-IDF: conserva <ip>, <correo>, <num> y palabras de
# una letra frecuentes en abreviaturas ("x", "q").
PATRON_TOKENS = r"<\w+>|\b\w+\b"


def quitar_tildes(texto: str) -> str:
    """Quita tildes y diéresis, pero conserva la ñ."""
    texto = texto.replace("ñ", "\0")
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", texto).replace("\0", "ñ")


def limpiar_texto(texto) -> str:
    if texto is None or (isinstance(texto, float) and pd.isna(texto)):
        return ""
    texto = str(texto).lower().replace("<", " ").replace(">", " ")
    texto = _RE_CORREO.sub(" <correo> ", texto)
    texto = _RE_IP.sub(" <ip> ", texto)
    texto = _RE_NUM.sub(" <num> ", texto)
    texto = quitar_tildes(texto)
    texto = _RE_SIMBOLOS.sub(" ", texto)
    return _RE_ESPACIOS.sub(" ", texto).strip()


def normalizar_categoria(valor) -> str:
    """Normaliza área/sede: minúsculas, sin tildes, espacios como guion bajo."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return ""
    valor = quitar_tildes(str(valor).strip().lower())
    return re.sub(r"[\s-]+", "_", valor)


def preparar_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Limpia el dataset: descripción limpia, sin vacíos, sin etiqueta ni duplicados."""
    resumen = {"filas_iniciales": len(df)}
    df = df.copy()
    df["descripcion_limpia"] = df["descripcion"].map(limpiar_texto)
    df["area"] = df["area"].map(normalizar_categoria)
    df["sede"] = df["sede"].map(normalizar_categoria)

    sin_desc = df["descripcion_limpia"] == ""
    resumen["sin_descripcion"] = int(sin_desc.sum())
    df = df[~sin_desc]

    sin_etiqueta = df["tipo_problema"].isna() | ~df["tipo_problema"].isin(CLASES)
    resumen["sin_etiqueta"] = int(sin_etiqueta.sum())
    df = df[~sin_etiqueta]

    fuera_catalogo = ~df["area"].isin(AREAS) | ~df["sede"].isin(SEDES)
    resumen["fuera_de_catalogo"] = int(fuera_catalogo.sum())
    df = df[~fuera_catalogo]

    duplicados = df.duplicated(subset=["descripcion_limpia", "area", "sede"])
    resumen["duplicados"] = int(duplicados.sum())
    df = df[~duplicados].reset_index(drop=True)

    resumen["filas_finales"] = len(df)
    return df, resumen


def generar_ejemplos_limpieza(n: int = 5) -> pd.DataFrame:
    """Elige descripciones que muestren cada regla de limpieza."""
    df = pd.read_csv(RUTA_DATASET)
    df = df[df["descripcion"].notna()]
    criterios = [
        df["descripcion"].str.contains(r"\d{1,3}(?:\.\d{1,3}){3}", regex=True),
        df["descripcion"].str.contains("@", regex=False),
        df["descripcion"].str.contains(r"[áéíóú].*\d|\d.*[áéíóú]", regex=True),
        df["descripcion"].str.contains(r"[!¡?¿/()#:]", regex=True),
        df["descripcion"].str.contains(r"\bno\b", case=False, regex=True)
        & df["descripcion"].str.contains("ñ", case=False),
    ]
    filas, usados = [], set()
    for criterio in criterios:
        candidatos = df[criterio & ~df.index.isin(usados)]
        if not candidatos.empty:
            fila = candidatos.sample(1, random_state=42).iloc[0]
            usados.add(fila.name)
            filas.append(fila)
    restantes = df[~df.index.isin(usados)]
    while len(filas) < n and not restantes.empty:
        fila = restantes.sample(1, random_state=len(filas)).iloc[0]
        restantes = restantes.drop(fila.name)
        filas.append(fila)
    salida = pd.DataFrame(
        {
            "descripcion_original": [f["descripcion"] for f in filas],
            "descripcion_limpia": [limpiar_texto(f["descripcion"]) for f in filas],
        }
    )
    DIR_OUTPUTS.mkdir(exist_ok=True)
    salida.to_csv(DIR_OUTPUTS / "ejemplos_limpieza.csv", index=False, encoding="utf-8-sig")
    return salida


if __name__ == "__main__":
    print(generar_ejemplos_limpieza().to_string())
