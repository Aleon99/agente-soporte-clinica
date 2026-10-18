"""Entrenamiento y validación del clasificador de tipo_problema.

1. Limpia el dataset y separa un holdout estratificado del 20 % ANTES de
   cualquier ajuste (se guarda en data/holdout_ids.csv).
2. Validación cruzada estratificada (k=5) sobre el 80 % restante para comparar
   Regresión Logística (principal), SVM lineal (alternativa) y Naive Bayes
   Multinomial (línea base).
3. GridSearchCV de la Regresión Logística sobre el mismo 80 %.
4. Entrena el modelo final (RL con los mejores hiperparámetros) en el 80 % y lo
   guarda en models/modelo_rl.joblib con su versión.

Todo el preprocesamiento va dentro del Pipeline para evitar fuga de datos.

Uso:  python -m ml.entrenamiento
"""

import hashlib
import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from ml.config import (
    DIR_MODELOS,
    DIR_OUTPUTS,
    RUTA_DATASET,
    RUTA_HOLDOUT,
    RUTA_MODELO,
    RUTA_VERSION_MODELO,
    SEMILLA,
)
from ml.limpieza import PATRON_TOKENS, preparar_dataset

COLUMNAS_X = ["descripcion_limpia", "area", "sede"]
OBJETIVO = "tipo_problema"

ESCORES = {
    "f1_macro": "f1_macro",
    "accuracy": "accuracy",
    "recall_paneles_totems": make_scorer(
        recall_score, labels=["paneles_totems"], average="macro", zero_division=0
    ),
}

REJILLA_RL = {
    "clasificador__C": [0.5, 1, 2, 5, 10, 20],
    "preprocesador__palabras__min_df": [1, 2],
    "preprocesador__caracteres__min_df": [1, 2],
}


def crear_preprocesador() -> ColumnTransformer:
    """TF-IDF de palabras (1-2) + TF-IDF de caracteres char_wb (3-5) + One-Hot de área y sede."""
    return ColumnTransformer(
        [
            ("palabras", TfidfVectorizer(ngram_range=(1, 2), token_pattern=PATRON_TOKENS,
                                         sublinear_tf=True), "descripcion_limpia"),
            ("caracteres", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                           sublinear_tf=True), "descripcion_limpia"),
            ("categorias", OneHotEncoder(handle_unknown="ignore"), ["area", "sede"]),
        ]
    )


def crear_modelos() -> dict[str, Pipeline]:
    clasificadores = {
        "regresion_logistica": LogisticRegression(
            class_weight="balanced", max_iter=5000, random_state=SEMILLA),
        "svm_lineal": LinearSVC(class_weight="balanced", random_state=SEMILLA),
        "naive_bayes_multinomial": MultinomialNB(),
    }
    return {
        nombre: Pipeline([("preprocesador", crear_preprocesador()), ("clasificador", clf)])
        for nombre, clf in clasificadores.items()
    }


def cargar_datos() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(RUTA_DATASET)
    return preparar_dataset(df)


def separar_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Holdout estratificado del 20 %; se guarda la lista de ids para reutilizarla."""
    entrenamiento, prueba = train_test_split(
        df, test_size=0.2, stratify=df[OBJETIVO], random_state=SEMILLA
    )
    prueba[["id_ticket"]].to_csv(RUTA_HOLDOUT, index=False)
    return entrenamiento, prueba


def cargar_particiones() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruye las particiones a partir de data/holdout_ids.csv."""
    df, _ = cargar_datos()
    ids = set(pd.read_csv(RUTA_HOLDOUT)["id_ticket"])
    es_prueba = df["id_ticket"].isin(ids)
    return df[~es_prueba], df[es_prueba]


def cargar_holdout_con_predicciones() -> pd.DataFrame:
    """Holdout con tipo_predicho y probabilidad del modelo final guardado."""
    _, prueba = cargar_particiones()
    paquete = joblib.load(RUTA_MODELO)
    modelo = paquete["modelo"]
    probabilidades = modelo.predict_proba(prueba[COLUMNAS_X])
    prueba = prueba.copy()
    prueba["tipo_predicho"] = modelo.classes_[probabilidades.argmax(axis=1)]
    prueba["probabilidad"] = probabilidades.max(axis=1)
    return prueba


def validacion_cruzada(entrenamiento: pd.DataFrame) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    filas = []
    for nombre, pipeline in crear_modelos().items():
        resultado = cross_validate(
            pipeline, entrenamiento[COLUMNAS_X], entrenamiento[OBJETIVO],
            cv=cv, scoring=ESCORES, n_jobs=-1,
        )
        fila = {"modelo": nombre}
        for metrica in ESCORES:
            valores = resultado[f"test_{metrica}"]
            fila[f"{metrica}_media"] = round(float(np.mean(valores)), 4)
            fila[f"{metrica}_desv"] = round(float(np.std(valores)), 4)
        fila["tiempo_entrenamiento_s"] = round(float(np.mean(resultado["fit_time"])), 3)
        filas.append(fila)
        print(f"{nombre}: F1 macro {fila['f1_macro_media']:.4f} ± {fila['f1_macro_desv']:.4f}")
    return pd.DataFrame(filas)


def busqueda_hiperparametros(entrenamiento: pd.DataFrame) -> GridSearchCV:
    pipeline = crear_modelos()["regresion_logistica"]
    busqueda = GridSearchCV(
        pipeline, REJILLA_RL,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA),
        scoring=ESCORES, refit="f1_macro", n_jobs=-1,
    )
    busqueda.fit(entrenamiento[COLUMNAS_X], entrenamiento[OBJETIVO])
    return busqueda


def main() -> None:
    DIR_OUTPUTS.mkdir(exist_ok=True)
    DIR_MODELOS.mkdir(exist_ok=True)

    df, resumen_limpieza = cargar_datos()
    print(f"Limpieza: {resumen_limpieza}")
    entrenamiento, prueba = separar_holdout(df)
    print(f"Entrenamiento: {len(entrenamiento)}  Holdout: {len(prueba)}")

    comparacion = validacion_cruzada(entrenamiento)
    comparacion.to_csv(DIR_OUTPUTS / "comparacion_modelos.csv", index=False, encoding="utf-8-sig")

    inicio = time.perf_counter()
    busqueda = busqueda_hiperparametros(entrenamiento)
    duracion = time.perf_counter() - inicio
    i = busqueda.best_index_
    mejores = {
        "modelo": "regresion_logistica",
        "metrica_refit": "f1_macro",
        "mejores_hiperparametros": busqueda.best_params_,
        "f1_macro_cv_media": round(float(busqueda.cv_results_["mean_test_f1_macro"][i]), 4),
        "f1_macro_cv_desv": round(float(busqueda.cv_results_["std_test_f1_macro"][i]), 4),
        "recall_paneles_totems_cv_media": round(
            float(busqueda.cv_results_["mean_test_recall_paneles_totems"][i]), 4),
        "rejilla": REJILLA_RL,
        "combinaciones_evaluadas": len(busqueda.cv_results_["params"]),
        "duracion_busqueda_s": round(duracion, 1),
    }
    (DIR_OUTPUTS / "mejores_hiperparametros.json").write_text(
        json.dumps(mejores, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Mejores hiperparámetros: {busqueda.best_params_} (F1 CV {mejores['f1_macro_cv_media']})")

    # GridSearchCV con refit ya entrenó el mejor pipeline sobre todo el 80 %.
    modelo_final = busqueda.best_estimator_
    huella = hashlib.sha1(RUTA_DATASET.read_bytes()).hexdigest()[:8]
    version = f"rl-{datetime.now():%Y%m%d}-{huella}"
    joblib.dump({"modelo": modelo_final, "version": version}, RUTA_MODELO)
    RUTA_VERSION_MODELO.write_text(json.dumps({
        "version": version,
        "entrenado": datetime.now().isoformat(timespec="seconds"),
        "scikit_learn": sklearn.__version__,
        "dataset_sha1_8": huella,
        "filas_entrenamiento": len(entrenamiento),
        "hiperparametros": busqueda.best_params_,
        "umbral": None,  # lo fija ml.evaluacion a partir de analisis_umbral.csv
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    from ml.evaluacion import actualizar_metricas
    actualizar_metricas({
        "dataset": {"limpieza": resumen_limpieza, "entrenamiento": len(entrenamiento),
                    "holdout": len(prueba)},
        "validacion_cruzada": comparacion.set_index("modelo").to_dict(orient="index"),
        "grid_search": {k: v for k, v in mejores.items() if k != "rejilla"},
        "version_modelo": version,
    }, reiniciar=True)
    print(f"Modelo guardado: {RUTA_MODELO} ({version})")


if __name__ == "__main__":
    main()
