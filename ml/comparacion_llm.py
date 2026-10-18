"""Agente 3 — Comparación experimental: Qwen local zero-shot vs. Regresión Logística.

Clasifica el MISMO holdout (data/holdout_ids.csv) con el LLM local, con un
prompt que lista las 6 clases y pide salida JSON. Compara F1 macro, recall de
paneles_totems, accuracy, latencia promedio por ticket y % de respuestas mal
formadas. Las respuestas se guardan una a una en data/generacion/zero_shot.jsonl
(se puede reanudar).

Uso:  python -m ml.comparacion_llm [--limite N]
"""

import argparse
import json
import re
import time

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from ml.config import CLASES, DIR_GENERACION, DIR_OUTPUTS, DIR_PROMPTS, RUTA_MODELO, SEMILLA
from ml.entrenamiento import COLUMNAS_X, cargar_particiones

RUTA_RESPUESTAS = DIR_GENERACION / "zero_shot.jsonl"
TEMPERATURA = 0.0

PROMPT_SISTEMA = (
    "Eres un clasificador de tickets de soporte técnico de una clínica. "
    "Respondes únicamente con JSON válido."
)
PLANTILLA = """Clasifica el ticket en exactamente una de estas clases:
- impresion: impresoras, impresión o escaneo.
- encendido: equipos que no encienden, se apagan o no arrancan.
- bloqueo_usuario: usuarios bloqueados, contraseñas o acceso a la cuenta.
- aplicativos: fallas de programas o sistemas informáticos.
- paneles_totems: paneles de llamado de turnos, televisores de sala de espera o tótems de autoatención.
- otros: cualquier otro problema de TI (red, telefonía, periféricos, solicitudes).

Ticket: "{descripcion}"
Área: {area}
Sede: {sede}

Responde SOLO con JSON: {{"tipo": "<clase>"}}"""


def interpretar(texto: str) -> str | None:
    """Devuelve la clase si la respuesta es un JSON válido con una clase conocida."""
    coincidencia = re.search(r"\{.*?\}", texto or "", re.DOTALL)
    if not coincidencia:
        return None
    try:
        tipo = json.loads(coincidencia.group(0)).get("tipo")
    except (json.JSONDecodeError, AttributeError):
        return None
    return tipo if tipo in CLASES else None


def clasificar_con_llm(prueba: pd.DataFrame) -> pd.DataFrame:
    from ml.llm_local import crear_cliente

    DIR_PROMPTS.mkdir(parents=True, exist_ok=True)
    (DIR_PROMPTS / "zero_shot_sistema.txt").write_text(PROMPT_SISTEMA, encoding="utf-8")
    (DIR_PROMPTS / "zero_shot_plantilla.txt").write_text(PLANTILLA, encoding="utf-8")

    hechos = {}
    if RUTA_RESPUESTAS.exists():
        for linea in RUTA_RESPUESTAS.read_text(encoding="utf-8").splitlines():
            if linea.strip():
                r = json.loads(linea)
                hechos[r["id_ticket"]] = r
    pendientes = prueba[~prueba["id_ticket"].isin(hechos)]
    if not pendientes.empty:
        cliente, modelo = crear_cliente()
        with RUTA_RESPUESTAS.open("a", encoding="utf-8") as salida:
            for n, fila in enumerate(pendientes.itertuples(), start=1):
                prompt = PLANTILLA.format(descripcion=fila.descripcion, area=fila.area, sede=fila.sede)
                inicio = time.perf_counter()
                try:
                    respuesta = cliente.chat.completions.create(
                        model=modelo, temperature=TEMPERATURA, seed=SEMILLA, max_tokens=40,
                        messages=[{"role": "system", "content": PROMPT_SISTEMA},
                                  {"role": "user", "content": prompt}],
                    )
                    texto = respuesta.choices[0].message.content
                except Exception as error:
                    texto = f"ERROR: {error}"
                registro = {"id_ticket": int(fila.id_ticket), "respuesta": texto,
                            "tipo_llm": interpretar(texto), "modelo": modelo,
                            "latencia_s": round(time.perf_counter() - inicio, 3)}
                salida.write(json.dumps(registro, ensure_ascii=False) + "\n")
                salida.flush()
                hechos[registro["id_ticket"]] = registro
                if n % 25 == 0:
                    print(f"{n}/{len(pendientes)} tickets", flush=True)
    resultado = pd.DataFrame([hechos[i] for i in prueba["id_ticket"] if i in hechos])
    return prueba.merge(resultado, on="id_ticket")


def metricas(y_real, y_pred) -> dict:
    return {
        "f1_macro": round(float(f1_score(y_real, y_pred, labels=CLASES, average="macro", zero_division=0)), 4),
        "recall_paneles_totems": round(float(recall_score(
            y_real, y_pred, labels=["paneles_totems"], average="macro", zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_real, y_pred)), 4),
    }


def main(limite: int | None = None) -> pd.DataFrame:
    from ml.evaluacion import actualizar_metricas

    _, prueba = cargar_particiones()
    if limite:
        prueba = prueba.sample(limite, random_state=SEMILLA)
    llm = clasificar_con_llm(prueba)
    # Las respuestas mal formadas cuentan como error (clase "invalida").
    y_llm = llm["tipo_llm"].fillna("invalida")

    modelo = joblib.load(RUTA_MODELO)["modelo"]
    tiempos = []
    for i in range(len(prueba)):
        inicio = time.perf_counter()
        modelo.predict_proba(prueba[COLUMNAS_X].iloc[[i]])
        tiempos.append(time.perf_counter() - inicio)
    y_rl = modelo.predict(prueba[COLUMNAS_X])

    filas = [
        {"modelo": "regresion_logistica", **metricas(prueba["tipo_problema"], y_rl),
         "latencia_promedio_ms": round(1000 * sum(tiempos) / len(tiempos), 2),
         "pct_respuestas_mal_formadas": 0.0, "tickets": len(prueba)},
        {"modelo": f"llm_zero_shot ({llm['modelo'].iloc[0]})", **metricas(llm["tipo_problema"], y_llm),
         "latencia_promedio_ms": round(1000 * llm["latencia_s"].mean(), 2),
         "pct_respuestas_mal_formadas": round(100 * llm["tipo_llm"].isna().mean(), 2),
         "tickets": len(llm)},
    ]
    tabla = pd.DataFrame(filas)
    tabla.to_csv(DIR_OUTPUTS / "comparacion_llm.csv", index=False, encoding="utf-8-sig")
    if not limite:
        actualizar_metricas({"comparacion_llm": tabla.to_dict(orient="records")})
    print(tabla.to_string(index=False))
    return tabla


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limite", type=int, default=None, help="Probar con N tickets del holdout")
    main(parser.parse_args().limite)
