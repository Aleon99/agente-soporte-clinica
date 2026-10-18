"""Agente 1 — Generador de datos sintéticos con Qwen local (Foundry Local).

Fase 1 (generar):   pide a Qwen lotes de 15 descripciones en JSON por cada
                    combinación (tipo, área) y estilo de redacción. Guarda cada
                    lote en data/generacion/lotes.jsonl apenas llega, así que si
                    se corta se puede reanudar con el mismo comando.
Fase 2 (ensamblar): proceso determinista (semilla 42) que toma las descripciones
                    generadas, deduplica, inyecta errores de digitación (~15 %),
                    tickets ambiguos y etiquetas ruidosas (~5 %), asigna sede y
                    fecha/hora y guarda data/tickets_sinteticos.csv.

Uso (desde la raíz del repositorio):
    uv run --project src/agent-framework-agent-basic-responses python -m ml.generador
    ... python -m ml.generador --fase generar      (solo LLM)
    ... python -m ml.generador --fase ensamblar    (solo ensamblado)
"""

import argparse
import json
import math
import random
import re
import time
from datetime import datetime, timedelta

import pandas as pd

from ml.config import (
    AREAS,
    CLASES,
    DIR_GENERACION,
    DIR_OUTPUTS,
    DIR_PROMPTS,
    RUTA_DATASET,
    SEDES,
    SEMILLA,
    VOLUMEN_SEMANAL,
)
from ml.limpieza import limpiar_texto

# ---------------------------------------------------------------------------
# Parámetros documentados de la generación
# ---------------------------------------------------------------------------
TEMPERATURA = 0.9
TOP_P = 0.95
MAX_TOKENS = 1400
TAM_LOTE = 15
TOTAL_OBJETIVO = 3300          # ~6 meses de tickets (ver README)
MARGEN_POOL = 1.2              # se genera 20 % extra para absorber duplicados
PROP_ERRORES = 0.15            # textos con errores de digitación inyectados por código
PROP_AMBIGUOS = 0.025          # tickets genéricos ("no funciona", "urgente pc")
PROP_RUIDO = 0.025             # tickets con etiqueta ruidosa
FECHA_INICIO = datetime(2026, 4, 1)
SEMANAS = 26

RUTA_LOTES = DIR_GENERACION / "lotes.jsonl"

# Distribución desbalanceada realista (paneles_totems ~8 %).
P_TIPO = {
    "impresion": 0.20,
    "encendido": 0.14,
    "bloqueo_usuario": 0.19,
    "aplicativos": 0.27,
    "paneles_totems": 0.08,
    "otros": 0.12,
}
P_AREA = {"consultorio": 0.30, "triaje": 0.15, "counter": 0.28, "administrativo": 0.27}
# Los paneles de llamados y tótems están sobre todo en admisión y salas de espera.
P_AREA_PANELES = {"consultorio": 0.20, "triaje": 0.25, "counter": 0.53, "administrativo": 0.02}

DESCRIPCION_TIPO = {
    "impresion": "problemas con impresoras, impresión o escaneo",
    "encendido": "equipos que no encienden, se apagan o no arrancan",
    "bloqueo_usuario": "usuarios bloqueados, contraseñas o acceso a la cuenta",
    "aplicativos": "fallas de programas o sistemas informáticos de la clínica",
    "paneles_totems": "paneles de llamado de turnos, televisores de sala de espera o tótems de autoatención",
    "otros": "otros problemas de TI: red, telefonía, periféricos o solicitudes",
}

SUBTEMAS = {
    "impresion": [
        "la impresora no imprime nada", "atasco de papel", "la impresora de recetas no responde",
        "la impresora de etiquetas de laboratorio imprime corrido", "falta tóner o imprime en blanco",
        "la cola de impresión está trabada", "la impresora no aparece en la red",
        "imprime manchado o con rayas", "la impresora térmica de boletas no imprime",
        "el escáner de la multifuncional no envía al correo", "sale error de controlador de impresora",
        "la impresora de pulseras de pacientes no imprime", "imprime en otra impresora por defecto",
    ],
    "encendido": [
        "la computadora no prende", "el monitor no da señal", "la pc se apaga sola",
        "pantalla azul al arrancar", "se queda cargando y no entra a Windows",
        "la laptop no carga la batería", "el estabilizador o UPS pita y no enciende",
        "la pc prende pero no muestra imagen", "se reinicia a cada rato",
        "el botón de encendido no responde", "huele a quemado y se apagó",
        "la all-in-one no enciende después del corte de luz",
    ],
    "bloqueo_usuario": [
        "la cuenta se bloqueó por intentos fallidos", "la contraseña venció",
        "olvidó su contraseña", "el usuario aparece deshabilitado",
        "no puede iniciar sesión en Windows", "usuario bloqueado en el sistema de historias clínicas",
        "pide cambio de contraseña y no la acepta", "personal nuevo sin usuario creado",
        "dice usuario o contraseña incorrectos", "no puede entrar al correo por bloqueo",
        "la sesión dice que la cuenta está bloqueada por el administrador",
    ],
    "aplicativos": [
        "el sistema de historias clínicas está lento o se cuelga", "error al guardar la receta",
        "el sistema de citas no carga", "la facturación electrónica da error al enviar a SUNAT",
        "Outlook no sincroniza los correos", "Teams no tiene audio en la reunión",
        "Excel no abre un archivo", "el sistema de laboratorio no muestra resultados",
        "el visor de imágenes no abre las placas", "una actualización del sistema falló",
        "el módulo de caja se cierra solo", "el navegador no abre la intranet",
        "el sistema de farmacia no descuenta stock", "error de conexión a la base de datos del sistema",
    ],
    "paneles_totems": [
        "el panel de llamados no muestra los turnos", "el televisor de la sala de espera se congeló",
        "el tótem de autoatención no responde al tacto", "el tótem no imprime el ticket de turno",
        "el tótem aparece fuera de línea", "la pantalla de llamados no tiene sonido",
        "el panel llama a un paciente equivocado o repite el llamado",
        "el tótem se quedó en pantalla negra", "el panel de turnos muestra la cola desactualizada",
        "los pacientes no pueden sacar turno en el tótem", "la pantalla de llamados de consultorio no cambia",
    ],
    "otros": [
        "no hay internet o la red está lenta", "el teléfono IP o anexo no tiene tono",
        "el mouse o el teclado no funcionan", "solicitud de instalación de un programa",
        "la cámara web no se detecta", "el proyector de la sala de reuniones no conecta",
        "el punto de red está suelto", "mover un equipo a otro ambiente",
        "el wifi no conecta en el celular institucional", "solicitud de un cable o accesorio",
        "el lector de código de barras no lee", "el lector de huella no reconoce",
    ],
}

CONTEXTO_AREA = {
    "consultorio": "médicos o asistentes en un consultorio médico, a veces con un paciente en consulta",
    "triaje": "enfermeras en triaje, mientras toman signos vitales y hay pacientes esperando",
    "counter": "personal de admisión, caja o counter de citas, con cola de pacientes",
    "administrativo": "personal administrativo (contabilidad, RR. HH., logística, compras o gerencia)",
}

ESTILOS = {
    "formal": ("redacción formal y completa, como un correo al área de soporte", 12, 35),
    "apurado": ("persona apurada: frases cortas, a veces en mayúsculas o pidiendo urgencia", 4, 12),
    "abreviaturas": ("con algunas abreviaturas de chat usadas con naturalidad, no en cada palabra "
                     "(q = que, x = por, xfa = por favor, tb = también, dr = doctor, sist = sistema, "
                     "impr = impresora)", 5, 15),
    "errores": ("escrito rápido desde el celular: sin tildes y con algunos errores de digitación", 5, 15),
    "corto_ambiguo": ("muy corto y vago, a veces sin decir exactamente qué equipo falla", 2, 6),
    "narrativo": ("narra el contexto: desde cuándo pasa, qué se intentó y a cuántos afecta", 15, 40),
}
P_ESTILO = {
    "formal": 0.15, "apurado": 0.22, "abreviaturas": 0.18,
    "errores": 0.15, "corto_ambiguo": 0.12, "narrativo": 0.18,
}

PROMPT_SISTEMA = (
    "Eres un generador de datos para un proyecto académico. Redactas descripciones "
    "realistas de tickets de soporte técnico que el personal de una clínica de Lima "
    "(Perú) escribe en Jira, en español peruano coloquial. Respondes solo con JSON válido."
)

PLANTILLA_USUARIO = """Genera {n} descripciones DISTINTAS de tickets de soporte técnico.
- Tipo de problema (no lo nombres literalmente): {tipo}
- Situaciones posibles (varía entre ellas y agrega otras parecidas): {subtemas}
- Quién escribe: {contexto}
- Estilo: {estilo}
Reglas:
- Cada descripción entre {min_pal} y {max_pal} palabras.
- Cada descripción trata una situación distinta; no repitas frases entre ellas.
- Varía el inicio de las frases; no empieces todas igual.
- No menciones siempre el área ni la sede.
- Algunas pueden incluir un código de equipo, un número de anexo, una IP o un correo inventado.
- No incluyas nombres de pacientes.
Responde SOLO con JSON: {{"descripciones": ["...", "..."]}}"""

# Piezas para los tickets genéricos/ambiguos que se inyectan por código.
AMBIGUO_INICIO = ["", "urgente", "ayuda", "por favor", "buenos dias", "hola", "URGENTE!!", "xfa"]
AMBIGUO_NUCLEO = [
    "no funciona", "no anda", "falla el equipo", "problema con la pc", "urgente pc",
    "algo pasa con el sistema", "no sale nada", "se colgo todo", "error", "no puedo trabajar",
    "revisar equipo", "otra vez falla", "no responde", "sale un mensaje raro",
]
AMBIGUO_FIN = ["", "en mi modulo", "desde hoy", "otra vez", "revisar", "ya", "x favor", "hace rato"]

# Confusiones plausibles para la etiqueta ruidosa (error humano del técnico).
CONFUSIONES = {
    "impresion": ["paneles_totems", "otros"],
    "encendido": ["otros", "aplicativos"],
    "bloqueo_usuario": ["aplicativos"],
    "aplicativos": ["bloqueo_usuario", "otros"],
    "paneles_totems": ["impresion", "otros"],
    "otros": ["aplicativos", "encendido"],
}

TECLAS_VECINAS = {
    "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg", "g": "fh", "h": "gj",
    "i": "uo", "j": "hk", "k": "jl", "l": "kñ", "m": "n", "n": "bm", "o": "ip", "p": "o",
    "q": "w", "r": "et", "s": "ad", "t": "ry", "u": "yi", "v": "cb", "w": "qe", "x": "zc",
    "y": "tu", "z": "x",
}


# ---------------------------------------------------------------------------
# Objetivos por combinación
# ---------------------------------------------------------------------------
def objetivos_por_combinacion() -> dict[tuple[str, str], int]:
    total_llm = round(TOTAL_OBJETIVO * (1 - PROP_AMBIGUOS))
    objetivos = {}
    for tipo in CLASES:
        p_area = P_AREA_PANELES if tipo == "paneles_totems" else P_AREA
        for area in AREAS:
            objetivos[(tipo, area)] = max(1, round(total_llm * P_TIPO[tipo] * p_area[area]))
    return objetivos


# ---------------------------------------------------------------------------
# Fase 1: generación con el LLM
# ---------------------------------------------------------------------------
def guardar_prompts() -> None:
    DIR_PROMPTS.mkdir(parents=True, exist_ok=True)
    (DIR_PROMPTS / "generador_sistema.txt").write_text(PROMPT_SISTEMA, encoding="utf-8")
    (DIR_PROMPTS / "generador_usuario_plantilla.txt").write_text(PLANTILLA_USUARIO, encoding="utf-8")
    (DIR_PROMPTS / "generador_catalogos.json").write_text(
        json.dumps(
            {
                "descripcion_tipo": DESCRIPCION_TIPO,
                "subtemas": SUBTEMAS,
                "contexto_area": CONTEXTO_AREA,
                "estilos": {k: {"descripcion": v[0], "min_palabras": v[1], "max_palabras": v[2]}
                            for k, v in ESTILOS.items()},
                "p_estilo": P_ESTILO,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )


def armar_prompt(tipo: str, area: str, estilo: str, rng: random.Random) -> str:
    descripcion, min_pal, max_pal = ESTILOS[estilo]
    subtemas = rng.sample(SUBTEMAS[tipo], k=min(4, len(SUBTEMAS[tipo])))
    return PLANTILLA_USUARIO.format(
        n=TAM_LOTE, tipo=DESCRIPCION_TIPO[tipo], subtemas="; ".join(subtemas),
        contexto=CONTEXTO_AREA[area], estilo=descripcion, min_pal=min_pal, max_pal=max_pal,
    )


def extraer_descripciones(texto: str) -> list[str] | None:
    """Parsea la respuesta JSON del modelo; None si está mal formada."""
    coincidencia = re.search(r"\{.*\}", texto or "", re.DOTALL)
    if not coincidencia:
        return None
    try:
        datos = json.loads(coincidencia.group(0))
    except json.JSONDecodeError:
        return None
    lista = datos.get("descripciones") if isinstance(datos, dict) else None
    if not isinstance(lista, list):
        return None
    return [d.strip() for d in lista if isinstance(d, str) and 1 <= len(d.split()) <= 60]


def leer_lotes() -> list[dict]:
    if not RUTA_LOTES.exists():
        return []
    with RUTA_LOTES.open(encoding="utf-8") as archivo:
        return [json.loads(linea) for linea in archivo if linea.strip()]


def fase_generar() -> None:
    from ml.llm_local import crear_cliente

    guardar_prompts()
    DIR_GENERACION.mkdir(parents=True, exist_ok=True)
    cliente, modelo = crear_cliente()
    (DIR_PROMPTS / "generador_parametros.json").write_text(
        json.dumps(
            {"modelo": modelo, "temperature": TEMPERATURA, "top_p": TOP_P,
             "max_tokens": MAX_TOKENS, "tam_lote": TAM_LOTE,
             "seed": f"{SEMILLA} * 100000 + numero_de_llamada"},
            indent=2,
        ),
        encoding="utf-8",
    )

    # Reanudación: reconstruye lo ya generado.
    lotes = leer_lotes()
    unicos: dict[tuple[str, str], set[str]] = {c: set() for c in objetivos_por_combinacion()}
    for lote in lotes:
        for d in lote["descripciones"] or []:
            unicos[(lote["tipo"], lote["area"])].add(limpiar_texto(d))
    n_llamada = len(lotes)
    print(f"Reanudando con {n_llamada} lotes previos." if n_llamada else "Inicio de la generación.")

    estilos, pesos = list(P_ESTILO), list(P_ESTILO.values())
    with RUTA_LOTES.open("a", encoding="utf-8") as salida:
        for (tipo, area), objetivo in objetivos_por_combinacion().items():
            meta = math.ceil(objetivo * MARGEN_POOL)
            intentos = 0
            while len(unicos[(tipo, area)]) < meta and intentos < meta // 5 + 6:
                rng = random.Random(SEMILLA * 100000 + n_llamada)
                estilo = rng.choices(estilos, weights=pesos)[0]
                prompt = armar_prompt(tipo, area, estilo, rng)
                inicio = time.perf_counter()
                try:
                    respuesta = cliente.chat.completions.create(
                        model=modelo, temperature=TEMPERATURA, top_p=TOP_P,
                        max_tokens=MAX_TOKENS, seed=SEMILLA * 100000 + n_llamada,
                        messages=[{"role": "system", "content": PROMPT_SISTEMA},
                                  {"role": "user", "content": prompt}],
                    )
                    texto = respuesta.choices[0].message.content
                except Exception as error:  # se registra y se continúa
                    texto = f"ERROR: {error}"
                latencia = time.perf_counter() - inicio
                descripciones = extraer_descripciones(texto)
                registro = {
                    "llamada": n_llamada, "tipo": tipo, "area": area, "estilo": estilo,
                    "prompt": prompt, "respuesta": texto, "descripciones": descripciones,
                    "latencia_s": round(latencia, 2),
                }
                salida.write(json.dumps(registro, ensure_ascii=False) + "\n")
                salida.flush()
                for d in descripciones or []:
                    unicos[(tipo, area)].add(limpiar_texto(d))
                n_llamada += 1
                intentos += 1
                print(f"[{n_llamada}] {tipo}/{area}/{estilo}: "
                      f"{len(unicos[(tipo, area)])}/{meta} únicos ({latencia:.1f} s)", flush=True)


# ---------------------------------------------------------------------------
# Fase 2: ensamblado determinista
# ---------------------------------------------------------------------------
def inyectar_errores(texto: str, rng: random.Random) -> str:
    """Introduce 1–2 errores de digitación: intercambio, omisión, duplicado o tecla vecina."""
    palabras = texto.split()
    candidatas = [i for i, p in enumerate(palabras) if len(p) >= 4 and p.isalpha()]
    if not candidatas:
        return texto
    for i in rng.sample(candidatas, k=min(len(candidatas), rng.choice([1, 1, 2]))):
        p = palabras[i]
        j = rng.randrange(1, len(p) - 1)
        operacion = rng.choice(["intercambio", "omision", "duplicado", "vecina"])
        if operacion == "intercambio":
            p = p[:j] + p[j + 1] + p[j] + p[j + 2:]
        elif operacion == "omision":
            p = p[:j] + p[j + 1:]
        elif operacion == "duplicado":
            p = p[:j] + p[j] + p[j:]
        else:
            vecinas = TECLAS_VECINAS.get(p[j].lower())
            if vecinas:
                p = p[:j] + rng.choice(vecinas) + p[j + 1:]
        palabras[i] = p
    return " ".join(palabras)


def ticket_ambiguo(rng: random.Random) -> str:
    partes = [rng.choice(AMBIGUO_INICIO), rng.choice(AMBIGUO_NUCLEO), rng.choice(AMBIGUO_FIN)]
    return " ".join(p for p in partes if p)


def fechas_plausibles(n: int, rng: random.Random) -> list[datetime]:
    """Fechas en 26 semanas: más tickets de lunes a viernes y en horas punta."""
    peso_dia = [1.25, 1.1, 1.05, 1.05, 1.0, 0.65, 0.3]           # lun..dom
    horas = list(range(7, 21))
    peso_hora = [0.6, 1.4, 1.6, 1.5, 1.2, 0.9, 0.7, 0.9, 1.2, 1.2, 1.0, 0.8, 0.6, 0.4]
    dias = [FECHA_INICIO + timedelta(days=d) for d in range(SEMANAS * 7)]
    pesos = [peso_dia[d.weekday()] for d in dias]
    fechas = []
    for _ in range(n):
        dia = rng.choices(dias, weights=pesos)[0]
        hora = rng.choices(horas, weights=peso_hora)[0]
        fechas.append(dia + timedelta(hours=hora, minutes=rng.randrange(60), seconds=rng.randrange(60)))
    return fechas


def fase_ensamblar() -> pd.DataFrame:
    rng = random.Random(SEMILLA)
    lotes = leer_lotes()
    if not lotes:
        raise RuntimeError("No hay lotes generados. Ejecuta primero --fase generar.")

    # Pool único de descripciones (la primera aparición manda).
    vistos, pool = set(), {c: [] for c in objetivos_por_combinacion()}
    for lote in lotes:
        for d in lote["descripciones"] or []:
            clave = limpiar_texto(d)
            if clave and clave not in vistos:
                vistos.add(clave)
                pool[(lote["tipo"], lote["area"])].append((d, lote["estilo"]))

    filas, faltantes = [], {}
    for (tipo, area), objetivo in objetivos_por_combinacion().items():
        candidatos = pool[(tipo, area)]
        if len(candidatos) < objetivo:
            faltantes[f"{tipo}/{area}"] = objetivo - len(candidatos)
        for descripcion, estilo in rng.sample(candidatos, k=min(objetivo, len(candidatos))):
            filas.append({"descripcion": descripcion, "area": area, "tipo_problema": tipo,
                          "estilo": estilo, "ambiguo": 0})

    # Tickets genéricos/ambiguos: el técnico los etiquetó después de revisar el equipo.
    n_ambiguos = round(TOTAL_OBJETIVO * PROP_AMBIGUOS)
    for _ in range(n_ambiguos):
        tipo = rng.choices(list(P_TIPO), weights=list(P_TIPO.values()))[0]
        p_area = P_AREA_PANELES if tipo == "paneles_totems" else P_AREA
        area = rng.choices(list(p_area), weights=list(p_area.values()))[0]
        filas.append({"descripcion": ticket_ambiguo(rng), "area": area, "tipo_problema": tipo,
                      "estilo": "generico_ambiguo", "ambiguo": 1})

    df = pd.DataFrame(filas)

    # Errores de digitación inyectados por código (~15 %).
    df["errores_inyectados"] = 0
    idx = rng.sample(range(len(df)), k=round(len(df) * PROP_ERRORES))
    for i in idx:
        df.at[i, "descripcion"] = inyectar_errores(df.at[i, "descripcion"], rng)
        df.at[i, "errores_inyectados"] = 1

    # Etiqueta ruidosa (~2,5 %): confusión plausible del técnico al registrar.
    df["etiqueta_ruidosa"] = 0
    idx = rng.sample(range(len(df)), k=round(len(df) * PROP_RUIDO))
    for i in idx:
        df.at[i, "tipo_problema"] = rng.choice(CONFUSIONES[df.at[i, "tipo_problema"]])
        df.at[i, "etiqueta_ruidosa"] = 1

    # Sede proporcional al volumen semanal (60 grandes / 25 pequeñas) y fecha.
    sedes = list(SEDES)
    pesos_sede = [VOLUMEN_SEMANAL[SEDES[s]] for s in sedes]
    df["sede"] = rng.choices(sedes, weights=pesos_sede, k=len(df))
    df["fecha"] = [f.isoformat(timespec="seconds") for f in fechas_plausibles(len(df), rng)]

    # Deduplicación final (los errores inyectados pueden generar coincidencias).
    df["_clave"] = df["descripcion"].map(limpiar_texto)
    df = df.drop_duplicates(subset=["_clave", "area", "sede"]).drop(columns="_clave")

    df = df.sort_values("fecha").reset_index(drop=True)
    df.insert(0, "id_ticket", range(1, len(df) + 1))
    columnas = ["id_ticket", "fecha", "descripcion", "area", "sede", "tipo_problema",
                "estilo", "errores_inyectados", "ambiguo", "etiqueta_ruidosa"]
    df = df[columnas]
    df.to_csv(RUTA_DATASET, index=False, encoding="utf-8")

    guardar_resumen(df)
    guardar_ejemplos(df, rng)
    print(f"Dataset: {len(df)} tickets -> {RUTA_DATASET}")
    if faltantes:
        print(f"Aviso: combinaciones con menos descripciones que el objetivo: {faltantes}")
    return df


def guardar_resumen(df: pd.DataFrame) -> None:
    filas = []
    for dimension in ["tipo_problema", "area", "sede"]:
        conteo = df[dimension].value_counts()
        for valor, n in conteo.items():
            filas.append({"dimension": dimension, "valor": valor, "tickets": int(n),
                          "porcentaje": round(100 * n / len(df), 2)})
    cruce = df.groupby(["tipo_problema", "area"]).size()
    for (tipo, area), n in cruce.items():
        filas.append({"dimension": "tipo_problema x area", "valor": f"{tipo} | {area}",
                      "tickets": int(n), "porcentaje": round(100 * n / len(df), 2)})
    DIR_OUTPUTS.mkdir(exist_ok=True)
    pd.DataFrame(filas).to_csv(DIR_OUTPUTS / "dataset_resumen.csv", index=False, encoding="utf-8-sig")


def guardar_ejemplos(df: pd.DataFrame, rng: random.Random) -> None:
    """10 tickets de muestra: normales, con errores, ambiguos y con etiqueta ruidosa."""
    limpio = (df["ambiguo"] == 0) & (df["errores_inyectados"] == 0) & (df["etiqueta_ruidosa"] == 0)
    grupos = [
        (limpio & (df["tipo_problema"] == "paneles_totems"), 1),
        (limpio & (df["estilo"] == "formal"), 1),
        (limpio & (df["estilo"] == "narrativo"), 1),
        (limpio & (df["estilo"] == "abreviaturas"), 1),
        (limpio & (df["estilo"] == "corto_ambiguo"), 1),
        (df["errores_inyectados"] == 1, 2),
        (df["ambiguo"] == 1, 2),
        (df["etiqueta_ruidosa"] == 1, 1),
    ]
    partes = [df[m].sample(min(n, int(m.sum())), random_state=SEMILLA) for m, n in grupos]
    ejemplos = pd.concat(partes).drop_duplicates("id_ticket").head(10)
    ejemplos.to_csv(DIR_OUTPUTS / "ejemplos_dataset.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fase", choices=["generar", "ensamblar", "todo"], default="todo")
    args = parser.parse_args()
    if args.fase in ("generar", "todo"):
        fase_generar()
    if args.fase in ("ensamblar", "todo"):
        fase_ensamblar()
