"""Rutas, catálogos cerrados y constantes compartidas por todo el proyecto."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_DATA = RAIZ / "data"
DIR_PROMPTS = DIR_DATA / "prompts"
DIR_GENERACION = DIR_DATA / "generacion"
DIR_MODELOS = RAIZ / "models"
DIR_OUTPUTS = RAIZ / "outputs"
DIR_AGENTE = RAIZ / "src" / "agent-framework-agent-basic-responses"

RUTA_DATASET = DIR_DATA / "tickets_sinteticos.csv"
RUTA_DB = DIR_DATA / "tickets.db"
RUTA_MODELO = DIR_MODELOS / "modelo_rl.joblib"
RUTA_VERSION_MODELO = DIR_MODELOS / "version_modelo.json"
RUTA_HOLDOUT = DIR_DATA / "holdout_ids.csv"

SEMILLA = 42

# Clases del clasificador (tipo_problema), en el orden del informe.
CLASES = [
    "impresion",
    "encendido",
    "bloqueo_usuario",
    "aplicativos",
    "paneles_totems",
    "otros",
]

# Catálogo cerrado de áreas: son las únicas que cubre la matriz de priorización.
AREAS = ["consultorio", "triaje", "counter", "administrativo"]

# Sinónimos aceptados al validar el área: solo traducen a una de las 4 áreas
# del catálogo (Admisión cuenta como counter); no crean áreas nuevas.
SINONIMOS_AREA = {
    "admision": "counter",
    "counters": "counter",
    "caja": "counter",
    "recepcion": "counter",
    "consultorios": "consultorio",
    "administrativa": "administrativo",
    "administracion": "administrativo",
    "contabilidad": "administrativo",
    "recursos_humanos": "administrativo",
    "logistica": "administrativo",
}

# Catálogo cerrado de sedes: 2 grandes y 3 pequeñas.
SEDES = {
    "san_isidro": "grande",
    "miraflores": "grande",
    "surco": "pequena",
    "la_molina": "pequena",
    "san_miguel": "pequena",
}

# Tickets por semana según el tamaño de la sede (sección 1 del informe).
VOLUMEN_SEMANAL = {"grande": 60, "pequena": 25}

# Umbrales evaluados para derivar al técnico.
UMBRALES = [0.5, 0.6, 0.7, 0.8]
