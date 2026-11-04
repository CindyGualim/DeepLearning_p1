"""Constantes de diseno. Copiadas literalmente de las celdas 3 y 7 del notebook.

Si un valor de aqui deja de coincidir con el notebook, los artefactos entrenados
dejan de ser validos: `pruebas/test_paridad.py` lo detecta.
"""
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
DIR_DATOS = RAIZ / "datos"
DIR_ARTEFACTOS = RAIZ / "artefactos"
DIR_DEMO = RAIZ / "datos_demo"

SEED = 2026

# --- generador ---
N_TARJETAS = 6_000
DIAS_SIMULACION = 120
FECHA_INICIO = "2025-01-01"
TASA_FRAUDE_OBJETIVO = 0.012
MEZCLA_FRAUDE = {"F1_escalada": 0.45, "F2_rafaga": 0.35, "F3_atipico": 0.20}

# --- secuencias ---
LARGO_SECUENCIA = 20
MIN_EVENTOS_PREVIOS = 3

# --- particion temporal ---
CORTE_TRAIN, CORTE_VAL = 0.60, 0.80
DIAS_EMBARGO = 2

# --- economia de la decision ---
COSTO_FN, COSTO_FP = 4_200, 180
CARTERA_TARJETAS = 1_400_000

# --- catalogos ---
CATEGORIAS = ("supermercado", "restaurante", "combustible", "farmacia", "ropa",
              "electronica", "viajes", "servicios", "entretenimiento")
CANALES = ("pos", "ecommerce", "atm", "recurrente")

ESCALA_CATEGORIA = np.array([1.4, 0.6, 0.9, 0.5, 1.1, 2.6, 3.2, 1.0, 0.7])

P_CANAL = np.array([
    [0.88, 0.06, 0.03, 0.03],   # supermercado
    [0.90, 0.08, 0.01, 0.01],   # restaurante
    [0.95, 0.01, 0.03, 0.01],   # combustible
    [0.90, 0.07, 0.01, 0.02],   # farmacia
    [0.55, 0.43, 0.01, 0.01],   # ropa
    [0.40, 0.58, 0.01, 0.01],   # electronica
    [0.15, 0.83, 0.01, 0.01],   # viajes
    [0.10, 0.25, 0.02, 0.63],   # servicios
    [0.30, 0.60, 0.02, 0.08],   # entretenimiento
])

N_COMERCIOS = 1_200
COMERCIOS_HABITUALES = 8
P_COMERCIO_HABITUAL = 0.85
MINUTOS_POR_DIA = 1_440

TAM_ESCALADA, TAM_RAFAGA = 5.0, 10.5
DISTRACTORES_POR_ESCALADA = 2.0

COLUMNAS_TX = ["id_transaccion", "id_tarjeta", "timestamp", "monto", "id_comercio",
               "categoria", "canal", "extranjero", "es_fraude", "mecanismo", "id_episodio"]

# --- caracteristicas ---
CANALES_NUM = ["log_monto", "log_dt", "hora_sin", "hora_cos", "extranjero", "comercio_nuevo"]
CANALES_CAT = ["canal", "categoria"]
DT_MAX_MIN = 60 * 24 * 30
PARTICIONES = ("train", "val", "test")

VENTANAS = {"1h": 60, "24h": 1_440, "7d": 10_080}

COLUMNAS_AG_NUM = [
    "log_monto", "log_dt", "hora_sin", "hora_cos", "extranjero", "comercio_nuevo",
    "n_1h", "n_24h", "n_7d", "monto_medio_24h", "monto_max_24h", "monto_medio_7d",
    "comercios_24h", "comercios_7d", "razon_media_24h", "razon_max_24h",
    "z_monto_tarjeta", "n_previas"]
COLUMNAS_AG_CAT = ["canal", "categoria"]
COLUMNAS_AG = COLUMNAS_AG_NUM + COLUMNAS_AG_CAT

# --- red ---
CARDINALIDADES = (len(CANALES), len(CATEGORIAS))
DIM_EMBEDDING = (3, 4)

MECANISMOS_LEGITIMOS = ("legitimo", "sesion_legitima")
