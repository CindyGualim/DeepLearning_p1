"""Configuracion global del proyecto.

Todas las constantes de diseno viven aqui para que el notebook no las redefina.
Ninguna de estas constantes se ajusta mirando el conjunto de prueba: se declaran
antes de generar los datos y no se tocan despues.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Semilla global
# --------------------------------------------------------------------------
# Una sola semilla gobierna generacion de datos, particion, inicializacion de
# pesos y barajado de lotes. Cambiarla debe cambiar todo el experimento de forma
# reproducible; mantenerla debe reproducirlo bit a bit.
SEED = 2026

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "datos"
DIR_ARTEFACTOS = RAIZ / "artefactos"
DIR_FIGURAS = RAIZ / "figuras"

# --------------------------------------------------------------------------
# Generador sintetico (Ruta A) - se usa en T2
# --------------------------------------------------------------------------
N_TARJETAS = 6_000            # tarjetas simuladas
DIAS_SIMULACION = 120         # ventana temporal total simulada
FECHA_INICIO = "2025-01-01"   # inicio de la simulacion

# Prevalencia objetivo de fraude a nivel de transaccion. El enunciado describe un
# problema desbalanceado; 1.2% mantiene el desbalance realista sin dejar tan
# pocos positivos que la validacion se vuelva ruido puro.
TASA_FRAUDE_OBJETIVO = 0.012

# Peso relativo de cada mecanismo dentro del total de fraude.
#   F1 escalada  -> depende del ORDEN (compras pequenas de prueba, luego una grande)
#   F2 rafaga    -> depende parcialmente del orden (muchos eventos muy juntos)
#   F3 atipico   -> NO depende del orden (una transaccion aislada de monto raro).
#                   Es el CONTROL: la linea base A deberia detectarlo igual que B.
MEZCLA_FRAUDE = {"F1_escalada": 0.45, "F2_rafaga": 0.35, "F3_atipico": 0.20}

# --------------------------------------------------------------------------
# Construccion de secuencias - se usa en T3
# --------------------------------------------------------------------------
# Cada ejemplo es la historia de la tarjeta hasta el evento actual, inclusive.
# Se predice si el evento actual es fraudulento (horizonte = el evento en curso,
# que es la decision real de un motor antifraude: autorizar o bloquear ahora).
LARGO_SECUENCIA = 20          # maximo de eventos por secuencia (padding a la izquierda)
MIN_EVENTOS_PREVIOS = 3       # se descartan tarjetas con historia mas corta

# --------------------------------------------------------------------------
# Particion temporal - se usa en T3
# --------------------------------------------------------------------------
# Cortes por TIEMPO, no aleatorios: lo mas antiguo entrena, lo intermedio valida,
# lo mas reciente prueba. Se expresan como fraccion de la linea de tiempo total.
CORTE_TRAIN = 0.60            # [0.00, 0.60) -> entrenamiento
CORTE_VAL = 0.80              # [0.60, 0.80) -> validacion
                              # [0.80, 1.00] -> prueba (se mira UNA sola vez)

# Dias de separacion entre particiones. Evita que una secuencia que cruza el
# corte aporte eventos a dos particiones distintas (fuga por solapamiento).
DIAS_EMBARGO = 2

# --------------------------------------------------------------------------
# Economia de la decision - se usa en T12
# --------------------------------------------------------------------------
COSTO_FN = 4_200              # quetzales: fraude no detectado
COSTO_FP = 180                # quetzales: bloquear una transaccion legitima
CARTERA_TARJETAS = 1_400_000  # tarjetas reales del Banco del Altiplano

# --------------------------------------------------------------------------
# Metrica principal
# --------------------------------------------------------------------------
# AUC-PR. La exactitud NO se usa como metrica principal: con 1.2% de positivos,
# predecir "todo legitimo" ya da 98.8% de exactitud y cero valor.
METRICA_PRINCIPAL = "auc_pr"
