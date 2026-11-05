"""Puntuador: convierte la historia de una tarjeta en los puntajes de A, B y C.

Es la pieza que faltaba para servir el modelo. El notebook puntua matrices enteras
porque evalua particiones; un servicio recibe *una* tarjeta a la vez. Las dos rutas
comparten el mismo codigo de caracteristicas y los mismos artefactos, y
`pruebas/test_paridad.py` verifica que dan el mismo numero.

Contrato publico (ver README, seccion 4):
  entrada -> historia de UNA tarjeta, ordenada por timestamp, con al menos
             MIN_EVENTOS_PREVIOS + 1 transacciones. La ultima es la que se puntua.
  salida  -> puntaje continuo en [0, 1] por cada modelo, mas la decision operativa.

Dos manipulaciones de la historia quedan expuestas aqui porque son las pruebas de
falsificacion del Proyecto 1, y el servicio las ofrece en vivo:
  `permutar_orden`  baraja la historia sin cambiar los eventos (celda 76)
  `recortar`        deja visibles solo los ultimos k eventos (celda 85)
"""
import time

import joblib
import numpy as np
import torch

from .caracteristicas import agregadas, normalizar_historia, tabla_eventos
from .config import (COLUMNAS_AG_NUM, DIR_ARTEFACTOS, LARGO_SECUENCIA,
                     MIN_EVENTOS_PREVIOS, COSTO_FN, COSTO_FP)
from .modelos import ARQUITECTURAS, RedHibrida

N_AG = len(COLUMNAS_AG_NUM)


def permutar_orden(idx_mat, mascara, seed, fijar_objetivo=True):
    """Baraja el orden DENTRO de cada secuencia sin cambiar sus eventos. Celda 76.

    El padding lleva clave -inf y se queda a la izquierda; el evento objetivo lleva
    +inf y se queda al final cuando `fijar_objetivo`, para que la prueba mida el valor
    del orden de la historia y no el de saber cual transaccion se esta puntuando.
    """
    rng = np.random.default_rng(seed)
    claves = rng.random(idx_mat.shape)
    claves[~mascara] = -np.inf
    if fijar_objetivo:
        claves[:, -1] = np.inf
    orden = np.argsort(claves, axis=1, kind="stable")
    return np.take_along_axis(idx_mat, orden, axis=1)


def recortar(mascara, k):
    """Deja visibles solo los ultimos k eventos de cada secuencia. Celda 85."""
    m = mascara.copy()
    if k < mascara.shape[1]:
        m[:, :-k] = False
    return m


class Puntuador:
    """Carga los tres artefactos una sola vez y los mantiene en memoria.

    Es deliberadamente un objeto y no funciones sueltas: cargar `modelo_a.joblib`
    cuesta ~0.3 s y hacerlo por peticion dominaria la latencia del servicio.
    """

    def __init__(self, dir_artefactos=DIR_ARTEFACTOS):
        self.dir = dir_artefactos
        faltantes = [n for n in ("modelo_a.joblib", "modelo_b.pt", "modelo_c.pt",
                                 "escalador_eventos.joblib", "escalador_agregadas.joblib")
                     if not (dir_artefactos / n).exists()]
        if faltantes:
            raise FileNotFoundError(
                f"faltan artefactos en {dir_artefactos}: {faltantes}. "
                "Se producen corriendo el notebook del Proyecto 1 de principio a fin.")

        bulto_a = joblib.load(dir_artefactos / "modelo_a.joblib")
        self.estimador_a = bulto_a["estimador"]
        self.familia_a = bulto_a["familia"]
        self.columnas_a = bulto_a["columnas"]
        self.umbral_a = float(bulto_a["umbral_val"])

        self.esc_eventos = joblib.load(dir_artefactos / "escalador_eventos.joblib")
        self.esc_agregadas = joblib.load(dir_artefactos / "escalador_agregadas.joblib")

        bulto_b = torch.load(dir_artefactos / "modelo_b.pt", map_location="cpu",
                             weights_only=False)
        self.red_b = ARQUITECTURAS[bulto_b["arquitectura"]](bulto_b["n_num"], bulto_b["oculto"])
        self.red_b.load_state_dict(bulto_b["pesos"])
        self.red_b.eval()
        self.umbral_b = float(bulto_b["umbral_val"])
        self.cfg_b = f"{bulto_b['arquitectura']}-{bulto_b['oculto']}"

        bulto_c = torch.load(dir_artefactos / "modelo_c.pt", map_location="cpu",
                             weights_only=False)
        self.red_c = RedHibrida(bulto_c["n_num"], bulto_c["n_agg"], oculto=bulto_c["oculto"])
        self.red_c.load_state_dict(bulto_c["pesos"])
        self.red_c.eval()
        self.umbral_c = float(bulto_c["umbral_val"])

        self.umbrales = {"A": self.umbral_a, "B": self.umbral_b, "C": self.umbral_c}

    # ---- preparacion ----

    def preparar(self, tx):
        """Historia cruda -> las tres matrices que consumen los modelos."""
        num, cat = tabla_eventos(tx)
        X = agregadas(tx).to_numpy()
        num_z = self.esc_eventos.transform(num).astype(np.float32)
        agg_z = self.esc_agregadas.transform(X)[:, :N_AG].astype(np.float32)
        return num_z, cat, X, agg_z

    @staticmethod
    def indices_secuencia(objetivos, largo=LARGO_SECUENCIA):
        """(m, L) de indices y mascara para los objetivos dados dentro de UNA tarjeta.

        Alineado a la derecha, padding a la izquierda: identico a
        `construir_secuencias`, pero para un subconjunto de posiciones.
        """
        objetivos = np.asarray(objetivos, dtype=np.int64)
        desplazamiento = np.arange(largo - 1, -1, -1)
        idx = objetivos[:, None] - desplazamiento[None, :]
        mascara = idx >= 0
        return np.where(mascara, idx, 0).astype(np.int32), mascara

    # ---- puntuacion ----

    def _redes(self, num_z, cat, idx_mat, mascara, agg_z_filas):
        """Pasa un lote por B y C. Devuelve dos vectores de probabilidad."""
        m = mascara
        num = torch.from_numpy(num_z[idx_mat] * m[:, :, None])
        cod = torch.from_numpy(((cat[idx_mat] + 1) * m[:, :, None]).astype(np.int64))
        with torch.no_grad():
            s_b = torch.sigmoid(self.red_b(num, cod)).numpy()
            s_c = torch.sigmoid(self.red_c(num, cod, torch.from_numpy(agg_z_filas))).numpy()
        return s_b.astype(np.float32), s_c.astype(np.float32)

    def puntuar_matrices(self, num_z, cat, X, agg_z, idx_mat, mascara, filas_tx):
        """Ruta por lote: la que usan el constructor del banco y la prueba de paridad."""
        s_a = self.estimador_a.predict_proba(X[filas_tx])[:, 1].astype(np.float32)
        s_b, s_c = self._redes(num_z, cat, idx_mat, mascara, agg_z[filas_tx])
        return {"A": s_a, "B": s_b, "C": s_c}

    def puntuar(self, historia, semilla_permutacion=None, k_visible=None,
                fijar_objetivo=True):
        """Puntua la ULTIMA transaccion de `historia` con los tres modelos.

        `semilla_permutacion` baraja la historia antes de puntuar (prueba de
        permutacion en vivo); `k_visible` deja visibles solo los ultimos k eventos
        (prueba de recorte). Las dos afectan unicamente a B y C: el modelo A lee
        agregadas de ventana, que por definicion no dependen del orden. Que A no se
        mueva mientras B se desploma es justamente lo que la prueba demuestra.
        """
        t0 = time.perf_counter()
        tx = normalizar_historia(historia)
        if len(tx) <= MIN_EVENTOS_PREVIOS:
            raise ValueError(
                f"la tarjeta necesita al menos {MIN_EVENTOS_PREVIOS + 1} transacciones "
                f"({MIN_EVENTOS_PREVIOS} previas mas la actual); llegaron {len(tx)}. "
                "Por debajo de eso la politica es camino de reglas, no modelo.")

        num_z, cat, X, agg_z = self.preparar(tx)
        objetivo = len(tx) - 1
        idx_mat, mascara = self.indices_secuencia([objetivo])

        if k_visible is not None:
            mascara = recortar(mascara, int(k_visible))
        if semilla_permutacion is not None:
            idx_mat = permutar_orden(idx_mat, mascara, int(semilla_permutacion),
                                     fijar_objetivo=fijar_objetivo)

        puntajes = self.puntuar_matrices(num_z, cat, X, agg_z, idx_mat, mascara,
                                         np.array([objetivo]))
        puntajes = {k: float(v[0]) for k, v in puntajes.items()}
        return {
            "puntajes": puntajes,
            "umbrales": self.umbrales,
            "alertas": {k: bool(puntajes[k] >= self.umbrales[k]) for k in puntajes},
            "decision": self.decidir(puntajes),
            "eventos_visibles": int(mascara.sum()),
            # Posiciones de la historia, en el orden exacto en que la red las leyo.
            # La consola las usa para dibujar la secuencia barajada tal cual se puntuo,
            # en lugar de pedirle al usuario que confie en que se barajo.
            "orden_eventos": idx_mat[0][mascara[0]].tolist(),
            "latencia_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def trayectoria(self, historia):
        """Puntaje de los tres modelos en CADA transaccion puntuable de la tarjeta.

        Es lo que dibuja la linea de tiempo de la consola: deja ver como el puntaje
        sube evento por evento mientras una escalada se construye, en lugar de dar un
        unico numero al final.
        """
        tx = normalizar_historia(historia)
        if len(tx) <= MIN_EVENTOS_PREVIOS:
            raise ValueError(f"se necesitan mas de {MIN_EVENTOS_PREVIOS} transacciones")

        num_z, cat, X, agg_z = self.preparar(tx)
        objetivos = np.arange(MIN_EVENTOS_PREVIOS, len(tx))
        idx_mat, mascara = self.indices_secuencia(objetivos)
        puntajes = self.puntuar_matrices(num_z, cat, X, agg_z, idx_mat, mascara, objetivos)
        return {"posiciones": objetivos.tolist(),
                "puntajes": {k: v.tolist() for k, v in puntajes.items()}}

    # ---- politica de decision ----

    def decidir(self, puntajes):
        """La politica recomendada al comite en el Proyecto 1: A decide, C prioriza.

        A es la incumbente y la mas barata en prueba; C tiene el mejor AUC-PR en las
        dos particiones pero su ventaja en dinero no es distinguible del ruido. Asi
        que A conserva la decision de alertar y C ordena la cola de revision manual.
        Complementar, no reemplazar.
        """
        alerta = puntajes["A"] >= self.umbral_a
        return {
            "alertar": bool(alerta),
            "decide": "A",
            "prioriza": "C",
            "prioridad": float(puntajes["C"]),
            "motivo": ("A supera su umbral de costo" if alerta
                       else "A no supera su umbral de costo"),
        }

    def ficha(self):
        """Metadatos del modelo servido, para `/salud` y para el pie de la consola."""
        return {
            "modelos": {
                "A": {"familia": self.familia_a, "umbral": round(self.umbral_a, 4),
                      "entrada": "20 agregadas de ventana", "ve_orden": False},
                "B": {"familia": self.cfg_b, "umbral": round(self.umbral_b, 4),
                      "entrada": "20 eventos ordenados", "ve_orden": True},
                "C": {"familia": "hibrida gru-128 + 18 agregadas",
                      "umbral": round(self.umbral_c, 4),
                      "entrada": "20 eventos ordenados + 18 agregadas", "ve_orden": True},
            },
            "politica": {"decide": "A", "prioriza": "C"},
            "costos": {"falso_negativo_Q": COSTO_FN, "falso_positivo_Q": COSTO_FP},
            "minimo_transacciones": MIN_EVENTOS_PREVIOS + 1,
            "largo_secuencia": LARGO_SECUENCIA,
        }


_PUNTUADOR = None


def obtener_puntuador():
    """Instancia unica. FastAPI la pide en cada peticion; cargar es caro, puntuar no."""
    global _PUNTUADOR
    if _PUNTUADOR is None:
        _PUNTUADOR = Puntuador()
    return _PUNTUADOR
