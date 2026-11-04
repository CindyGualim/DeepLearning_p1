"""Construccion de caracteristicas. Celdas 20, 28 y 32 del notebook, sin cambios de logica.

Dos representaciones sobre las mismas transacciones:

  `tabla_eventos`  -> 6 canales numericos + 2 categoricos POR EVENTO. Es lo que leen
                      los modelos secuenciales (B y la rama GRU de C).
  `agregadas`      -> 20 variables de ventana de la transaccion actual. Es lo que lee
                      el modelo A y la rama densa de C.

Las dos son causales por construccion: cada columna usa la fila actual y las
anteriores de la misma tarjeta, nunca una estadistica del conjunto completo. Por eso
pueden calcularse antes de partir los datos sin introducir fuga; la estandarizacion,
que si requiere estadisticas globales, vive en los escaladores de `artefactos/`.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (CANALES, CATEGORIAS, COLUMNAS_AG, COLUMNAS_AG_CAT, COLUMNAS_AG_NUM,
                     CORTE_TRAIN, CORTE_VAL, DIAS_EMBARGO, DT_MAX_MIN, LARGO_SECUENCIA,
                     MIN_EVENTOS_PREVIOS, PARTICIONES, VENTANAS)


@dataclass
class Datos:
    """`idx` y `mascara` indexan las tablas por transaccion (`num`, `cat`): la fila j de
    la secuencia i es la transaccion idx[i, j], valida solo si mascara[i, j]."""
    idx: np.ndarray          # (n, L) int32
    mascara: np.ndarray      # (n, L) bool
    num: np.ndarray          # (n_tx, 6) float32
    cat: np.ndarray          # (n_tx, 2) int16
    y: np.ndarray
    t: np.ndarray            # timestamp del evento objetivo
    particion: np.ndarray
    mecanismo: np.ndarray
    id_episodio: np.ndarray
    fila_tx: np.ndarray      # fila de la transaccion objetivo
    longitud: np.ndarray

    @property
    def n(self):
        return self.y.size

    def sel(self, particion):
        return np.flatnonzero(self.particion == particion)


def tabla_eventos(tx):
    """(n_tx, 6) numericos y (n_tx, 2) codigos categoricos, uno por transaccion."""
    dt = (tx.groupby("id_tarjeta", observed=True).timestamp.diff()
          .dt.total_seconds().div(60).fillna(DT_MAX_MIN).clip(upper=DT_MAX_MIN))
    hora = tx.timestamp.dt.hour + tx.timestamp.dt.minute / 60.0
    num = np.column_stack([
        np.log1p(tx.monto.to_numpy()),
        np.log1p(dt.to_numpy()),
        np.sin(2 * np.pi * hora.to_numpy() / 24.0),
        np.cos(2 * np.pi * hora.to_numpy() / 24.0),
        tx.extranjero.to_numpy(),
        (~tx.duplicated(["id_tarjeta", "id_comercio"])).to_numpy(),
    ]).astype(np.float32)
    cat = np.column_stack([tx.canal.cat.codes.to_numpy(),
                           tx.categoria.cat.codes.to_numpy()]).astype(np.int16)
    return num, cat


def cortes_temporales(tx):
    t0, t1 = tx.timestamp.min(), tx.timestamp.max()
    return t0 + (t1 - t0) * CORTE_TRAIN, t0 + (t1 - t0) * CORTE_VAL


def asignar_particion(t, corte_train, corte_val):
    """Las franjas de embargo quedan fuera para que ningun episodio se parta."""
    embargo = pd.Timedelta(days=DIAS_EMBARGO)
    p = np.full(len(t), "descarte", dtype=object)
    p[t < corte_train] = "train"
    p[(t >= corte_train + embargo) & (t < corte_val)] = "val"
    p[t >= corte_val + embargo] = "test"
    return p.astype("<U8")


def construir_secuencias(tx, largo=LARGO_SECUENCIA, min_previos=MIN_EVENTOS_PREVIOS):
    """Un ejemplo = historia de la tarjeta hasta el evento actual inclusive, recortada
    a `largo` eventos y alineada a la derecha (el padding va a la izquierda)."""
    num, cat = tabla_eventos(tx)
    pos = tx.groupby("id_tarjeta", observed=True).cumcount().to_numpy()

    objetivo = np.flatnonzero(pos >= min_previos)      # arranque en frio
    desplazamiento = np.arange(largo - 1, -1, -1)
    idx = objetivo[:, None] - desplazamiento[None, :]
    mascara = (pos[objetivo][:, None] - desplazamiento[None, :]) >= 0
    idx = np.where(mascara, idx, 0).astype(np.int32)

    corte_train, corte_val = cortes_temporales(tx)
    t_obj = tx.timestamp.to_numpy()[objetivo]
    return Datos(idx=idx, mascara=mascara, num=num, cat=cat,
                 y=tx.es_fraude.to_numpy()[objetivo].astype(np.int8), t=t_obj,
                 particion=asignar_particion(pd.Series(t_obj), corte_train, corte_val),
                 mecanismo=tx.mecanismo.to_numpy()[objetivo],
                 id_episodio=tx.id_episodio.to_numpy()[objetivo],
                 fila_tx=objetivo.astype(np.int64),
                 longitud=np.minimum(pos[objetivo] + 1, largo).astype(np.int16))


def ventana_previa(tarjeta, minuto, monto, comercio, ancho):
    """Conteo, suma, maximo y comercios distintos sobre los eventos *anteriores* de la
    misma tarjeta dentro de [t - ancho, t). El evento actual queda fuera a proposito:
    asi `monto / maximo_24h` no se vuelve trivialmente 1 cuando la actual es la mayor."""
    n = minuto.size
    cuenta = np.zeros(n, np.float32); suma = np.zeros(n, np.float32)
    maximo = np.zeros(n, np.float32); distintos = np.zeros(n, np.float32)
    acumulado = np.concatenate([[0.0], np.cumsum(monto)])
    frecuencias, mayores, ini = {}, deque(), 0

    for i in range(n):
        if i and tarjeta[i] != tarjeta[i - 1]:
            ini = i; frecuencias.clear(); mayores.clear()
        limite = minuto[i] - ancho
        while ini < i and minuto[ini] < limite:
            c = comercio[ini]
            if frecuencias[c] == 1:
                del frecuencias[c]
            else:
                frecuencias[c] -= 1
            ini += 1
        while mayores and mayores[0] < ini:
            mayores.popleft()

        cuenta[i] = i - ini
        suma[i] = acumulado[i] - acumulado[ini]
        distintos[i] = len(frecuencias)
        maximo[i] = monto[mayores[0]] if mayores else 0.0

        while mayores and monto[mayores[-1]] <= monto[i]:
            mayores.pop()
        mayores.append(i)
        frecuencias[comercio[i]] = frecuencias.get(comercio[i], 0) + 1
    return cuenta, suma, maximo, distintos


def estadisticos_previos(tx, log_monto):
    """Media y desviacion del log-monto sobre la historia estricta de la tarjeta."""
    g = tx.assign(_lm=log_monto, _lm2=log_monto ** 2).groupby("id_tarjeta", observed=True)
    n_previas = g.cumcount().to_numpy().astype(np.float32)
    suma = g._lm.cumsum().to_numpy() - log_monto
    suma_cuad = g._lm2.cumsum().to_numpy() - log_monto ** 2
    media = np.where(n_previas > 0, suma / np.maximum(n_previas, 1), 0.0)
    var = np.where(n_previas > 1, suma_cuad / np.maximum(n_previas, 1) - media ** 2, 0.0)
    return n_previas, media, np.sqrt(np.maximum(var, 0.0))


def agregadas(tx):
    """Las 20 variables de ventana. Toda columna usa la fila actual y las anteriores
    de la misma tarjeta; no interviene ninguna estadistica del conjunto completo."""
    tarjeta = tx.id_tarjeta.to_numpy()
    # El notebook escribia `astype("int64") // 60_000_000_000`, que asume resolucion de
    # nanosegundos. pandas 3 infiere datetime64[us] y esa division daba minutos/1000:
    # las ventanas de 1h/24h/7d quedaban 1000 veces mas anchas y los conteos absorbian
    # toda la historia de la tarjeta. Convertir a datetime64[m] da el mismo numero bajo
    # cualquier resolucion.
    minuto = tx.timestamp.to_numpy().astype("datetime64[m]").astype("int64")
    monto = tx.monto.to_numpy().astype(np.float64)
    comercio = tx.id_comercio.to_numpy()

    log_monto = np.log1p(monto)
    dt = (tx.groupby("id_tarjeta", observed=True).timestamp.diff().dt.total_seconds()
          .div(60).fillna(DT_MAX_MIN).clip(upper=DT_MAX_MIN).to_numpy())
    hora = (tx.timestamp.dt.hour + tx.timestamp.dt.minute / 60.0).to_numpy()

    v = {k: ventana_previa(tarjeta, minuto, monto, comercio, a) for k, a in VENTANAS.items()}
    n_previas, media_prev, desv_prev = estadisticos_previos(tx, log_monto)
    medio_24h = np.where(v["24h"][0] > 0, v["24h"][1] / np.maximum(v["24h"][0], 1), 0.0)
    medio_7d = np.where(v["7d"][0] > 0, v["7d"][1] / np.maximum(v["7d"][0], 1), 0.0)

    return pd.DataFrame({
        "log_monto": log_monto,
        "log_dt": np.log1p(dt),
        "hora_sin": np.sin(2 * np.pi * hora / 24.0),
        "hora_cos": np.cos(2 * np.pi * hora / 24.0),
        "extranjero": tx.extranjero.to_numpy(),
        "comercio_nuevo": (~tx.duplicated(["id_tarjeta", "id_comercio"])).to_numpy(),
        "n_1h": v["1h"][0], "n_24h": v["24h"][0], "n_7d": v["7d"][0],
        "monto_medio_24h": medio_24h, "monto_max_24h": v["24h"][2], "monto_medio_7d": medio_7d,
        "comercios_24h": v["24h"][3], "comercios_7d": v["7d"][3],
        "razon_media_24h": log_monto - np.log1p(medio_24h),
        "razon_max_24h": log_monto - np.log1p(v["24h"][2]),
        # Se acota: con una tarjeta de historia casi constante la desviacion tiende a
        # cero y el cociente explota a miles, que es ruido numerico, no senal.
        "z_monto_tarjeta": np.clip(np.where(desv_prev > 1e-3,
                                            (log_monto - media_prev) / np.maximum(desv_prev, 1e-3),
                                            0.0), -20.0, 20.0),
        "n_previas": n_previas,
        "canal": tx.canal.cat.codes.to_numpy(),
        "categoria": tx.categoria.cat.codes.to_numpy(),
    }).astype(np.float32)[COLUMNAS_AG]


def normalizar_historia(historia):
    """Lista de dicts -> DataFrame con los dtypes que espera el resto del modulo.

    Es la puerta de entrada del servicio: valida el contrato publico de `/puntuar`.
    Lanza ValueError con un mensaje accionable si algo no cuadra, porque el que llama
    es un cliente HTTP y necesita saber que corregir.
    """
    if not isinstance(historia, (list, tuple)) or not historia:
        raise ValueError("la historia debe ser una lista no vacia de transacciones")

    filas = []
    for i, e in enumerate(historia):
        faltantes = {"timestamp", "monto", "id_comercio", "categoria", "canal"} - set(e)
        if faltantes:
            raise ValueError(f"transaccion {i}: faltan campos {sorted(faltantes)}")
        if e["categoria"] not in CATEGORIAS:
            raise ValueError(f"transaccion {i}: categoria '{e['categoria']}' no existe. "
                             f"Validas: {list(CATEGORIAS)}")
        if e["canal"] not in CANALES:
            raise ValueError(f"transaccion {i}: canal '{e['canal']}' no existe. "
                             f"Validos: {list(CANALES)}")
        monto = float(e["monto"])
        if not np.isfinite(monto) or monto <= 0:
            raise ValueError(f"transaccion {i}: monto debe ser un numero positivo")
        filas.append({
            "id_tarjeta": int(e.get("id_tarjeta", 0)),
            "timestamp": pd.Timestamp(e["timestamp"]),
            "monto": monto,
            "id_comercio": int(e["id_comercio"]),
            "categoria": e["categoria"],
            "canal": e["canal"],
            "extranjero": int(e.get("extranjero", 0)),
        })

    tx = pd.DataFrame(filas)
    if tx.id_tarjeta.nunique() > 1:
        raise ValueError("la historia debe ser de una sola tarjeta")
    if not tx.timestamp.is_monotonic_increasing:
        raise ValueError("las transacciones deben venir ordenadas por timestamp ascendente")

    tx["categoria"] = pd.Categorical(tx.categoria, categories=CATEGORIAS)
    tx["canal"] = pd.Categorical(tx.canal, categories=CANALES)
    return tx
