"""Construccion de secuencias por tarjeta y particion temporal con embargo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

CANALES_NUM = ["log_monto", "log_dt", "hora_sin", "hora_cos", "extranjero", "comercio_nuevo"]
CANALES_CAT = ["canal", "categoria"]

# Tope para el hueco desde la transaccion anterior; tambien es el valor que recibe
# la primera transaccion de cada tarjeta, que no tiene anterior.
DT_MAX_MIN = 60 * 24 * 30

PARTICIONES = ("train", "val", "test")


@dataclass
class Datos:
    """Secuencias listas para entrenar.

    `idx` y `mascara` indexan las tablas por transaccion (`num`, `cat`): la fila j
    de la secuencia i es la transaccion `idx[i, j]`, valida solo si `mascara[i, j]`.
    Se guarda el indice y no el tensor expandido para no materializar (n, L, C).
    """

    idx: np.ndarray          # (n, L) int32
    mascara: np.ndarray      # (n, L) bool
    num: np.ndarray          # (n_tx, len(CANALES_NUM)) float32
    cat: np.ndarray          # (n_tx, len(CANALES_CAT)) int16
    y: np.ndarray            # (n,) int8
    t: np.ndarray            # (n,) datetime64: timestamp del evento objetivo
    particion: np.ndarray    # (n,) <U5
    mecanismo: np.ndarray    # (n,) object
    id_episodio: np.ndarray  # (n,) int64
    fila_tx: np.ndarray      # (n,) int64: fila de la transaccion objetivo
    longitud: np.ndarray     # (n,) int16: eventos reales en la secuencia

    @property
    def n(self) -> int:
        return self.y.size

    def sel(self, particion: str) -> np.ndarray:
        return np.flatnonzero(self.particion == particion)

    def tensor_num(self, filas: np.ndarray) -> np.ndarray:
        """(m, L, C) numerico, con ceros donde la mascara es falsa."""
        return self.num[self.idx[filas]] * self.mascara[filas][:, :, None]

    def tensor_cat(self, filas: np.ndarray) -> np.ndarray:
        """(m, L, 2) de indices; el padding queda en 0 y se reserva ese codigo."""
        return (self.cat[self.idx[filas]] + 1) * self.mascara[filas][:, :, None]


def tabla_eventos(tx: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Canales por transaccion, todos causales: usan la fila actual y las previas
    de la misma tarjeta, nunca estadisticas del conjunto."""
    dt = (tx.groupby("id_tarjeta", observed=True).timestamp.diff()
          .dt.total_seconds().div(60).fillna(DT_MAX_MIN).clip(upper=DT_MAX_MIN))
    hora = tx.timestamp.dt.hour + tx.timestamp.dt.minute / 60.0
    comercio_nuevo = ~tx.duplicated(["id_tarjeta", "id_comercio"])

    num = np.column_stack([
        np.log1p(tx.monto.to_numpy()),
        np.log1p(dt.to_numpy()),
        np.sin(2 * np.pi * hora.to_numpy() / 24.0),
        np.cos(2 * np.pi * hora.to_numpy() / 24.0),
        tx.extranjero.to_numpy(),
        comercio_nuevo.to_numpy(),
    ]).astype(np.float32)

    cat = np.column_stack([tx.canal.cat.codes.to_numpy(),
                           tx.categoria.cat.codes.to_numpy()]).astype(np.int16)
    return num, cat


def cortes_temporales(tx: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    t0, t1 = tx.timestamp.min(), tx.timestamp.max()
    span = t1 - t0
    return t0 + span * config.CORTE_TRAIN, t0 + span * config.CORTE_VAL


def asignar_particion(t: pd.Series, corte_train, corte_val) -> np.ndarray:
    """Etiqueta cada evento objetivo. Las franjas de embargo quedan fuera para que
    ningun episodio quede partido entre dos particiones."""
    embargo = pd.Timedelta(days=config.DIAS_EMBARGO)
    p = np.full(len(t), "descarte", dtype=object)
    p[t < corte_train] = "train"
    p[(t >= corte_train + embargo) & (t < corte_val)] = "val"
    p[t >= corte_val + embargo] = "test"
    return p.astype("<U8")


def construir(tx: pd.DataFrame, largo: int = None, min_previos: int = None) -> Datos:
    """Una secuencia por transaccion elegible: la historia de la tarjeta hasta ese
    evento inclusive, recortada a `largo` y alineada a la derecha."""
    largo = config.LARGO_SECUENCIA if largo is None else largo
    min_previos = config.MIN_EVENTOS_PREVIOS if min_previos is None else min_previos

    num, cat = tabla_eventos(tx)
    pos = tx.groupby("id_tarjeta", observed=True).cumcount().to_numpy()

    # Cold start: sin historia suficiente no hay nada que leer en la secuencia.
    objetivo = np.flatnonzero(pos >= min_previos)
    desplazamiento = np.arange(largo - 1, -1, -1)

    idx = objetivo[:, None] - desplazamiento[None, :]
    mascara = (pos[objetivo][:, None] - desplazamiento[None, :]) >= 0
    idx = np.where(mascara, idx, 0).astype(np.int32)

    corte_train, corte_val = cortes_temporales(tx)
    t_obj = tx.timestamp.to_numpy()[objetivo]

    return Datos(
        idx=idx,
        mascara=mascara,
        num=num,
        cat=cat,
        y=tx.es_fraude.to_numpy()[objetivo].astype(np.int8),
        t=t_obj,
        particion=asignar_particion(pd.Series(t_obj), corte_train, corte_val),
        mecanismo=tx.mecanismo.to_numpy()[objetivo],
        id_episodio=tx.id_episodio.to_numpy()[objetivo],
        fila_tx=objetivo.astype(np.int64),
        longitud=np.minimum(pos[objetivo] + 1, largo).astype(np.int16),
    )


def resumen(d: Datos) -> pd.DataFrame:
    df = pd.DataFrame({"particion": d.particion, "y": d.y, "t": d.t,
                       "mecanismo": d.mecanismo, "largo": d.longitud})
    orden = [*PARTICIONES, "descarte"]
    tabla = (df.groupby("particion", observed=True)
             .agg(secuencias=("y", "size"), fraudes=("y", "sum"),
                  desde=("t", "min"), hasta=("t", "max"), largo_medio=("largo", "mean"))
             .reindex([p for p in orden if p in set(df.particion)]))
    tabla["tasa_fraude_%"] = (tabla.fraudes / tabla.secuencias * 100).round(3)
    tabla["largo_medio"] = tabla.largo_medio.round(1)
    return tabla


def verificar(d: Datos, tx: pd.DataFrame) -> list[str]:
    """Controles antifuga. Devuelve la lista de controles ejecutados; lanza si alguno falla."""
    hechos = []

    t = pd.Series(d.t)
    limites = {p: (t[d.particion == p].min(), t[d.particion == p].max()) for p in PARTICIONES}
    assert limites["train"][1] < limites["val"][0], "train se solapa con val"
    assert limites["val"][1] < limites["test"][0], "val se solapa con test"
    hechos.append("el orden temporal train < val < test se cumple sin solape")

    embargo = pd.Timedelta(days=config.DIAS_EMBARGO)
    assert limites["val"][0] - limites["train"][1] >= embargo
    assert limites["test"][0] - limites["val"][1] >= embargo
    hechos.append(f"hay al menos {config.DIAS_EMBARGO} dias de embargo entre particiones")

    reales = d.particion != "descarte"
    ep = pd.DataFrame({"ep": d.id_episodio[reales], "p": d.particion[reales]})
    ep = ep[ep.ep >= 0]
    assert ep.groupby("ep").p.nunique().max() == 1, "un episodio quedo partido entre particiones"
    hechos.append("ningun episodio de fraude aparece en mas de una particion")

    # El padding va a la izquierda: se rellena con el evento real mas antiguo de
    # cada fila para que no rompa la comprobacion de monotonia.
    ts = tx.timestamp.to_numpy()[d.idx]
    tope = np.datetime64("2262-01-01")
    ts = np.where(d.mascara, ts, np.where(d.mascara, ts, tope).min(axis=1, keepdims=True))
    assert (np.diff(ts, axis=1) >= np.timedelta64(0)).all(), "una secuencia no esta ordenada"
    assert (ts <= d.t[:, None]).all(), "una secuencia contiene eventos posteriores al objetivo"
    hechos.append("ninguna secuencia contiene eventos posteriores a su evento objetivo")

    tarjeta = tx.id_tarjeta.to_numpy()[d.idx]
    misma = np.where(d.mascara, tarjeta, tarjeta[:, -1:])
    assert (misma == misma[:, -1:]).all(), "una secuencia mezcla tarjetas"
    hechos.append("cada secuencia contiene transacciones de una sola tarjeta")

    for p in PARTICIONES:
        assert d.y[d.particion == p].sum() > 0, f"la particion {p} no tiene fraude"
    hechos.append("las tres particiones contienen casos de fraude")

    return hechos
