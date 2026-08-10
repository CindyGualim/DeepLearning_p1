"""Variables agregadas por ventana y escaladores ajustados solo con entrenamiento."""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config

# Ventanas del motor actual del banco, en minutos.
VENTANAS = {"1h": 60, "24h": 1_440, "7d": 10_080}

COLUMNAS_NUM = [
    "log_monto", "log_dt", "hora_sin", "hora_cos", "extranjero", "comercio_nuevo",
    "n_1h", "n_24h", "n_7d",
    "monto_medio_24h", "monto_max_24h", "monto_medio_7d",
    "comercios_24h", "comercios_7d",
    "razon_media_24h", "razon_max_24h", "z_monto_tarjeta", "n_previas",
]
COLUMNAS_CAT = ["canal", "categoria"]
COLUMNAS = COLUMNAS_NUM + COLUMNAS_CAT

DT_MAX_MIN = 60 * 24 * 30


def _ventana_previa(tarjeta, minuto, monto, comercio, ancho):
    """Conteo, suma, maximo y comercios distintos sobre los eventos *anteriores*
    de la misma tarjeta dentro de [t - ancho, t).

    El evento actual queda fuera a proposito: asi las razones monto/referencia no
    se vuelven triviales y toda la ventana es historia estricta.
    """
    n = minuto.size
    cuenta = np.zeros(n, np.float32)
    suma = np.zeros(n, np.float32)
    maximo = np.zeros(n, np.float32)
    distintos = np.zeros(n, np.float32)

    acumulado = np.concatenate([[0.0], np.cumsum(monto)])
    frecuencias: dict[int, int] = {}
    mayores: deque[int] = deque()
    ini = 0

    for i in range(n):
        if i and tarjeta[i] != tarjeta[i - 1]:
            ini = i
            frecuencias.clear()
            mayores.clear()

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


def _estadisticos_previos(tx: pd.DataFrame, log_monto: np.ndarray):
    """Media y desviacion del log-monto sobre la historia estricta de la tarjeta."""
    g = tx.assign(_lm=log_monto, _lm2=log_monto ** 2).groupby("id_tarjeta", observed=True)
    n_previas = g.cumcount().to_numpy().astype(np.float32)
    suma = g._lm.cumsum().to_numpy() - log_monto
    suma_cuad = g._lm2.cumsum().to_numpy() - log_monto ** 2

    with np.errstate(invalid="ignore", divide="ignore"):
        media = np.where(n_previas > 0, suma / np.maximum(n_previas, 1), 0.0)
        var = np.where(n_previas > 1, suma_cuad / np.maximum(n_previas, 1) - media ** 2, 0.0)
    return n_previas, media, np.sqrt(np.maximum(var, 0.0))


def agregados(tx: pd.DataFrame) -> pd.DataFrame:
    """Variables agregadas por transaccion.

    Toda columna se calcula con la fila actual y las filas anteriores de la misma
    tarjeta. No interviene ninguna estadistica del conjunto completo, asi que
    calcularlas antes de partir los datos no introduce fuga.
    """
    tarjeta = tx.id_tarjeta.to_numpy()
    minuto = (tx.timestamp.astype("int64") // 60_000_000_000).to_numpy()
    monto = tx.monto.to_numpy().astype(np.float64)
    comercio = tx.id_comercio.to_numpy()

    log_monto = np.log1p(monto)
    dt = (tx.groupby("id_tarjeta", observed=True).timestamp.diff()
          .dt.total_seconds().div(60).fillna(DT_MAX_MIN).clip(upper=DT_MAX_MIN).to_numpy())
    hora = (tx.timestamp.dt.hour + tx.timestamp.dt.minute / 60.0).to_numpy()

    v = {k: _ventana_previa(tarjeta, minuto, monto, comercio, ancho)
         for k, ancho in VENTANAS.items()}
    n_previas, media_prev, desv_prev = _estadisticos_previos(tx, log_monto)

    medio_24h = np.where(v["24h"][0] > 0, v["24h"][1] / np.maximum(v["24h"][0], 1), 0.0)
    medio_7d = np.where(v["7d"][0] > 0, v["7d"][1] / np.maximum(v["7d"][0], 1), 0.0)

    ag = pd.DataFrame({
        "log_monto": log_monto,
        "log_dt": np.log1p(dt),
        "hora_sin": np.sin(2 * np.pi * hora / 24.0),
        "hora_cos": np.cos(2 * np.pi * hora / 24.0),
        "extranjero": tx.extranjero.to_numpy(),
        "comercio_nuevo": (~tx.duplicated(["id_tarjeta", "id_comercio"])).to_numpy(),
        "n_1h": v["1h"][0],
        "n_24h": v["24h"][0],
        "n_7d": v["7d"][0],
        "monto_medio_24h": medio_24h,
        "monto_max_24h": v["24h"][2],
        "monto_medio_7d": medio_7d,
        "comercios_24h": v["24h"][3],
        "comercios_7d": v["7d"][3],
        "razon_media_24h": log_monto - np.log1p(medio_24h),
        "razon_max_24h": log_monto - np.log1p(v["24h"][2]),
        # Se acota: con una tarjeta de historia casi constante la desviacion tiende
        # a cero y el cociente explota a miles, que es ruido numerico, no senal.
        "z_monto_tarjeta": np.clip(
            np.where(desv_prev > 1e-3, (log_monto - media_prev) / np.maximum(desv_prev, 1e-3), 0.0),
            -20.0, 20.0),
        "n_previas": n_previas,
        "canal": tx.canal.cat.codes.to_numpy(),
        "categoria": tx.categoria.cat.codes.to_numpy(),
    })
    return ag.astype(np.float32)[COLUMNAS]


def matriz(ag: pd.DataFrame, filas: np.ndarray) -> np.ndarray:
    """Matriz de diseno para las transacciones objetivo indicadas."""
    return ag.to_numpy()[filas]


def ajustar_escaladores(x_agregados: np.ndarray, eventos_num: np.ndarray,
                        filas_train_tx: np.ndarray, idx_train: np.ndarray):
    """Ajusta ambos escaladores usando exclusivamente el periodo de entrenamiento.

    `filas_train_tx` son las transacciones objetivo de train: sus estadisticas son
    las unicas que el modelo puede conocer antes de ver validacion o prueba.
    """
    esc_agregados = StandardScaler().fit(x_agregados[idx_train])
    esc_eventos = StandardScaler().fit(eventos_num[filas_train_tx])
    return esc_agregados, esc_eventos


def guardar_escaladores(esc_agregados, esc_eventos, destino=None) -> dict:
    import joblib

    destino = config.DIR_ARTEFACTOS if destino is None else destino
    rutas = {"agregados": destino / "escalador_agregados.joblib",
             "eventos": destino / "escalador_eventos.joblib"}
    joblib.dump(esc_agregados, rutas["agregados"])
    joblib.dump(esc_eventos, rutas["eventos"])
    return rutas


def verificar_causalidad(tx: pd.DataFrame, ag: pd.DataFrame, fraccion: float = 0.6) -> str:
    """Recalcula las agregadas sobre un recorte temporal y exige valores identicos.

    Si alguna columna mirara el futuro, borrar el futuro cambiaria su valor.
    """
    corte = tx.timestamp.min() + (tx.timestamp.max() - tx.timestamp.min()) * fraccion
    recorte = tx[tx.timestamp < corte]
    ag_recorte = agregados(recorte.reset_index(drop=True))
    referencia = ag.loc[recorte.index].reset_index(drop=True)

    iguales = np.isclose(ag_recorte.to_numpy(), referencia.to_numpy(),
                         rtol=1e-5, atol=1e-5, equal_nan=True)
    if not iguales.all():
        culpables = ag.columns[~iguales.all(axis=0)].tolist()
        raise AssertionError(f"columnas que dependen del futuro: {culpables}")
    return (f"las {ag.shape[1]} columnas son identicas al recalcularlas sin los datos "
            f"posteriores a {corte:%Y-%m-%d} ({len(recorte):,} filas comparadas)")
