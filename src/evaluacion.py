"""Metricas comunes a todos los modelos y decision de umbral por costo."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from . import config


def auc_pr(y: np.ndarray, puntaje: np.ndarray) -> float:
    """Area bajo la curva precision-exhaustividad.

    Es la metrica principal: con 1.2 % de positivos, la exactitud premia predecir
    "todo legitimo" y el AUC-ROC se infla con los negativos faciles.
    """
    return float(average_precision_score(y, puntaje))


def metricas(y: np.ndarray, puntaje: np.ndarray, umbral: float) -> dict:
    alerta = puntaje >= umbral
    vp = int((alerta & (y == 1)).sum())
    fp = int((alerta & (y == 0)).sum())
    fn = int((~alerta & (y == 1)).sum())
    precision = vp / (vp + fp) if vp + fp else 0.0
    exhaustividad = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precision * exhaustividad / (precision + exhaustividad) if precision + exhaustividad else 0.0
    return {"umbral": umbral, "precision": precision, "exhaustividad": exhaustividad,
            "f1": f1, "alertas": int(alerta.sum()), "vp": vp, "fp": fp, "fn": fn}


def costo(y: np.ndarray, puntaje: np.ndarray, umbral: float,
          costo_fn: float = None, costo_fp: float = None) -> float:
    """Costo esperado en quetzales de operar con ese umbral."""
    costo_fn = config.COSTO_FN if costo_fn is None else costo_fn
    costo_fp = config.COSTO_FP if costo_fp is None else costo_fp
    m = metricas(y, puntaje, umbral)
    return costo_fn * m["fn"] + costo_fp * m["fp"]


def curva_costo(y: np.ndarray, puntaje: np.ndarray,
                costo_fn: float = None, costo_fp: float = None) -> pd.DataFrame:
    """Costo total para cada umbral posible, ordenado de mas a menos estricto."""
    costo_fn = config.COSTO_FN if costo_fn is None else costo_fn
    costo_fp = config.COSTO_FP if costo_fp is None else costo_fp

    orden = np.argsort(-puntaje, kind="mergesort")
    y_ord = y[orden].astype(np.int64)
    vp = np.cumsum(y_ord)
    fp = np.cumsum(1 - y_ord)
    positivos = int(y.sum())

    return pd.DataFrame({
        "umbral": puntaje[orden],
        "alertas": np.arange(1, y.size + 1),
        "vp": vp,
        "fp": fp,
        "fn": positivos - vp,
        "costo": costo_fn * (positivos - vp) + costo_fp * fp,
    })


def umbral_optimo_costo(y: np.ndarray, puntaje: np.ndarray,
                        costo_fn: float = None, costo_fp: float = None) -> float:
    """Umbral que minimiza el costo esperado. Se calcula SIEMPRE sobre validacion."""
    curva = curva_costo(y, puntaje, costo_fn, costo_fp)
    return float(curva.umbral.iloc[int(curva.costo.values.argmin())])


def resumen(y: np.ndarray, puntajes: dict, umbrales: dict = None) -> pd.DataFrame:
    """Una fila por modelo: AUC-PR y, en su umbral, precision, exhaustividad y F1."""
    filas = []
    for nombre, s in puntajes.items():
        u = (umbrales or {}).get(nombre)
        fila = {"modelo": nombre, "auc_pr": auc_pr(y, s)}
        if u is not None:
            fila.update(metricas(y, s, u))
            fila["costo_Q"] = costo(y, s, u)
        filas.append(fila)
    return pd.DataFrame(filas).set_index("modelo")


def por_mecanismo(y: np.ndarray, puntajes: dict, mecanismo: np.ndarray,
                  legitimos=("legitimo", "sesion_legitima")) -> pd.DataFrame:
    """AUC-PR calculado por mecanismo de fraude.

    Cada columna enfrenta un solo mecanismo contra TODOS los negativos, de modo que
    las cifras son comparables entre mecanismos y contra el AUC-PR global.
    """
    negativos = np.isin(mecanismo, legitimos)
    filas = []
    for mec in [m for m in pd.unique(mecanismo) if m not in legitimos]:
        sel = negativos | (mecanismo == mec)
        positivos = int((mecanismo == mec).sum())
        # La prevalencia cambia de un mecanismo a otro, y el AUC-PR depende de ella:
        # las filas se comparan entre modelos, no entre mecanismos.
        fila = {"mecanismo": mec, "positivos": positivos, "azar": positivos / int(sel.sum())}
        for nombre, s in puntajes.items():
            fila[nombre] = auc_pr(y[sel], s[sel])
        filas.append(fila)
    return pd.DataFrame(filas).set_index("mecanismo").sort_index()


def puntos_pr(y: np.ndarray, puntaje: np.ndarray):
    p, r, _ = precision_recall_curve(y, puntaje)
    return r, p
