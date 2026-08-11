"""Modelo A: linea base sobre variables agregadas, sin leer el orden."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from . import config
from .caracteristicas import COLUMNAS, COLUMNAS_CAT, COLUMNAS_NUM
from .evaluacion import auc_pr

IDX_CAT = [COLUMNAS.index(c) for c in COLUMNAS_CAT]
IDX_NUM = [COLUMNAS.index(c) for c in COLUMNAS_NUM]

# Rejilla deliberadamente corta: se recorre entera sobre validacion y no se toca
# el conjunto de prueba en ningun punto de la seleccion.
REJILLA_ARBOLES = [
    {"learning_rate": 0.10, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 40},
    {"learning_rate": 0.10, "max_iter": 400, "max_leaf_nodes": 31, "min_samples_leaf": 40},
    {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 31, "min_samples_leaf": 40},
    {"learning_rate": 0.05, "max_iter": 600, "max_leaf_nodes": 63, "min_samples_leaf": 20},
    {"learning_rate": 0.10, "max_iter": 300, "max_leaf_nodes": 63, "min_samples_leaf": 20},
    {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 15, "min_samples_leaf": 80},
]

REJILLA_LOGISTICA = [{"C": c} for c in (0.03, 0.1, 0.3, 1.0)]


def _una_caliente(x: np.ndarray) -> np.ndarray:
    """Numericas tal cual, mas indicadores para canal y categoria."""
    partes = [x[:, IDX_NUM]]
    for j, n_niveles in zip(IDX_CAT, (4, 9)):
        codigos = x[:, j].astype(int)
        partes.append(np.eye(n_niveles, dtype=np.float32)[codigos])
    return np.hstack(partes)


def arboles(**kwargs) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        categorical_features=IDX_CAT,
        class_weight="balanced",
        early_stopping=False,
        random_state=config.SEED,
        **kwargs,
    )


def logistica(escalador, **kwargs) -> LogisticRegression:
    return LogisticRegression(class_weight="balanced", max_iter=2000,
                              random_state=config.SEED, **kwargs)


def buscar(X: np.ndarray, y: np.ndarray, idx_train: np.ndarray, idx_val: np.ndarray,
           escalador) -> tuple[pd.DataFrame, object, np.ndarray]:
    """Recorre las dos familias sobre validacion y devuelve la tabla y el ganador."""
    x_tr, y_tr = X[idx_train], y[idx_train]
    x_va, y_va = X[idx_val], y[idx_val]

    z_tr = _una_caliente(escalador.transform(x_tr))
    z_va = _una_caliente(escalador.transform(x_va))

    filas, modelos, puntajes = [], [], []

    for params in REJILLA_ARBOLES:
        m = arboles(**params).fit(x_tr, y_tr)
        s = m.predict_proba(x_va)[:, 1]
        filas.append({"familia": "arboles", **params, "auc_pr_val": auc_pr(y_va, s)})
        modelos.append(m)
        puntajes.append(s)

    for params in REJILLA_LOGISTICA:
        m = logistica(escalador, **params).fit(z_tr, y_tr)
        s = m.predict_proba(z_va)[:, 1]
        filas.append({"familia": "logistica", **params, "auc_pr_val": auc_pr(y_va, s)})
        modelos.append(m)
        puntajes.append(s)

    tabla = pd.DataFrame(filas).sort_values("auc_pr_val", ascending=False)
    mejor = int(tabla.index[0])
    return tabla.reset_index(drop=True), modelos[mejor], puntajes[mejor]


class ModeloA:
    """Envoltura que fija la transformacion de entrada junto con el estimador.

    Guardar el estimador suelto obligaria a recordar si esperaba la matriz cruda o
    la escalada con indicadores; asi el artefacto es autosuficiente.
    """

    def __init__(self, estimador, escalador, familia: str):
        self.estimador = estimador
        self.escalador = escalador
        self.familia = familia

    def puntaje(self, X: np.ndarray) -> np.ndarray:
        if self.familia == "arboles":
            return self.estimador.predict_proba(X)[:, 1]
        return self.estimador.predict_proba(_una_caliente(self.escalador.transform(X)))[:, 1]

    def guardar(self, ruta=None):
        import joblib

        ruta = (config.DIR_ARTEFACTOS / "modelo_a.joblib") if ruta is None else ruta
        joblib.dump(self, ruta)
        return ruta
