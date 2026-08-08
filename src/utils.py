"""Utilidades transversales: semillas, reproducibilidad y dispositivo."""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import pandas as pd

from . import config


def fijar_semillas(seed: int | None = None) -> int:
    """Fija la semilla de random, numpy y torch (si esta instalado).

    Se llama al inicio del notebook y de nuevo antes de cada entrenamiento, para
    que reentrenar un modelo no dependa de cuantas celdas se corrieron antes.
    """
    seed = config.SEED if seed is None else seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Determinismo a costa de algo de velocidad: sin esto, cuDNN elige
        # algoritmos distintos entre corridas y los pesos no se reproducen.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
    return seed


def hash_df(df: pd.DataFrame) -> str:
    """Huella estable de un DataFrame, para verificar reproducibilidad.

    Se usa en T2: dos corridas con la misma semilla deben dar el mismo hash.
    """
    filas = pd.util.hash_pandas_object(df, index=True).values
    columnas = ",".join(map(str, df.columns)).encode()
    h = hashlib.sha256()
    h.update(filas.tobytes())
    h.update(columnas)
    return h.hexdigest()[:16]


def dispositivo():
    """Devuelve el dispositivo de torch a usar (cuda si hay, si no cpu)."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def asegurar_directorios() -> None:
    """Crea datos/, artefactos/ y figuras/ si no existen."""
    for d in (config.DIR_DATOS, config.DIR_ARTEFACTOS, config.DIR_FIGURAS):
        d.mkdir(parents=True, exist_ok=True)


def resumen_entorno() -> pd.DataFrame:
    """Tabla de versiones para el README y la seccion de reproducibilidad."""
    import platform

    filas = [("python", platform.python_version()), ("plataforma", platform.platform())]
    for nombre in ("numpy", "pandas", "sklearn", "torch", "matplotlib"):
        try:
            mod = __import__(nombre)
            filas.append((nombre, getattr(mod, "__version__", "?")))
        except ImportError:
            filas.append((nombre, "no instalado"))
    return pd.DataFrame(filas, columns=["componente", "version"])
