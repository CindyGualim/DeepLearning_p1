"""Generador sintetico de transacciones. Celdas 7-10 del notebook, sin cambios de logica.

Tres mecanismos de fraude y un control:
  F1_escalada     sondas pequenas en orden creciente que rematan en el golpe (depende del orden)
  F2_rafaga       muchos cargos en pocos minutos en comercios distintos (semi-orden)
  F3_atipico      un solo cargo muy por encima del monto habitual (NO depende del orden)
  sesion_legitima control de F1: mismos montos y ventana, en orden no creciente
"""
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (CATEGORIAS, CANALES, ESCALA_CATEGORIA, P_CANAL, N_COMERCIOS,
                     COMERCIOS_HABITUALES, P_COMERCIO_HABITUAL, MINUTOS_POR_DIA,
                     TAM_ESCALADA, TAM_RAFAGA, DISTRACTORES_POR_ESCALADA, COLUMNAS_TX,
                     MEZCLA_FRAUDE, SEED, N_TARJETAS, DIAS_SIMULACION, FECHA_INICIO,
                     TASA_FRAUDE_OBJETIVO)


@dataclass
class Poblacion:
    tasa: np.ndarray          # transacciones por dia
    mu: np.ndarray            # log-monto tipico
    sigma: np.ndarray         # dispersion del log-monto
    hora_pico: np.ndarray
    habituales: np.ndarray    # (n_tarjetas, COMERCIOS_HABITUALES)
    cat_comercio: np.ndarray  # (N_COMERCIOS,)

    @property
    def n(self):
        return self.tasa.size


def hash_df(df):
    """Huella estable de un DataFrame, para verificar reproducibilidad."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    h.update(",".join(map(str, df.columns)).encode())
    return h.hexdigest()[:16]


def muestrear_fila(rng, probs):
    """Muestreo categorico con una distribucion distinta por fila."""
    acum = np.cumsum(probs, axis=1)
    u = rng.random((probs.shape[0], 1))
    return np.minimum((u > acum).sum(axis=1), probs.shape[1] - 1)


def construir_poblacion(rng, n_tarjetas):
    cat_comercio = rng.integers(0, len(CATEGORIAS), N_COMERCIOS)
    prefs = rng.dirichlet(np.full(len(CATEGORIAS), 0.8), n_tarjetas)

    cat_slots = muestrear_fila(rng, np.repeat(prefs, COMERCIOS_HABITUALES, axis=0))
    habituales = np.empty(cat_slots.size, dtype=np.int64)
    for c in range(len(CATEGORIAS)):
        candidatos = np.flatnonzero(cat_comercio == c)
        sel = cat_slots == c
        habituales[sel] = rng.choice(candidatos, sel.sum())

    return Poblacion(
        tasa=rng.gamma(2.0, 0.225, n_tarjetas),
        mu=rng.normal(4.2, 0.45, n_tarjetas),
        sigma=rng.uniform(0.45, 0.80, n_tarjetas),
        hora_pico=rng.normal(13.5, 2.0, n_tarjetas),
        habituales=habituales.reshape(n_tarjetas, COMERCIOS_HABITUALES),
        cat_comercio=cat_comercio,
    )


def canal_y_pais(rng, categorias):
    canal = muestrear_fila(rng, P_CANAL[categorias])
    p_ext = np.where(canal == 1, 0.09, 0.012)
    return canal, (rng.random(canal.size) < p_ext).astype(np.int8)


def base_legitima(rng, pob, dias):
    """Trafico normal: llegadas por tarjeta, hora preferida y comercios habituales."""
    n_tx = rng.poisson(pob.tasa * dias)
    tarjeta = np.repeat(np.arange(pob.n), n_tx)
    n = tarjeta.size

    dia = rng.integers(0, dias, n)
    hora = np.clip(rng.normal(pob.hora_pico[tarjeta], 3.5), 0.0, 23.99)
    minuto = dia * MINUTOS_POR_DIA + hora * 60.0

    habitual = rng.random(n) < P_COMERCIO_HABITUAL
    col = rng.integers(0, COMERCIOS_HABITUALES, n)
    comercio = np.where(habitual, pob.habituales[tarjeta, col], rng.integers(0, N_COMERCIOS, n))

    categoria = pob.cat_comercio[comercio]
    canal, extranjero = canal_y_pais(rng, categoria)
    monto = np.exp(rng.normal(pob.mu[tarjeta], pob.sigma[tarjeta])) * ESCALA_CATEGORIA[categoria]

    return dict(id_tarjeta=tarjeta, minuto=minuto, monto=monto, id_comercio=comercio,
                categoria=categoria, canal=canal, extranjero=extranjero,
                es_fraude=np.zeros(n, np.int8),
                mecanismo=np.full(n, "legitimo", dtype=object),
                id_episodio=np.full(n, -1, np.int64))


def fila_episodio(pob, tarjeta, minutos, montos, comercios, canal, extranjero,
                  es_fraude, mecanismo, id_episodio):
    k = montos.size
    return dict(
        id_tarjeta=np.full(k, tarjeta, np.int64), minuto=minutos, monto=montos,
        id_comercio=comercios, categoria=pob.cat_comercio[comercios],
        canal=np.full(k, canal, np.int64), extranjero=np.full(k, extranjero, np.int8),
        es_fraude=np.full(k, es_fraude, np.int8),
        mecanismo=np.full(k, mecanismo, dtype=object),
        id_episodio=np.full(k, id_episodio, np.int64))


def montos_sesion(rng, tipico, k):
    """Sondas pequenas mas un golpe grande, sin ordenar."""
    return np.append(rng.uniform(5.0, 45.0, k), tipico * rng.uniform(8.0, 25.0))


def escaladas(rng, pob, n_episodios, dias, id0):
    """F1: fraude dependiente del orden, montos crecientes que rematan en el golpe."""
    eps = []
    for i in range(n_episodios):
        t = int(rng.integers(0, pob.n))
        montos = np.sort(montos_sesion(rng, float(np.exp(pob.mu[t])), int(rng.integers(3, 6))))
        inicio = rng.uniform(0, dias) * MINUTOS_POR_DIA
        minutos = inicio + np.cumsum(rng.uniform(1.0, 8.0, montos.size))
        comercios = rng.choice(N_COMERCIOS, montos.size, replace=False)
        eps.append(fila_episodio(pob, t, minutos, montos, comercios, 1, 1, 1,
                                 "F1_escalada", id0 + i))
    return eps


def sesiones_legitimas(rng, pob, n_episodios, dias, id0):
    """Control de F1: mismos montos, ventana y comercios, en orden no creciente."""
    eps = []
    for i in range(n_episodios):
        t = int(rng.integers(0, pob.n))
        montos = montos_sesion(rng, float(np.exp(pob.mu[t])), int(rng.integers(3, 6)))
        while True:
            rng.shuffle(montos)
            if not np.all(np.diff(montos) > 0):
                break
        inicio = rng.uniform(0, dias) * MINUTOS_POR_DIA
        minutos = inicio + np.cumsum(rng.uniform(1.0, 8.0, montos.size))
        comercios = rng.choice(N_COMERCIOS, montos.size, replace=False)
        eps.append(fila_episodio(pob, t, minutos, montos, comercios, 1, 1, 0,
                                 "sesion_legitima", id0 + i))
    return eps


def rafagas(rng, pob, n_episodios, dias, id0):
    """F2: muchos cargos en pocos minutos, en comercios distintos."""
    eps = []
    for i in range(n_episodios):
        t = int(rng.integers(0, pob.n))
        k = int(rng.integers(7, 15))
        inicio = rng.uniform(0, dias) * MINUTOS_POR_DIA
        minutos = inicio + np.sort(rng.uniform(0, rng.uniform(3.0, 20.0), k))
        montos = float(np.exp(pob.mu[t])) * rng.uniform(0.5, 2.0, k)
        comercios = rng.choice(N_COMERCIOS, k, replace=False)
        eps.append(fila_episodio(pob, t, minutos, montos, comercios, 0, 1, 1,
                                 "F2_rafaga", id0 + i))
    return eps


def atipicos(rng, pob, n_episodios, dias, id0):
    """F3: cargo unico y aislado, muy por encima del monto habitual de la tarjeta."""
    t = rng.integers(0, pob.n, n_episodios)
    minutos = (rng.integers(0, dias, n_episodios) * MINUTOS_POR_DIA
               + rng.uniform(0, MINUTOS_POR_DIA, n_episodios))
    montos = np.exp(pob.mu[t]) * rng.uniform(15.0, 40.0, n_episodios)
    comercios = rng.integers(0, N_COMERCIOS, n_episodios)
    categoria = pob.cat_comercio[comercios]
    canal, _ = canal_y_pais(rng, categoria)
    return dict(id_tarjeta=t, minuto=minutos, monto=montos, id_comercio=comercios,
                categoria=categoria, canal=canal,
                extranjero=(rng.random(n_episodios) < 0.35).astype(np.int8),
                es_fraude=np.ones(n_episodios, np.int8),
                mecanismo=np.full(n_episodios, "F3_atipico", dtype=object),
                id_episodio=id0 + np.arange(n_episodios))


def contar_episodios(n_base, tasa):
    """Reparte la tasa objetivo entre mecanismos. Los distractores suman filas
    legitimas, asi que entran en el denominador."""
    extra = 1.0 + DISTRACTORES_POR_ESCALADA * MEZCLA_FRAUDE["F1_escalada"]
    total = tasa * n_base / (1.0 - tasa * extra)
    n_f1 = round(MEZCLA_FRAUDE["F1_escalada"] * total / TAM_ESCALADA)
    return {"F1_escalada": n_f1,
            "F2_rafaga": round(MEZCLA_FRAUDE["F2_rafaga"] * total / TAM_RAFAGA),
            "F3_atipico": round(MEZCLA_FRAUDE["F3_atipico"] * total),
            "sesion_legitima": round(DISTRACTORES_POR_ESCALADA * n_f1)}


def generar(seed=SEED, n_tarjetas=N_TARJETAS, dias=DIAS_SIMULACION):
    """Historial completo de transacciones, ordenado por tarjeta y tiempo."""
    rng = np.random.default_rng(seed)
    pob = construir_poblacion(rng, n_tarjetas)
    base = base_legitima(rng, pob, dias)

    n = contar_episodios(base["monto"].size, TASA_FRAUDE_OBJETIVO)
    partes, id0 = [base], 0
    for constructor, clave in ((escaladas, "F1_escalada"), (rafagas, "F2_rafaga"),
                               (sesiones_legitimas, "sesion_legitima")):
        partes.extend(constructor(rng, pob, n[clave], dias, id0))
        id0 += n[clave]
    partes.append(atipicos(rng, pob, n["F3_atipico"], dias, id0))

    df = pd.DataFrame({k: np.concatenate([p[k] for p in partes]) for k in base})
    df["timestamp"] = pd.Timestamp(FECHA_INICIO) + pd.to_timedelta(df.pop("minuto").round(), unit="m")
    df["monto"] = df["monto"].round(2)
    df["categoria"] = pd.Categorical.from_codes(df["categoria"], CATEGORIAS)
    df["canal"] = pd.Categorical.from_codes(df["canal"], CANALES)
    df["mecanismo"] = df["mecanismo"].astype("category")
    df = df.sort_values(["id_tarjeta", "timestamp"], kind="mergesort").reset_index(drop=True)
    df.insert(0, "id_transaccion", np.arange(len(df), dtype=np.int64))
    return df[COLUMNAS_TX]
