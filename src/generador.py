"""Generador sintetico de transacciones de tarjeta con tres mecanismos de fraude."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

CATEGORIAS = ("supermercado", "restaurante", "combustible", "farmacia", "ropa",
              "electronica", "viajes", "servicios", "entretenimiento")
CANALES = ("pos", "ecommerce", "atm", "recurrente")

# Factor de escala del monto tipico por categoria.
ESCALA_CATEGORIA = np.array([1.4, 0.6, 0.9, 0.5, 1.1, 2.6, 3.2, 1.0, 0.7])

# P(canal | categoria); filas en el orden de CATEGORIAS.
P_CANAL = np.array([
    [0.88, 0.06, 0.03, 0.03],
    [0.90, 0.08, 0.01, 0.01],
    [0.95, 0.01, 0.03, 0.01],
    [0.90, 0.07, 0.01, 0.02],
    [0.55, 0.43, 0.01, 0.01],
    [0.40, 0.58, 0.01, 0.01],
    [0.15, 0.83, 0.01, 0.01],
    [0.10, 0.25, 0.02, 0.63],
    [0.30, 0.60, 0.02, 0.08],
])

N_COMERCIOS = 1_200
COMERCIOS_HABITUALES = 8
P_COMERCIO_HABITUAL = 0.85
MINUTOS_POR_DIA = 1_440

# Tamano medio de episodio, usado para traducir la tasa de fraude objetivo a un
# numero de episodios antes de generarlos.
TAM_ESCALADA = 5.0
TAM_RAFAGA = 10.5

# Sesiones legitimas que replican monto, ventana y comercios de una escalada,
# pero en otro orden. Sin ellas, "varias compras chicas y una grande" seria
# separable por agregados y el orden no tendria nada que aportar.
DISTRACTORES_POR_ESCALADA = 2.0

COLUMNAS = ["id_transaccion", "id_tarjeta", "timestamp", "monto", "id_comercio",
            "categoria", "canal", "extranjero", "es_fraude", "mecanismo", "id_episodio"]


@dataclass
class Poblacion:
    tasa: np.ndarray          # transacciones por dia
    mu: np.ndarray            # log-monto tipico
    sigma: np.ndarray         # dispersion del log-monto
    hora_pico: np.ndarray
    habituales: np.ndarray    # (n_tarjetas, COMERCIOS_HABITUALES)
    cat_comercio: np.ndarray  # (N_COMERCIOS,)

    @property
    def n(self) -> int:
        return self.tasa.size


def _muestrear_fila(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """Muestreo categorico con una distribucion distinta por fila."""
    acum = np.cumsum(probs, axis=1)
    u = rng.random((probs.shape[0], 1))
    return np.minimum((u > acum).sum(axis=1), probs.shape[1] - 1)


def _poblacion(rng: np.random.Generator, n_tarjetas: int) -> Poblacion:
    cat_comercio = rng.integers(0, len(CATEGORIAS), N_COMERCIOS)
    prefs = rng.dirichlet(np.full(len(CATEGORIAS), 0.8), n_tarjetas)

    # Los comercios habituales se sortean por categoria preferida, de modo que el
    # gusto de cada tarjeta queda implicito en su lista de comercios.
    cat_slots = _muestrear_fila(rng, np.repeat(prefs, COMERCIOS_HABITUALES, axis=0))
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


def _monto_tipico(pob: Poblacion, tarjeta: int) -> float:
    return float(np.exp(pob.mu[tarjeta]))


def _canal_y_pais(rng: np.random.Generator, categorias: np.ndarray):
    canal = _muestrear_fila(rng, P_CANAL[categorias])
    p_ext = np.where(canal == 1, 0.09, 0.012)
    return canal, (rng.random(canal.size) < p_ext).astype(np.int8)


def _base_legitima(rng: np.random.Generator, pob: Poblacion, dias: int) -> dict:
    n_tx = rng.poisson(pob.tasa * dias)
    tarjeta = np.repeat(np.arange(pob.n), n_tx)
    n = tarjeta.size

    dia = rng.integers(0, dias, n)
    hora = np.clip(rng.normal(pob.hora_pico[tarjeta], 3.5), 0.0, 23.99)
    minuto = dia * MINUTOS_POR_DIA + hora * 60.0

    habitual = rng.random(n) < P_COMERCIO_HABITUAL
    col = rng.integers(0, COMERCIOS_HABITUALES, n)
    comercio = np.where(habitual, pob.habituales[tarjeta, col],
                        rng.integers(0, N_COMERCIOS, n))

    categoria = pob.cat_comercio[comercio]
    canal, extranjero = _canal_y_pais(rng, categoria)
    monto = np.exp(rng.normal(pob.mu[tarjeta], pob.sigma[tarjeta])) * ESCALA_CATEGORIA[categoria]

    return dict(id_tarjeta=tarjeta, minuto=minuto, monto=monto, id_comercio=comercio,
                categoria=categoria, canal=canal, extranjero=extranjero,
                es_fraude=np.zeros(n, np.int8),
                mecanismo=np.full(n, "legitimo", dtype=object),
                id_episodio=np.full(n, -1, np.int64))


def _episodio(pob, tarjeta, minutos, montos, comercios, canal, extranjero,
              es_fraude, mecanismo, id_episodio) -> dict:
    k = montos.size
    return dict(
        id_tarjeta=np.full(k, tarjeta, np.int64), minuto=minutos, monto=montos,
        id_comercio=comercios, categoria=pob.cat_comercio[comercios],
        canal=np.full(k, canal, np.int64),
        extranjero=np.full(k, extranjero, np.int8),
        es_fraude=np.full(k, es_fraude, np.int8),
        mecanismo=np.full(k, mecanismo, dtype=object),
        id_episodio=np.full(k, id_episodio, np.int64),
    )


def _montos_sesion(rng: np.random.Generator, tipico: float, k: int) -> np.ndarray:
    """Sondas pequenas mas un golpe grande, sin ordenar."""
    sondas = rng.uniform(5.0, 45.0, k)
    golpe = tipico * rng.uniform(8.0, 25.0)
    return np.append(sondas, golpe)


def _escaladas(rng, pob, n_episodios, dias, id0):
    """Fraude dependiente del orden: montos crecientes que rematan en el golpe."""
    eps = []
    for i in range(n_episodios):
        t = int(rng.integers(0, pob.n))
        montos = np.sort(_montos_sesion(rng, _monto_tipico(pob, t), int(rng.integers(3, 6))))
        inicio = rng.uniform(0, dias) * MINUTOS_POR_DIA
        minutos = inicio + np.cumsum(rng.uniform(1.0, 8.0, montos.size))
        comercios = rng.choice(N_COMERCIOS, montos.size, replace=False)
        eps.append(_episodio(pob, t, minutos, montos, comercios, canal=1, extranjero=1,
                             es_fraude=1, mecanismo="F1_escalada", id_episodio=id0 + i))
    return eps


def _sesiones_legitimas(rng, pob, n_episodios, dias, id0):
    """Control de la escalada: mismos montos y ventana, orden no creciente."""
    eps = []
    for i in range(n_episodios):
        t = int(rng.integers(0, pob.n))
        montos = _montos_sesion(rng, _monto_tipico(pob, t), int(rng.integers(3, 6)))
        while True:
            rng.shuffle(montos)
            if not np.all(np.diff(montos) > 0):
                break
        inicio = rng.uniform(0, dias) * MINUTOS_POR_DIA
        minutos = inicio + np.cumsum(rng.uniform(1.0, 8.0, montos.size))
        comercios = rng.choice(N_COMERCIOS, montos.size, replace=False)
        eps.append(_episodio(pob, t, minutos, montos, comercios, canal=1, extranjero=1,
                             es_fraude=0, mecanismo="sesion_legitima", id_episodio=id0 + i))
    return eps


def _rafagas(rng, pob, n_episodios, dias, id0):
    """Muchos cargos en pocos minutos, en comercios distintos."""
    eps = []
    for i in range(n_episodios):
        t = int(rng.integers(0, pob.n))
        k = int(rng.integers(7, 15))
        ventana = rng.uniform(3.0, 20.0)
        inicio = rng.uniform(0, dias) * MINUTOS_POR_DIA
        minutos = inicio + np.sort(rng.uniform(0, ventana, k))
        montos = _monto_tipico(pob, t) * rng.uniform(0.5, 2.0, k)
        comercios = rng.choice(N_COMERCIOS, k, replace=False)
        eps.append(_episodio(pob, t, minutos, montos, comercios, canal=0, extranjero=1,
                             es_fraude=1, mecanismo="F2_rafaga", id_episodio=id0 + i))
    return eps


def _atipicos(rng, pob, n_episodios, dias, id0) -> dict:
    """Cargo unico y aislado, muy por encima del monto habitual de la tarjeta."""
    t = rng.integers(0, pob.n, n_episodios)
    minutos = (rng.integers(0, dias, n_episodios) * MINUTOS_POR_DIA
               + rng.uniform(0, MINUTOS_POR_DIA, n_episodios))
    montos = np.exp(pob.mu[t]) * rng.uniform(15.0, 40.0, n_episodios)
    comercios = rng.integers(0, N_COMERCIOS, n_episodios)
    categoria = pob.cat_comercio[comercios]
    canal, _ = _canal_y_pais(rng, categoria)
    return dict(id_tarjeta=t, minuto=minutos, monto=montos, id_comercio=comercios,
                categoria=categoria, canal=canal,
                extranjero=(rng.random(n_episodios) < 0.35).astype(np.int8),
                es_fraude=np.ones(n_episodios, np.int8),
                mecanismo=np.full(n_episodios, "F3_atipico", dtype=object),
                id_episodio=id0 + np.arange(n_episodios))


def _n_episodios(n_base: int, tasa: float) -> dict:
    """Reparte la tasa de fraude objetivo entre mecanismos.

    Los distractores suman filas legitimas, asi que entran en el denominador.
    """
    mezcla = config.MEZCLA_FRAUDE
    extra = 1.0 + DISTRACTORES_POR_ESCALADA * mezcla["F1_escalada"]
    total_fraude = tasa * n_base / (1.0 - tasa * extra)
    n_f1 = round(mezcla["F1_escalada"] * total_fraude / TAM_ESCALADA)
    return {
        "F1_escalada": n_f1,
        "F2_rafaga": round(mezcla["F2_rafaga"] * total_fraude / TAM_RAFAGA),
        "F3_atipico": round(mezcla["F3_atipico"] * total_fraude),
        "sesion_legitima": round(DISTRACTORES_POR_ESCALADA * n_f1),
    }


def generar(seed: int = None, n_tarjetas: int = None, dias: int = None) -> pd.DataFrame:
    """Devuelve el historial completo de transacciones, ordenado por tarjeta y tiempo."""
    seed = config.SEED if seed is None else seed
    n_tarjetas = config.N_TARJETAS if n_tarjetas is None else n_tarjetas
    dias = config.DIAS_SIMULACION if dias is None else dias

    rng = np.random.default_rng(seed)
    pob = _poblacion(rng, n_tarjetas)
    base = _base_legitima(rng, pob, dias)

    n = _n_episodios(base["monto"].size, config.TASA_FRAUDE_OBJETIVO)
    partes = [base]
    id0 = 0
    for constructor, clave in ((_escaladas, "F1_escalada"), (_rafagas, "F2_rafaga"),
                               (_sesiones_legitimas, "sesion_legitima")):
        partes.extend(constructor(rng, pob, n[clave], dias, id0))
        id0 += n[clave]
    partes.append(_atipicos(rng, pob, n["F3_atipico"], dias, id0))

    df = pd.DataFrame({k: np.concatenate([p[k] for p in partes]) for k in base})
    df["timestamp"] = (pd.Timestamp(config.FECHA_INICIO)
                       + pd.to_timedelta(df.pop("minuto").round(), unit="m"))
    df["monto"] = df["monto"].round(2)
    df["categoria"] = pd.Categorical.from_codes(df["categoria"], CATEGORIAS)
    df["canal"] = pd.Categorical.from_codes(df["canal"], CANALES)
    df["mecanismo"] = df["mecanismo"].astype("category")

    df = df.sort_values(["id_tarjeta", "timestamp"], kind="mergesort").reset_index(drop=True)
    df.insert(0, "id_transaccion", np.arange(len(df), dtype=np.int64))
    return df[COLUMNAS]
