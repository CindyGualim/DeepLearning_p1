"""Paridad: el codigo que sirve el modelo reproduce el notebook.

Es la prueba que sostiene todo lo demas. El notebook del Proyecto 1 es la fuente de
verdad del entrenamiento y publico seis AUC-PR en el informe; `src/` es una
reescritura del mismo pipeline para poder servirlo. Si las dos rutas se separan, los
numeros que el servicio muestra al comite dejan de ser los que el informe defiende.

Es lenta (~15 s): regenera las 324,603 transacciones y puntua tres modelos sobre dos
particiones. Se salta con `-m "not lenta"`.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.caracteristicas import agregadas, construir_secuencias   # noqa: E402
from src.config import COLUMNAS_AG                                # noqa: E402
from src.generador import generar                                 # noqa: E402
from src.puntuador import Puntuador                               # noqa: E402

pytestmark = pytest.mark.lenta

# Publicados en el informe del Proyecto 1.
AUC_PR_ESPERADO = {
    ("val", "A"): 0.9171, ("val", "B"): 0.9226, ("val", "C"): 0.9424,
    ("test", "A"): 0.9092, ("test", "B"): 0.9206, ("test", "C"): 0.9361,
}
# Celda 38 del notebook: perfil de agregadas por mecanismo sobre entrenamiento.
PERFIL_ESPERADO = {
    "F1_escalada":     {"razon_max_24h": 0.760, "z_monto_tarjeta": -0.006,
                        "n_1h": 2.204, "comercios_24h": 2.619},
    "F2_rafaga":       {"razon_max_24h": -0.181, "z_monto_tarjeta": 0.082,
                        "n_1h": 5.179, "comercios_24h": 5.651},
    "F3_atipico":      {"razon_max_24h": 5.774, "z_monto_tarjeta": 4.655,
                        "n_1h": 0.025, "comercios_24h": 0.520},
    "sesion_legitima": {"razon_max_24h": -0.844, "z_monto_tarjeta": -0.211,
                        "n_1h": 2.205, "comercios_24h": 2.598},
}


@pytest.fixture(scope="module")
def mundo():
    tx = generar()
    datos = construir_secuencias(tx)
    X = agregadas(tx).to_numpy()
    p = Puntuador()
    num_z = p.esc_eventos.transform(datos.num).astype(np.float32)
    agg_z = p.esc_agregadas.transform(X)[:, :18].astype(np.float32)
    return {"tx": tx, "datos": datos, "X": X, "p": p, "num_z": num_z, "agg_z": agg_z}


def test_el_generador_reproduce_el_conjunto(mundo):
    tx = mundo["tx"]
    assert len(tx) == 324_603
    assert tx.id_tarjeta.nunique() == 5_991
    assert tx.es_fraude.mean() * 100 == pytest.approx(1.197, abs=0.001)


def test_el_generador_es_reproducible():
    from src.generador import hash_df
    assert hash_df(generar()) == hash_df(generar())
    assert hash_df(generar()) != hash_df(generar(seed=2027))


def test_particion_temporal_sin_solape(mundo):
    d = mundo["datos"]
    import pandas as pd
    t = pd.Series(d.t)
    lim = {p: (t[d.particion == p].min(), t[d.particion == p].max())
           for p in ("train", "val", "test")}
    assert lim["train"][1] < lim["val"][0]
    assert lim["val"][1] < lim["test"][0]


def test_perfil_de_agregadas_coincide_con_la_celda_38(mundo):
    """Si una columna de ventana se rompe, esto lo dice antes que el AUC-PR."""
    import pandas as pd
    d, X = mundo["datos"], mundo["X"]
    idx_tr = d.sel("train")
    Xtr = X[d.fila_tx[idx_tr]]
    perfil = pd.DataFrame({c: Xtr[:, COLUMNAS_AG.index(c)]
                           for c in ("razon_max_24h", "z_monto_tarjeta", "n_1h", "comercios_24h")})
    perfil["mecanismo"] = d.mecanismo[idx_tr]
    medias = perfil.groupby("mecanismo", observed=True).mean()

    for mec, esperado in PERFIL_ESPERADO.items():
        for col, valor in esperado.items():
            assert medias.loc[mec, col] == pytest.approx(valor, abs=0.001), \
                f"{mec}.{col}: esperado {valor}, obtenido {medias.loc[mec, col]:.3f}"


@pytest.mark.parametrize("particion,modelo", list(AUC_PR_ESPERADO))
def test_auc_pr_coincide_con_el_informe(mundo, particion, modelo):
    d, p = mundo["datos"], mundo["p"]
    sel = d.sel(particion)
    s = p.puntuar_matrices(mundo["num_z"], d.cat, mundo["X"], mundo["agg_z"],
                           d.idx[sel], d.mascara[sel], d.fila_tx[sel])[modelo]
    obtenido = float(average_precision_score(d.y[sel], s))
    assert obtenido == pytest.approx(AUC_PR_ESPERADO[(particion, modelo)], abs=5e-4)


def test_la_ruta_por_tarjeta_coincide_con_la_ruta_por_lote(mundo):
    """El servicio puntua una tarjeta a la vez; el notebook puntua matrices enteras.

    Tienen que dar el mismo numero: es lo que permite mostrarle al comite los AUC-PR
    del informe junto a los puntajes que la consola calcula en vivo.
    """
    d, p, tx = mundo["datos"], mundo["p"], mundo["tx"]
    sel = d.sel("test")
    rng = np.random.default_rng(0)
    muestra = rng.choice(sel, 15, replace=False)

    lote = p.puntuar_matrices(mundo["num_z"], d.cat, mundo["X"], mundo["agg_z"],
                              d.idx[muestra], d.mascara[muestra], d.fila_tx[muestra])
    tarjeta_tx = tx.id_tarjeta.to_numpy()

    for k, i in enumerate(muestra):
        objetivo = int(d.fila_tx[i])
        filas = np.flatnonzero(tarjeta_tx[:objetivo + 1] == tarjeta_tx[objetivo])
        sub = tx.iloc[filas]
        historia = [{"timestamp": t.isoformat(sep=" "), "monto": float(m),
                     "id_comercio": int(c), "categoria": str(cat), "canal": str(can),
                     "extranjero": int(e)}
                    for t, m, c, cat, can, e in zip(sub.timestamp, sub.monto,
                                                    sub.id_comercio, sub.categoria,
                                                    sub.canal, sub.extranjero)]
        vivo = p.puntuar(historia)["puntajes"]
        for modelo in ("A", "B", "C"):
            assert vivo[modelo] == pytest.approx(float(lote[modelo][k]), abs=1e-5), \
                f"caso {k}, modelo {modelo}: por tarjeta {vivo[modelo]} != por lote {lote[modelo][k]}"
