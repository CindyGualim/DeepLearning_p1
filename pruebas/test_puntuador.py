"""Contrato del puntuador: lo que el servicio promete y lo que rechaza.

Estas pruebas no miden calidad del modelo —eso lo hace `test_paridad.py`— sino que el
servicio se comporte como dice el README: valida su entrada, no se cae con casos
limite, y responde a las dos manipulaciones de historia de la forma que la evidencia
del Proyecto 1 exige.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DIR_DEMO, MIN_EVENTOS_PREVIOS       # noqa: E402
from src.puntuador import Puntuador                        # noqa: E402


@pytest.fixture(scope="module")
def p():
    return Puntuador()


@pytest.fixture(scope="module")
def banco():
    ruta = DIR_DEMO / "banco_casos.json"
    if not ruta.exists():
        pytest.skip("falta datos_demo/banco_casos.json (python scripts/construir_banco.py)")
    return json.loads(ruta.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def caso_f1(banco):
    """Un caso de escalada: el mecanismo que depende del orden por construccion."""
    for c in banco["casos"]:
        if c["verdad"]["mecanismo"] == "F1_escalada" and not c["verdad"]["sonda_temprana"]:
            return c
    pytest.skip("el banco no tiene casos F1_escalada")


HISTORIA_MINIMA = [
    {"timestamp": "2025-04-20 09:14:00", "monto": 82.5, "id_comercio": 41,
     "categoria": "supermercado", "canal": "pos", "extranjero": 0},
    {"timestamp": "2025-04-20 11:02:00", "monto": 15.0, "id_comercio": 902,
     "categoria": "farmacia", "canal": "pos", "extranjero": 0},
    {"timestamp": "2025-04-21 19:40:00", "monto": 240.0, "id_comercio": 77,
     "categoria": "ropa", "canal": "ecommerce", "extranjero": 0},
    {"timestamp": "2025-04-21 19:52:00", "monto": 3150.0, "id_comercio": 512,
     "categoria": "electronica", "canal": "ecommerce", "extranjero": 1},
]


# ---------- contrato de salida ----------

def test_puntua_la_historia_minima(p):
    r = p.puntuar(HISTORIA_MINIMA)
    assert set(r["puntajes"]) == {"A", "B", "C"}
    for m, v in r["puntajes"].items():
        assert 0.0 <= v <= 1.0, f"{m} fuera de [0,1]: {v}"
    assert r["eventos_visibles"] == len(HISTORIA_MINIMA)
    assert r["decision"]["decide"] == "A" and r["decision"]["prioriza"] == "C"
    assert r["latencia_ms"] > 0


def test_la_decision_sigue_el_umbral_de_a(p, caso_f1):
    r = p.puntuar(caso_f1["historia"])
    assert r["decision"]["alertar"] == (r["puntajes"]["A"] >= p.umbral_a)


def test_el_orden_de_eventos_termina_en_el_objetivo(p, caso_f1):
    r = p.puntuar(caso_f1["historia"])
    assert r["orden_eventos"][-1] == len(caso_f1["historia"]) - 1


# ---------- validacion de entrada ----------

def test_rechaza_historia_demasiado_corta(p):
    with pytest.raises(ValueError, match="al menos"):
        p.puntuar(HISTORIA_MINIMA[:MIN_EVENTOS_PREVIOS])


def test_rechaza_historia_vacia(p):
    with pytest.raises(ValueError):
        p.puntuar([])


def test_rechaza_desordenada(p):
    revuelta = [HISTORIA_MINIMA[2], HISTORIA_MINIMA[0], HISTORIA_MINIMA[1], HISTORIA_MINIMA[3]]
    with pytest.raises(ValueError, match="ordenadas"):
        p.puntuar(revuelta)


def test_rechaza_categoria_inexistente(p):
    mala = [dict(t) for t in HISTORIA_MINIMA]
    mala[-1]["categoria"] = "criptomonedas"
    with pytest.raises(ValueError, match="categoria"):
        p.puntuar(mala)


def test_rechaza_canal_inexistente(p):
    mala = [dict(t) for t in HISTORIA_MINIMA]
    mala[-1]["canal"] = "telepatia"
    with pytest.raises(ValueError, match="canal"):
        p.puntuar(mala)


def test_rechaza_monto_no_positivo(p):
    mala = [dict(t) for t in HISTORIA_MINIMA]
    mala[-1]["monto"] = 0
    with pytest.raises(ValueError, match="positivo"):
        p.puntuar(mala)


def test_rechaza_campo_faltante(p):
    mala = [dict(t) for t in HISTORIA_MINIMA]
    del mala[1]["monto"]
    with pytest.raises(ValueError, match="faltan campos"):
        p.puntuar(mala)


def test_rechaza_dos_tarjetas(p):
    mezcla = [dict(t, id_tarjeta=i % 2) for i, t in enumerate(HISTORIA_MINIMA)]
    with pytest.raises(ValueError, match="una sola tarjeta"):
        p.puntuar(mezcla)


# ---------- las dos pruebas de falsificacion ----------

def test_el_modelo_a_es_ciego_al_orden(p, banco):
    """A lee agregados de ventana. Barajar la historia NO puede moverlo.

    Es el control de las dos pruebas: si A se moviera, la caida de B no probaria nada
    sobre el orden, solo que la consola cambio los datos.
    """
    for caso in banco["casos"][:25]:
        base = p.puntuar(caso["historia"])
        for semilla in (1, 7, 99):
            perm = p.puntuar(caso["historia"], semilla_permutacion=semilla)
            assert perm["puntajes"]["A"] == pytest.approx(base["puntajes"]["A"], abs=1e-9), \
                f"{caso['id']}: el modelo A se movio al barajar"


def test_barajar_degrada_al_modelo_secuencial_en_escaladas(p, banco):
    """Sobre escaladas —el mecanismo que depende del orden— barajar tiene que doler."""
    escaladas = [c for c in banco["casos"]
                 if c["verdad"]["mecanismo"] == "F1_escalada" and not c["verdad"]["sonda_temprana"]]
    if len(escaladas) < 5:
        pytest.skip("hacen falta al menos 5 escaladas en el banco")

    caidas = []
    for caso in escaladas:
        base = p.puntuar(caso["historia"])["puntajes"]["B"]
        perm = p.puntuar(caso["historia"], semilla_permutacion=7)["puntajes"]["B"]
        caidas.append(base - perm)
    media = sum(caidas) / len(caidas)
    assert media > 0.10, f"la caida media de B al barajar fue solo {media:.4f}"


def test_permutar_conserva_los_eventos(p, caso_f1):
    base = p.puntuar(caso_f1["historia"])
    perm = p.puntuar(caso_f1["historia"], semilla_permutacion=3)
    assert sorted(perm["orden_eventos"]) == sorted(base["orden_eventos"])
    assert perm["orden_eventos"][-1] == base["orden_eventos"][-1], "se movio el objetivo"


def test_recortar_reduce_los_eventos_visibles(p, caso_f1):
    for k in (1, 3, 6, 20):
        r = p.puntuar(caso_f1["historia"], k_visible=k)
        assert r["eventos_visibles"] == min(k, len(caso_f1["historia"]), 20)


def test_recortar_no_mueve_al_modelo_a(p, caso_f1):
    """Las agregadas usan ventanas de tiempo reales, no la ventana de secuencia."""
    base = p.puntuar(caso_f1["historia"])["puntajes"]["A"]
    for k in (1, 4, 12):
        assert p.puntuar(caso_f1["historia"], k_visible=k)["puntajes"]["A"] == \
            pytest.approx(base, abs=1e-9)


# ---------- trayectoria ----------

def test_trayectoria_cubre_cada_transaccion_puntuable(p, caso_f1):
    t = p.trayectoria(caso_f1["historia"])
    n = len(caso_f1["historia"])
    assert t["posiciones"] == list(range(MIN_EVENTOS_PREVIOS, n))
    for m in ("A", "B", "C"):
        assert len(t["puntajes"][m]) == n - MIN_EVENTOS_PREVIOS


def test_la_trayectoria_termina_en_el_puntaje_del_caso(p, caso_f1):
    """El ultimo punto de la trayectoria es, por definicion, el puntaje del caso."""
    t = p.trayectoria(caso_f1["historia"])
    r = p.puntuar(caso_f1["historia"])
    for m in ("A", "B", "C"):
        assert t["puntajes"][m][-1] == pytest.approx(r["puntajes"][m], abs=1e-6)


# ---------- determinismo ----------

def test_puntuar_es_determinista(p, caso_f1):
    a = p.puntuar(caso_f1["historia"])["puntajes"]
    b = p.puntuar(caso_f1["historia"])["puntajes"]
    assert a == b
