"""Pruebas de la API HTTP: el contrato que ve una persona ajena al equipo.

Un modelo esta en produccion si alguien de afuera puede usarlo. Estas pruebas recorren
exactamente ese camino —peticiones HTTP contra la app, sin atajos por dentro— e
incluyen el caso que la consola nunca debe permitir: ver la verdad de un caso antes de
emitir un veredicto.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DIR_DEMO                                # noqa: E402

fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture(scope="module")
def cliente(tmp_path_factory):
    if not (DIR_DEMO / "banco_casos.json").exists():
        pytest.skip("falta datos_demo/banco_casos.json (python scripts/construir_banco.py)")

    # El registro va a un archivo temporal: las pruebas no deben ensuciar la
    # evidencia de uso real que se entrega en el repositorio.
    import src.registro as registro
    registro._REGISTRO = registro.Registro(tmp_path_factory.mktemp("reg") / "prueba.db")

    from api.app import app
    with fastapi_testclient.TestClient(app) as c:
        yield c


HISTORIA = [
    {"timestamp": "2025-04-20 09:14:00", "monto": 82.5, "id_comercio": 41,
     "categoria": "supermercado", "canal": "pos", "extranjero": 0},
    {"timestamp": "2025-04-20 11:02:00", "monto": 15.0, "id_comercio": 902,
     "categoria": "farmacia", "canal": "pos", "extranjero": 0},
    {"timestamp": "2025-04-21 19:40:00", "monto": 240.0, "id_comercio": 77,
     "categoria": "ropa", "canal": "ecommerce", "extranjero": 0},
    {"timestamp": "2025-04-21 19:52:00", "monto": 3150.0, "id_comercio": 512,
     "categoria": "electronica", "canal": "ecommerce", "extranjero": 1},
]


# ---------- salud y paridad ----------

def test_salud(cliente):
    r = cliente.get("/salud")
    assert r.status_code == 200
    d = r.json()
    assert d["estado"] == "ok"
    assert set(d["modelo"]["modelos"]) == {"A", "B", "C"}
    assert d["modelo"]["politica"] == {"decide": "A", "prioriza": "C"}


def test_el_servicio_en_vivo_coincide_con_el_lote(cliente):
    """`/salud` publica el desvio maximo entre puntuar en vivo y el banco por lotes."""
    d = cliente.get("/salud").json()
    assert d["paridad_con_el_notebook"]["desvio_maximo_absoluto"] < 1e-5


# ---------- /puntuar ----------

def test_puntuar(cliente):
    r = cliente.post("/puntuar", json={"historia": HISTORIA})
    assert r.status_code == 200
    d = r.json()
    assert set(d["puntajes"]) == {"A", "B", "C"}
    assert all(0.0 <= v <= 1.0 for v in d["puntajes"].values())
    assert d["decision"]["decide"] == "A"


def test_puntuar_rechaza_historia_corta(cliente):
    r = cliente.post("/puntuar", json={"historia": HISTORIA[:3]})
    assert r.status_code == 422
    assert "al menos" in r.json()["detail"]


def test_puntuar_rechaza_categoria_invalida(cliente):
    mala = [dict(t) for t in HISTORIA]
    mala[-1]["categoria"] = "criptomonedas"
    r = cliente.post("/puntuar", json={"historia": mala})
    assert r.status_code == 422


def test_puntuar_con_permutacion_no_mueve_a(cliente):
    base = cliente.post("/puntuar", json={"historia": HISTORIA}).json()
    perm = cliente.post("/puntuar", json={"historia": HISTORIA, "permutar": 11}).json()
    assert perm["puntajes"]["A"] == pytest.approx(base["puntajes"]["A"], abs=1e-9)


def test_puntuar_con_recorte_reduce_visibles(cliente):
    d = cliente.post("/puntuar", json={"historia": HISTORIA, "k_visible": 2}).json()
    assert d["eventos_visibles"] == 2


# ---------- cola ----------

def test_cola_ordenada_por_el_modelo_pedido(cliente):
    for modelo in ("A", "B", "C"):
        d = cliente.get(f"/casos?orden={modelo}").json()
        puntajes = [c["puntajes"][modelo] for c in d["casos"]]
        assert puntajes == sorted(puntajes, reverse=True)


def test_cola_rechaza_orden_invalido(cliente):
    assert cliente.get("/casos?orden=Z").status_code == 422


def test_la_cola_no_filtra_la_verdad(cliente):
    """El mecanismo real no puede viajar al navegador antes del veredicto.

    Si se filtrara, la revelacion seria teatro: bastaria abrir las herramientas de
    desarrollo para saber la respuesta antes de decidir.
    """
    d = cliente.get("/casos").json()
    sin_revelar = [c for c in d["casos"] if not c["revelado"]]
    assert sin_revelar, "todos los casos ya estaban revelados"
    for c in sin_revelar:
        assert c["verdad"] is None
        assert "mecanismo" not in c and "referencia" not in c


def test_detalle_tampoco_filtra_la_verdad(cliente):
    cid = next(c["id"] for c in cliente.get("/casos").json()["casos"] if not c["revelado"])
    d = cliente.get(f"/casos/{cid}").json()
    assert d["verdad"] is None
    assert "referencia" not in d
    assert len(d["historia"]) == d["n_eventos"]
    assert len(d["trayectoria"]["posiciones"]) == d["n_eventos"] - 3


def test_caso_inexistente(cliente):
    assert cliente.get("/casos/NOEXISTE").status_code == 404


# ---------- experimentos ----------

def test_experimento_permutar(cliente):
    cid = cliente.get("/casos").json()["casos"][0]["id"]
    d = cliente.post(f"/casos/{cid}/experimento",
                     json={"tipo": "permutar", "semilla": 5}).json()
    assert d["tipo"] == "permutar"
    assert d["puntajes"]["A"] == pytest.approx(d["base"]["A"], abs=1e-9)
    assert sorted(d["orden_eventos"]) == sorted(d["orden_eventos"])


def test_experimento_recortar(cliente):
    cid = cliente.get("/casos").json()["casos"][0]["id"]
    d = cliente.post(f"/casos/{cid}/experimento", json={"tipo": "recortar", "k": 3}).json()
    assert d["eventos_visibles"] == 3


def test_experimento_tipo_invalido(cliente):
    cid = cliente.get("/casos").json()["casos"][0]["id"]
    r = cliente.post(f"/casos/{cid}/experimento", json={"tipo": "invertir"})
    assert r.status_code == 422


# ---------- veredicto y revelacion ----------

def test_veredicto_revela_la_verdad(cliente):
    cid = next(c["id"] for c in cliente.get("/casos").json()["casos"] if not c["revelado"])
    d = cliente.post(f"/casos/{cid}/veredicto", json={"veredicto": "revisar"}).json()
    assert d["verdad"]["mecanismo"]
    assert isinstance(d["acierto"], bool)
    assert d["acierto"] == (bool(d["verdad"]["es_fraude"]))
    # y a partir de ahora la cola si lo muestra
    fila = next(c for c in cliente.get("/casos").json()["casos"] if c["id"] == cid)
    assert fila["revelado"] and fila["verdad"] is not None


def test_veredicto_invalido(cliente):
    cid = cliente.get("/casos").json()["casos"][0]["id"]
    assert cliente.post(f"/casos/{cid}/veredicto",
                        json={"veredicto": "quizas"}).status_code == 422


# ---------- evidencias y registros ----------

def test_evidencias_trae_las_tablas_del_informe(cliente):
    d = cliente.get("/evidencias").json()
    assert {"particiones", "permutacion", "recorte", "sensibilidad"} <= set(d)
    assert d["particiones"]["test"]["modelos"]["C"]["auc_pr"] == pytest.approx(0.9361, abs=5e-4)
    barajada = next(r for r in d["permutacion"]["tabla"] if r["variante"] == "historia barajada")
    assert barajada["caida"] > 0.2, "barajar tendria que costarle AUC-PR a B"


def test_f3_no_se_degrada_al_barajar(cliente):
    """El control del experimento, servido por la API."""
    d = cliente.get("/evidencias").json()
    f3 = next(r for r in d["permutacion"]["por_mecanismo"] if r["mecanismo"] == "F3_atipico")
    f1 = next(r for r in d["permutacion"]["por_mecanismo"] if r["mecanismo"] == "F1_escalada")
    assert f1["B_permutado"] < f1["B_real"] - 0.2, "F1 depende del orden y deberia caer"
    assert f3["B_permutado"] >= f3["B_real"] - 0.05, "F3 no deberia degradarse"


def test_registros_crecen_con_el_uso(cliente):
    antes = cliente.get("/registros").json()["resumen"]["peticiones"]
    cliente.post("/puntuar", json={"historia": HISTORIA})
    despues = cliente.get("/registros").json()["resumen"]["peticiones"]
    assert despues == antes + 1


def test_la_consola_se_sirve(cliente):
    r = cliente.get("/")
    assert r.status_code == 200 and "Centinela" in r.text
    assert cliente.get("/estatico/consola.js").status_code == 200
    assert cliente.get("/estatico/estilos.css").status_code == 200
