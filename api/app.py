"""Centinela — servicio HTTP del modelo del Proyecto 1.

Sirve tres cosas sobre el mismo puntuador:

  1. `/puntuar`  la API publica. Cualquiera puede mandar la historia de una tarjeta y
     recibir los puntajes de A, B y C mas la decision operativa. Es el contrato que
     hace que el modelo este "en produccion".
  2. `/casos/*`  la cola de revision del analista, con los casos preparados de la
     particion de prueba. La verdad de cada caso la retiene el servidor hasta que el
     analista emite su veredicto: asi la revelacion es honesta y no una animacion.
  3. `/evidencias` y `/registros`  las tablas del informe y el registro de uso real.

Arranque:  uvicorn api.app:app --reload
"""
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DIR_DEMO, MIN_EVENTOS_PREVIOS                      # noqa: E402
from src.puntuador import obtener_puntuador                               # noqa: E402
from src.registro import obtener_registro                                 # noqa: E402

ESTATICO = Path(__file__).resolve().parent / "estatico"

ESTADO = {}


# ---------- modelos de peticion ----------

class Transaccion(BaseModel):
    timestamp: str
    monto: float
    id_comercio: int
    categoria: str
    canal: str
    extranjero: int = 0


class PeticionPuntuar(BaseModel):
    historia: list[Transaccion] = Field(
        ..., description=f"Historia de UNA tarjeta ordenada por timestamp. "
                         f"Minimo {MIN_EVENTOS_PREVIOS + 1} transacciones.")
    permutar: int | None = Field(
        None, description="Semilla: baraja la historia antes de puntuar (prueba de permutacion).")
    k_visible: int | None = Field(
        None, description="Deja visibles solo los ultimos k eventos (prueba de recorte).")


class PeticionExperimento(BaseModel):
    tipo: str = Field(..., description="'permutar' | 'recortar' | 'normal'")
    semilla: int = 1
    k: int = 20


class PeticionVeredicto(BaseModel):
    veredicto: str = Field(..., description="'revisar' | 'descartar'")


# ---------- arranque ----------

def arrancar():
    """Carga artefactos, banco y registro una sola vez, al levantar el proceso.

    Cargar `modelo_a.joblib` cuesta ~0.3 s y puntuar los 85 casos de la cola otros
    ~4 s; hacerlo por peticion dominaria la latencia. Se paga una vez.
    """
    t0 = time.perf_counter()
    ESTADO["puntuador"] = obtener_puntuador()
    ESTADO["registro"] = obtener_registro()

    ruta_banco = DIR_DEMO / "banco_casos.json"
    ruta_metricas = DIR_DEMO / "metricas.json"
    if not ruta_banco.exists():
        raise RuntimeError(
            f"falta {ruta_banco}. Generarlo con: python scripts/construir_banco.py")
    ESTADO["banco"] = json.loads(ruta_banco.read_text(encoding="utf-8"))
    ESTADO["evidencias"] = (json.loads(ruta_metricas.read_text(encoding="utf-8"))
                            if ruta_metricas.exists() else {})
    ESTADO["casos"] = {c["id"]: c for c in ESTADO["banco"]["casos"]}

    # Se puntua cada caso con el servicio EN VIVO y se compara con el puntaje que
    # produjo el pipeline por lotes. Es la prueba de paridad corriendo en produccion:
    # si el servicio se desviara del notebook, `/salud` lo diria.
    p = ESTADO["puntuador"]
    vivo, desvio = {}, 0.0
    for cid, caso in ESTADO["casos"].items():
        r = p.puntuar([t for t in caso["historia"]])
        vivo[cid] = r
        desvio = max(desvio, max(abs(r["puntajes"][m] - caso["referencia"][m])
                                 for m in ("A", "B", "C")))
    ESTADO["vivo"] = vivo
    ESTADO["desvio_maximo"] = desvio
    ESTADO["arranque_s"] = round(time.perf_counter() - t0, 2)
    ESTADO["revelados"] = set()
    print(f"[centinela] {len(vivo)} casos puntuados en vivo | "
          f"desvio maximo contra el lote: {desvio:.2e} | arranque {ESTADO['arranque_s']} s")


@asynccontextmanager
async def ciclo_de_vida(app):
    arrancar()
    yield


app = FastAPI(
    title="Centinela",
    description="Monitoreo transaccional — modelo del Proyecto 1 en produccion",
    version="1.0.0",
    lifespan=ciclo_de_vida,
)


# ---------- API publica ----------

@app.get("/salud")
def salud():
    """Ficha del modelo servido y estado del servicio."""
    return {
        "estado": "ok",
        "servicio": "Centinela 1.0.0",
        "modelo": ESTADO["puntuador"].ficha(),
        "casos_en_cola": len(ESTADO["casos"]),
        "paridad_con_el_notebook": {
            "desvio_maximo_absoluto": ESTADO["desvio_maximo"],
            "interpretacion": ("el servicio reproduce el pipeline por lotes del "
                               "notebook dentro de la precision de float32"),
        },
        "arranque_s": ESTADO["arranque_s"],
    }


@app.post("/puntuar")
def puntuar(peticion: PeticionPuntuar):
    """Puntua la ULTIMA transaccion de la historia recibida con los tres modelos."""
    historia = [t.model_dump() for t in peticion.historia]
    try:
        resultado = ESTADO["puntuador"].puntuar(
            historia, semilla_permutacion=peticion.permutar, k_visible=peticion.k_visible)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    variante = ("permutada" if peticion.permutar is not None
                else "recortada" if peticion.k_visible is not None else "normal")
    ESTADO["registro"].anotar_peticion("/puntuar", resultado, n_eventos=len(historia),
                                       variante=variante, origen="api")
    return resultado


@app.get("/evidencias")
def evidencias():
    """Las tablas del informe, precalculadas sobre validacion y prueba."""
    return ESTADO["evidencias"]


@app.get("/registros")
def registros(limite: int = 50):
    """Registro de uso: resumen agregado y las ultimas peticiones."""
    return {"resumen": ESTADO["registro"].resumen(),
            "ultimas": ESTADO["registro"].ultimas(limite)}


# ---------- cola de revision ----------

def _vista_cola(cid):
    """Lo que el navegador puede ver ANTES del veredicto: ni mecanismo ni etiqueta."""
    caso = ESTADO["casos"][cid]
    r = ESTADO["vivo"][cid]
    return {
        "id": cid,
        "tarjeta": caso["tarjeta"],
        "ts": caso["ts"],
        "monto": caso["monto"],
        "n_eventos": caso["n_eventos"],
        "puntajes": r["puntajes"],
        "decision": r["decision"],
        "alertas": r["alertas"],
        "revelado": cid in ESTADO["revelados"],
        "verdad": caso["verdad"] if cid in ESTADO["revelados"] else None,
    }


@app.get("/casos")
def casos(orden: str = "C"):
    """Cola de revision, priorizada por el puntaje del modelo que prioriza (C)."""
    if orden not in ("A", "B", "C"):
        raise HTTPException(status_code=422, detail="orden debe ser A, B o C")
    cola = [_vista_cola(cid) for cid in ESTADO["casos"]]
    cola.sort(key=lambda c: c["puntajes"][orden], reverse=True)
    return {"orden": orden, "umbrales": ESTADO["puntuador"].umbrales,
            "total": len(cola), "casos": cola}


@app.get("/casos/{cid}")
def caso(cid: str):
    """Detalle de un caso: historia, puntajes y trayectoria evento por evento."""
    if cid not in ESTADO["casos"]:
        raise HTTPException(status_code=404, detail=f"no existe el caso {cid}")
    caso = ESTADO["casos"][cid]
    trayectoria = ESTADO["puntuador"].trayectoria(caso["historia"])
    ESTADO["registro"].anotar_peticion("/casos/{id}", ESTADO["vivo"][cid], caso=cid,
                                       n_eventos=caso["n_eventos"], origen="consola")
    return {**_vista_cola(cid), "historia": caso["historia"], "trayectoria": trayectoria}


@app.post("/casos/{cid}/experimento")
def experimento(cid: str, peticion: PeticionExperimento):
    """Re-puntua el caso con la historia barajada o recortada.

    Es la prueba de falsificacion del informe, en vivo y sobre un caso concreto. El
    modelo A no se mueve —lee agregadas de ventana, que no dependen del orden— y esa
    inmovilidad es parte de la evidencia, no un error de la consola.
    """
    if cid not in ESTADO["casos"]:
        raise HTTPException(status_code=404, detail=f"no existe el caso {cid}")
    historia = ESTADO["casos"][cid]["historia"]
    kw = {}
    if peticion.tipo == "permutar":
        kw["semilla_permutacion"] = peticion.semilla
    elif peticion.tipo == "recortar":
        kw["k_visible"] = peticion.k
    elif peticion.tipo != "normal":
        raise HTTPException(status_code=422,
                            detail="tipo debe ser 'permutar', 'recortar' o 'normal'")

    resultado = ESTADO["puntuador"].puntuar(historia, **kw)
    ESTADO["registro"].anotar_peticion("/casos/{id}/experimento", resultado, caso=cid,
                                       n_eventos=len(historia),
                                       variante=peticion.tipo, origen="consola")
    return {"caso": cid, "tipo": peticion.tipo,
            "base": ESTADO["vivo"][cid]["puntajes"], **resultado}


@app.post("/casos/{cid}/veredicto")
def veredicto(cid: str, peticion: PeticionVeredicto):
    """El analista decide, y RECIEN AHI el servidor revela la verdad del caso."""
    if cid not in ESTADO["casos"]:
        raise HTTPException(status_code=404, detail=f"no existe el caso {cid}")
    if peticion.veredicto not in ("revisar", "descartar"):
        raise HTTPException(status_code=422,
                            detail="veredicto debe ser 'revisar' o 'descartar'")

    caso = ESTADO["casos"][cid]
    verdad = caso["verdad"]
    acierto = (peticion.veredicto == "revisar") == bool(verdad["es_fraude"])
    ESTADO["revelados"].add(cid)
    ESTADO["registro"].anotar_revision(cid, peticion.veredicto,
                                       ESTADO["vivo"][cid]["puntajes"]["C"],
                                       acierto, verdad["mecanismo"])
    return {
        "caso": cid,
        "veredicto": peticion.veredicto,
        "verdad": verdad,
        "acierto": acierto,
        "modelo": {
            "alertaria": ESTADO["vivo"][cid]["decision"]["alertar"],
            "acierto_modelo": (ESTADO["vivo"][cid]["decision"]["alertar"]
                               == bool(verdad["es_fraude"])),
            "puntajes": ESTADO["vivo"][cid]["puntajes"],
        },
    }


@app.get("/marcador")
def marcador():
    """Aciertos del analista contra aciertos del modelo, sobre los casos revelados."""
    revelados = sorted(ESTADO["revelados"])
    filas = []
    for cid in revelados:
        caso = ESTADO["casos"][cid]
        filas.append({"caso": cid, "mecanismo": caso["verdad"]["mecanismo"],
                      "es_fraude": caso["verdad"]["es_fraude"],
                      "alertaria": ESTADO["vivo"][cid]["decision"]["alertar"]})
    aciertos_modelo = sum(f["alertaria"] == bool(f["es_fraude"]) for f in filas)
    resumen = ESTADO["registro"].resumen()
    humano = {r["veredicto"]: r for r in resumen["revisiones"]}
    revisados = sum(r["n"] for r in resumen["revisiones"]) or 0
    aciertos_humano = sum((r["aciertos"] or 0) for r in resumen["revisiones"])
    return {"revelados": len(filas), "modelo_aciertos": aciertos_modelo,
            "analista_revisiones": revisados, "analista_aciertos": aciertos_humano,
            "detalle": filas, "por_veredicto": humano}


# ---------- consola ----------

@app.get("/")
def consola():
    return FileResponse(ESTATICO / "index.html")


app.mount("/estatico", StaticFiles(directory=ESTATICO), name="estatico")
