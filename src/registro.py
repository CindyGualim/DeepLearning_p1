"""Registro de uso del servicio en SQLite.

Es un entregable, no instrumentacion opcional: la rubrica pide "registros de uso" como
evidencia del pase a produccion. Cada puntuacion y cada decision de analista queda
escrita, y `/registros` las devuelve para que el comite las vea sin entrar al servidor.

Se usa SQLite y no un archivo de texto porque hace falta agregar (conteos por modelo,
latencia mediana, desacuerdos) y porque no agrega dependencias: viene con Python.
"""
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import RAIZ

RUTA_POR_DEFECTO = RAIZ / "datos_demo" / "registro.db"

ESQUEMA = """
CREATE TABLE IF NOT EXISTS peticiones (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    ruta          TEXT    NOT NULL,
    caso          TEXT,
    n_eventos     INTEGER,
    eventos_vistos INTEGER,
    variante      TEXT,
    puntaje_a     REAL,
    puntaje_b     REAL,
    puntaje_c     REAL,
    alerta        INTEGER,
    latencia_ms   REAL,
    origen        TEXT
);
CREATE TABLE IF NOT EXISTS revisiones (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    caso       TEXT NOT NULL,
    veredicto  TEXT NOT NULL,
    puntaje_c  REAL,
    acierto    INTEGER,
    mecanismo  TEXT
);
CREATE INDEX IF NOT EXISTS ix_peticiones_ts ON peticiones(ts);
CREATE INDEX IF NOT EXISTS ix_revisiones_caso ON revisiones(caso);
"""


def _ahora():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Registro:
    """Escrituras serializadas con un lock: uvicorn puede atender en varios hilos y
    una conexion SQLite no es segura entre hilos sin `check_same_thread=False` + lock.
    """

    def __init__(self, ruta=RUTA_POR_DEFECTO):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.con = sqlite3.connect(self.ruta, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        with self._lock:
            self.con.executescript(ESQUEMA)
            self.con.commit()

    def anotar_peticion(self, ruta, resultado, caso=None, n_eventos=None,
                        variante="normal", origen="api"):
        p = resultado.get("puntajes", {})
        with self._lock:
            self.con.execute(
                "INSERT INTO peticiones (ts, ruta, caso, n_eventos, eventos_vistos, "
                "variante, puntaje_a, puntaje_b, puntaje_c, alerta, latencia_ms, origen) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (_ahora(), ruta, caso, n_eventos, resultado.get("eventos_visibles"),
                 variante, p.get("A"), p.get("B"), p.get("C"),
                 int(bool(resultado.get("decision", {}).get("alertar"))),
                 resultado.get("latencia_ms"), origen))
            self.con.commit()

    def anotar_revision(self, caso, veredicto, puntaje_c, acierto, mecanismo):
        with self._lock:
            self.con.execute(
                "INSERT INTO revisiones (ts, caso, veredicto, puntaje_c, acierto, mecanismo) "
                "VALUES (?,?,?,?,?,?)",
                (_ahora(), caso, veredicto, puntaje_c,
                 None if acierto is None else int(acierto), mecanismo))
            self.con.commit()

    # ---- lectura ----

    def resumen(self):
        """Lo que pinta la pestana de monitoreo."""
        with self._lock:
            fila = self.con.execute(
                "SELECT COUNT(*) n, AVG(latencia_ms) lat_media, "
                "MIN(ts) desde, MAX(ts) hasta, SUM(alerta) alertas FROM peticiones"
            ).fetchone()
            por_variante = self.con.execute(
                "SELECT variante, COUNT(*) n FROM peticiones GROUP BY variante ORDER BY n DESC"
            ).fetchall()
            revisiones = self.con.execute(
                "SELECT veredicto, COUNT(*) n, SUM(acierto) aciertos "
                "FROM revisiones GROUP BY veredicto"
            ).fetchall()
            latencias = [r[0] for r in self.con.execute(
                "SELECT latencia_ms FROM peticiones WHERE latencia_ms IS NOT NULL "
                "ORDER BY latencia_ms").fetchall()]

        def percentil(xs, q):
            if not xs:
                return None
            return round(xs[min(int(q * len(xs)), len(xs) - 1)], 2)

        return {
            "peticiones": fila["n"] or 0,
            "alertas": fila["alertas"] or 0,
            "desde": fila["desde"],
            "hasta": fila["hasta"],
            "latencia_ms": {"media": round(fila["lat_media"], 2) if fila["lat_media"] else None,
                            "p50": percentil(latencias, 0.50),
                            "p95": percentil(latencias, 0.95)},
            "por_variante": [dict(r) for r in por_variante],
            "revisiones": [dict(r) for r in revisiones],
        }

    def ultimas(self, limite=50):
        with self._lock:
            filas = self.con.execute(
                "SELECT ts, ruta, caso, variante, puntaje_a, puntaje_b, puntaje_c, "
                "alerta, latencia_ms, origen FROM peticiones ORDER BY id DESC LIMIT ?",
                (int(limite),)).fetchall()
        return [dict(f) for f in filas]

    def exportar_jsonl(self, destino):
        """Volcado plano para adjuntar como evidencia en el repositorio."""
        destino = Path(destino)
        with self._lock:
            filas = self.con.execute("SELECT * FROM peticiones ORDER BY id").fetchall()
        with destino.open("w", encoding="utf-8") as fh:
            for f in filas:
                fh.write(json.dumps(dict(f), ensure_ascii=False) + "\n")
        return len(filas)


_REGISTRO = None


def obtener_registro():
    global _REGISTRO
    if _REGISTRO is None:
        _REGISTRO = Registro()
    return _REGISTRO
