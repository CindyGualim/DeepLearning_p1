"""Exporta el registro de uso a un archivo plano que se versiona como evidencia.

`datos_demo/registro.db` es estado de ejecucion y no se versiona (cambia en cada
arranque). Lo que si se entrega es el volcado: `datos_demo/registro_uso.jsonl` mas un
resumen legible. Asi el comite ve el uso real del servicio sin tener que levantarlo.

Uso:  python scripts/exportar_registros.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DIR_DEMO                 # noqa: E402
from src.registro import Registro               # noqa: E402


def exportar():
    ruta_db = DIR_DEMO / "registro.db"
    if not ruta_db.exists():
        raise SystemExit(f"no existe {ruta_db}: el servicio todavia no se ha usado.")

    reg = Registro(ruta_db)
    destino = DIR_DEMO / "registro_uso.jsonl"
    n = reg.exportar_jsonl(destino)

    resumen = reg.resumen()
    (DIR_DEMO / "registro_resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{destino.name}: {n} peticiones ({destino.stat().st_size/1e3:.1f} KB)")
    print(f"  ventana      : {resumen['desde']}  ->  {resumen['hasta']}")
    print(f"  alertas       : {resumen['alertas']}")
    print(f"  latencia p50  : {resumen['latencia_ms']['p50']} ms")
    print(f"  latencia p95  : {resumen['latencia_ms']['p95']} ms")
    print(f"  por variante  : {resumen['por_variante']}")
    print(f"  revisiones    : {resumen['revisiones']}")
    return resumen


if __name__ == "__main__":
    exportar()
