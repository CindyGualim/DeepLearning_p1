"""Construye el banco de casos y las metricas que sirve la consola Centinela.

Corre fuera de linea (no en el arranque del servicio) porque regenerar 324,603
transacciones y puntuar tres modelos sobre todo el conjunto toma ~40 s. El servicio
solo lee los dos JSON que este script escribe, asi que arranca en menos de 2 s.

Produce:
  datos_demo/banco_casos.json  casos de la particion de PRUEBA, con la historia
                               completa de la tarjeta y la verdad oculta
  datos_demo/metricas.json     evidencias precalculadas: AUC-PR por particion y por
                               mecanismo, prueba de permutacion, curva de recorte,
                               tabla de costo y sensibilidad

Y verifica la paridad con el notebook: si algun AUC-PR se desvia de lo publicado en el
informe, el script FALLA en lugar de escribir un banco silenciosamente incorrecto.

Uso:  python scripts/construir_banco.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.caracteristicas import agregadas, construir_secuencias            # noqa: E402
from src.config import (COSTO_FN, COSTO_FP, DIR_DEMO, LARGO_SECUENCIA,     # noqa: E402
                        MECANISMOS_LEGITIMOS, MIN_EVENTOS_PREVIOS)
from src.generador import generar, hash_df                                 # noqa: E402
from src.puntuador import Puntuador, permutar_orden, recortar              # noqa: E402

# Valores publicados en el informe del Proyecto 1. Son el contrato de paridad.
ESPERADO = {("val", "A"): 0.9171, ("val", "B"): 0.9226, ("val", "C"): 0.9424,
            ("test", "A"): 0.9092, ("test", "B"): 0.9206, ("test", "C"): 0.9361}
TOLERANCIA = 5e-4

# Composicion de la cola que vera el comite. Se fija aqui, no al azar en el navegador,
# para que la demostracion sea la misma en los dos dias de presentacion.
CUPOS = {"F1_escalada": 12, "F2_rafaga": 10, "F3_atipico": 8,
         "sesion_legitima": 15, "legitimo": 35}
CUPO_SONDA_TEMPRANA = 5
LARGOS_RECORTE = [1, 2, 3, 4, 6, 8, 12, 16, 20]
SEMILLAS_PERMUTACION = range(1, 6)


def auc_pr(y, s):
    return float(average_precision_score(y, s))


def metricas_umbral(y, s, umbral):
    alerta = s >= umbral
    vp = int((alerta & (y == 1)).sum())
    fp = int((alerta & (y == 0)).sum())
    fn = int((~alerta & (y == 1)).sum())
    precision = vp / (vp + fp) if vp + fp else 0.0
    exhaustividad = vp / (vp + fn) if vp + fn else 0.0
    f1 = (2 * precision * exhaustividad / (precision + exhaustividad)
          if precision + exhaustividad else 0.0)
    return {"umbral": round(float(umbral), 4), "precision": round(precision, 4),
            "exhaustividad": round(exhaustividad, 4), "f1": round(f1, 4),
            "alertas": int(alerta.sum()), "vp": vp, "fp": fp, "fn": fn,
            "costo_Q": COSTO_FN * fn + COSTO_FP * fp}


def umbral_optimo_costo(y, puntaje, costo_fn=COSTO_FN, costo_fp=COSTO_FP):
    """Umbral que minimiza el costo esperado. Se calcula SIEMPRE sobre validacion."""
    orden = np.argsort(-puntaje, kind="mergesort")
    y_ord = y[orden].astype(np.int64)
    vp = np.cumsum(y_ord)
    fp = np.cumsum(1 - y_ord)
    positivos = int(y.sum())
    costo = costo_fn * (positivos - vp) + costo_fp * fp
    return float(puntaje[orden][int(costo.argmin())])


def por_mecanismo(y, puntajes, mecanismo):
    """Cada mecanismo se enfrenta a TODOS los negativos. Las filas se comparan entre
    modelos, no entre mecanismos: la prevalencia cambia y el AUC-PR depende de ella."""
    negativos = np.isin(mecanismo, MECANISMOS_LEGITIMOS)
    filas = []
    for mec in sorted(m for m in pd.unique(mecanismo) if m not in MECANISMOS_LEGITIMOS):
        sel = negativos | (mecanismo == mec)
        fila = {"mecanismo": mec, "positivos": int((mecanismo == mec).sum())}
        for nombre, s in puntajes.items():
            fila[nombre] = round(auc_pr(y[sel], s[sel]), 4)
        filas.append(fila)
    return filas


def construir():
    print("1/6  generando datos...", flush=True)
    tx = generar()
    datos = construir_secuencias(tx)
    ag = agregadas(tx)
    X = ag.to_numpy()
    print(f"     {len(tx):,} transacciones | {datos.n:,} secuencias | huella {hash_df(tx)}")

    print("2/6  cargando artefactos y puntuando...", flush=True)
    p = Puntuador()
    num_z = p.esc_eventos.transform(datos.num).astype(np.float32)
    agg_z = p.esc_agregadas.transform(X)[:, :18].astype(np.float32)

    puntajes, indices = {}, {}
    for part in ("val", "test"):
        sel = datos.sel(part)
        indices[part] = sel
        puntajes[part] = p.puntuar_matrices(num_z, datos.cat, X, agg_z,
                                            datos.idx[sel], datos.mascara[sel],
                                            datos.fila_tx[sel])

    print("3/6  verificando paridad con el notebook...", flush=True)
    desvios = []
    for (part, modelo), esperado in ESPERADO.items():
        obtenido = auc_pr(datos.y[indices[part]], puntajes[part][modelo])
        marca = "ok " if abs(obtenido - esperado) <= TOLERANCIA else "MAL"
        print(f"     [{marca}] {part:4s} {modelo}  esperado {esperado:.4f}  "
              f"obtenido {obtenido:.4f}")
        if marca == "MAL":
            desvios.append((part, modelo, esperado, obtenido))
    if desvios:
        raise SystemExit(
            "\nPARIDAD ROTA. El pipeline de src/ ya no reproduce el notebook:\n  "
            + "\n  ".join(f"{p_} {m}: esperado {e:.4f}, obtenido {o:.4f}"
                          for p_, m, e, o in desvios)
            + "\nNo se escribio el banco. Revisar src/caracteristicas.py.")

    print("4/6  calculando evidencias...", flush=True)
    evidencias = calcular_evidencias(p, datos, num_z, agg_z, X, puntajes, indices)

    print("5/6  curando el banco de casos...", flush=True)
    casos = curar_casos(tx, datos, puntajes["test"], indices["test"], p)

    print("6/6  escribiendo...", flush=True)
    DIR_DEMO.mkdir(parents=True, exist_ok=True)
    ruta_casos = DIR_DEMO / "banco_casos.json"
    ruta_metricas = DIR_DEMO / "metricas.json"
    ruta_casos.write_text(json.dumps(casos, ensure_ascii=False), encoding="utf-8")
    ruta_metricas.write_text(json.dumps(evidencias, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    print(f"     {ruta_casos.name}: {len(casos['casos'])} casos, "
          f"{ruta_casos.stat().st_size/1e3:.0f} KB")
    print(f"     {ruta_metricas.name}: {ruta_metricas.stat().st_size/1e3:.0f} KB")
    return casos, evidencias


def calcular_evidencias(p, datos, num_z, agg_z, X, puntajes, indices):
    """Las cinco tablas que la consola muestra en la pestana de evidencias."""
    ev = {"generado": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
          "costos": {"FN": COSTO_FN, "FP": COSTO_FP}}

    # --- AUC-PR y metricas en el umbral, por particion ---
    ev["particiones"] = {}
    for part in ("val", "test"):
        sel = indices[part]
        y = datos.y[sel]
        ev["particiones"][part] = {
            "secuencias": int(sel.size), "fraudes": int(y.sum()),
            "prevalencia": round(float(y.mean()), 5),
            "modelos": {m: {"auc_pr": round(auc_pr(y, s), 4),
                            **metricas_umbral(y, s, p.umbrales[m])}
                        for m, s in puntajes[part].items()},
            "por_mecanismo": por_mecanismo(y, puntajes[part], datos.mecanismo[sel]),
        }

    # --- prueba de permutacion sobre PRUEBA (celda 77) ---
    sel = indices["test"]
    y = datos.y[sel]
    base = auc_pr(y, puntajes["test"]["B"])
    filas = [{"variante": "orden real", "auc_pr": round(base, 4), "desviacion": 0.0,
              "caida": 0.0, "caida_pct": 0.0}]
    perm_por_mecanismo = None
    for etiqueta, fijar in (("historia barajada", True), ("secuencia completa", False)):
        aucs, ultima = [], None
        for semilla in SEMILLAS_PERMUTACION:
            ip = permutar_orden(datos.idx[sel], datos.mascara[sel], semilla,
                                fijar_objetivo=fijar)
            s = p.puntuar_matrices(num_z, datos.cat, X, agg_z, ip,
                                   datos.mascara[sel], datos.fila_tx[sel])["B"]
            aucs.append(auc_pr(y, s))
            ultima = s
        filas.append({"variante": etiqueta, "auc_pr": round(float(np.mean(aucs)), 4),
                      "desviacion": round(float(np.std(aucs)), 4),
                      "caida": round(base - float(np.mean(aucs)), 4),
                      "caida_pct": round((base - float(np.mean(aucs))) / base * 100, 1)})
        if fijar:
            perm_por_mecanismo = por_mecanismo(
                y, {"B_real": puntajes["test"]["B"], "B_permutado": ultima},
                datos.mecanismo[sel])
    ev["permutacion"] = {"tabla": filas, "por_mecanismo": perm_por_mecanismo,
                         "semillas": len(list(SEMILLAS_PERMUTACION))}

    # --- curva de recorte (celda 85) ---
    curva = []
    for k in LARGOS_RECORTE:
        m = recortar(datos.mascara[sel], k)
        s = p.puntuar_matrices(num_z, datos.cat, X, agg_z, datos.idx[sel], m,
                               datos.fila_tx[sel])["B"]
        fila = {"k": k, "auc_pr": round(auc_pr(y, s), 4)}
        for d in por_mecanismo(y, {"b": s}, datos.mecanismo[sel]):
            fila[d["mecanismo"]] = d["b"]
        curva.append(fila)
    ev["recorte"] = curva

    # --- sensibilidad al costo del falso negativo (celda 110) ---
    # El umbral se REOPTIMIZA sobre validacion para cada costo y recien ahi se aplica a
    # prueba. Dejarlo fijo mediria otra cosa: como envejece un umbral, no que modelo
    # conviene si el costo del fraude fuera distinto.
    y_val = datos.y[indices["val"]]
    sens = []
    for costo_fn in (1500, 2500, 4200, 6000, 9000):
        fila = {"costo_fn": costo_fn}
        for m in ("A", "B", "C"):
            u = umbral_optimo_costo(y_val, puntajes["val"][m], costo_fn=costo_fn)
            s = puntajes["test"][m]
            alerta = s >= u
            fn = int((~alerta & (y == 1)).sum())
            fp = int((alerta & (y == 0)).sum())
            fila[m] = costo_fn * fn + COSTO_FP * fp
            fila[f"umbral_{m}"] = round(float(u), 4)
        fila["mejor"] = min("ABC", key=lambda m: fila[m])
        sens.append(fila)
    ev["sensibilidad"] = sens
    return ev


def _historia_de(tx, filas):
    """Serializa las transacciones `filas` como la historia que consume `/puntuar`."""
    sub = tx.iloc[filas]
    return [{"timestamp": t.isoformat(sep=" "), "monto": round(float(m), 2),
             "id_comercio": int(c), "categoria": str(cat), "canal": str(can),
             "extranjero": int(ext)}
            for t, m, c, cat, can, ext in zip(
                sub.timestamp, sub.monto, sub.id_comercio, sub.categoria,
                sub.canal, sub.extranjero)]


def curar_casos(tx, datos, puntajes_test, sel_test, p):
    """Elige los casos de la cola con cupos fijos por mecanismo.

    Todos salen de la particion de PRUEBA: el comite revisa casos que ningun modelo vio
    durante el entrenamiento ni durante la eleccion del umbral. Dentro de cada
    mecanismo se toman los de puntaje mas alto segun C junto con una muestra del resto,
    para que la cola tenga casos faciles y casos dudosos y no solo los obvios.
    """
    rng = np.random.default_rng(20261108)
    tarjeta_tx = tx.id_tarjeta.to_numpy()
    fila_tx = datos.fila_tx[sel_test]
    mecanismo = datos.mecanismo[sel_test]
    y = datos.y[sel_test]
    s_c = puntajes_test["C"]

    # Primera sonda de un episodio de escalada: etiquetada fraude pero sin historia que
    # la delate. Es el error irreducible declarado en el informe.
    id_ep = datos.id_episodio[sel_test]
    primera_del_episodio = np.zeros(sel_test.size, bool)
    for ep in np.unique(id_ep[(mecanismo == "F1_escalada")]):
        en_ep = np.flatnonzero((id_ep == ep) & (mecanismo == "F1_escalada"))
        if en_ep.size:
            primera_del_episodio[en_ep[np.argmin(fila_tx[en_ep])]] = True

    elegidos = []
    for mec, cupo in CUPOS.items():
        cand = np.flatnonzero((mecanismo == mec) & ~primera_del_episodio)
        if cand.size == 0:
            continue
        orden = cand[np.argsort(-s_c[cand])]
        mitad = min(cupo // 2, orden.size)
        top = orden[:mitad]
        resto = np.setdiff1d(orden, top)
        extra = rng.choice(resto, min(cupo - mitad, resto.size), replace=False) if resto.size else []
        elegidos.extend(np.concatenate([top, extra]).astype(int).tolist())

    sondas = np.flatnonzero(primera_del_episodio)
    if sondas.size:
        elegidos.extend(rng.choice(sondas, min(CUPO_SONDA_TEMPRANA, sondas.size),
                                   replace=False).astype(int).tolist())

    # Se barajan para que el orden de la cola no delate el mecanismo.
    elegidos = list(dict.fromkeys(elegidos))
    rng.shuffle(elegidos)

    casos = []
    for n, i in enumerate(elegidos, start=1):
        objetivo = int(fila_tx[i])
        tarjeta = int(tarjeta_tx[objetivo])
        # Historia COMPLETA de la tarjeta hasta el objetivo: las agregadas usan
        # ventanas de 7 dias y estadisticos sobre toda la historia, asi que recortarla
        # cambiaria el puntaje. Es tambien el contrato real del servicio.
        filas = np.flatnonzero(tarjeta_tx[:objetivo + 1] == tarjeta)
        historia = _historia_de(tx, filas)
        casos.append({
            "id": f"C{n:03d}",
            "tarjeta": f"T{tarjeta:05d}",
            "historia": historia,
            "n_eventos": len(historia),
            "monto": historia[-1]["monto"],
            "ts": historia[-1]["timestamp"],
            "referencia": {m: round(float(puntajes_test[m][i]), 6) for m in ("A", "B", "C")},
            "verdad": {
                "es_fraude": int(y[i]),
                "mecanismo": str(mecanismo[i]),
                "sonda_temprana": bool(primera_del_episodio[i]),
            },
        })

    return {
        "generado": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
        "particion": "test",
        "nota": ("Los campos `verdad` y `referencia` NO se envian al navegador hasta "
                 "que el analista emite su veredicto: la API los retiene."),
        "umbrales": {m: round(v, 4) for m, v in p.umbrales.items()},
        "casos": casos,
    }


if __name__ == "__main__":
    construir()
    print("\nlisto.")
