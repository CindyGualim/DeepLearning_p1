/* Centinela — consola de monitoreo transaccional.
   Sin dependencias ni paso de compilacion: el servicio se sirve con `uvicorn` y
   nada mas. Las graficas son SVG construido a mano, por la misma razon. */

"use strict";

const ESTADO = {
  casos: [], caso: null, orden: "C", umbrales: {}, evidencias: null,
  experimento: null, ficha: null,
};

const SERIES = { A: "var(--serie-a)", B: "var(--serie-b)", C: "var(--serie-c)" };
const NOMBRES = {
  A: "A · agregadas, no ve el orden",
  B: "B · GRU sobre la secuencia",
  C: "C · híbrido (secuencia + agregadas)",
};
const MECANISMOS = {
  F1_escalada: "Escalada — sondas crecientes y golpe final (depende del orden)",
  F2_rafaga: "Ráfaga — muchos cargos en pocos minutos",
  F3_atipico: "Atípico aislado — un solo cargo fuera del patrón (NO depende del orden)",
  sesion_legitima: "Sesión legítima — mismos montos que una escalada, en otro orden",
  legitimo: "Tráfico normal",
};

const $ = (s, r = document) => r.querySelector(s);
const el = (t, props = {}, hijos = []) => {
  const n = document.createElement(t);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  for (const h of [].concat(hijos)) if (h) n.append(h);
  return n;
};
const q = (x) => "Q" + Math.round(x).toLocaleString("es-GT");
const f4 = (x) => (x === null || x === undefined) ? "—" : Number(x).toFixed(4);
const f3 = (x) => (x === null || x === undefined) ? "—" : Number(x).toFixed(3);

async function api(ruta, opciones) {
  const r = await fetch(ruta, opciones);
  if (!r.ok) {
    let d; try { d = (await r.json()).detail; } catch { d = r.statusText; }
    throw new Error(typeof d === "string" ? d : JSON.stringify(d));
  }
  return r.json();
}

/* ============================ grafica de lineas ============================ */

const NS = "http://www.w3.org/2000/svg";
const svgEl = (t, a = {}) => {
  const n = document.createElementNS(NS, t);
  for (const [k, v] of Object.entries(a)) if (v !== null && v !== undefined) n.setAttribute(k, v);
  return n;
};

/**
 * Grafica de lineas con crucero y pista al pasar el cursor.
 * series: [{clave, nombre, color, puntos:[y...]}], x: [valores]
 * reglas: [{y, texto, color}] para los umbrales
 */
function graficaLineas(contenedor, { x, series, reglas = [], etiquetaX, etiquetaY,
                                     yMin = 0, yMax = 1, alto = 210, formatoY = f3 }) {
  contenedor.innerHTML = "";
  const env = el("div", { class: "envoltura-grafica" });
  const W = 720, H = alto, M = { t: 10, r: 58, b: 30, l: 42 };
  const ix = (i) => M.l + (x.length === 1 ? 0 : i * (W - M.l - M.r) / (x.length - 1));
  const iy = (v) => M.t + (1 - (v - yMin) / (yMax - yMin)) * (H - M.t - M.b);

  // Sin `preserveAspectRatio="none"`: estirar el viewBox deformaria el texto de los
  // ejes. El SVG escala a lo ancho y la altura sale de la relacion de aspecto.
  const svg = svgEl("svg", { class: "grafica", viewBox: `0 0 ${W} ${H}`, role: "img" });

  // rejilla horizontal, recesiva
  const pasos = 4;
  for (let i = 0; i <= pasos; i++) {
    const v = yMin + (yMax - yMin) * i / pasos;
    svg.append(svgEl("line", { class: "rejilla", x1: M.l, x2: W - M.r, y1: iy(v), y2: iy(v) }));
    svg.append(Object.assign(svgEl("text", { x: M.l - 7, y: iy(v) + 3.5, "text-anchor": "end" }),
                             { textContent: formatoY(v) }));
  }
  svg.append(svgEl("line", { class: "eje", x1: M.l, x2: W - M.r, y1: iy(yMin), y2: iy(yMin) }));

  // reglas (umbrales)
  for (const r of reglas) {
    svg.append(svgEl("line", { class: "umbral-linea", x1: M.l, x2: W - M.r,
                               y1: iy(r.y), y2: iy(r.y), stroke: r.color }));
  }

  // etiquetas del eje x: primera, ultima y algunas intermedias
  const cada = Math.max(1, Math.ceil(x.length / 10));
  x.forEach((v, i) => {
    if (i % cada === 0 || i === x.length - 1) {
      svg.append(Object.assign(
        svgEl("text", { x: ix(i), y: H - M.b + 15, "text-anchor": "middle" }),
        { textContent: v }));
    }
  });
  if (etiquetaX) {
    svg.append(Object.assign(
      svgEl("text", { x: (M.l + W - M.r) / 2, y: H - 2, "text-anchor": "middle" }),
      { textContent: etiquetaX }));
  }

  // lineas + etiqueta directa al final de cada serie
  for (const s of series) {
    const d = s.puntos.map((v, i) => `${i ? "L" : "M"}${ix(i).toFixed(1)},${iy(v).toFixed(1)}`).join("");
    svg.append(svgEl("path", { class: "linea", d, stroke: s.color }));
    const ultimo = s.puntos.length - 1;
    svg.append(Object.assign(
      svgEl("text", { class: "etiqueta-serie", x: W - M.r + 6, y: iy(s.puntos[ultimo]) + 3.5,
                      fill: s.color }),
      { textContent: s.clave }));
  }

  // capa de interaccion
  const crucero = svgEl("line", { class: "eje", y1: M.t, y2: H - M.b, opacity: 0 });
  svg.append(crucero);
  const marcas = series.map((s) => {
    const c = svgEl("circle", { class: "punto", r: 4.5, fill: s.color, opacity: 0 });
    svg.append(c); return c;
  });

  const pista = el("div", { class: "pista" });
  pista.style.display = "none";
  env.append(svg, pista);

  const mover = (ev) => {
    const caja = svg.getBoundingClientRect();
    const px = (ev.clientX - caja.left) / caja.width * W;
    let i = Math.round((px - M.l) / ((W - M.l - M.r) / Math.max(1, x.length - 1)));
    i = Math.max(0, Math.min(x.length - 1, i));
    crucero.setAttribute("x1", ix(i)); crucero.setAttribute("x2", ix(i));
    crucero.setAttribute("opacity", .6);
    series.forEach((s, k) => {
      marcas[k].setAttribute("cx", ix(i));
      marcas[k].setAttribute("cy", iy(s.puntos[i]));
      marcas[k].setAttribute("opacity", 1);
    });
    pista.innerHTML = "";
    pista.append(el("div", { class: "t", text: `${etiquetaX || ""} ${x[i]}`.trim() }));
    for (const s of series) {
      pista.append(el("div", { class: "r", html:
        `<span style="color:${s.color}">■</span> ${s.nombre || s.clave}` +
        `<b>${formatoY(s.puntos[i])}</b>` }));
    }
    pista.style.display = "block";
    const rel = ix(i) / W * caja.width;
    pista.style.left = Math.min(caja.width - pista.offsetWidth - 4, Math.max(0, rel + 12)) + "px";
    pista.style.top = "6px";
  };
  svg.addEventListener("pointermove", mover);
  svg.addEventListener("pointerleave", () => {
    pista.style.display = "none";
    crucero.setAttribute("opacity", 0);
    marcas.forEach((m) => m.setAttribute("opacity", 0));
  });

  const leyenda = el("div", { class: "leyenda" });
  for (const s of series) {
    leyenda.append(el("span", { html: `<i style="background:${s.color}"></i>${s.nombre || s.clave}` }));
  }
  for (const r of reglas) {
    leyenda.append(el("span", { html: `<i style="background:${r.color};opacity:.6"></i>${r.texto}` }));
  }
  contenedor.append(env, leyenda);
  if (etiquetaY) contenedor.append(el("div", { class: "leyenda", text: etiquetaY }));
}

/* ================================== cola ================================== */

function filaCaso(c) {
  const sel = ESTADO.caso && ESTADO.caso.id === c.id;
  const p = c.puntajes[ESTADO.orden];
  const insignias = el("div", { class: "linea1" }, [
    el("span", { class: "cid", text: c.id }),
    el("span", { class: "monto", text: q(c.monto) }),
    c.decision.alertar ? el("span", { class: "insignia alerta", text: "alerta" })
                       : el("span", { class: "insignia limpio", text: "sin alerta" }),
    c.revelado ? el("span", {
      class: "insignia " + (c.verdad.es_fraude ? "fraude" : "legitimo"),
      text: c.verdad.es_fraude ? "fraude" : "legítimo" }) : null,
  ]);
  return el("button", {
    class: "fila-caso", role: "option", "aria-selected": sel ? "true" : "false",
    onclick: () => abrirCaso(c.id),
  }, [
    insignias,
    el("div", { class: "prio" }, [
      el("span", { class: "medidor" }, [
        el("i", { style: `width:${Math.max(2, p * 100).toFixed(1)}%` })]),
      el("span", { class: "valor", text: f3(p) }),
    ]),
    el("div", { class: "linea2", text: `${c.tarjeta} · ${c.ts} · ${c.n_eventos} eventos` }),
  ]);
}

async function cargarCola() {
  const d = await api(`/casos?orden=${ESTADO.orden}`);
  ESTADO.casos = d.casos;
  ESTADO.umbrales = d.umbrales;
  const cola = $("#cola");
  cola.innerHTML = "";
  d.casos.forEach((c) => cola.append(filaCaso(c)));
}

/* ================================= detalle ================================= */

function barraPuntaje(clave, valor, base) {
  const u = ESTADO.umbrales[clave] ?? 0;
  const hijos = [
    el("i", { style: `width:${(valor * 100).toFixed(1)}%` }),
    el("span", { class: "umbral", style: `left:${(u * 100).toFixed(1)}%`,
                 title: `umbral ${f4(u)}` }),
  ];
  if (base !== undefined && Math.abs(base - valor) > 1e-6) {
    hijos.push(el("span", { class: "fantasma", style: `left:${(base * 100).toFixed(1)}%`,
                            title: `puntaje original ${f4(base)}` }));
  }
  const d = base === undefined ? null : valor - base;
  return el("div", { class: `puntaje ${clave.toLowerCase()}` }, [
    el("span", { class: `etiqueta ${clave.toLowerCase()}`, text: clave, title: NOMBRES[clave] }),
    el("span", { class: "barra" }, hijos),
    el("span", { class: "num" }, [
      el("span", { text: f3(valor) }),
      d !== null && Math.abs(d) > 1e-6
        ? el("div", { class: "delta " + (d < 0 ? "baja" : "sube"),
                      text: (d > 0 ? "+" : "") + d.toFixed(3) })
        : null,
    ]),
  ]);
}

/**
 * Tabla de la historia, en el orden EXACTO en que la red la leyo.
 * `ordenAbs` son las posiciones que devolvio el servicio (null = orden natural);
 * `kVisible` atenua los eventos que quedaron fuera de la ventana recortada.
 */
function tablaHistoria(historia, ordenAbs, kVisible) {
  const n = historia.length;
  const inicioVentana = Math.max(0, n - 20);
  const visiblesDesde = kVisible ? Math.max(inicioVentana, n - kVisible) : inicioVentana;

  let ordenPos = null;
  if (ordenAbs && ordenAbs.length) {
    // El servicio indexa sobre la historia recibida; el desfase corrige cualquier
    // recorte para que el ultimo elemento sea siempre la transaccion objetivo.
    const desfase = ordenAbs[ordenAbs.length - 1] - (n - 1);
    ordenPos = ordenAbs.map((a) => a - desfase);
  }

  const filas = [];
  for (let i = 0; i < visiblesDesde; i++) filas.push({ i, dentro: false, movida: false });
  if (ordenPos) {
    ordenPos.forEach((p, k) => filas.push({ i: p, dentro: true, movida: p !== visiblesDesde + k }));
  } else {
    for (let i = visiblesDesde; i < n; i++) filas.push({ i, dentro: true, movida: false });
  }

  const cuerpo = el("tbody");
  for (const { i, dentro, movida } of filas) {
    const t = historia[i];
    const esObjetivo = i === n - 1;
    cuerpo.append(el("tr", {
      class: [esObjetivo ? "objetivo" : "", !dentro ? "oculta" : "", movida && !esObjetivo ? "movida" : ""]
        .filter(Boolean).join(" "),
    }, [
      el("td", { class: "txt", text: t.timestamp }),
      el("td", { text: q(t.monto) }),
      el("td", { class: "txt", text: t.categoria }),
      el("td", { class: "txt", text: t.canal }),
      el("td", { text: t.id_comercio }),
      el("td", { text: t.extranjero ? "sí" : "—" }),
    ]));
  }
  return el("div", { class: "historia" }, [
    el("table", {}, [
      el("thead", {}, [el("tr", {}, [
        el("th", { text: "fecha y hora" }), el("th", { text: "monto" }),
        el("th", { text: "categoría" }), el("th", { text: "canal" }),
        el("th", { text: "comercio" }), el("th", { text: "ext." }),
      ])]),
      cuerpo,
    ]),
  ]);
}

async function abrirCaso(cid) {
  const c = await api(`/casos/${cid}`);
  ESTADO.caso = c;
  ESTADO.experimento = null;
  await cargarCola();
  pintarDetalle();
}

function pintarDetalle() {
  const c = ESTADO.caso;
  if (!c) return;
  const exp = ESTADO.experimento;
  const puntajes = exp ? exp.puntajes : c.puntajes;
  const base = exp ? c.puntajes : undefined;
  const cont = $("#detalle");
  cont.innerHTML = "";

  /* --- puntajes y decisión --- */
  const panelPuntajes = el("div", { class: "panel" }, [
    el("h2", {}, [
      document.createTextNode(`Caso ${c.id} · ${c.tarjeta}`),
      el("span", { class: "controles-cola", text: `${q(c.monto)} · ${c.ts}` }),
    ]),
    el("div", { class: "cuerpo" }, [
      exp ? el("p", { class: "nota", html:
        exp.tipo === "permutar"
          ? `<strong>Historia barajada</strong> (semilla ${exp.semilla}). Los eventos son los mismos y
             la transacción que se puntúa sigue siendo la última: lo único que cambió es el orden.`
          : `<strong>Historia recortada</strong> a los últimos ${exp.k} eventos
             (${exp.eventos_visibles} visibles).` }) : null,
      el("div", { class: "puntajes" },
         ["A", "B", "C"].map((k) => barraPuntaje(k, puntajes[k], base && base[k]))),
      el("div", { class: "leyenda", html:
        `<span><i style="background:var(--texto-3)"></i>umbral por costo (validación)</span>` +
        (exp ? `<span><i style="background:var(--texto)"></i>puntaje con la historia real</span>` : "") }),
      el("div", { style: "margin-top:12px", class: "acciones" }, [
        el("span", { class: "insignia " + ((exp || c).decision.alertar ? "alerta" : "limpio"),
                     text: (exp || c).decision.alertar ? "A alertaría" : "A no alertaría" }),
        el("span", { class: "insignia", text: `prioridad C ${f3(puntajes.C)}` }),
        el("span", { class: "insignia", text: `${(exp || c).latencia_ms ?? "—"} ms` }),
      ]),
    ]),
  ]);

  /* --- experimentos --- */
  const deslizador = el("input", { type: "range", min: 1, max: 20,
                                   value: exp && exp.tipo === "recortar" ? exp.k : 20 });
  const salida = el("output", { text: deslizador.value });
  deslizador.addEventListener("input", () => { salida.textContent = deslizador.value; });
  deslizador.addEventListener("change", () => correrExperimento("recortar", { k: +deslizador.value }));

  const panelExp = el("div", { class: "panel" }, [
    el("h2", { text: "Pruebas de falsificación, en vivo" }),
    el("div", { class: "cuerpo" }, [
      el("p", { class: "nota", html:
        `Barajar la historia destruye el orden sin tocar los eventos. Si el modelo B de verdad usa
         la secuencia, su puntaje tiene que moverse — y el de <strong>A no debería moverse nunca</strong>,
         porque lee agregados de ventana que no dependen del orden. Esa asimetría es la evidencia.` }),
      el("div", { class: "acciones" }, [
        el("button", { class: "btn", text: "Barajar historia",
                       onclick: () => correrExperimento("permutar",
                         { semilla: Math.floor(Math.random() * 9000) + 1 }) }),
        el("button", { class: "btn", text: "Restablecer", disabled: exp ? null : "",
                       onclick: () => { ESTADO.experimento = null; pintarDetalle(); } }),
      ]),
      el("div", { class: "deslizador", style: "margin-top:12px" }, [
        el("label", { text: "eventos visibles" }), deslizador, salida,
      ]),
    ]),
  ]);

  /* --- trayectoria --- */
  const panelTray = el("div", { class: "panel" }, [
    el("h2", { text: "Puntaje evento por evento" }),
    el("div", { class: "cuerpo" }, [
      el("p", { class: "nota", html:
        `Qué habría puntuado cada modelo en <em>cada</em> transacción de esta tarjeta, no solo en la
         última. En una escalada el puntaje trepa conforme el patrón se construye.` }),
      el("div", { id: "g-trayectoria" }),
    ]),
  ]);

  /* --- historia --- */
  const panelHist = el("div", { class: "panel" }, [
    el("h2", {}, [
      document.createTextNode("Historia de la tarjeta"),
      el("span", { class: "controles-cola", text: `${c.n_eventos} transacciones` }),
    ]),
    el("div", { class: "cuerpo sin-aire" },
       [tablaHistoria(c.historia, exp && exp.tipo === "permutar" ? exp.orden_eventos : null,
                      exp && exp.tipo === "recortar" ? exp.k : null)]),
  ]);

  /* --- veredicto --- */
  const panelVer = el("div", { class: "panel" }, [
    el("h2", { text: "Veredicto del analista" }),
    el("div", { class: "cuerpo", id: "zona-veredicto" }, [
      c.revelado ? revelacion(c.verdad, null) : el("div", {}, [
        el("p", { class: "nota", html:
          `El servidor retiene el mecanismo real hasta que usted decida. Revise la historia,
           use las pruebas de arriba y después escale o descarte.` }),
        el("div", { class: "acciones" }, [
          el("button", { class: "btn peligro", text: "Escalar a revisión",
                         onclick: () => votar("revisar") }),
          el("button", { class: "btn", text: "Descartar",
                         onclick: () => votar("descartar") }),
        ]),
      ]),
    ]),
  ]);

  cont.append(panelPuntajes, panelExp, panelTray, panelHist, panelVer);
  for (const p of [panelExp, panelTray, panelHist, panelVer]) p.style.marginTop = "16px";

  const t = c.trayectoria;
  graficaLineas($("#g-trayectoria"), {
    x: t.posiciones.map((p) => p + 1),
    series: ["A", "B", "C"].map((k) => ({
      clave: k, nombre: k, color: SERIES[k], puntos: t.puntajes[k],
    })),
    reglas: [{ y: ESTADO.umbrales.A, texto: "umbral de A", color: "var(--serie-a)" }],
    etiquetaX: "transacción nº",
  });
}

async function correrExperimento(tipo, params) {
  const c = ESTADO.caso;
  const r = await api(`/casos/${c.id}/experimento`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tipo, semilla: params.semilla ?? 1, k: params.k ?? 20 }),
  });
  ESTADO.experimento = { ...r, ...params, tipo };
  pintarDetalle();
}

function revelacion(verdad, resultado) {
  const esFraude = !!verdad.es_fraude;
  const hijos = [
    el("h3", {}, [
      document.createTextNode("Caso revelado: "),
      el("span", { class: "insignia " + (esFraude ? "fraude" : "legitimo"),
                   text: esFraude ? "FRAUDE" : "LEGÍTIMO" }),
    ]),
    el("p", { text: MECANISMOS[verdad.mecanismo] || verdad.mecanismo }),
  ];
  if (verdad.sonda_temprana) {
    hijos.push(el("p", { html:
      `<strong>Es una primera sonda.</strong> Está etiquetada como fraude porque la tarjeta ya está
       comprometida, pero todavía no existe historia que la delate. El informe la declara como
       error irreducible: ningún modelo debería detectarla, y no detectarla no es una falla.` }));
  }
  if (resultado) {
    hijos.push(el("p", {}, [
      document.createTextNode("Su veredicto: "),
      el("span", { class: "insignia " + (resultado.acierto ? "ok" : "mal"),
                   text: resultado.acierto ? "acertó" : "falló" }),
      document.createTextNode("  ·  El modelo: "),
      el("span", { class: "insignia " + (resultado.modelo.acierto_modelo ? "ok" : "mal"),
                   text: resultado.modelo.acierto_modelo ? "acertó" : "falló" }),
    ]));
  }
  return el("div", { class: "revelacion" }, hijos);
}

async function votar(veredicto) {
  const c = ESTADO.caso;
  const r = await api(`/casos/${c.id}/veredicto`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ veredicto }),
  });
  const zona = $("#zona-veredicto");
  zona.innerHTML = "";
  zona.append(revelacion(r.verdad, r));
  ESTADO.caso.revelado = true;
  ESTADO.caso.verdad = r.verdad;
  await cargarCola();
}

/* =============================== evidencias =============================== */

function tabla(cabeceras, filas, opciones = {}) {
  return el("div", { class: "tabla-envoltura" }, [
    el("table", {}, [
      el("thead", {}, [el("tr", {}, cabeceras.map((h) => el("th", { text: h })))]),
      el("tbody", {}, filas.map((f, i) => el("tr", { class: opciones.destacar?.(f, i) ? "destacada" : "" },
        f.map((v, j) => el("td", { class: j === 0 ? "txt" : "", html: String(v) }))))),
    ]),
  ]);
}

async function cargarEvidencias() {
  const ev = ESTADO.evidencias ?? (ESTADO.evidencias = await api("/evidencias"));
  const cont = $("#evidencias");
  cont.innerHTML = "";

  /* permutación */
  const perm = el("div", { class: "panel" }, [
    el("h2", { text: "1 · Permutación controlada — ¿el modelo usa el orden?" }),
    el("div", { class: "cuerpo" }, [
      el("p", { class: "nota", html:
        `Se baraja la historia dentro de cada secuencia de prueba, sin cambiar sus eventos, con
         ${ev.permutacion.semillas} semillas. Si el AUC-PR no cayera, la ventaja de B no vendría del orden.` }),
      tabla(["variante", "AUC-PR", "desv.", "caída", "caída %"],
        ev.permutacion.tabla.map((r) => [r.variante, f4(r.auc_pr), f4(r.desviacion),
                                         f4(r.caida), r.caida_pct + " %"])),
      el("p", { class: "nota", style: "margin-top:14px", html:
        `<strong>El control del experimento.</strong> F3 es el mecanismo diseñado para no depender
         del orden. Es el único que no se degrada al barajar — y eso descarta que la caída sea un
         error de código: un bug habría hundido a los tres por igual.` }),
      tabla(["mecanismo", "positivos", "B con orden real", "B barajado", "caída"],
        ev.permutacion.por_mecanismo.map((r) => [
          r.mecanismo, r.positivos, f4(r.B_real), f4(r.B_permutado),
          `<span class="delta ${r.B_permutado < r.B_real ? "baja" : "sube"}">` +
          `${(r.B_permutado - r.B_real >= 0 ? "+" : "")}${(r.B_permutado - r.B_real).toFixed(4)}</span>`]),
        { destacar: (f) => f[0] === "F3_atipico" }),
    ]),
  ]);

  /* recorte */
  const rec = el("div", { class: "panel" }, [
    el("h2", { text: "2 · Recorte de la historia — ¿cuánta memoria hace falta?" }),
    el("div", { class: "cuerpo" }, [
      el("p", { class: "nota", html:
        `Se deja visible solo los últimos <em>k</em> eventos. Cada mecanismo satura donde su largo de
         episodio predice, y el máximo global no está en k=20 sino en <strong>k=6</strong>:
         hallazgo operativo, porque acortar la ventana abarata el servicio sin costar desempeño.` }),
      el("div", { id: "g-recorte" }),
    ]),
  ]);

  /* por mecanismo */
  const test = ev.particiones.test;
  const mec = el("div", { class: "panel" }, [
    el("h2", { text: "3 · Desempeño por mecanismo (prueba)" }),
    el("div", { class: "cuerpo" }, [
      el("p", { class: "nota", html:
        `Cada mecanismo se enfrenta a todos los negativos. Las filas se comparan <em>entre modelos</em>,
         no entre mecanismos: la prevalencia cambia y el AUC-PR depende de ella.` }),
      tabla(["mecanismo", "positivos", "A", "B", "C"],
        test.por_mecanismo.map((r) => [r.mecanismo, r.positivos, f4(r.A), f4(r.B), f4(r.C)])),
    ]),
  ]);

  /* costo */
  const costo = el("div", { class: "panel" }, [
    el("h2", { text: "4 · La decisión económica" }),
    el("div", { class: "cuerpo" }, [
      el("p", { class: "nota", html:
        `El umbral se eligió por costo sobre <strong>validación</strong> y la prueba se abrió una sola vez.
         Ordenar modelos por AUC-PR no equivale a ordenarlos por dinero: C tiene el mejor AUC-PR y
         es el más caro. Toda la diferencia entre el mejor y el peor son 7 fraudes sobre
         ${test.fraudes}, y en el informe los intervalos de confianza de esa diferencia cruzan el cero.` }),
      tabla(["modelo", "AUC-PR", "umbral", "VP", "FP", "FN", "costo"],
        ["A", "B", "C"].map((m) => {
          const d = test.modelos[m];
          return [m, f4(d.auc_pr), f4(d.umbral), d.vp, d.fp, d.fn, q(d.costo_Q)];
        }), { destacar: (f) => f[0] === "A" }),
      el("p", { class: "nota", style: "margin-top:14px", html:
        `<strong>Sensibilidad.</strong> El umbral se reoptimiza en validación para cada supuesto de
         costo y recién ahí se aplica a prueba. La recomendación cambia con el precio del fraude:
         si un falso negativo costara Q1,500 o Q2,500 convendría B, no A.` }),
      tabla(["costo de un FN", "A", "B", "C", "más barato"],
        ev.sensibilidad.map((r) => [q(r.costo_fn), q(r.A), q(r.B), q(r.C),
                                    `<span class="mejor">${r.mejor}</span>`])),
    ]),
  ]);

  /* particiones */
  const part = el("div", { class: "panel" }, [
    el("h2", { text: "5 · Paridad con el notebook" }),
    el("div", { class: "cuerpo" }, [
      el("p", { class: "nota", html:
        `Estos AUC-PR los produjo el código de <code>src/</code> que sirve este servicio, no el
         notebook. Coinciden con los publicados en el informe hasta el cuarto decimal, y
         <code>scripts/construir_banco.py</code> falla si dejan de coincidir.` }),
      tabla(["partición", "secuencias", "fraudes", "prevalencia", "A", "B", "C"],
        ["val", "test"].map((p) => {
          const d = ev.particiones[p];
          return [p, d.secuencias.toLocaleString("es-GT"), d.fraudes,
                  (d.prevalencia * 100).toFixed(3) + " %",
                  f4(d.modelos.A.auc_pr), f4(d.modelos.B.auc_pr), f4(d.modelos.C.auc_pr)];
        })),
    ]),
  ]);

  cont.append(perm, rec, mec, costo, part);
  for (const p of [rec, mec, costo, part]) p.style.marginTop = "16px";

  const series = [
    { clave: "global", nombre: "global", color: SERIES.A },
    { clave: "F1", nombre: "F1 escalada", color: SERIES.B },
    { clave: "F2", nombre: "F2 ráfaga", color: SERIES.C },
    { clave: "F3", nombre: "F3 atípico", color: "var(--serie-4)" },
  ];
  const claves = { global: "auc_pr", F1: "F1_escalada", F2: "F2_rafaga", F3: "F3_atipico" };
  graficaLineas($("#g-recorte"), {
    x: ev.recorte.map((r) => r.k),
    series: series.map((s) => ({ ...s, puntos: ev.recorte.map((r) => r[claves[s.clave]]) })),
    etiquetaX: "eventos visibles (k)", yMin: 0.2, yMax: 1, alto: 240,
  });
}

/* =============================== monitoreo =============================== */

function cifra(k, v, d) {
  return el("div", { class: "cifra" }, [
    el("div", { class: "k", text: k }), el("div", { class: "v", text: v }),
    d ? el("div", { class: "d", text: d }) : null,
  ]);
}

async function cargarMonitoreo() {
  const [reg, marc] = await Promise.all([api("/registros?limite=40"), api("/marcador")]);
  const r = reg.resumen;
  const cont = $("#monitoreo");
  cont.innerHTML = "";

  const resumen = el("div", { class: "panel" }, [
    el("h2", { text: "Uso del servicio" }),
    el("div", { class: "cuerpo" }, [
      el("div", { class: "cifras" }, [
        cifra("peticiones", (r.peticiones || 0).toLocaleString("es-GT"),
              r.desde ? `desde ${r.desde.replace("T", " ").replace("+00:00", " UTC")}` : null),
        cifra("alertas emitidas", (r.alertas || 0).toLocaleString("es-GT"),
              "modelo A sobre su umbral"),
        cifra("latencia p50", (r.latencia_ms.p50 ?? "—") + " ms", "puntuar una tarjeta"),
        cifra("latencia p95", (r.latencia_ms.p95 ?? "—") + " ms", "cola completa de 20 eventos"),
        cifra("casos revelados", marc.revelados, `${marc.modelo_aciertos} aciertos del modelo`),
        cifra("veredictos", marc.analista_revisiones, `${marc.analista_aciertos} aciertos del analista`),
      ]),
    ]),
  ]);

  const variantes = el("div", { class: "panel" }, [
    el("h2", { text: "Peticiones por variante" }),
    el("div", { class: "cuerpo" }, [
      el("p", { class: "nota", html:
        `<code>normal</code> es una puntuación corriente; <code>permutar</code> y <code>recortar</code>
         son las pruebas de falsificación corridas desde la consola. Quedan registradas igual que
         cualquier otra llamada.` }),
      tabla(["variante", "peticiones"], r.por_variante.map((v) => [v.variante, v.n])),
    ]),
  ]);

  const ultimas = el("div", { class: "panel" }, [
    el("h2", { text: "Últimas peticiones" }),
    el("div", { class: "cuerpo sin-aire" }, [
      tabla(["hora (UTC)", "ruta", "caso", "variante", "A", "B", "C", "alerta", "ms"],
        reg.ultimas.map((u) => [
          (u.ts || "").replace("T", " ").replace("+00:00", ""), u.ruta, u.caso || "—",
          u.variante, f3(u.puntaje_a), f3(u.puntaje_b), f3(u.puntaje_c),
          u.alerta ? "sí" : "—", u.latencia_ms ?? "—"])),
    ]),
  ]);

  cont.append(resumen, variantes, ultimas);
  for (const p of [variantes, ultimas]) p.style.marginTop = "16px";
}

/* ================================== api ================================== */

async function cargarFicha() {
  const s = ESTADO.ficha ?? (ESTADO.ficha = await api("/salud"));
  const m = s.modelo;
  const cont = $("#ficha");
  cont.innerHTML = "";
  cont.append(
    tabla(["modelo", "arquitectura", "ve el orden", "umbral"],
      Object.entries(m.modelos).map(([k, v]) => [k, v.familia, v.ve_orden ? "sí" : "no", f4(v.umbral)])),
    el("p", { class: "nota", style: "margin-top:12px", html:
      `<strong>Política:</strong> decide ${m.politica.decide}, prioriza ${m.politica.prioriza}.
       <strong>Costos:</strong> falso negativo ${q(m.costos.falso_negativo_Q)}, falso positivo
       ${q(m.costos.falso_positivo_Q)}. <strong>Mínimo:</strong> ${m.minimo_transacciones}
       transacciones por tarjeta.` }),
    el("p", { class: "nota", html:
      `<strong>Paridad:</strong> el servicio reproduce el pipeline por lotes del notebook con un
       desvío máximo de ${s.paridad_con_el_notebook.desvio_maximo_absoluto.toExponential(1)}
       sobre los ${s.casos_en_cola} casos de la cola.` }),
  );

  $("#ejemplo-curl").textContent =
`curl -X POST ${location.origin}/puntuar \\
  -H "Content-Type: application/json" \\
  -d '{
    "historia": [
      {"timestamp":"2025-04-20 09:14:00","monto":82.50,"id_comercio":41,
       "categoria":"supermercado","canal":"pos","extranjero":0},
      {"timestamp":"2025-04-20 11:02:00","monto":15.00,"id_comercio":902,
       "categoria":"farmacia","canal":"pos","extranjero":0},
      {"timestamp":"2025-04-21 19:40:00","monto":240.00,"id_comercio":77,
       "categoria":"ropa","canal":"ecommerce","extranjero":0},
      {"timestamp":"2025-04-21 19:52:00","monto":3150.00,"id_comercio":512,
       "categoria":"electronica","canal":"ecommerce","extranjero":1}
    ]
  }'

# Barajar la historia antes de puntuar (prueba de permutacion):
#   agregar  "permutar": 7
# Dejar visibles solo los ultimos k eventos (prueba de recorte):
#   agregar  "k_visible": 6`;

  $("#pie").innerHTML =
    `Centinela 1.0.0 · modelo candidato <code>artefactos/modelo_c.pt</code> · ` +
    `datos sintéticos, prevalencia 1.2 % · Cindy Gualim y Gadiel Ocaña · ` +
    `Universidad del Valle, Deep Learning 2026`;
}

/* ================================ arranque ================================ */

const CARGADORES = { evidencias: cargarEvidencias, monitoreo: cargarMonitoreo, api: cargarFicha };

function mostrar(vista) {
  document.querySelectorAll(".vista").forEach((v) => v.classList.toggle("activa", v.id === "vista-" + vista));
  document.querySelectorAll("nav button").forEach((b) =>
    b.setAttribute("aria-current", b.dataset.vista === vista ? "true" : "false"));
  CARGADORES[vista]?.().catch((e) => console.error(e));
}

document.querySelectorAll("nav button").forEach((b) =>
  b.addEventListener("click", () => mostrar(b.dataset.vista)));

$("#sel-orden").addEventListener("change", (e) => {
  ESTADO.orden = e.target.value;
  cargarCola();
});

$("#btn-refrescar").addEventListener("click", () => cargarMonitoreo());

$("#btn-tema").addEventListener("click", () => {
  const oscuro = document.documentElement.dataset.theme === "dark" ||
    (!document.documentElement.dataset.theme &&
     matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = oscuro ? "light" : "dark";
  try { localStorage.setItem("tema", document.documentElement.dataset.theme); } catch {}
});

try {
  const t = localStorage.getItem("tema");
  if (t) document.documentElement.dataset.theme = t;
} catch {}

(async function iniciar() {
  try {
    await cargarCola();
    await cargarFicha();
    if (ESTADO.casos.length) abrirCaso(ESTADO.casos[0].id);
  } catch (e) {
    $("#cola").innerHTML = `<div class="vacio">No se pudo cargar la cola.<br>${e.message}</div>`;
  }
})();
