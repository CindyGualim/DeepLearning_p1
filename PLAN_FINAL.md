# Plan de trabajo — Proyecto Final: el modelo del Proyecto 1 en producción

**Universidad del Valle · Deep Learning y Sistemas Inteligentes 2026 · Kevin Recinos**

Equipo de dos: **Cindy Gualim** (21226) y **Gadiel Ocaña** (231270).
Presentación **lunes 9 de noviembre**. Entrega límite **domingo 15 de noviembre, 23:59**.

> Este es el plan que se ejecutó, no una propuesta. La columna *estado* dice dónde quedó
> cada etapa al cerrar el 8 de noviembre.

---

## Decisión de arranque (lunes 3 de noviembre)

**Modelo base: el Proyecto 1 de Gualim y Ocaña** — monitoreo transaccional, tres modelos
comparables (A agregadas, B secuencial, C híbrido). Se eligió sobre los otros proyectos del
equipo por tres razones:

1. Ya tenía un **candidato declarado** y su artefacto versionado (`artefactos/modelo_c.pt`),
   más un contrato de entrada/salida escrito en la sección "Candidato al Proyecto Final" del
   README original. No había que inventar qué poner en producción.
2. Tenía **tres modelos que no coinciden**, lo que permite servir una política
   (*A decide, C prioriza*) en vez de un solo número.
3. Tenía un **generador sintético propio**, así que se puede producir una cola de casos de
   prueba con el mecanismo de fraude etiquetado. Ningún conjunto descargado permite eso.

**Qué se decidió NO hacer.** No se reentrena. El encargo dice "puede mejorar el modelo; no se
pide uno nuevo", y los pesos del Proyecto 1 ya están defendidos con evidencia en el informe.
Reentrenar habría obligado a rehacer la validación completa en seis días y habría puesto en
riesgo lo único que de verdad se califica aquí: que el modelo quede **usable por alguien ajeno
al equipo**.

**Producto elegido: Centinela**, una consola de monitoreo transaccional sobre una API pública.
No un formulario de predicción: el puesto de trabajo del analista de riesgos, que es quien usa
el puntaje según la problemática del Proyecto 1.

---

## Etapas

| # | Etapa | Fecha | Responsable | Entregable | Estado |
|---|---|---|---|---|---|
| 1 | Decisión de alcance y diseño del producto | mar 3 nov | ambos | Este plan | hecho |
| 2 | Extraer el pipeline del notebook a `src/` | mié 4 nov | Cindy | `config`, `generador`, `caracteristicas`, `modelos` | hecho |
| 3 | Puntuador de una tarjeta + contrato de entrada | mié 4 – jue 5 nov | Cindy | `src/puntuador.py` | hecho |
| 4 | **Prueba de paridad con el notebook** | jue 5 nov | ambos | `pruebas/test_paridad.py` | hecho |
| 5 | Banco de casos y evidencias precalculadas | jue 5 nov | Gadiel | `scripts/construir_banco.py` | hecho |
| 6 | API HTTP y registro de uso | vie 6 nov | Gadiel | `api/app.py`, `src/registro.py` | hecho |
| 7 | Consola: cola, caso, experimentos en vivo | vie 6 – sáb 7 nov | Cindy | `api/estatico/` | hecho |
| 8 | Pestañas de evidencias y monitoreo | sáb 7 nov | Gadiel | `api/estatico/consola.js` | hecho |
| 9 | Pruebas de contrato y de API | sáb 7 nov | ambos | `pruebas/` (51 pruebas) | hecho |
| 10 | Empaque para despliegue | dom 8 nov | Gadiel | `Dockerfile`, `requisitos-servicio.txt` | hecho |
| 11 | Verificación de uso y exportación del registro | dom 8 nov | Cindy | `datos_demo/registro_uso.jsonl` | hecho |
| 12 | README como documento técnico (7 secciones) | dom 8 nov | ambos | `README.md` | hecho |
| 13 | Despliegue público y captura de evidencias | dom 8 nov | Cindy | enlace al servicio | **pendiente** |
| 14 | Presentación (10 min) | dom 8 nov | ambos | `presentacion_final.pdf` | **pendiente** |
| 15 | Ensayo cronometrado y reparto de preguntas | dom 8 nov, noche | ambos | — | **pendiente** |

### Ruta crítica

```
2 → 3 → 4 → 5 → 6 → (7 ∥ 8) → 9 → 10 → 13
                                  ↘ 11 → 12
```

La etapa 4 es la que gobierna todo lo demás: hasta que `src/` no reprodujo los seis AUC-PR
publicados en el informe, nada construido encima tenía valor. Fue también donde apareció el
único defecto real del proyecto (ver abajo).

---

## Lo que salió mal, y qué costó

**El pipeline extraído no reproducía el notebook.** Al correr la etapa 4 por primera vez, el
modelo B coincidía exacto (0.9226 / 0.9206) pero A y C quedaban muy por debajo (0.77 contra
0.9171). Como B es el único de los tres que **no** usa variables agregadas, el defecto tenía que
estar en `agregadas()`.

La causa: el notebook calculaba el minuto de cada transacción con
`timestamp.astype("int64") // 60_000_000_000`, que supone resolución de nanosegundos. La versión
de pandas del entorno de servicio infiere `datetime64[us]`, así que esa misma división devolvía
minutos divididos entre mil. Las ventanas de 1 h, 24 h y 7 días quedaban mil veces más anchas y
absorbían toda la historia de la tarjeta: `n_1h` valía 16 en lugar de 2.2.

El arreglo es una línea (`astype("datetime64[m]")`, que da el mismo número bajo cualquier
resolución) y está comentado en `src/caracteristicas.py`. **Costó media jornada de la etapa 4** y
es la razón por la que esa prueba existe: sin ella, el servicio habría mostrado al comité
puntajes silenciosamente incorrectos con la misma cara de confianza.

---

## Riesgos que se gestionaron

| Riesgo | Mitigación | Resultado |
|---|---|---|
| El servicio muestra números distintos a los del informe | Paridad automatizada que **falla el build** (`scripts/construir_banco.py` aborta y no escribe el banco) | 6/6 AUC-PR coinciden al 4º decimal |
| Arranque lento: regenerar 324,603 transacciones por petición | Banco de casos y evidencias precalculados fuera de línea | arranque 6.4 s; puntuar una tarjeta, 39 ms (p50) |
| La revelación de casos parece teatro | La verdad vive en el servidor y solo viaja tras el veredicto; hay una prueba que lo verifica | `test_la_cola_no_filtra_la_verdad` |
| La demostración no se ve igual en los dos días de presentación | Cupos fijos por mecanismo y semilla fija en la curación del banco | cola idéntica el 9 y el 13 |
| Torch con CUDA infla la imagen a ~2.5 GB | Índice de CPU explícito en `requisitos-servicio.txt` | imagen manejable |

---

## Reparto para la presentación (10 minutos)

| Bloque | Minutos | Quién |
|---|---|---|
| Problemática, usuario y decisión que depende del modelo | 1.5 | Gadiel |
| Modelo base y por qué no se reentrenó | 1.5 | Cindy |
| **Demostración en vivo**: cola → caso C082 → barajar historia | 3.0 | Cindy |
| Validación de hipótesis y el control F3 | 2.0 | Gadiel |
| Arquitectura, monitoreo y registros de uso | 1.5 | Gadiel |
| Límites y recomendación al comité | 0.5 | Cindy |

**Caso de demostración: C082.** Barajar su historia hunde al modelo B de 1.000 a 0.041 mientras
el modelo A no se mueve ni un diezmilésimo. Alternativas si falla: C060 (1.000 → 0.072) o
C030 (1.000 → 0.147).

**Preguntas previstas y quién las toma**

- *¿Por qué C si A es más barato en prueba?* — Cindy. Porque la ventaja de A no es
  estadísticamente distinguible y C gana AUC-PR en ambas particiones; por eso la política sirve
  a los dos, con A decidiendo.
- *¿Cómo sabemos que el modelo usa el orden y no otra cosa?* — Gadiel. Por el control F3, que es
  el único mecanismo que no se degrada al barajar.
- *¿Esto sirve con datos reales?* — ambos. No todavía: la metodología está validada, el
  comportamiento del fraude real no. Está declarado en los límites.
- *¿Por qué el umbral no se recalibra solo?* — Gadiel. Porque el propio informe mostró que un
  umbral puntual no generaliza entre períodos (4 FN en validación → 15 en prueba); recalibrar sin
  un conjunto mayor sería sobreajustar otra vez.
