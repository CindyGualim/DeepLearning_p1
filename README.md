# Proyecto 1 — Monitoreo transaccional: detectar lo que el orden revela

**Universidad del Valle · Deep Learning 2026 · Kevin Recinos**

Integrantes:
**Cindy Gualim** (21226) 
· **Gadiel Ocaña** (231270) 
· Entrega: 4 de septiembre de 2026

Entregable principal: [`proyecto1_Gualim_Ocaña.ipynb`](proyecto1_Gualim_Ocaña.ipynb) — **113 celdas,
autocontenido**. No importa código propio: todo lo que produce un número está a la vista.

---

## 1. La pregunta y la respuesta corta

> ¿El orden de las transacciones aporta información que las variables agregadas no capturan, bajo
> qué condiciones y cuánto vale esa información en quetzales?

**Tres respuestas, y no dicen lo mismo:**

| Pregunta | Respuesta | Dónde se ve |
|---|---|---|
| ¿El orden aporta información? | **Sí, y está demostrado.** Barajar el orden le cuesta al modelo secuencial el 28 % de su AUC-PR (0.9226 → 0.6647). | celdas **77**, **80** |
| ¿Bajo qué condiciones? | **Solo en un mecanismo.** En la escalada de montos el modelo secuencial saca 0.78 contra 0.74 de los agregados; en el fraude de monto atípico empatan (0.754 contra 0.754). | celdas **80**, **96** |
| ¿Cuánto vale en quetzales? | **No se puede afirmar que valga algo todavía.** En prueba ningún modelo es más barato que otro: los tres intervalos de confianza del costo cruzan el cero. | celdas **104**, **106** |

**Recomendación al comité: complementar, no reemplazar.** Conservar el motor de agregados como
decisor y usar el puntaje secuencial como señal secundaria de priorización. El razonamiento completo
está en la celda **111**.

La aparente contradicción entre la respuesta 1 y la 3 es el hallazgo central del trabajo, no un
error: *que un modelo use el orden no implica que ese uso se pague solo*.

---

## 2. Qué pedía el encargo y dónde está cumplido

> **Cómo leer los números de celda.** Son la posición de la celda en el notebook contando desde 0 e
> incluyendo las de texto. Para ubicarlas rápido, buscar el título de sección (`1.1`, `2.3`, `3.1`…)
> con `Ctrl+F`: el mapa de abajo va en el mismo orden que el notebook.

### 2.1 Núcleo común (A, B y la apuesta C)

| Pieza | Qué pedía | Cómo se resolvió | Celdas |
|---|---|---|---|
| **A — línea base sin orden** | Modelo competitivo sobre variables agregadas | `HistGradientBoostingClassifier` sobre 20 agregadas de ventana. Se compitió contra regresión logística en la misma rejilla | **43** (rejilla), **45** (ganador), **53** (artefacto) |
| **B — modelo secuencial** | Modelo que reciba eventos ordenados | GRU de 1 capa (128 unidades) con embeddings de canal y categoría. Se compitió contra CNN causal dilatada | **55–56** (arquitecturas), **58** (comparación), **60** (reentrenamiento), **63** (artefacto) |
| **C — apuesta del equipo** | Extensión con hipótesis propia, control y métrica declarada antes de ver prueba | Híbrido: estado final de la GRU + 18 agregadas → cabeza MLP | **91** (criterio), **93–94** (entrenamiento), **95** (veredicto), **97** (artefacto) |

Los tres devuelven **puntaje continuo** (`predict_proba` / `sigmoid`), y el umbral se decide después,
en el bloque 5.

### 2.2 Las dos pruebas obligatorias de falsificación

| Prueba | Celdas | Qué muestra |
|---|---|---|
| **Permutación controlada** (obligatoria) | **76** (función), **77** (5 semillas × 2 variantes), **80** (por mecanismo), **81** (figura) | El AUC-PR cae de 0.9226 a 0.6647 al barajar la historia. F1 se desploma (−0.47), **F3 no cae** (+0.03) |
| **Recorte de la historia** (elegida) | **85** (función y barrido), **86** (figura), **87** (saturación) | Cada mecanismo satura donde su largo de episodio predice: F1 en k=6 (episodios de 5.03), F3 ya en k=1 |

### 2.3 Las seis evidencias del informe

| # | Evidencia | Celdas | Qué se ve ahí |
|---|---|---|---|
| 1 | Integridad de datos | **7–18**, **20–29**, **32–38** | Composición, huella de reproducibilidad, 6 controles antifuga, control de causalidad |
| 2 | Comparación común A vs B | **65** (tabla), **67** (curva PR), **69** (bootstrap), **72** (presupuesto fijo) | AUC-PR + precisión/exhaustividad/F1 en el umbral, con intervalo de confianza |
| 3 | Valor del orden | **77**, **80**, **81**, **85**, **86** | Permutación y recorte, con sus figuras |
| 4 | Apuesta del equipo | **90–91** (hipótesis previa), **94–96** (resultado), **98** (veredicto) | Hipótesis, control, métrica y veredicto explícito |
| 5 | Decisión económica | **101–102** (validación), **104** (prueba), **106** (intervalos), **109–110** (mensual y sensibilidad) | Umbral por costo, aplicación única a prueba, proyección |
| 6 | Recomendación y límites | **111** | Complementar; patrón de error; condiciones que la invertirían |
| — | **Matriz de evidencias** | **112** | Tabla de una página: evidencia / figura / conclusión / limitación |

### 2.4 Penalizaciones evitadas y dónde comprobarlo

| Penalización | Cómo se evitó | Celda que lo demuestra |
|---|---|---|
| −20 partición aleatoria | Cortes por tiempo con embargo de 2 días | **25** (tabla por partición), **28** (aserciones) |
| −15 estadísticas del conjunto completo | Agregadas causales + escalador ajustado solo en train | **34** (control de causalidad), **36** (train en media 0, val y test no) |
| −15 exactitud como métrica principal | Nunca se reporta; la métrica es AUC-PR | **3** (declaración), **41** (definición) |
| −10 mirar prueba para decidir | Prueba se abre una sola vez, en la celda **104** | **101–102** (todo decidido antes) |
| −10 afirmar el orden sin permutación | La permutación está corrida con 5 semillas | **77** |

---

## 3. Qué dicen los resultados, sección por sección

### 3.1 Los datos (celdas 7–18)

324,603 transacciones · 5,991 tarjetas · 120 días · **1.197 %** de fraude · huella `2a9dc26504bd733d`.

**La decisión de diseño más importante del proyecto está en la celda 9**: junto a cada escalada
fraudulenta se generan **dos sesiones legítimas** con los mismos montos, la misma ventana, los mismos
comercios nuevos y el mismo canal, pero en orden **no creciente**.

La celda **16** demuestra que el control funciona:

| | eventos | suma | promedio | máximo | comercios | duración | % creciente |
|---|---|---|---|---|---|---|---|
| F1_escalada | 5.03 | 1302.55 | 265.03 | 1203.14 | 5.03 | 18.2 min | **99.7** |
| sesion_legitima | 5.03 | 1327.35 | 270.40 | 1225.12 | 5.03 | 18.1 min | **0.0** |

Los agregados son estadísticamente iguales. **Lo único que separa fraude de no fraude en F1 es la
secuencia.**

### 3.2 Protocolo temporal (celdas 25–29)

| partición | secuencias | fraudes | desde | hasta | tasa |
|---|---|---|---|---|---|
| train | 177,033 | 2,291 | 01-01 | 03-13 | 1.294 % |
| val | 59,495 | 660 | 03-16 | 04-06 | 1.109 % |
| test | 59,410 | 644 | 04-09 | 04-30 | 1.084 % |
| descarte | 10,732 | 111 | (embargo) | | |

Los seis controles de la celda **28** son `assert`, no mensajes. El más importante para la defensa
oral es el cuarto —*ninguna secuencia contiene eventos posteriores a su evento objetivo*— porque
anticipa la objeción de que las secuencias de validación incluyen eventos del período de
entrenamiento. **Eso no es fuga**: en producción, al puntuar una transacción el historial ya existe.
La fuga sería usar información *posterior*.

### 3.3 Predicción escrita antes de entrenar (celda 38)

| mecanismo | razon_max_24h | z_monto_tarjeta | n_1h | comercios_24h |
|---|---|---|---|---|
| F3_atípico | 5.774 | **4.655** | 0.025 | 0.520 |
| F2_ráfaga | −0.181 | 0.082 | **5.179** | **5.651** |
| F1_escalada | 0.760 | −0.006 | 2.204 | 2.619 |
| sesión_legítima | −0.844 | −0.211 | **2.205** | **2.598** |

F1 y su control son **el mismo vector** para los agregados. De ahí la predicción: *A debería igualar
a B en F3, acercarse en F2 y quedarse corto en F1*. Todo el bloque 3 es la comprobación.

### 3.4 A contra B (celdas 65–72)

| modelo | AUC-PR val | precisión | exhaustividad | F1 | costo |
|---|---|---|---|---|---|
| A | 0.9171 | 0.5670 | 0.9879 | 0.7204 | **Q123,240** |
| B | 0.9226 | 0.5863 | 0.9833 | 0.7346 | Q128,640 |

**Dos lecturas opuestas y las dos válidas.** B tiene mejor AUC-PR pero cuesta más: cambia 40 falsos
positivos menos por 3 falsos negativos más, y a Q4,200 contra Q180 ese canje pierde Q5,400.
**Ordenar modelos por AUC-PR no equivale a ordenarlos por dinero.**

Y la diferencia de 0.0055 no sobrevive la incertidumbre (celda **69**):

```
diferencia observada B - A : +0.0055
IC 95 % (bootstrap x400)   : [-0.0148, +0.0254]
P(B > A)                   : 0.688
```

El bootstrap remuestrea **tarjetas, no filas**: las secuencias de una misma tarjeta están
correlacionadas y tratarlas como independientes estrecharía el intervalo artificialmente.

### 3.5 La permutación (celdas 77, 80) — la evidencia más fuerte

| variante | AUC-PR | caída | F1 en umbral |
|---|---|---|---|
| orden real | 0.9226 | — | 0.7346 |
| historia barajada | 0.6647 ± 0.0113 | **−0.2580 (28 %)** | 0.5998 |
| secuencia completa | 0.0503 ± 0.0009 | −0.8723 (94.5 %) | 0.1347 |

| mecanismo | A | B | B permutado | caída |
|---|---|---|---|---|
| F1_escalada | 0.7355 | 0.7806 | 0.3109 | **−0.4698** |
| F2_ráfaga | 0.9494 | 0.9489 | 0.4452 | −0.5037 |
| F3_atípico | 0.7541 | 0.7540 | 0.7887 | **+0.0346** |

**F3 —el mecanismo diseñado sin dependencia del orden— es el único que no se degrada.** Eso descarta
que la caída venga de un error técnico: si el bug fuera de código, F3 caería igual que los demás.

### 3.6 El recorte (celdas 85–87)

| k visible | global | F1 | F2 | F3 |
|---|---|---|---|---|
| 1 | 0.5970 | 0.2811 | 0.2299 | **0.7745** |
| 3 | 0.8431 | 0.5478 | **0.9317** | 0.7817 |
| **6** | **0.9258** | **0.7958** | 0.9474 | 0.7687 |
| 20 | 0.9226 | 0.7806 | 0.9489 | 0.7540 |

Largo de episodio por construcción: F1 = 5.03, F2 = 10.32, F3 = 1.00. **Cada curva satura donde su
propio episodio predice.** Con k = 1, B cae a 0.597 contra 0.917 de A: sin historia el modelo
secuencial es peor que la línea base.

*Hallazgo operativo:* el máximo global está en **k = 6, no en k = 20**.

### 3.7 La apuesta C (celdas 91–98)

| condición declarada en la celda 91 | exigido | obtenido | ¿cumple? |
|---|---|---|---|
| AUC-PR en validación | ≥ 0.9426 | **0.9424** | **no, por 0.0002** |
| IC 95 % contra `max(A,B)` excluye cero | sí | [+0.0098, +0.0337] | sí |

**Veredicto: no cumple.** Falla por dos diezmilésimas y así queda escrito. Mover el listón después de
ver el resultado destruiría el valor de haberlo declarado antes; el historial de git tiene la prueba
(commit `612b2fc`, anterior a `9337a49`).

| mecanismo | A | B | C | control |
|---|---|---|---|---|
| F1_escalada | 0.7355 | 0.7806 | **0.8347** | 0.8107 |
| F2_ráfaga | 0.9494 | 0.9489 | 0.9614 | **0.9635** |
| F3_atípico | 0.7541 | 0.7540 | **0.8049** | 0.7718 |

**La lección metodológica que más conviene defender:** comparar el híbrido contra B —lo intuitivo—
habría atribuido a las agregadas una mejora de +0.0198. El control con agregadas en ceros muestra que
solo **+0.0094** viene de ellas. **Sin control experimental habríamos exagerado el efecto al doble.**

### 3.8 La decisión económica (celdas 101–111)

Validación → candidato **C** (Q89,520, el más barato). Prueba, una sola vez:

| modelo | AUC-PR test | FN | costo FN | FP | costo FP | total |
|---|---|---|---|---|---|---|
| A | 0.9092 | **8** | Q33,600 | 429 | Q77,220 | **Q110,820** |
| B | 0.9206 | 12 | Q50,400 | 419 | Q75,420 | Q125,820 |
| C | **0.9361** | 15 | Q63,000 | 379 | Q68,220 | Q131,220 |

**El orden por costo se invirtió.** Toda la diferencia entre el mejor y el peor son **7 fraudes sobre
644**, y los intervalos lo confirman (celda **106**):

```
B - A: observado Q+15,000 | IC95 [Q-28,510, Q+65,224] | P(B más barato) = 0.287
C - A: observado Q+20,400 | IC95 [Q-29,889, Q+91,083] | P(C más barato) = 0.263
C - B: observado Q+5,400  | IC95 [Q-48,902, Q+76,146] | P(C más barato) = 0.439
```

**Ninguna diferencia es distinguible del ruido.** En cambio el AUC-PR mantuvo el orden en las dos
particiones (C > B > A). Además el híbrido pasó de 4 falsos negativos en validación a 15 en prueba
con el mismo umbral: el umbral puntual se sobreajusta al período en que se elige.

Proyección mensual (factor 318.7 = 1.4 M / 5,991 tarjetas × 30 / 22 días): A ≈ Q35.3 M, B ≈ Q40.1 M,
C ≈ Q41.8 M, sin modelo ≈ Q862 M. **Estas cifras están infladas** por la prevalencia sintética de
1.2 %: léanse diferencias relativas, no montos.

Sensibilidad (celda **110**): con FN a Q1,500 o Q2,500 **el mejor es B**; desde Q4,200 en adelante es
A.

---


---

##  Candidato al Proyecto Final

**Modelo que conservaríamos: C (híbrido).** Artefacto en **`artefactos/modelo_c.pt`** (262 KB), con
`state_dict`, arquitectura, dimensiones y umbral de validación. Se acompaña de
`artefactos/escalador_eventos.joblib` y `artefactos/escalador_agregadas.joblib`, los parámetros de
preparación necesarios para reproducir los puntajes.

Se elige C y no A pese a que A fue más barato en prueba, por tres razones: su AUC-PR es el mejor en
**ambas** particiones (0.9424 val, 0.9361 test), es el único que combina las dos fuentes de señal, y
la ventaja de A en costo no es estadísticamente distinguible. La incumbente A queda como campeona a
batir, no como descartada.

**Quién usa el puntaje y qué decide.** El equipo de monitoreo transaccional del área de riesgos. En
la etapa inmediata, **priorizar la cola de revisión manual** —no bloquear de forma autónoma— hasta que
un conjunto de prueba mayor resuelva la pregunta de costo.

**Contrato preliminar**

*Entrada:* las últimas 20 transacciones de la tarjeta hasta la actual inclusive, cada una con
`log_monto`, `log_dt`, `hora_sin`, `hora_cos`, `extranjero`, `comercio_nuevo` más los índices de
`canal` y `categoria`; adicionalmente 18 variables agregadas causales de la transacción actual.
Mínimo 3 transacciones previas (por debajo de eso, camino de reglas).

*Salida:* un puntaje continuo en [0, 1]. Umbral operativo inicial **0.2263**, elegido por costo
esperado sobre validación y **recalibrable** — el análisis mostró que un umbral puntual no generaliza
bien entre períodos.

**Límites, riesgos y datos que faltan**

- Los datos son **sintéticos**: los mecanismos de fraude son los que nosotros diseñamos. La
  metodología está validada; el comportamiento del fraude real no.
- La prevalencia (1.2 % por transacción) está muy por encima de la real, así que las cifras
  monetarias no son trasladables.
- El generador asigna episodios uniformemente entre tarjetas, lo que produce más fraude por
  transacción en tarjetas poco activas (2.9 % contra 0.6 %). El efecto medido es de 0.0003 de AUC-PR
  (celda **51**), pero existe.
- **Error irreducible conocido:** las primeras sondas de una escalada están etiquetadas como fraude
  pero no tienen historia que las delate. Ningún modelo puede detectarlas (celda **23**).
- **Falta:** datos reales etiquetados, un conjunto de prueba con ~10× más positivos para resolver la
  comparación de costos, y validación de que la escalada existe en la cartera del banco con la
  frecuencia que aquí se supuso.
