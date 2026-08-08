# Plan de trabajo — Proyecto 1: Monitoreo transaccional

Entrega: viernes 4 de septiembre de 2026, 23:59 · Presentación 8 min + 4 de preguntas · Parejas · 8 pts

**Decisión base:** Ruta A (generador sintético propio). Da control total sobre los mecanismos
de fraude, garantiza que al menos uno dependa del orden, y no depende de descargar ni limpiar
un dataset externo con el tiempo que queda.

---

## Bloque 1 — Datos y protocolo (rúbrica: 15 pts)

### T1. Estructura del repo y entorno  ✅ HECHA
- `datos/`, `artefactos/`, `figuras/`, `src/`, `proyecto1_<apellidos>.ipynb`.
- `requirements.txt` con versiones fijadas. Semilla global en `src/config.py`.
- **Hecho cuando:** el notebook corre de arriba a abajo con las celdas vacías de estructura.

### T2. Generador sintético reproducible
- `src/generador.py`: población de tarjetas, comercios, canales, montos, timestamps.
- Tres mecanismos de fraude, al menos uno **dependiente del orden**:
  - F1 *escalada*: varias compras pequeñas de prueba seguidas de una compra grande (ORDEN).
  - F2 *ráfaga*: muchas transacciones en pocos minutos en comercios distintos (semi-orden).
  - F3 *monto atípico aislado*: una sola transacción anómala (NO depende del orden — es el control:
    la línea base A debería detectarlo igual de bien).
- Misma semilla ⇒ mismos datos, verificado con `hash_df`.
- **Hecho cuando:** dos corridas con la misma semilla producen el mismo hash y la tasa de fraude
  queda cerca de `TASA_FRAUDE_OBJETIVO`.

### T3. Construcción de secuencias y partición temporal
- Secuencia = últimas N transacciones de la tarjeta hasta el evento actual inclusive;
  se predice si ese evento es fraude.
- Partición **por tiempo** con embargo entre cortes. Sin solapamiento entre particiones.
- **Hecho cuando:** hay tabla de tamaño/rango/tasa por partición y una aserción que falla si
  `max(fecha_train) >= min(fecha_val)`.

### T4. Features agregadas + escalado sin fuga
- Agregadas por ventana (promedio 24h, conteo/hora, monto máx del día, diversidad de comercios)
  calculadas **solo con el pasado** de cada tarjeta.
- `StandardScaler` ajustado **solo en train**. Guardar en `artefactos/`.
- **Hecho cuando:** existe una celda de "controles antifuga" que lista cada control y lo verifica.

---

## Bloque 2 — Núcleo A y B (rúbrica: 20 pts)

### T5. Modelo A — línea base sin orden
- `HistGradientBoostingClassifier` sobre las agregadas de T4. Hiperparámetros por validación.
- **Hecho cuando:** AUC-PR de validación reportada y modelo serializado en `artefactos/`.

### T6. Modelo B — modelo secuencial
- GRU chica (1 capa, hidden ~64 — el entorno es CPU) sobre eventos ordenados.
- Manejo de desbalance (`pos_weight`), early stopping por AUC-PR de validación.
- **Hecho cuando:** curva de entrenamiento guardada y AUC-PR de validación registrada.

### T7. Comparación común A vs B
- Mismos datos, misma partición, mismo horizonte. Curva PR conjunta.
- Umbral elegido **con validación**, nunca con test.
- Tabla: AUC-PR y, en el umbral, precisión/exhaustividad/F1. **Sin exactitud como métrica principal.**
- **Hecho cuando:** existe `figuras/pr_A_vs_B.png` y la tabla en el notebook.

---

## Bloque 3 — Valor del orden (rúbrica: 20 pts)

### T8. Prueba 1 — permutación controlada (obligatoria)
- Barajar el orden **dentro de cada secuencia** de test, sin cambiar eventos ni agregadas.
- Re-evaluar B con el mismo modelo y umbral. Varias semillas, media ± dispersión.
- **Hecho cuando:** tabla original vs permutado con interpretación honesta (si no cae, decirlo).

### T9. Prueba 2 — elegida por el equipo
- **Evaluación por mecanismo de fraude** (F1/F2/F3 por separado). Predicción de la teoría:
  B gana claramente en F1, poco en F2, y nada en F3.
- **Hecho cuando:** desglose por mecanismo con AUC-PR de A y B lado a lado.

---

## Bloque 4 — Apuesta C (rúbrica: 15 pts)

### T10. Hipótesis previa escrita ANTES de entrenar
- Celda literal: "Creemos que ___ mejorará ___ porque ___. Lo consideraremos útil si ___."
- Control experimental y métrica de éxito con número concreto.
- Sugerencia: híbrido (agregadas + estado final de la GRU), control = B solo,
  éxito = +0.03 AUC-PR en validación.
- **Hecho cuando:** la celda está escrita y commiteada antes de cualquier resultado de C.

### T11. Entrenar y juzgar C
- C y su control con el mismo protocolo. Veredicto explícito **aunque haya fallado**.
- **Hecho cuando:** tabla C vs control y línea de veredicto "cumple / no cumple".

---

## Bloque 5 — Decisión y comunicación (rúbrica: 15 + 15 pts)

### T12. Análisis económico y umbral
- FN = Q4,200; FP = Q180. Barrido de umbral sobre **validación**, minimizando costo esperado;
  recién ahí una sola aplicación a test.
- Proyección mensual con la cartera de 1.4M tarjetas.
- **Hecho cuando:** hay una cifra en quetzales/mes y `figuras/costo_umbral.png`.

### T13. Informe (≤7 págs) + matriz de evidencias + presentación (≤8 slides)
- Las seis evidencias localizables sin adivinarlas.
- Matriz final: evidencia | figura o tabla | conclusión | limitación.
- Al menos un patrón de error concreto y el caso donde se espera que el modelo falle.
- Recomendación: reemplazar / complementar / conservar, con condiciones para cambiarla.
- **Hecho cuando:** `informe.pdf` y `presentacion.pdf` exportados, sin código adentro.

### T14. README, artefactos y cierre
- README: reproducción, versiones, **declaración de uso de IA**, y las **tres decisiones
  técnicas** con alternativas y evidencia (cualquiera de los dos debe poder defenderlas).
- Sección "Candidato al Proyecto Final": modelo conservado y su artefacto, quién usa el puntaje
  y qué decide, contrato preliminar de entrada/salida, límites y riesgos.
- **Hecho cuando:** un tercero clona, sigue el README y obtiene los mismos puntajes.

---

## Penalizaciones a vigilar (descuento sin excepción)
- −20 partición aleatoria → T3 lo previene.
- −15 escalar/seleccionar/ventanear con estadísticas del conjunto completo → T4.
- −15 exactitud como métrica principal → T7.
- −10 elegir arquitectura, umbral o apuesta mirando test → T7, T10, T12.
- −10 afirmar que el orden aporta sin permutación → T8.

## Reparto sugerido en pareja
- Persona 1: T2, T3, T4, T6, T8, T9
- Persona 2: T1, T5, T7, T10, T11, T12
- Ambos: T13, T14

## Ruta crítica
T1 → T2 → T3 → T4 → (T5 ∥ T6) → T7 → (T8 ∥ T9 ∥ T10→T11) → T12 → T13 → T14
