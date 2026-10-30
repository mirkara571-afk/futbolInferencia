# futbolInferencia — Objetivos, restricciones y criterios de éxito

**Documento:** 01 · Especificación de objetivos, reglas y aceptación

**Proyecto:** futbolInferencia

**Rol emisor:** Delimitador de Reglas y Lógica

**Fecha de auditoría:** 28 de octubre de 2026

**Workspace:** `C:\Users\USUARIO\Trabajo\proyectos\mundiales\futbolInferencia`

**Destinatario:** Analista de Solución (y, a través de él, Programador Experto y QA)

**Estado:** listo para análisis técnico; contiene decisiones que requieren confirmación del dueño (marcadas con 🔴).

---

## 0. Fuentes, precedencia y convenciones

### 0.1 Fuentes obligatorias leídas

| Fuente | Uso en este documento |
|---|---|
| `futbolInferencia — Guía por fases v2.pdf` (24-oct-2026), texto en `.spec-tmp/futbolInferencia_-_Guía_por_fases_v2.txt` | Spec vigente para las etapas A–H y para las reglas de validación. Manda sobre todo lo demás. |
| `futbolInferencia — Guía por fases v1.pdf` (5-oct-2026), texto en `.spec-tmp/futbolInferencia_-_Guía_por_fases_v1.txt` | Solo se usa para el detalle de las fases 4, 5 y 6 que la v2 cita en la Etapa H, y como historia del proyecto. |
| `README.md` | Documentación descriptiva del repositorio. No es fuente de verdad: se contrasta con código y artefactos. |
| `requirements.txt`, `requirements-dev.txt`, `.gitignore`, estructura del repo | Estado de dependencias, datos y exclusiones. |
| `web/fases/predictores.py`, `evaluacion.py`, `dataset.py`, `contexto.py`, `modulo.py`, `views.py`, `tests.py`, comandos `evaluar_modelos.py` y `construir_dataset.py` | Evidencia de lo que existe y de cómo está implementado. |
| `experimentos/resultados_baseline.json` (25-oct-2026), `data/modelos/lstm-2026-10-23/`, `data/modelos/lstm-2026-10-25/`, `data/datasets/partidos_features.csv` | Evidencia de lo que está medido y de qué modelo usa la app. |
| `git log --oneline -15`, `git status`, ramas | Estado de integración y disciplina de trabajo. |

### 0.2 Precedencia documental

> **Regla de precedencia.** La guía v2 gobierna la predicción (Etapas A–G) y las reglas de validación. La guía v1 solo aporta el detalle de las fases 4, 5 y 6 que la v2 delega en la Etapa H. El README y los notebooks son descriptivos: si contradicen a la v2 o a los artefactos medidos, se corrigen; nunca se usan para declarar “terminado” algo que el código o los datos no demuestran. El código demuestra que algo **existe**, no que esté **validado**.

### 0.3 Convenciones de estado

| Marca | Significado |
|---|---|
| `[H]` | Existe en el repositorio y es ejecutable. |
| `[V]` | Validado con pruebas, comandos o artefactos reproducibles. |
| `[T]` | Terminado según el criterio de “Terminado cuando” de la guía, con pruebas y documentación coherentes. |
| `[P]` | Parcial: existe una parte, pero no cumple el criterio completo. |
| `[—]` | No existe o no está iniciado. |
| 🔴 | Requiere decisión del dueño antes de que el Analista cierre el plan. |

“Terminado” no significa “funciona en mi máquina”. En este proyecto significa: cumple el criterio de la etapa, tiene prueba automatizada o evidencia reproducible, no rompe la app, respeta el esquema fijo de métricas y deja el README/notebook/JSON coherentes con el modelo que se usa.

---

## 1. Resumen ejecutivo y diagnóstico de estado

### 1.1 Veredicto en una página

El repositorio no está en el punto que sugiere el README ni exactamente en el que sugiere la v2. Está en un punto intermedio: **la infraestructura de datos, simulación y evaluación está madura; la calidad probabilística del modelo está empezando**. La Etapa A se implementó después de que se escribiera la v2 (commit `1c725a9`, 26-oct-2026) y, al añadir el Elo puro como línea base, **cambió el veredicto sobre el LSTM**: el modelo que la app usa no demuestra ser mejor que el Elo puro.

Evidencias principales de esta auditoría:

- `python manage.py check` → “System check identified no issues”.
- `python manage.py test fases` → **41 tests, OK** (9,5 s; `SimpleTestCase`, sin base de datos de prueba).
- `GET /` con `Client(HTTP_HOST="localhost")` → **200**, 40.763 bytes, contiene el botón “Predecir”.
- Base de datos operativa: **1.051 partidos**, **49.945 internacionales**, **524 filas de proxy**.
- `experimentos/resultados_baseline.json` (25-oct-2026): **9.833 partidos** de dataset; configuración elegida `k=1`, `con_rival=False`, `con_contexto=True`, `extras=["elo"]`, `epocas=6`, `umbral_empate=0.01`.
- **El Elo puro no es superado de forma demostrable por el LSTM**: en validación móvil el LSTM queda en 0,8406 de log-loss frente a 0,8409 del Elo (diferencia práctica nula); en 2018+2022 el LSTM mejora 0,0074 de log-loss pero con `p_dm=0,817`; en 2026 el Elo obtiene 0,8541 frente a 0,8772 del LSTM y 66,35 % de acierto frente a 63,46 %.
- La app sigue usando `MODEL_DIR=lstm-2026-10-23` por defecto (`web/mundial/settings.py:19`), no el modelo evaluado el 25-oct (`lstm-2026-10-25`).
- La Etapa A está implementada en código, pero **no cumple aún su “Terminado cuando”** porque la validación móvil no guarda intervalos ni tabla de calibración, y porque README/notebook siguen con cifras antiguas.
- Las Etapas B, C, D (como etapa), E.2/E.3, F, G y H **no están iniciadas**. E.1 tiene el scraper escrito pero no ejecutado.
- La “fase 3 al 70 %” de la v2 es defendible en cuanto a código, pero **3.6 no es funcional**: `VALOR_NEUTRO` está a `None` y cualquier intento de desactivar una métrica devuelve error 400 en vez de neutralizarla.
- El README afirma “1930–2026” y “22 Mundiales”, pero los datos y la app cubren **1934–2026** (22 torneos contando 2026); 1930 existe en CSV suelto pero no entra en `partidos.csv`, en `load_data` ni en `agnos`.

> **Conclusión ejecutiva.** El problema ya no es “cómo hacer más profundo el LSTM”. El problema es que **la vara de medir de la Etapa A desbancó al modelo actual**. El siguiente paso no es entrenar más redes: es cerrar la verdad de la medición, construir modelos clásicos/tabulares que puedan ganarle al Elo, y decidir con evidencia si el producto final debe ser un LSTM, un ensamble o el propio Elo puro.

### 1.2 Lo que está sano

- **Pipeline de datos**: `etl/`, `load_data`, `construir_dataset`; dataset de 9.833 partidos; datos de 1934–2026.
- **Simulación**: todos los formatos históricos, incluido 2026 (48 equipos, 12 grupos, 8 mejores terceros, tabla oficial de 495 combinaciones). Tests de `FormatoTorneo` y de las llaves 1986/1990/1994/2026.
- **Motor actual**: `PredictorKeras` con Elo y esquema fijo de 14 métricas; app conectada a `cargar_predictor`; `MODEL_DIR` configurable.
- **Evaluación**: `evaluacion.py` con log-loss, Brier, RPS, calibración de 10 tramos, bootstrap, Diebold-Mariano y McNemar; `evaluar_modelos` con validación por origen móvil 2006/2010/2014.
- **Contrato de modelos**: `Predictor`, `Lote`, `REGISTRO`, `esquema.json`, rechazo de modelos con métricas o versión distintas.
- **Pruebas**: 41 tests en verde; casos calculados a mano para A.1; tests de fuga temporal en dataset; tests de Elo y de formatos.
- **Repositorio**: ramas por feature, PRs fusionados, `main` limpio; `agents/` y `.spec-tmp/` sin seguimiento (no se deben tocar).

### 1.3 Lo que está a medias

| Elemento | Qué existe | Qué falta | Evidencia |
|---|---|---|---|
| Etapa A | Métricas, Elo, validación móvil, bootstrap, DM/McNemar, tests | Guardar calibración e intervalos también en validación; actualizar README/notebook; protocolo de “una sola mirada”; promover el modelo evaluado | `evaluacion.py`, `evaluar_modelos.py`, `resultados_baseline.json`, `README.md:233-242` |
| Fase 3.6 (selección de métricas) | Checkboxes en la plantilla, `drop_metricas`, `neutralizar` | `VALOR_NEUTRO` con valores reales; probar que desactivar una métrica no rompe la predicción | `modulo.py:37-38`, `modulo.py:652-669`, `views.py:98-100` |
| E.1 (datos reales FBref) | `etl/busca_metricas.py`, `etl/construye_proxy.py`, marcas `pk_real`/`sot_real` | Ejecutar la descarga real, reconstruir proxy, regenerar dataset y medir impacto | `README.md:259-260`, `proxy_desempegno` |
| 1930 | `data/partidos/partidos_1930.csv`, `data/partidos/fixture_1930.csv` | Cargarlo en `partidos.csv`/`load_data` y en `agnos`, o corregir el README | `load_data.py:14`, `modulo.py:19`, `views.py:11` |
| Refactor H.2 | Nada | Test de foto, paquete, logging, ruff | `modulo.py` 3.056 líneas, 21 `print` |
| Documentación de resultados | README y notebook con tabla antigua | Tabla con Elo puro, log-loss, RPS, intervalos y modelo realmente desplegado | `README.md:233-242`, `notebooks/evaluacion.ipynb` |

### 1.4 Lo que falta por completo

- **Etapa B**: `PredictorPoissonGLM`, `PredictorDixonColes`, `PredictorOrdinal`; `statsmodels` no está en `requirements.txt`; no hay clases en `predictores.py`.
- **Etapa C**: `PredictorLightGBM`, `PredictorRandomForest`; `lightgbm` no está en `requirements.txt`; `scikit-learn` solo está en `requirements-dev.txt`.
- **Etapa D como etapa**: no hay ensamble de semillas, cabeza de resultado, dropout/L2/BN, Keras Tuner, ni secuencias largas; el LSTM actual usa `k=1`.
- **E.2/E.3**: no existe `etl/busca_statsbomb.py`; no hay tabla fuente × dato × años × licencia; no hay ranking FIFA, valor de plantilla ni edad media.
- **Etapa F**: no existe `PredictorEnsamble`; el umbral de empate sigue siendo 0,01 elegido por acierto, no por RPS/acierto esperado; no hay publicación final.
- **Etapa G**: no existe `web/fases/montecarlo.py`; no hay ruta `/probabilidades/`; no hay cuadro de favoritos.
- **Etapa H**: no hay DRF, Swagger, Docker, CI, refactor, bot ni material de portafolio; la plantilla actual existe, pero no cumple H.4.

### 1.5 Estado etapa por etapa (A–H)

| Etapa | Qué existe en el repo | Qué está validado | Qué falta para “terminado” | Veredicto |
|---|---|---|---|---|
| **A** · Vara de probabilidades | `evaluacion.py` con log-loss/Brier/RPS/calibración; `PredictorElo`; validación móvil; bootstrap; DM/McNemar; tests | 41 tests; JSON con las tres líneas base + Elo; comparaciones pareadas | Guardar calibración e intervalos de validación; actualizar README/notebook; fijar protocolo de pruebas; promover modelo evaluado | `[P]` ≈85–90 %; **no `[T]`** |
| **B** · Modelos clásicos | Solo `PredictorElo` (A.2); no hay GLM, Dixon-Coles ni ordinal | — | Las tres clases, coeficientes en `esquema.json`, evaluación con la vara A, comparación contra Elo | `[—]` 0 % |
| **C** · Árboles | No hay LightGBM ni Random Forest; `scikit-learn` solo en dev | — | Dos variantes de LightGBM, RF de contraste, importancia de variables, evaluación | `[—]` 0 % |
| **D** · Red a fondo | `PredictorKeras` base; `k=1`; entrenamiento con semillas; sin ensamble, sin cabeza de resultado, sin tuning, sin secuencias largas | Entrenamiento/guardado/carga en tests | D.1–D.5 y decisión de si la red gana a B/C | `[P]` solo el motor base; etapa no iniciada |
| **E** · Datos nuevos | Proxy con `pk_real`/`sot_real`; 49.945 internacionales en BD sin usar al entrenar por defecto; `busca_metricas.py` escrito | Inventario parcial; tests de marcas reales | E.1 ejecutado y medido; E.2 StatsBomb; E.3 tabla de fuentes/licencias | `[P]` infraestructura; contenido no |
| **F** · Ensamble y elección | Nada de ensamble; umbral 0,01; no hay publicación final | — | F.1, F.2, F.3; modelo final en app; README con log-loss y margen vs Elo | `[—]` 0 % |
| **G** · 5 campeones | Nada | — | Montecarlo, ruta, vista, cuadro, tests, tiempo y variación entre semillas | `[—]` 0 % |
| **H** · Producto | Plantilla antigua; no API/Docker/CI/refactor | — | H.1–H.7, con H.5 opcional | `[—]` 0 %; H.2 puede empezar ya |

### 1.6 Estado fase por fase (v1)

| Fase | Estado | Evidencia | Matiz |
|---|---|---|---|
| **0** · Seguridad y arranque | `[H]` en repo; `[V]` parcial | `settings.py` usa `python-decouple`; `.env.example`; `.gitignore`; `load_data`; `check` limpio | La revocación del token y la rotación de contraseñas son manuales y **no auditables desde el repo** |
| **1** · README, dependencias, orden | `[H]` | README bilingüe, `requirements*.txt`, `etl/`, `notebooks/`, `experimentos/`, `web/`; un solo `busca_grupos.py` | No se re-ejecutó una instalación desde cero; el README contiene afirmaciones desactualizadas |
| **2** · Medir contra baselines | `[H]` en código; `[P]` en documentación | `evaluar_modelos`, `notebooks/evaluacion.ipynb`, `resultados_baseline.json` | El notebook y el README conservan la tabla de 53,9/63,8 y no incluyen Elo puro ni log-loss |
| **3** · Propuestas | `[P]` ≈70 % | Formatos 2026; métricas acumuladas; esquema fijo; reentrenamiento; 2026 en la app; UI de métricas | 3.6 no funciona por `VALOR_NEUTRO=None`; E.1 (scraper real) pendiente; 1930 fuera |
| **4** · Refactor modular | `[—]` 0 % | `modulo.py` 3.056 líneas, 21 `print`, sin paquete, sin test de foto | Es la fuente de H.2; no empezar sin snapshot |
| **5** · Docker, despliegue, UI, CI | `[—]` 0 % | No hay `Dockerfile`, `docker-compose.yml`, `.github/workflows`, `ruff` | La UI actual no es H.4 |
| **6** · LinkedIn y Fiverr | `[—]` 0 % | No hay material ni enlaces | Depende de H.7 y de la demo |

### 1.7 La tabla que contradice al README

Resultados del `experimentos/resultados_baseline.json` del **25-oct-2026** (no se re-ejecutó `evaluar_modelos` en esta auditoría; habría tardado ~1 h). `LSTM app` es el modelo con el umbral 0,01 que la app usa; `LSTM sin umbral` es la variante sin ese umbral.

| Modelo | 2018+2022: acierto | 2018+2022: log-loss | 2018+2022: RPS | 2026: acierto | 2026: log-loss | 2026: RPS |
|---|---:|---:|---:|---:|---:|---:|
| (a) Más puntos por partido | 46,09 % | 2,223 | 0,396 | 50,96 % | 2,025 | 0,360 |
| (b) Poisson con medias | 42,97 % | 1,109 | 0,252 | 47,12 % | 1,067 | 0,234 |
| (c) Siempre 1-1 | 21,88 % | 3,206 | 0,378 | 23,08 % | 3,157 | 0,372 |
| **Elo puro (A.2)** | **51,56 %** | **1,020** | **0,217** | **66,35 %** | **0,854** | **0,158** |
| LSTM app (`umbral=0,01`) | 51,56 % | 1,013 | 0,216 | 63,46 % | 0,877 | 0,167 |
| LSTM sin umbral | 53,91 % | 1,013 | 0,216 | — | — | — |

Lectura correcta:

- En **2018+2022**, el LSTM con umbral **empata** con el Elo en acierto (51,56 %) y mejora 0,0074 de log-loss, pero la comparación pareada no es significativa (`p_dm=0,817`, `p_mcnemar=1,0`). El 53,9 % del README corresponde a **LSTM sin umbral**, no al modelo que la app usa.
- En **2026**, el Elo puro obtiene mejor punto estimado que el LSTM en las tres métricas; la diferencia de log-loss es +0,0231 a favor del Elo, con `p_dm=0,451` (no significativa, pero **no hay evidencia de que el LSTM sea mejor**).
- En **validación móvil**, Elo 0,8409 y LSTM 0,8406: diferencia de 0,0003, ruido de semilla.
- El README dice “mejor línea base: 46,1 %”; eso ignora al Elo puro, que es la línea base que la propia Etapa A.2 define como referencia (“cualquier modelo que no le gane a este no aporta”).

### 1.8 Discrepancias documentales y de estado

| ID | Afirmación | Evidencia | Veredicto |
|---|---|---|---|
| H-01 | v2: “fases 0, 1 y 2 terminadas; 3 al 70 %” | Código y tests lo sostienen en general; 3.6 está bloqueado | **Aceptable con matices**: 0–2 terminadas en repo; 3 al ~70 %, no cerrada |
| H-02 | README: “el LSTM con Elo supera a las baselines” y 53,9 % / 63,8 % | JSON 25-oct: 51,56 % con umbral; 63,46 % en 2026; Elo 66,35 % | **Desactualizado y omite al Elo**; la frase “mejor baseline 46,1 %” es falsa al incluir Elo |
| H-03 | README: “el umbral 0,01 apenas cambia los resultados” | 2018+2022: 53,91 % sin umbral vs 51,56 % con umbral | **Falso**: cambia 2,35 puntos de acierto |
| H-04 | README: “1930–2026” y “22 Mundiales” | `partidos.csv` cubre 1934–2026; `agnos` no incluye 1930; `load_data` no carga 1930 | **Inconsistente**: o se carga 1930 o se corrige el README |
| H-05 | README: “una métrica descartada llega con `VALOR_NEUTRO`” | `VALOR_NEUTRO = {m: None}`; `validar_valores_neutros` lanza error | **No funcional**: 3.6 no se puede usar |
| H-06 | README: `MODEL_DIR=lstm-2026-10-23` | `settings.py:19` usa ese valor por defecto; el modelo evaluado es `lstm-2026-10-25` | **Modelo no promovido**: la app no usa el último modelo evaluado |
| H-07 | v2: LSTM 63,8 % en 2026 | JSON 63,46 %; esquema de `lstm-2026-10-23` 66,35 %; README 63,8 % | **Tres cifras públicas distintas**; falta trazabilidad |
| H-08 | v2: “con k=1 el LSTM funciona como red densa”; D.4 propone k=5–20 | `secuencias_desde_tabla` lanza `ValueError` si `k != 1` | **Limitación real de la app**; D.4 exige adaptar el estado de la simulación |
| H-09 | v1: `modulo_11.py` 2.451 líneas | `modulo.py` actual: 3.056 líneas | **Crecimiento y refactor pendiente** |
| H-10 | README: `evaluar_modelos` ~30 min | Docstring del comando: “alrededor de una hora” | **Menor**: la estimación no es estable |
| H-11 | v2: pruebas “se miran una sola vez al final de cada etapa” | Ya hay al menos tres cifras de 2026 y el README las publica desde octubre | **Protocolo incumplido o mal definido**; requiere decisión |
| H-12 | README: “all 21 World Cups” (roadmap) vs “all 22” (status) | 21 antes de 2026; 22 contando 2026; 1930 fuera | **Contradicción interna** del README |

### 1.9 Lo que la v1 aporta a la Etapa H (y que la v2 solo cita)

La v2 delega en la v1 el detalle de las fases 4, 5 y 6. Esta spec lo recupera, pero **no reabre** las fases 0–3 de la v1: la v2 las sustituye. Los detalles que sí se usan son:

| Bloque v1 | Detalle | Uso en esta spec |
|---|---|---|
| **Fase 4.1** | Test de “foto” de `consulta_general` para 2018 y 1986, guardado en JSON, que debe pasar antes y después de cada refactor | Precondición de H.2; R-12 y D-16 |
| **Fase 4.2** | Partir `modulo.py` en `web/prediccion/` con `datos.py`, `metricas.py`, `formatos.py`, `modelo.py` y `simulacion.py` | O-15 / H.2 |
| **Fase 4.3** | Migrar `CharField` a `IntegerField` para goles, puntos y año, con índice en `agno` | Opcional dentro de H.2; decisión D-03 |
| **Fase 4.4** | `logging` en vez de `print`, type hints y `ruff` | H.2; R-12 |
| **Fase 4.5** | Tests de reglas de negocio: tabla de grupos, desempates, mejores terceros 1986/2026, `calc_*` | H.2; R-12 |
| **Fase 5.1** | `Dockerfile` (Python 3.12, `tensorflow-cpu`, gunicorn, whitenoise) y `docker-compose.yml` con PostgreSQL; `migrate` y `load_data` al arrancar | H.3 |
| **Fase 5.2** | Despliegue en Render/Railway, PostgreSQL gestionado, `DEBUG=False`, `ALLOWED_HOSTS`, TFLite/ONNX si TF no cabe | H.3; D-18 |
| **Fase 5.3** | Rediseño de la plantilla: selector, cuadro eliminatorio visual, probabilidades por partido y modo “¿qué hubiera pasado si…?” | H.4 |
| **Fase 5.4** | DRF + Swagger (la v2 lo hace obligatorio, no opcional) | H.1 |
| **Fase 5.5** | GitHub Actions con `ruff` y `pytest`; badge en README | H.6 |
| **Fase 5.6** | GIF de 15 s y 3 capturas limpias para el portafolio | H.7 |
| **Fase 6.1–6.4** | Titular y “Acerca de” de LinkedIn, 5 posts, 3 gigs de Fiverr, checklist de publicación | H.7 |

> **Aviso de obsolescencia:** la v1 habla de `modulo_11.py` (2.451 líneas) y de scripts por Mundial; eso ya no existe. El módulo actual es `modulo.py` (3.056 líneas) y el scraper está unificado. Se conserva el **requisito**, no la ruta histórica.

---

## 2. Objetivos priorizados

### 2.1 Objetivo general

> **Objetivo general.** Que cualquier usuario pueda elegir un Mundial y una fase y obtener, en la web o en una API, **probabilidades de resultado y de campeón calibradas, reproducibles y mejores que una línea base Elo pura**, con una demo pública y un README que lo demuestre con números, intervalos y límites explícitos.

Ese objetivo tiene tres dimensiones que no se deben confundir:

1. **Predicción (Etapas A–G):** que las probabilidades sean buenas, estén calibradas y superen al Elo puro con evidencia pareada.
2. **Producto/portafolio (Etapa H):** que el motor se pueda consumir por API, web o bot, esté desplegado y sea presentable.
3. **Calidad interna (transversal):** que el repositorio sea reproducible, testeado y mantenible; sin esto, ni la predicción ni el producto se sostienen.

### 2.2 Objetivo de predicción vs objetivo de producto

| Dimensión | Qué persigue | Cómo se mide | Qué **no** es |
|---|---|---|---|
| **Predicción A–G** | Probabilidades de local/empate/visitante y de campeón, calibradas, con incertidumbre | Log-loss, RPS, Brier, calibración, empates predichos, comparación pareada vs Elo | No es “acertar el marcador exacto” ni “tener una red más grande” |
| **Producto H** | API primero; web actual, sitio propio o bot; demo pública; README y material de portafolio | Endpoints funcionando, demo accesible, README con resultados, enlaces en GitHub/LinkedIn/Fiverr | No es solo “una plantilla bonita” sin modelo validado |
| **Calidad interna** | Reproducibilidad, tests, schema, logging, CI | `check`, `test fases`, `evaluar_modelos`, Docker/CI | No es refactor cosmético sin snapshot |

> **No confundir éxito de producto con éxito de predicción.** Una demo pública que usa un modelo que pierde contra el Elo no cumple el criterio de la v2; un buen modelo sin demo no cumple la Etapa H. Ambas cosas son necesarias para declarar el proyecto “presentable”.

### 2.3 Objetivos priorizados

El orden no es el orden de las letras: primero se arregla la verdad de la medición; después se construyen candidatos; al final se elige y se publica. Lo que desbloquea lo demás va primero.

| Prioridad | ID | Objetivo | Por qué ahora | Depende de | Terminado cuando |
|---|---|---|---|---|---|
| **P0** | O-01 | **Cerrar la Etapa A**: métricas, calibración e intervalos en validación y pruebas; tabla única; tests a mano | Sin una vara confiable no se puede comparar B, C, D ni F; hoy el README contradice al JSON | — | Tabla con log-loss, RPS, Brier, acierto y calibración de las 3 baselines + Elo, con intervalos, en validación móvil y en 2018+2022/2026; JSON y docs coherentes |
| **P0** | O-02 | **Fijar el Elo puro como referencia oficial** y persistirlo/compararlo en todos los experimentos | La Etapa A ya demostró que el Elo es la vara real; el LSTM no lo supera | O-01 | `PredictorElo` en `REGISTRO`, en `evaluar_modelos` y en la tabla de decisión; diferencia con intervalos y p-valor por modelo |
| **P0** | O-03 | **Protocolo de pruebas único**: 2018+2022 y 2026 congelados hasta F.3; registro de cada mirada | Ya hay tres cifras de 2026 y el test se ha usado para narrativa; si se sigue mirando, deja de ser test | O-01 | Documento de registro de miradas; `evaluar_modelos` con modo “solo validación” y modo “prueba final”; decisión del dueño |
| **P0** | O-04 | **Protocolo de fuga de datos**: auditoría y tests de que ninguna fila ve su resultado ni el futuro | Es la restricción más cara de reparar; el README la declara, pero hay que demostrarla con tests | — | Tests de fuga en dataset, Elo, proxy y simulación; revisión de `construir_dataset` y `contexto` |
| **P1** | O-05 | **Etapa B**: Poisson GLM, Dixon-Coles y ordinal | Son rápidos, explicables y los que más probablemente le ganen al Elo y arreglen los empates | O-01/O-02 | Tres clases en `REGISTRO`, coeficientes en `esquema.json`, evaluación con la vara A y comparación vs Elo |
| **P1** | O-06 | **Etapa C**: LightGBM y Random Forest | En datos tabulares suelen igualar o superar a las redes; dan importancia de variables | O-01/O-02 | Dos variantes de LightGBM, RF de contraste, evaluación y ranking de variables |
| **P1** | O-07 | **Etapa D**: ensamble de semillas, cabeza de resultado, regularización/tuning, secuencias y calibración | Es la ruta para exprimir la red, pero solo tiene sentido si B/C dejan un hueco | O-01/O-02 | Mejor combinación por validación móvil; respuesta explícita a “¿gana a Dixon-Coles y LightGBM?” |
| **P1** | O-08 | **E.1**: ejecutar FBref real, reconstruir proxy, regenerar dataset y medir | Es el dato que más directamente mejora las 14 métricas; el scraper ya existe | O-01 | Penales 1998+ y tiros 2018+ reales; `pk_real`/`sot_real`; comparación antes/después |
| **P2** | O-09 | **E.2/E.3**: StatsBomb, ranking FIFA, valor de plantilla, edad media; tabla de licencias | Puede aportar, pero es lo más caro y lo más dependiente de licencias | O-01/O-08 | Cada fuente medida; tabla fuente × dato × años × licencia; entra solo si mejora validación |
| **P1** | O-10 | **F.1**: ensamble de probabilidades y goles esperados con pesos en validación | Promediar modelos distintos suele superar al mejor individual | O-05/O-06/O-07 | `PredictorEnsamble` en `REGISTRO`, pesos en `esquema.json`, evaluación |
| **P1** | O-11 | **F.2**: reemplazar el umbral de empate por la decisión que maximiza acierto esperado o minimiza RPS | Hoy el 0,01 empeora el acierto y predice 3/28 y 1/24 empates | O-01/O-10 | Regla de decisión documentada; empates predichos vs ~22 % real; comparación |
| **P1** | O-12 | **F.3**: elección final, prueba única, modelo en app, README/notebook/`MODEL_DIR` | Es el cierre de la predicción y la frase vendible | O-10/O-11 | App usa el mejor modelo; README con log-loss y acierto vs Elo con intervalos; una sola mirada final |
| **P2** | O-13 | **G**: cuadro de los 5 campeones más probables | Es lo más vendible y no bloquea F; puede ir con el modelo actual | O-12 (o antes, con el modelo actual) | `montecarlo.py`, `/probabilidades/`, cuadro en la página, tests, tiempo y variación entre semillas |
| **P1** | O-14 | **H.1**: API DRF `/api/prediccion/` y `/api/probabilidades/` con Swagger | Es la decisión nueva de la v2: API primero para web, sitio y bot | O-12 (para el modelo final); puede empezar con el actual | Endpoints documentados, mismos parámetros que “Predecir”, tests |
| **P0/P1** | O-15 | **H.2**: partir `modulo.py` con test de foto, logging y ruff | 3.056 líneas y 21 `print`; sin snapshot, el refactor es riesgo puro; puede empezar ya | — | Paquete `web/prediccion/` (datos, métricas, formatos, modelo, simulación), test de foto 2018/1986 verde, sin `print` de depuración, ruff limpio |
| **P2** | O-16 | **H.3/H.4/H.6**: Docker, despliegue, UI, CI | Convierte el proyecto en demo pública y portafolio | O-14/O-15 | Demo pública; UI H.4; CI verde con badge |
| **P2** | O-17 | **H.7**: LinkedIn/Fiverr | Es el objetivo de portafolio de la v2 | O-16 | Enlace en GitHub, LinkedIn y Fiverr; al menos un gig y el post 5 |

### 2.4 Imprescindible vs opcional

**Imprescindible para declarar la spec cumplida:**

- O-01, O-02, O-03, O-04 (verdad de la medición, Elo, protocolo de test y no fuga).
- Al menos **una** familia de modelos que supere al Elo con evidencia (B o C; D si gana).
- O-10/O-11/O-12 (ensamble/decisión de empates y elección final publicada).
- O-14/O-15 (API y refactor con snapshot) y la app sigue arrancando.
- H.7 si el objetivo es portafolio; la v2 lo considera vital.

**Recomendado, pero recortable si el tiempo aprieta:**

- O-05/O-06/O-07 en paralelo (B, C, D); cuanto más se pruebe, más probable es encontrar algo que bata al Elo.
- O-08 (E.1) porque es barato comparado con E.2/E.3 y ataca directamente las 14 métricas.
- O-13 (G) porque es muy vendible y desbloquea H.4/H.5.

**Opcional o explícitamente recortable:**

- B.4 (Elo afinado y Bradley-Terry): la propia v2 lo marca opcional.
- D.4/D.5 si la app no puede adoptar `k>1`; en ese caso, documentar la limitación y no fingir que se probó.
- E.2/E.3 si las licencias o el tiempo no alcanzan; no meter datos sin licencia.
- H.5 (bot de Telegram): la v2 lo marca opcional.
- H.3/H.4 pueden hacerse en versión mínima (Docker + una pantalla) si la fecha aprieta; H.1/H.2 no.

> **Decisión del dueño 🔴:** si el tiempo es de una sola persona, no se pueden ejecutar B, C y D en paralelo. La secuencia recomendada es **A → B → C → F → G → H**, dejando D y E.2/E.3 como “si hay tiempo”. La v2 asume paralelismo; el repositorio no tiene señales de un equipo grande.

### 2.5 Secuencia y presupuesto

Duraciones de la v2: A 1 semana; B 1–2; C 1; D 1–2; E continuo; F 1; G 1; H continuo. En solitario, la suma realista es **8–11 semanas**, no 6.

| Tramo | Trabajo | Duración realista | Resultado |
|---|---|---|---|
| Semana 1 | O-01/O-02/O-03/O-04 + arranque de O-15 (snapshot H.2) | 1 semana | Verdad de la medición y red de seguridad del refactor |
| Semanas 2–3 | O-05 (B) y O-06 (C) | 2–3 semanas | Dos familias de candidatos y comparación contra Elo |
| Semanas 3–4 | O-08 (E.1) en paralelo; O-07 (D) si hay hueco | 1–2 semanas | Datos reales y red mejorada |
| Semana 5 | O-10/O-11 (F.1/F.2) | 1 semana | Ensamble y regla de empates |
| Semana 6 | O-12 (F.3) + O-13 (G) | 1 semana | Modelo final, prueba única, cuadro de favoritos |
| Semanas 6–8 | O-14/O-16 (API, UI, Docker, CI) | 2–3 semanas | Demo pública |
| Semana 8+ | O-17 (portafolio) | continuo | Enlaces y material |

**Corte mínimo vendible:** A + B + F + H.1/H.2 + H.7 (5–6 semanas). **Corte recomendado:** A + B + C + F + G + H (8 semanas). **Completo:** A–H (10–12 semanas). Elegir uno es decisión del dueño 🔴.

---

## 3. Restricciones lógicas y de negocio

Estas reglas no son sugerencias: cualquier tarea del Analista que las viole debe rechazarse en revisión. Cada una cita su fundamento.

| ID | Restricción | Fundamento | Cómo se verifica |
|---|---|---|---|
| **R-01** | **Regla de oro:** ningún modelo o dato nuevo entra a la app si no gana en la validación móvil; las pruebas 2018+2022 y 2026 se miran una sola vez al final de cada etapa. **Interpretación operativa:** para no convertir la prueba en validación encubierta, las etapas A–E usan solo validación y la mirada confirmatoria única se hace en F.3 (ver D-20). | v2, “Cómo usar esta guía” (“Regla de oro”); v2 §A.3, §E, §F.3 | `resultados_baseline.json` con validación y prueba separadas; checklist de PR; registro de miradas |
| **R-02** | **Los tests no se usan para elegir hiperparámetros, features, umbrales ni modelos.** La selección se hace solo con validación móvil (2006/2010/2014), y la prueba es confirmatoria. | v2 §A.3, §D.3, §F.3 | `evaluar_modelos` debe poder correr “solo validación”; PR que use métricas de test para decidir se rechaza |
| **R-03** | **No fuga de datos (leakage):** cada fila de entrenamiento solo contiene información anterior al partido; el Elo se calcula antes del partido; el histórico de métricas abstractas solo suma Mundiales anteriores; 2026 no entra en el entrenamiento que predice 2026; la calibración se ajusta en validación, nunca en test. | v1, Fase 3 (tabla de veredictos: “cuidado con la fuga de datos”); v2, “Datos disponibles”; README:111 | Tests de `dataset.py` (métricas previas, histórico anterior); revisión de `contexto.calcular_elo`; auditoría de `agregar_contexto`; test de que `hasta` excluye el futuro |
| **R-04** | **Esquema fijo de 14 métricas por equipo:** `PTS, PJ, PG, PP, PE, GF, GC, D, gkps, mds, mos, mms, rate_GC, rate_GF`, en ese orden. Nunca se añade ni se quita una columna de la entrada del modelo. | v2, tabla “14 métricas por equipo”; README:70 y 113; `modulo.py:22-28` | `ALL_METRICS` y `COLS_MODELO_POR_EQUIPO`; tests de forma; `cargar_predictor` rechaza esquemas distintos |
| **R-05** | **El esquema se versiona:** cambiar una fórmula de métrica obliga a subir `VERSION_METRICAS` y reentrenar todos los modelos; un modelo con otra versión no se carga. | `modulo.py:32-34`; `predictores.py:399-401` | Test `test_version_de_metricas`; `esquema.json` con `version_metricas` |
| **R-06** | **La neutralización no puede ser silenciosa:** si el usuario descarta una métrica base o rechaza una opcional, el modelo recibe el `VALOR_NEUTRO`; si ese valor no está definido, se rechaza la petición con un error claro o se oculta el control, nunca se rellena con 0. | README:113; `modulo.py:37-38`, `652-669`; v1, Fase 3.6 | `VALOR_NEUTRO` completo y testeado, o UI deshabilitada; test de petición con `drop_metricas` |
| **R-07** | **Contrato `Predictor`/`REGISTRO`/`esquema.json`:** todo modelo nuevo implementa `predecir(Lote) -> (n, 2)`, opcionalmente `probabilidades(Lote) -> (n, 3)`, `a_esquema()`, `desde_esquema()`; se registra en `REGISTRO`; se guarda con `guardar_predictor`; nadie carga un modelo por fuera de `cargar_predictor`. | v2 §A.2, §B.1; README, “Framework-agnostic predictions”; `predictores.py:61-73`, `358-404` | Tests de guardar/cargar; `REGISTRO` con la clave; `esquema.json` con `backend`, `metricas`, `version_metricas` |
| **R-08** | **Trazabilidad del modelo desplegado:** `MODEL_DIR` debe apuntar al modelo evaluado y documentado; el `esquema.json` debe incluir fecha, datos de entrenamiento, métricas y bloque de prueba; no se deja un modelo viejo como predeterminado. | `settings.py:17-19`; README:258; H-06/H-07 de este documento | `MODEL_DIR` en `.env.example`; `cargar_predictor`; comparación schema vs JSON vs README |
| **R-09** | **Reproducibilidad:** comandos documentados (`construir_dataset`, `evaluar_modelos`, `entrenar_predictor`), semillas 0/1/2, versiones fijadas, dataset y modelo con fecha; los resultados deben poder repetirse dentro de la variación de semilla. | v1, Fase 1.1 y 2.3; v2 §A.3 | `requirements*.txt`; `resultados_baseline.json` con fecha/dataset/partidos; media ± desviación de semillas |
| **R-10** | **Ramas y commits:** cada etapa en su rama (`exp/...`, `feat/...`, `refactor/...`, `deploy/...`), desde `main` actualizado; `git status` limpio antes de empezar; tests y app antes de commit; PR/merge, no commit directo a `main`. | v2, “Cómo usar esta guía”; v1, “Cómo usar esta guía” | `git log`/PR; checklist de rama; no hay commits de este documento (solo el archivo) |
| **R-11** | **La app debe seguir arrancando:** después de cada cambio, `python manage.py check`, `python manage.py test fases`, y una petición a `/` (o `runserver`) deben funcionar. Si un modelo nuevo no carga, la app debe fallar con un error claro, no predecir con pesos aleatorios. | v2, “Cómo usar” (“la app sigue arrancando”); v1, Fase 0.4 | `check` limpio; 41 tests actuales; GET / 200; test de carga del modelo |
| **R-12** | **Pruebas:** cada métrica, modelo y regla nueva lleva test; los tests actuales no se debilitan; los casos de probabilidad se calculan a mano; antes de refactorizar `modulo.py` se escribe un test de foto de `consulta_general` para 2018 y 1986. | v2 §A.1; v1 Fase 4.1 y 4.5 | `tests.py` (41); casos a mano de log-loss/Brier/RPS; snapshot 2018/1986 antes de H.2 |
| **R-13** | **Datos externos y licencias:** toda fuente nueva debe tener tabla “fuente × dato × años × licencia” y respetar `robots.txt` y condiciones de uso; no se asume que “público” signifique “libre”; se mantienen las marcas `pk_real`/`sot_real` para separar dato real de imputado. | v2 §E.3 (“con sus condiciones de uso”); v1 Fase 3.3; README, “Data collection” | Tabla de fuentes antes de incorporar; revisión de licencia; flags en `proxy_desempegno` |
| **R-14** | **Datos históricos inmutables:** no se editan los CSV históricos a mano sin auditoría y re-descarga; los cambios se hacen con script reproducible y se documentan (como `corregir_penales.py`). | README, “Data collection”; v1 Fase 3.3 | Diff de CSV; script de corrección; `--reemplazar` de `load_data` |
| **R-15** | **Dependencias:** las de la app van en `requirements.txt` con versión fijada; las de notebooks/scraping en `requirements-dev.txt`; no se añade una dependencia sin justificar tamaño, licencia y compatibilidad con Python 3.12/TensorFlow CPU. | v1 Fase 1.1; `requirements*.txt` | PR de requirements; `pip install -r` en entorno limpio; Docker |
| **R-16** | **Presupuesto y alcance:** A 1 semana, B 1–2, C 1, D 1–2, E continuo, F 1, G 1, H continuo. No se abre Champions/Libertadores, LLM, app móvil ni cuentas de usuario en esta spec. | v2, “Orden recomendado y duración”; v1 Fase 3 (“Aplazar la tarea 11”); README roadmap | Plan del Analista con corte; PR fuera de alcance se rechaza |
| **R-17** | **Selección y reporte:** la métrica primaria es log-loss; RPS es secundaria; el acierto no decide por sí solo; toda comparación usa bootstrap pareado y Diebold-Mariano, y McNemar para el acierto; se reporta media ± desviación de semillas. | v2 §A.3, §F.2, criterio final | Tabla de `evaluar_modelos`; JSON con `contra_elo`; intervalos |
| **R-18** | **Seguridad:** secretos en `.env` (nunca en git), rotación de la `SECRET_KEY` y de la contraseña de PostgreSQL, `.gitignore` para `.env`; no se commitean credenciales. | v1 Fase 0 | `.env` no trackeado; `.env.example`; `git ls-files` sin `.env` |
| **R-19** | **Rendimiento operativo:** no se entrena dentro de una petición HTTP; el Elo se cachea por proceso; el Montecarlo de G es numpy y no pasa por `consulta_general`; si `/probabilidades/` falla, la predicción actual sigue funcionando. | v2 §G (“milisegundos”, “si falla, solo se ve el de siempre”); `contexto.contexto_desde_bd` | Prueba de latencia; test de fallo aislado; revisión de vista |
| **R-20** | **Verdad documental:** README, notebook, `esquema.json` y `resultados_baseline.json` deben contar la misma historia; una cifra sin artefacto que la respalde no se publica. | v2 §F.3 (“actualiza README, notebooks/evaluacion.ipynb y MODEL_DIR”); README:233-242 | Revisión de coherencia antes de cada merge; script de comparación de cifras |
| **R-21** | **No tocar `agents/`:** contiene los roles del equipo y está sin seguimiento en git; esta spec no lo modifica. | Instrucción del usuario; `git status` | `agents/` intacto; solo se crea el documento de `docs/especificacion/` |

---

## 4. Criterios de éxito globales y medibles

### 4.1 Criterio global de la v2, operacionalizado

La v2 cierra con esta frase:

> “Criterio de éxito de la predicción: un modelo (o ensamble) que supere al Elo puro en log-loss y RPS en la validación móvil y en las dos pruebas, con la ventaja confirmada por la comparación pareada, que prediga empates con una frecuencia razonable y cuyas probabilidades estén calibradas.” (v2, “Orden recomendado y duración”)

Para que sea verificable, se desglosa así:

| ID | Criterio | Umbral propuesto | Estado actual (28-oct-2026) |
|---|---|---|---|
| **G-01** | **Log-loss** menor que el Elo puro en validación móvil (media 2006/2010/2014) y en las dos pruebas (2018+2022, 2026) | Diferencia < 0 con IC 95 % que excluya 0 y `p_dm < 0,05` | **No cumple**: validación 0,8406 vs 0,8409 (ruido); 2018+2022 −0,0074 con `p=0,817`; 2026 +0,0231 (peor) |
| **G-02** | **RPS** menor que el Elo puro en los mismos conjuntos | Diferencia < 0 con IC 95 % que excluya 0 | **No cumple**: 2018+2022 −0,0012 con `p=0,902`; 2026 +0,0092 (peor) |
| **G-03** | **Ventaja confirmada por comparación pareada** | Bootstrap pareado + Diebold-Mariano; McNemar para acierto | **No cumple**: ninguna comparación es significativa a favor del LSTM |
| **G-04** | **Calibración**: en la tabla de 10 tramos, la probabilidad predicha y la frecuencia real se parecen | ECE ≤ 0,03 y ningún tramo con `n ≥ 20` desviado > 0,07 (propuesta) | **No medido en validación**; en prueba hay tabla, pero muchas casillas vacías |
| **G-05** | **Empates predichos** con frecuencia razonable | Tasa predicha dentro de ±5 puntos de la observada (~22 % en Mundiales) | **No cumple**: 3/28 en 2018+2022 y 1/24 en 2026 (v2, “Punto de partida”); el umbral 0,01 no lo arregla |
| **G-06** | **Cobertura de la tabla**: 3 baselines + Elo + candidatos, con intervalos, en validación y pruebas | Tabla completa y JSON | **Parcial**: prueba sí; validación sin intervalos ni calibración guardada |
| **G-07** | **App y tests**: `check`, `test fases`, GET/POST funcionan; modelo carga | 100 % tests; sin errores de arranque | **Parcial**: 41 tests, check limpio, GET / 200; falta test de POST y de carga del modelo |
| **G-08** | **Reproducibilidad**: mismo comando, misma configuración, resultados dentro de la variación de semilla | Media ± desviación documentada | **Parcial**: JSON y semillas, pero sin hash de dataset/commit ni regeneración verificada |
| **G-09** | **No fuga**: auditoría y tests | 0 hallazgos | **Parcial**: tests de dataset; falta auditoría formal y test de simulación |
| **G-10** | **Documentación coherente**: README/notebook/JSON/modelo | 0 contradicciones | **No cumple**: H-02 a H-07 |
| **G-11** | **Producto H**: demo pública con API, README con resultados y enlaces | Demo accesible; enlaces | **No cumple**: H no iniciada |

> **Nota sobre el umbral de significación.** La v2 no fija `p < 0,05`; se propone aquí porque es el estándar y porque el tamaño de prueba (128 y 104 partidos) hace que diferencias pequeñas no sean detectables. Si el dueño prefiere un criterio más laxo, debe fijarlo **antes** de mirar resultados, no después. 🔴

### 4.2 “Terminado” para cada etapa

| Etapa | Criterio de terminado (además de R-01 a R-21) |
|---|---|
| **A** | `evaluacion.py` tiene log-loss, Brier, RPS y calibración con tests a mano; `PredictorElo` está en `REGISTRO` y en `evaluar_modelos`; la validación móvil 2006/2010/2014 y las dos pruebas tienen intervalos y calibración; el JSON guarda todo; README/notebook describen la tabla real; existe registro de miradas. |
| **B** | `PredictorPoissonGLM`, `PredictorDixonColes` y `PredictorOrdinal` están en `REGISTRO`, guardan coeficientes/estado en `esquema.json`, se evalúan con la vara A y se sabe **cuál gana al Elo y por cuánto**, con intervalos. |
| **C** | `PredictorLightGBM` (Poisson y multiclase) y `PredictorRandomForest` están registrados, evaluados y con importancia de variables (SHAP o equivalente); hay lista ordenada de variables que aportan. |
| **D** | Se conoce la mejor combinación de arquitectura, regularización y datos por log-loss en validación móvil; se probó ensamble de semillas, cabeza de resultado y calibración; se responde si la red gana o no a Dixon-Coles y LightGBM; si se usa `k>1`, la app arma el mismo estado que el entrenamiento. |
| **E** | E.1 ejecutado y medido; E.2/E.3 con tabla fuente × dato × años × licencia; cada fuente entra solo si mejora validación o queda documentada como rechazada. |
| **F** | `PredictorEnsamble` con pesos en `esquema.json`; regla de empates que maximiza acierto esperado o minimiza RPS; modelo final elegido con validación; **una sola** medición final en 2018+2022 y 2026; README, notebook y `MODEL_DIR` actualizados; app usa el ganador. |
| **G** | `montecarlo.py` con `simular_torneo(lambdas_por_pareja, formato, n=5000)`; ruta `/probabilidades/`; cuadro de 5 favoritos en la página; tests de un torneo de 4 equipos (probabilidades suman 100 %, favorito primero); se reporta tiempo y variación entre semillas. |
| **H** | H.1 API con Swagger; H.2 paquete con test de foto, logging y ruff; H.3 Docker + despliegue; H.4 UI; H.5 opcional; H.6 CI; H.7 enlaces y material. Demo pública que usa la API y README con resultados. |

### 4.3 Umbrales cuantitativos

| Criterio | Valor | Origen |
|---|---|---|
| Métrica primaria | **log-loss** | v2 §A.3, §F.3 |
| Métrica secundaria | **RPS** | v2 §A.1, §F.2 |
| Comparación | bootstrap pareado + Diebold-Mariano; McNemar para acierto | v2 §A.3 |
| Significación | `p < 0,05` y IC 95 % sin 0 (propuesta) | Decisión del dueño 🔴 |
| Calibración | ECE ≤ 0,03; tramos con `n ≥ 20` y desviación ≤ 0,07 (propuesta) | Derivado de v2 §A.1/D.5; decisión del dueño 🔴 |
| Empates | Tasa predicha dentro de ±5 pp de la observada (~22 %) | v2 §F.2; decisión del dueño 🔴 |
| Acierto | No retroceder más de 2 pp frente al Elo (guardrail, no métrica primaria) | Derivado; decisión del dueño 🔴 |
| Cobertura | 3 baselines + Elo + cada candidato, con intervalos en validación y pruebas | v2 §A.3 |
| Tests | 100 % en verde; nuevos tests por métrica/modelo/regla | v2 §A.1; v1 Fase 4.1 |
| Documentación | 0 contradicciones entre README, notebook, JSON y `esquema.json` | v2 §F.3; R-20 |

### 4.4 Definición de “hecho” para cualquier tarea o PR

1. Rama creada desde `main` actualizado; `git status` limpio antes de empezar.
2. Cambio pequeño y revisable; sin `print` de depuración; logging si aplica.
3. Test nuevo o actualizado que falla antes y pasa después; los 41 tests actuales siguen verdes.
4. Sin fuga de datos; sin tocar el esquema de 14 métricas sin decisión explícita.
5. Si es modelo: `Predictor` + `REGISTRO` + `esquema.json` + carga con `cargar_predictor`.
6. Si es dato: fuente, licencia, años, marca de dato real/imputado y efecto en validación.
7. `python manage.py check` limpio; `python manage.py test fases` verde; la app responde.
8. README/notebook/JSON actualizados si cambia una cifra o un modelo.
9. Commit con mensaje que explique el qué y el porqué; PR con la evidencia (comando, JSON, captura si aplica).
10. Si el PR no cumple R-01 a R-21, se rechaza aunque el código “funcione”.

---

## 5. Límites de alcance: qué NO se debe hacer en esta spec

**No hacer ahora (aunque sea tentador):**

- **No añadir una métrica nueva a la entrada del modelo** (por ejemplo xG como decimoquinta métrica) sin decisión explícita. xG entra como `extra` de contexto o como nueva `VERSION_METRICAS` con reentrenamiento completo; no se mete “por la puerta de atrás” en `ALL_METRICS`.
- **No empezar por D** (red más grande) antes de cerrar A y probar B/C. La evidencia actual dice que el Elo es fuerte; una red más grande no es la prioridad.
- **No tocar los tests 2018+2022/2026 para elegir hiperparámetros**, ni cambiar su composición, ni entrenar con ellos.
- **No reescribir `modulo.py` de una vez.** H.2 empieza con test de foto 2018/1986; sin snapshot, no hay refactor.
- **No implementar Champions/Libertadores, clubes, mujeres, torneos continentales** ni otras competiciones: la v1 aplaza la tarea 11 y esta spec es de Mundiales.
- **No implementar explicaciones en lenguaje natural (LLM)**: está en el roadmap del README, no en la v2; sería otro proyecto.
- **No implementar cuentas de usuario, pagos, apuestas, gamificación ni app móvil.**
- **No cambiar el frontend a otro framework** (React/Vue); H.4 se hace sobre la plantilla Django actual.
- **No desplegar Kubernetes ni infraestructura de pago** sin decisión del dueño; H.3 es Docker + Render/Railway en plan gratuito.
- **No prometer “predecir el Mundial”** ni vender certeza; se venden probabilidades con intervalos y límites.
- **No usar datos sin licencia clara** ni scrapear sin respetar `robots.txt`; no paralelizar crawling.
- **No editar los CSV históricos a mano** ni “arreglar” un resultado sin script reproducible y auditoría.
- **No borrar modelos o datasets** sin archivarlos; `data/modelos/` tiene modelos antiguos que sirven de comparación.
- **No commitear** `.env`, datasets generados, PDFs ni `agents/`; no modificar `agents/`.
- **No ampliar a 1930** sin decisión del dueño: hoy los datos y la app cubren 1934–2026; 1930 está en CSV suelto y fuera de `load_data`/`agnos`. 🔴
- **No usar el bot de Telegram (H.5)** como excusa para retrasar H.1/H.2; es opcional.
- **No cambiar el contrato `Predictor`/`REGISTRO`/`esquema.json`** para “simplificar”; la compatibilidad es lo que permite comparar modelos y desplegar sin romper la app.

---

## 6. Riesgos, ambigüedades y decisiones pendientes

| ID | Riesgo o ambigüedad | Evidencia | Impacto | Decisión requerida | ¿Dueño? |
|---|---|---|---|---|---|
| **D-01** | **El Elo puro no es superado de forma demostrable por el LSTM**; el criterio de éxito de la v2 no se cumple | JSON 25-oct: validación 0,8406 vs 0,8409; 2018+2022 −0,0074 con p=0,817; 2026 +0,0231 (Elo mejor) | Si no se supera, el proyecto no puede afirmar “mejor modelo”; la app podría quedarse con Elo | Aceptar que el modelo final sea Elo si ningún candidato lo supera; no forzar LSTM. Definir si el objetivo es “mejor que Elo” o “mejor que las baselines antiguas” | **Sí 🔴** |
| **D-02** | **Contaminación de los tests**: ya hay al menos tres cifras de 2026 y el test se usó en README/narrativa | README 63,8 %; JSON 63,46 %; schema 10-23 66,35 %; v2 “una sola vez” | Cualquier “mejora” futura en 2026 puede ser ruido o sobreajuste; el test deja de ser hold-out | Congelar 2018+2022 y 2026 hasta F.3; registrar cada mirada; usar solo validación para seleccionar; declarar el estado actual como “histórico, no virgen” | **Sí 🔴** |
| **D-03** | **Alcance de H.2**: la v2 cita la fase 4 de la v1, que incluye 4.1 snapshot, 4.2 división, 4.3 tipos, 4.4 logging/ruff, 4.5 tests de negocio | v2 §H.2; v1 Fase 4 | Si se interpreta “todo”, H.2 crece y compite con A–F; si se interpreta “solo dividir”, quedan deudas | Confirmar: H.2 incluye 4.1/4.2/4.4 y, al menos, 4.5; 4.3 (CharField→IntegerField) puede posponerse | **Sí 🔴** |
| **D-04** | **`k=1` en la app vs D.4 (k=5–20)**: `secuencias_desde_tabla` lanza `ValueError` si `k != 1` | `predictores.py:267-268`; v2 §D.4 | D.4 no se puede desplegar sin adaptar el estado de la simulación; se puede entrenar y no usar | Elegir: mantener `k=1` y documentarlo, o implementar el estado de secuencia en la app antes de D.4 | **Sí 🔴** |
| **D-05** | **Umbral de empate 0,01**: empeora el acierto y no arregla los empates; F.2 pide otra regla | JSON: 53,91 % sin umbral vs 51,56 % con umbral; v2 §F.2 (~22 % de empates) | La app predice menos empates de los reales y pierde acierto; el README dice que “apenas cambia” | Reemplazar por la decisión que maximice acierto esperado o minimice RPS con probabilidades calibradas; aceptar que cambien los resultados mostrados | **Sí 🔴** |
| **D-06** | **Criterio de calibración sin número**: la v2 pide “calibradas” pero no fija ECE ni tamaño de tramo | v2 §A.1/D.5 | Sin umbral, “calibrado” es opinable y se puede declarar terminado sin evidencia | Fijar ECE ≤ 0,03 y tramos con `n ≥ 20` (o el criterio que el dueño prefiera) antes de medir | **Sí 🔴** |
| **D-07** | **`VALOR_NEUTRO=None` y 3.6 no funcional**: cualquier `drop_metricas` devuelve 400 | `modulo.py:37-38`, `652-669`; `views.py:98-100` | La UI de métricas existe pero no se puede usar; el README la presenta como hecha | Rellenar `VALOR_NEUTRO` con medias del entrenamiento (sin fuga) o deshabilitar los controles hasta H.2; probar ambos caminos | **Sí 🔴** |
| **D-08** | **xG/StatsBomb vs esquema fijo de 14**: E.2 pide “métrica opcional”, pero el esquema no admite una 15.ª sin romper modelos | v2 §E.2; README:70/113; `ALL_METRICS` | Si se añade sin decisión, todos los modelos guardados dejan de cargar; si no se añade, E.2 no se puede cumplir tal cual | Elegir: xG como `extra` de contexto (sin tocar las 14), o `VERSION_METRICAS=3` + reentrenar todo; documentar la decisión | **Sí 🔴** |
| **D-09** | **1930 fuera de la app**: hay CSV y fixture, pero no está en `partidos.csv`, `load_data` ni `agnos` | `partidos.csv` 1934–2026; `load_data.py:14`; `modulo.py:19`; `views.py:11` | El README dice “1930–2026” y “22 Mundiales”; la app no puede simular 1930 | Cargar 1930 y añadirlo a `agnos` (verificar formato) o corregir el README a 1934–2026 | **Sí 🔴** |
| **D-10** | **Modelo desplegado vs modelo evaluado**: `MODEL_DIR=lstm-2026-10-23`, JSON 10-25; tres cifras de 2026 | `settings.py:19`; `data/modelos/*/esquema.json`; README:242 | La app no usa el modelo que el README/JSON describen; cualquier mejora puede quedar sin desplegar | Promover `lstm-2026-10-25` o reentrenar el modelo final en F.3; actualizar README/notebook/`MODEL_DIR` en el mismo PR | **Sí 🔴** |
| **D-11** | **Licencias de datos externos**: FBref, StatsBomb Open Data, ranking FIFA, Transfermarkt/valor de plantilla | v2 §E.2/E.3; README, “Data collection” | Un dato sin licencia puede obligar a retirar el proyecto o la demo; el coste de limpiar es alto | Hacer tabla fuente × dato × años × licencia antes de E.2/E.3; decidir qué fuente entra primero; atribuir | **Sí 🔴** |
| **D-12** | **E.1 puede no ejecutarse**: requiere navegador/Firefox y FBref bloquea descargas que no parecen navegador | `README.md:259-260`; `etl/busca_metricas.py` | Las métricas siguen imputadas; el modelo no aprovecha datos reales | Ejecutar E.1 antes de F o declarar formalmente que se queda con proxy imputado y no comparar “real vs imputado” | **Sí 🔴** |
| **D-13** | **Tamaño de prueba y multiplicidad**: 128 y 104 partidos; IC de ±9 pp; muchas comparaciones | JSON: intervalos de acierto [42,97; 60,16] en 2018+2022 | Se puede “descubrir” una mejora que sea ruido; el p-valor se infla con múltiples modelos | Pre-registrar métrica primaria, alpha y tamaño de efecto mínimo; corregir por comparaciones múltiples si se prueban muchos modelos | **Sí 🔴** |
| **D-14** | **Presupuesto real vs paralelismo de la v2**: B, C y D en paralelo asumen más de una persona | v2, “Orden recomendado y duración”; repo de un solo autor | El plan puede prometer 6 semanas y entregar 10; se recorta calidad | Elegir corte: mínimo (A+B+F+H), recomendado (A+B+C+F+G+H) o completo (A–H); fijar fecha y recortes | **Sí 🔴** |
| **D-15** | **Reproducibilidad de TensorFlow**: semillas fijadas, pero sin `deterministic`; resultados varían entre corridas | `predictor.entrenar` usa `keras.utils.set_random_seed`; `modulo.py` fija numpy/tf al importar | No se puede exigir bit-exactitud; comparaciones de 0,001 pueden ser ruido | Reportar media ± desviación de 3 semillas; no exigir reproducibilidad bit a bit; documentar la variación | **No** (técnico) |
| **D-16** | **Refactor sin snapshot**: `modulo.py` 3.056 líneas, 21 `print`, sin test de `consulta_general` | `tests.py` no tiene snapshot; v1 Fase 4.1 lo exige | Cualquier división puede cambiar resultados sin que nadie lo note | Escribir el test de foto 2018/1986 antes de tocar `modulo.py`; bloquear H.2 sin él | **No** (técnico, pero bloqueante) |
| **D-17** | **La spec no está versionada**: `*.pdf` está en `.gitignore`; los PDF viven solo en la raíz; `.spec-tmp/` no se versiona | `.gitignore` final; `git ls-files` no lista PDFs | Si se pierde el PDF, se pierde la fuente de verdad; el equipo puede trabajar con una copia obsoleta | Añadir la spec a `docs/` (convertida o PDF) en una rama de documentación; no en este PR | **Sí 🔴** |
| **D-18** | **Despliegue y memoria de TensorFlow**: el plan gratuito puede no soportar TF + modelo | v2 §H.3 (“si TensorFlow no cabe… el modelo ganador de B o C”) | El despliegue puede fallar al final, con la demo a medias | Decidir desde el inicio si el modelo final puede ser B/C sin TF; preparar Docker con `tensorflow-cpu` y probar memoria | **Sí 🔴** |
| **D-19** | **No hay test de integración**: 41 tests son `SimpleTestCase`; no hay test de `consulta_general` ni de POST | `tests.py`; `views.py` | Un cambio en la simulación o en el contrato puede romper la app sin que los tests lo detecten | Añadir smoke test GET/POST y test de carga del modelo; el snapshot de H.2 cubre la simulación | **No** (técnico) |
| **D-20** | **“Una sola mirada” vs iteración por etapas**: si cada etapa mira los tests, se miran 8 veces | v2, “Cómo usar” y §F.3 | El test se convierte en validación encubierta; las conclusiones se inflan | Definir “mirada” como evaluación confirmatoria en F.3; las etapas A–E solo usan validación; registrar cualquier mirada extraordinaria | **Sí 🔴** |

---

## 7. Trazabilidad

### 7.1 Objetivos → etapa → guía → evidencia

| Objetivo | Etapa | Sección de la guía | Evidencia actual | Criterio de aceptación |
|---|---|---|---|---|
| O-01 Cerrar A | A | v2 §A.1–A.3 (txt 119–142) | `evaluacion.py`, `evaluar_modelos.py`, `resultados_baseline.json`, `tests.py` | G-01 a G-06; tabla A completa |
| O-02 Elo puro | A | v2 §A.2 (txt 129–134) | `PredictorElo`, `REGISTRO`, `evaluar_modelos.probar` | Elo en todas las tablas; diferencia con IC |
| O-03 Protocolo de test | A/F | v2 “Cómo usar” (txt 17–18); §F.3 | JSON con tres cifras de 2026; README | Registro de miradas; modo solo validación |
| O-04 No fuga | Transversal | v1 Fase 3 (tabla); v2 “Datos” (txt 47–118) | Tests de `dataset.py`; `contexto.py`; `construir_dataset` | G-09; auditoría + tests |
| O-05 B | B | v2 §B.1–B.3 (txt 147–172) | No existe; `statsmodels` ausente | Tres modelos en `REGISTRO`; comparación vs Elo |
| O-06 C | C | v2 §C.1–C.2 (txt 177–190) | No existe; `lightgbm` ausente | Dos LightGBM + RF; importancia |
| O-07 D | D | v2 §D.1–D.5 (txt 191–220) | `PredictorKeras` base; `k=1` | Mejor configuración; respuesta vs B/C |
| O-08 E.1 | E | v2 §E.1 (txt 228–231) | `busca_metricas.py`, `construye_proxy.py`; README pendiente | Datos reales; comparación antes/después |
| O-09 E.2/E.3 | E | v2 §E.2–E.3 (txt 232–240) | No existe `busca_statsbomb.py`; no hay tabla de licencias | Fuentes medidas; tabla licencias |
| O-10 F.1 | F | v2 §F.1 (txt 245–248) | No existe `PredictorEnsamble` | Pesos en `esquema.json`; evaluación |
| O-11 F.2 | F | v2 §F.2 (txt 249–251) | `umbral_empate=0,01`; `mejor_umbral` por acierto | Regla por acierto esperado/RPS; empates ~22 % |
| O-12 F.3 | F | v2 §F.3 (txt 252–257) | README/notebook desactualizados; `MODEL_DIR=10-23` | Modelo final en app; README con log-loss y margen |
| O-13 G | G | v2 §G (txt 262–289) | No existe | Cuadro de 5; tests; tiempo/variación |
| O-14 H.1 | H | v2 §H.1; v1 §5.4 | No hay DRF/Swagger | Endpoints + documentación + tests |
| O-15 H.2 | H | v2 §H.2; v1 Fase 4 | `modulo.py` 3.056 líneas; 21 `print` | Paquete + snapshot + logging + ruff |
| O-16 H.3/H.4/H.6 | H | v2 §H.3/H.4/H.6; v1 §5.1–5.3/5.5 | No hay Docker/CI; UI antigua | Demo pública; UI; CI verde |
| O-17 H.7 | H | v2 §H.7; v1 Fase 6 | No hay material | Enlaces y gig/post |

### 7.2 Restricciones → etapa → guía → evidencia

| Restricción | Etapas | Sección de la guía | Evidencia / verificación |
|---|---|---|---|
| R-01 Regla de oro | A–H | v2 “Cómo usar” (17–18); §A.3/E/F.3 | JSON, checklist, registro |
| R-02 Tests no se usan para elegir | A–F | v2 §A.3, §D.3, §F.3 | `evaluar_modelos` separado |
| R-03 No fuga | A–H | v1 Fase 3; v2 datos (47–118); README:111 | Tests dataset, Elo, proxy, simulación |
| R-04 14 métricas | A–H | v2 datos (52–60); README:70/113 | `ALL_METRICS`, `COLS_MODELO_POR_EQUIPO` |
| R-05 Versión de métricas | A–H | `modulo.py:32-34`; `predictores.py:399-401` | `VERSION_METRICAS`, test |
| R-06 Neutralización | 3.6/A–H | README:113; `modulo.py:652-669`; v1 Fase 3.6 | `VALOR_NEUTRO`, test |
| R-07 Contrato de predictor | A–H | v2 §A.2/B.1; README | `Predictor`, `REGISTRO`, `esquema.json` |
| R-08 Modelo desplegado | F/H | README:258; `settings.py:19` | `MODEL_DIR`, schema, JSON |
| R-09 Reproducibilidad | A–H | v1 Fase 1.1/2.3; v2 §A.3 | requirements, JSON, semillas |
| R-10 Ramas/commits | A–H | v2 “Cómo usar”; v1 “Cómo usar” | Git log/PR |
| R-11 App arranca | A–H | v2 “Cómo usar”; v1 Fase 0.4 | check/test/GET |
| R-12 Tests | A–H | v2 §A.1; v1 Fase 4.1/4.5 | 41 tests; nuevos |
| R-13 Licencias | E/H | v2 §E.3; v1 Fase 3.3 | Tabla fuente × licencia |
| R-14 Datos inmutables | E | README “Data collection”; v1 3.3 | Script/diff |
| R-15 Dependencias | A–H | v1 Fase 1.1 | requirements |
| R-16 Presupuesto/alcance | A–H | v2 orden/duras; v1 3.6 | Plan/corte |
| R-17 Selección/reporte | A–F | v2 §A.3/F.2 | JSON/intervalos |
| R-18 Seguridad | 0/H | v1 Fase 0 | `.env`, `.gitignore` |
| R-19 Rendimiento | G/H | v2 §G; `contexto` | Latencia/test |
| R-20 Verdad documental | A–H | v2 §F.3; README | README/notebook/JSON |
| R-21 No tocar `agents/` | — | Instrucción del usuario | `git status` |

### 7.3 Decisiones que el Analista no puede tomar solo

Las filas 🔴 de la sección 6 son decisiones del dueño, no del Analista ni del Programador. El plan técnico debe presentarlas como “supuestos” y no como hechos. Las más urgentes antes de escribir código:

1. **D-01**: ¿el objetivo es “mejor que Elo” o “mejor que las baselines antiguas”? Si es lo primero, el LSTM actual no sirve como narrativa.
2. **D-02**: ¿se congela 2026? Si no, no hay forma de afirmar una mejora.
3. **D-03**: ¿qué incluye H.2 exactamente?
4. **D-04**: ¿se permite `k>1` en la app?
5. **D-07**: ¿se arregla `VALOR_NEUTRO` o se oculta la UI?
6. **D-08**: ¿xG como `extra` o nueva versión de métricas?
7. **D-09**: ¿1930 dentro o fuera?
8. **D-14**: ¿qué corte de alcance y fecha?

---

## 8. Anexos

### 8.1 Evidencia de los comandos ejecutados

| Comando | Resultado |
|---|---|
| `python manage.py check` (desde `web/`) | “System check identified no issues (0 silenced).” |
| `python manage.py test fases -v 1` | “Ran 41 tests in 9.544s — OK” |
| `Client(HTTP_HOST="localhost").get("/")` | HTTP 200; 40.763 bytes; contiene “Predecir” |
| Consulta ORM de conteos | `partidos=1051`, `internacionales=49945`, `proxy=524` |
| `git status --short` | Solo `.spec-tmp/` y `agents/` sin seguimiento; sin cambios en archivos del proyecto antes de este documento |
| `git log --oneline -15` | Último merge `1c725a9` (26-oct-2026, `exp/metricas-probabilidad`); modelo/resultados del 25-oct |
| `evaluar_modelos` | **No re-ejecutado** en esta auditoría (≈1 h); se usó el JSON del 25-oct como evidencia |

> No se lanzaron entrenamientos ni evaluaciones largas. La única escritura de esta tarea es este archivo.

### 8.2 Archivos revisados

- `README.md`
- `requirements.txt`, `requirements-dev.txt`, `.gitignore`
- `web/mundial/settings.py`, `web/mundial/urls.py`
- `web/fases/predictores.py`, `evaluacion.py`, `dataset.py`, `contexto.py`, `modulo.py`, `views.py`, `urls.py`, `tests.py`, `models.py`
- `web/fases/management/commands/evaluar_modelos.py`, `construir_dataset.py`, `entrenar_predictor.py`, `load_data.py`
- `web/fases/templates/prediccion.html` (estructura y llamadas fetch)
- `experimentos/resultados_baseline.json`
- `data/modelos/lstm-2026-10-23/esquema.json`, `data/modelos/lstm-2026-10-25/esquema.json`, `data/modelos/medias/esquema.json`
- `data/partidos/partidos.csv`, `data/partidos/fixtures.csv` (años y conteos)
- `notebooks/evaluacion.ipynb` (conteo de menciones y tabla)
- `agents/*.md` (solo lectura; no se modificó)
- PDFs v1/v2 y sus extracciones `.spec-tmp/*.txt`

### 8.3 Inventario de artefactos

| Artefacto | Fecha / tamaño | Nota |
|---|---|---|
| `experimentos/resultados_baseline.json` | 25-oct-2026; 134.854 bytes | 9.833 partidos; configuración elegida `k=1`, `extras=["elo"]`, `umbral=0,01` |
| `data/modelos/lstm-2026-10-23/` | 24-oct-2026; `modelo.keras` 205.595 bytes | Modelo que usa la app por defecto; `prueba` 2026 = 66,35 % |
| `data/modelos/lstm-2026-10-25/` | 26-oct-2026; `modelo.keras` 205.605 bytes | Modelo evaluado en el JSON; 2026 = 63,46 %; log-loss/RPS/calibración |
| `data/modelos/medias/` | 19-oct-2026 | Baseline de medias |
| `data/modelos/modelo-LSTM-(series_temporales)-16-12-2024-0.h5` | 2024 | Modelo antiguo de 2024; no usar |
| `data/datasets/partidos_features.csv` | 23-oct-2026; 3,4 MB | 9.833 filas; generado, no versionado |
| `data/datasets/partidos_features_internacionales.csv` | 18-oct-2026; 14,9 MB | Ciclo internacionales; no usado por defecto |
| `data/partidos/partidos.csv` | 1.051 filas | 1934–2026; sin 1930 |
| `data/partidos/internacionales.csv` | 3,99 MB | 49.945 partidos |

### 8.4 Glosario mínimo

- **Elo puro**: predictor que usa solo la diferencia de Elo (más ventaja de local) para probabilidades y goles esperados; es la línea base fuerte de A.2.
- **Log-loss**: entropía cruzada media de la probabilidad asignada al resultado que ocurrió; menor es mejor; 1,099 = repartir 1/3 a cada resultado.
- **RPS**: ranked probability score; como el Brier pero con probabilidades acumuladas en el orden local < empate < visitante; respeta que equivocarse por empate cuesta menos que por visitante.
- **Brier**: suma de errores cuadráticos sobre las tres probabilidades; menor es mejor.
- **Calibración**: que “70 %” signifique aproximadamente 7 de cada 10; se mide con la tabla de 10 tramos y, si se cuantifica, con ECE.
- **Leakage**: usar información posterior al partido o al ciclo para entrenar/predecir; en este proyecto es la restricción más importante.
- **Validación por origen móvil**: validar cada ciclo (2006/2010/2014) entrenando solo con lo anterior; evita elegir con el futuro.
- **Hold-out**: conjunto que se mira una sola vez al final; aquí 2018+2022 y 2026.
- **Ensamble**: combinar probabilidades/goles de varios modelos con pesos elegidos en validación.
- **`VALOR_NEUTRO`**: valor con el que se reemplaza una métrica apagada para no cambiar la forma de la entrada; hoy está a `None` y bloquea 3.6.
- **`esquema.json`**: contrato de cada modelo guardado (backend, métricas, versión, configuración, prueba); `cargar_predictor` lo verifica.

### 8.5 Checklist de decisiones del dueño

- [ ] **D-01** ¿Aceptar Elo como modelo final si ningún candidato lo supera?
- [ ] **D-02** ¿Congelar 2018+2022 y 2026 hasta F.3 y registrar miradas?
- [ ] **D-03** ¿Qué incluye H.2 (4.1/4.2/4.4/4.5; 4.3 opcional)?
- [ ] **D-04** ¿Mantener `k=1` o adaptar la app para `k>1`?
- [ ] **D-05** ¿Reemplazar el umbral 0,01 por una regla de acierto esperado/RPS?
- [ ] **D-06** ¿Fijar ECE ≤ 0,03 y tramos con `n ≥ 20`?
- [ ] **D-07** ¿Rellenar `VALOR_NEUTRO` o deshabilitar los controles de métricas?
- [ ] **D-08** ¿xG como `extra` o nueva `VERSION_METRICAS` con reentrenamiento?
- [ ] **D-09** ¿Incluir 1930 o corregir el README a 1934–2026?
- [ ] **D-10** ¿Promover `lstm-2026-10-25` o reentrenar el modelo final?
- [ ] **D-11** ¿Qué fuente externa entra primero y con qué licencia?
- [ ] **D-12** ¿Se ejecuta E.1 antes de F?
- [ ] **D-13** ¿Métrica primaria, alpha y tamaño de efecto mínimo?
- [ ] **D-14** ¿Corte mínimo, recomendado o completo, y con qué fecha?
- [ ] **D-17** ¿Versionar la spec en `docs/`?
- [ ] **D-18** ¿El despliegue puede usar un modelo B/C sin TensorFlow?

---

**Fin del documento.** Cualquier cambio de alcance, de esquema de métricas o de criterio de éxito debe volver al Delimitador de Reglas y Lógica antes de que el Analista lo convierta en tareas.
