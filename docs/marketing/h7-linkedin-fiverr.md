# H.7 / Fase 6 · Portafolio: README de resultados, posts de LinkedIn y gigs de Fiverr

> Textos listos para publicar. Los números salen de `experimentos/resultados_*.json`
> y de `docs/especificacion/08-informe-impacto.md`. No hay enlaces a demo aún (la demo
> pública es H.3, fuera del alcance de este pase local).

---

## 1 · Frase de resultados para el README (Etapa F.3)

> En validación móvil (Mundiales 2006/2010/2014, entrenando siempre solo con lo anterior),
> el ensamble (Elo + Poisson GLM + LightGBM Poisson) logra un log-loss de **0,8327** frente
> al **0,8409** del Elo puro, con RPS 0,1601 vs 0,1624. Es una ventaja pequeña y no se
> promueve ningún modelo como ganador: el **Elo y el Poisson GLM siguen siendo la
> referencia** y la app usa el LSTM (`MODEL_DIR=lstm-2026-10-25`) hasta que una nueva
> validación justifique otro modelo.

---

## 2 · 5 posts de LinkedIn (150–250 palabras, en español, tono cercano)

### Post 1 · El gancho "¿qué hubiera pasado si…?"
**Título:** ¿Qué habría pasado si Argentina empataba con Arabia Saudita en 2022?

Construí una app que simula cualquier Mundial desde 1934, desde la fase que quieras, y te
deja cambiar un resultado para ver cómo cambia todo el torneo. No es un predictor de
resultados de apuestas: es un "¿qué hubiera pasado si…?" con 22 Mundiales y 22 formatos
distintos detrás.

Lo más difícil no fue el modelo (una LSTM que ve el torneo como una secuencia partido a
partido). Fue reunir los datos: cada Mundial cambió de reglas — eliminatoria directa en
1934, liguilla final en 1950, dos fases de grupos en 1974–82, mejores terceros en 1986–94,
y 48 equipos en 2026.

Medir antes de cambiar fue lo que me salvó: sin una vara clara no sabes si lo que añades
mejora o empeora. Hoy el modelo acierta ~52 % de resultados en 2018+2022, y lo comparo
contra baselines honestos. ¿Curioso? Te cuento más en el próximo post.

*Imagen propuesta: captura de la app con un marcador editado y el torneo resultante.*

---

### Post 2 · Lo difícil de reunir 22 Mundiales con formatos distintos
**Título:** 22 Mundiales, 22 conjuntos de reglas (y por qué casi me rindo)

Los Mundiales no se parecen entre sí: 1934 era eliminatoria directa; 1950 se definió con
una liguilla final; 1974 y 1978 tenían segunda fase de grupos; 1986–1994 elegían "mejores
terceros" con 15 combinaciones posibles; 2026 tiene 48 equipos y 495 combinaciones.

Mi primer intento fue un script de scraping por Mundial. Wikipedia cambió su HTML y varios
dejaron de funcionar sin avisar. La lección: no seleccionar por posición ni estilo, sino por
significado (una columna "Pts", un título "Grupo…"). Reescribí todo como un solo script
parametrizado y validé contra los datos que ya tenía.

Lo que más me costó entender es que el código que "varía por año" no es código: son datos.
Hoy eso es un registro de formatos (`FormatoTorneo`), no 20 ramas de lógica. Reunir datos
limpios y verificables sigue siendo la mitad del trabajo de cualquier modelo.

*Imagen propuesta: tabla con los formatos históricos y sus diferencias clave.*

---

### Post 3 · Lo que aprendí midiendo contra baselines
**Título:** Mi modelo de ML no le gana a una regresión de 3 líneas. Y está bien.

Medí mi LSTM contra tres baselines tontos: "gana el de más puntos", "Poisson con medias" y
"siempre 1-1". La sorpresa no fue el resultado —fue que un **Elo puro** (una sola variable,
~50.000 partidos internacionales) es casi imposible de batir.

El LSTM acierta 51,6 % en 2018+2022; el Elo, 51,6 % también. El Poisson GLM, 53,1 %. Con
intervalos y tests pareados (Diebold-Mariano, McNemar), la diferencia no es significativa.

Conclusión honesta: no promuevo ningún modelo como "ganador". El Elo y el GLM son la
referencia, y mi app sigue usando el LSTM hasta tener evidencia real de mejora. Medir antes
de presumir es lo que separa un proyecto serio de un post inflado. Los datos no mienten:
a veces la respuesta es "todavía no".

*Imagen propuesta: gráfico de barras con log-loss de los modelos vs Elo.*

---

### Post 4 · Antes y después de refactorizar un archivo de 2.400 líneas
**Título:** Un archivo de 2.400 líneas no es un proyecto, es una deuda

`modulo.py` llegó a tener ~2.400 líneas mezclando acceso a base de datos, métricas, formatos
de torneo y simulación. Funcionaba, pero tocarlo daba miedo.

Empecé el refactor con una **red de seguridad**: una "foto" de la simulación (2018 y 1986,
116 partidos) que falla si cambio un solo marcador. Con eso, mover código deja de ser
apostar. Extraje primero las reglas de formato a `formatos.py` y el snapshot siguió en verde.

La regla que me cambió la forma de trabajar: **git es mi carpeta de pruebas**. Si una rama
sale mal, se borra. Y ningún refactor entra sin que la foto y los 90+ tests pasen. El
refactor no es cosmética: es lo que te deja seguir avanzando sin romper lo que ya funciona.

*Imagen propuesta: captura del snapshot pasando antes y después del refactor.*

---

### Post 5 · Lanzamiento con enlace a la demo
**Título:** futbolInferencia: simulá cualquier Mundial, desde cualquier fase

Después de meses de datos y evaluación, la app está lista para probar: elegí un Mundial
(1934–2026), una fase, editá un resultado y mirá cómo cambia el resto del torneo.

Stack: Python 3.12, Django, PostgreSQL y TensorFlow/Keras, con la simulación respetando el
formato real de cada año (incluido 2026 con 48 equipos). La API expone la misma predicción
que la web, y el cuadro de "campeones más probables" simula el torneo miles de veces.

No prometo predicciones infalibles: prometo un motor honesto, medido contra baselines y con
la vara a la vista. El enlace a la demo y el código están en mi perfil. Probalo y decime
qué Mundial querés "reescribir".

*Imagen propuesta: GIF de 15 s del modo "¿qué hubiera pasado si…?".*

---

## 3 · 3 gigs de Fiverr (inglés)

### Gig A · Web scraping & data cleaning (Python)
**Title:** Scrape and clean sports data into production-ready CSVs and databases
**Description:** I scrape Wikipedia and FBref data and clean it into verified datasets,
exactly as I did for a project covering 22 FIFA World Cups (1934–2026). I validate every
scrape against a known source, keep raw and derived data separate, and respect robots.txt.
Deliverables include CSVs, a documented cleaning pipeline, and a reproducibility checklist.
**Packages:**
- Basic: one source, one table, cleaned CSV + validation report (2 days).
- Standard: up to 3 sources, deduplication, name normalization, SQLite/PostgreSQL load (5 days).
- Premium: full pipeline with tests, re-run script and docs (2 weeks).
**Tags:** web-scraping, data-cleaning, python, sports-data, pandas

### Gig B · Predictive model / sports data analysis
**Title:** Build and honestly evaluate a predictive model for sports data
**Description:** I build statistical and ML models (Poisson GLM, LightGBM, LSTM) and
evaluate them against strong baselines (e.g. Elo) with log-loss, RPS, calibration and
paired tests (Diebold-Mariano, McNemar). You get a model plus a truthful report of whether
it actually beats the baseline — no inflated accuracy claims.
**Packages:**
- Basic: one baseline + one model, validation metrics and a short report (3 days).
- Standard: temporal validation, bootstrap intervals, variable importance (1 week).
- Premium: ensemble + calibration + reproducible training script and docs (2 weeks).
**Tags:** machine-learning, sports-analytics, python, data-science, predictive-modeling

### Gig C · Django web app with dashboard
**Title:** Build a Django web app with a REST API and Swagger docs
**Description:** I build Django apps with a clean REST API (DRF + drf-spectacular), a
PostgreSQL backend and a simple UI. My case study is futbolInferencia, a World Cup
simulator serving the same prediction to the web and the API. Includes migrations, a
`load_data` command to rebuild the DB from CSVs, and a test suite.
**Packages:**
- Basic: Django project + models + one API endpoint + Swagger (3 days).
- Standard: full CRUD API, Postgres, data loader and 20+ tests (1 week).
- Premium: dashboard UI, Montecarlo-style feature, CI-ready and documented (2 weeks).
**Tags:** django, rest-api, postgresql, python, web-development

---

## 4 · Checklist de publicación (manual)

- [ ] Demo y repo en la sección Destacados de LinkedIn
- [ ] Portada de cada gig = captura de la interfaz, no código
- [ ] Vídeo corto (30–60 s) en la galería del gig
- [ ] Precios de entrada bajos hasta conseguir 3–5 reseñas
- [ ] Repo fijado (pinned) en GitHub con descripción y topics
