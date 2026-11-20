# 09 · Estado por fases (consolidado)

**Fecha:** 2026-11-18 · **Rama:** `main` · **Fuente:** guías v1/v2, `08-informe-impacto.md`, `11-cierre-sesion.md`, README y verificación en vivo (107 tests OK).

Resumen ejecutivo: la infraestructura, la medición y la API están maduras; la calidad
predictiva **no supera al Elo/GLM** de forma consistente, así que ningún modelo se promovió
y la app sigue con el LSTM (`lstm-2026-10-25`). La **Etapa G** está prácticamente completa
(Montecarlo 1934–2026, incluidos 1954 y el knockout de 2026), **D.1** quedó medido (sin
promoción), se añadió **Bradley-Terry (B.4)** y **H.2** avanzó mucho (print→logging, `import *`
retirado, `ruff` limpio).

---

## Guía v1 · fases 0–6

| Fase | Objetivo | Estado | Nota |
|---|---|---|---|
| 0 · Seguridad y arranque | Secretos a `.env`, `.gitignore`, arranque, `load_data` | 95 % | ✅ verificado en vivo (runserver + 107 tests OK) |
| 1 · README y orden | requirements, carpetas, scraper único, README | 100 % | — |
| 2 · Medir antes de cambiar | notebook, baselines, `resultados_baseline.json` | 100 % | — |
| 3 · Propuestas | formatos (2026), métricas opcionales, scraping | 90 % | E.1 real sigue bloqueado por sandbox |
| 4 · Refactor y tests | partir `modulo.py`, tipar, logging, ruff | ~40 % | print→logging ✅ · `import *` quitado ✅ · `ruff` limpio ✅ · `models.py` sin tipar |
| 5 · Docker, UI, CI | Dockerfile, deploy, UI, API, CI | ~40 % | API ✅ · G ✅ · UI parcial · deploy/CI sin ejecutar |
| 6 · Portafolio | LinkedIn y Fiverr | ~20 % | Textos listos; publicar es manual |

## Guía v2 · etapas A–H

| Etapa | Objetivo | Estado | Nota |
|---|---|---|---|
| A · Vara de medir | log-loss/RPS/Brier/ECE, Elo, validación móvil | 97 % | — |
| B · Clásicos | Poisson GLM, Dixon-Coles, ordinal, **Bradley-Terry** | 100 % | B.4 añadido |
| C · Árboles | LightGBM, Random Forest, importancia | 100 % | — |
| D · Exprimir la red | ensamble semillas, multitarea, tuner, calibración | ~55 % | D.1 medido (no promovido); D.3 con arnés; D.4/D.5 pendientes |
| E · Datos nuevos | FBref real, StatsBomb xG, plantel | 65 % | E.2 hecho; E.1 datos reales ya integrados (falta re-medir); E.3 script listo |
| F · Ensamble y elección | ensamble, decisión de empates, elección | 92 % | ensamble 0,8327; no se promueve |
| G · 5 campeones | Montecarlo + `/api/probabilidades/` + cuadro | ~95 % | ✅ 1934–2026 incl. knockout; 1954 aproximado |
| H · Producto | API, refactor, deploy, UI, bot, CI, portafolio | ~50 % | H.1 ✅ · H.2 avanzado · H.4 ✅ (llave visual) · D.3 ✅ |

**Totales:** v1 ~64 % · v2 ~76 % (estimación).

---

## Qué se hizo en el pase 2026-11-17/18 (local, sin deploy)

1. **Fix API**: `/api/prediccion/` ya no revienta con 500 ante un `recepcion` parcial:
   guard de lote vacío en `PredictorKeras.predecir` + validación en `fase_de_grupos` (→ 400).
2. **D.1 cerrado**: `medir_semillas` confirma que el ensamble de 5 LSTM mejora a la LSTM
   única (log-loss 0.8116 → 0.8044, significativo), pero no se promueve.
3. **G ampliado**: 1954 y las fases knockout de 2026 (dieciseisavos/octavos/…) en `montecarlo.py`.
4. **B.4**: `PredictorBradleyTerry` (extensión de Davidson) añadido y registrado.
5. **H.2**: `print`→`logging` en `simulacion.py`; `import *` retirado de `views.py`/`simulacion.py`;
   imports muertos eliminados y re-exportaciones marcadas → `ruff` **All checks passed!**.
6. **Docs**: corregido el estado de `/api/probabilidades/` (501→200) en README y `docs/05`.

Detalle en [`docs/especificacion/11-cierre-sesion.md`](11-cierre-sesion.md).

## Qué sigue

- **H.2**: terminar de partir `modulo.py`, tipar `models.py` (IntegerField + índice), type hints.
- **D.3/D.4/D.5**: Keras Tuner, secuencias k>1 (ciclo internacionales), calibración de la red.
- **E.1/E.3**: FBref real (fuera del sandbox) y ranking FIFA (descarga Kaggle).
- **H.4**: cuadro eliminatorio visual y modo "¿qué hubiera pasado si…?".
- **H.3/H.5/H.6/H.7**: deploy, bot, CI y publicación (fuera del alcance local).
