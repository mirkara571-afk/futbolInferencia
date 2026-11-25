# 09 · Estado por fases (consolidado)

**Fecha:** 2026-11-25 · **Rama:** `main` · **Fuente:** guías v1/v2, `08-informe-impacto.md`, `11-cierre-sesion.md`, README y verificación en vivo (**113 tests OK**, `ruff` limpio).

Resumen ejecutivo: la infraestructura, la medición, la API y la simulación están maduras. En el
último pase se cerraron **D.4 (secuencias k>1)**, **D.5 (calibración)**, **E.3 (ranking FIFA)** y
**H.2 (tipado de `models.py`)**. El ranking FIFA **mejora de forma estadísticamente significativa**
(log-loss −0.0086, IC 95 % [−0.0126, −0.0046], Diebold-Mariano p < 0.001 sobre tres pliegues).
El LSTM **k=8 con Elo sobre el ciclo internacionales** vence al Elo puro en el Mundial 2026
(acierto 67.3 % vs 66.3 %; log-loss 0.815 vs 0.862; p_dm 0.048) y **es el modelo de la app**
(`MODEL_DIR=lstm-2026-11-23`).

---

## Guía v1 · fases 0–6

| Fase | Objetivo | Estado | Nota |
|---|---|---|---|
| 0 · Seguridad y arranque | Secretos a `.env`, `.gitignore`, arranque, `load_data` | 100 % | verificado en vivo (runserver + 113 tests OK) |
| 1 · README y orden | requirements, carpetas, scraper único, README | 100 % | — |
| 2 · Medir antes de cambiar | notebook, baselines, `resultados_baseline.json` | 100 % | — |
| 3 · Propuestas | formatos (2026), métricas opcionales, scraping | 100 % | E.1 real y E.3 ranking integrados |
| 4 · Refactor y tests | partir `modulo.py`, tipar, logging, ruff | 100 % | `print`→`logging` ✅ · `import *` quitado ✅ · `ruff` limpio ✅ · `models.py` tipado (IntegerField + índices) ✅ |
| 5 · Docker, UI, CI | Dockerfile, deploy, UI, API, CI | ~60 % | API ✅ · G ✅ · UI ✅ · bot H.5 ✅ · **deploy/CI sin ejecutar** |
| 6 · Portafolio | LinkedIn y Fiverr | ~20 % | Textos listos; publicar es manual |

## Guía v2 · etapas A–H

| Etapa | Objetivo | Estado | Nota |
|---|---|---|---|
| A · Vara de medir | log-loss/RPS/Brier/ECE, Elo, validación móvil | 100 % | — |
| B · Clásicos | Poisson GLM, Dixon-Coles, ordinal, Bradley-Terry | 100 % | B.4 añadido |
| C · Árboles | LightGBM, Random Forest, importancia | 100 % | — |
| D · Exprimir la red | ensamble semillas, multitarea, tuner, calibración, secuencias | 100 % | D.1 medido · D.2 · D.3 · **D.4 k>1 (k=8)** · D.5 calibración |
| E · Datos nuevos | FBref real, StatsBomb xG, ranking | 100 % | E.1 real · E.2 proxy · **E.3 ranking (p<0.001)** |
| F · Ensamble y elección | ensamble, decisión de empates, elección | 95 % | ensamble 0,8327; no se promueve |
| G · 5 campeones | Montecarlo + `/api/probabilidades/` + cuadro | 95 % | ✅ 1934–2026 incl. knockout; 1954 aproximado |
| H · Producto | API, refactor, deploy, UI, bot, CI, portafolio | ~70 % | H.1 ✅ · H.2 ✅ · H.4 ✅ · **H.5 ✅ (bot)** · H.3/H.6/H.7 pendientes |

**Totales:** v1 ~80 % · **v2 ~92 %** (estimación).

---

## Qué se hizo en el pase 2026-11-19/25 (local, sin deploy)

1. **H.2 (tipado) cerrado**: `agno`, `score_0/1` y `Pts/PJ/PG/PE/PP/GF/GC` → `IntegerField`
   con índices (migraciones `0009` y `0010` aplicadas); `formar_dataset_real` y los filtros ORM
   ajustados al tipo entero.
2. **D.5**: calibración por temperatura en `PredictorKeras` (`temperatura` + `calibrar`).
3. **E.3 cerrado**: ranking FIFA como extra (`ranking_por_agno` en dataset y app). El ranking
   mejora el LSTM k=1 de forma **significativa** (log-loss −0.0086, RPS −0.0022, DM p < 0.001);
   sobre el modelo completo k=8+Elo es redundante (ya captura fuerza y forma).
4. **D.4 cerrado**: la app arma **secuencias k>1** con historial partido a partido (simulación y
   Montecarlo); `evaluar_modelos` gana `--ciclo` y `--configuracion`. El LSTM **k=8 + Elo sobre
   internacionales** gana al Elo en 2026 (log-loss 0.815 vs 0.862, p_dm 0.048) y es el modelo en uso.
5. **H.5**: bot de Telegram (`manage.py bot_telegram`, comandos `/favoritos` y `/partido`), listo
   a falta del token de `@BotFather`.
6. **Docs**: este archivo, README y `11-cierre-sesion.md` sincronizados con el estado real.

Detalle en [`docs/especificacion/11-cierre-sesion.md`](11-cierre-sesion.md).

## Qué sigue

- **H.3**: deploy (Docker/gunicorn) — requiere el entorno Docker del usuario.
- **H.6**: verificar que el workflow de CI salió verde en GitHub Actions (se dispara con cada push).
- **H.7**: publicar en LinkedIn/Fiverr (textos en `docs/marketing/h7-linkedin-fiverr.md`).
- **H.5**: correr el bot con el token (`TELEGRAM_TOKEN` en `.env`).
- **Opcional**: validar la significancia del k=8 con más semillas/pliegues.
