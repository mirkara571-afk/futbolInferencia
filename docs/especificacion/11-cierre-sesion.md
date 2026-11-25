# 11 · Cierre de sesión (17 nov – 25 nov 2026)

**Fecha:** 2026-11-25 · **Rama:** `main` · **Alcance:** arranque, robustez de la API, H.2, D.1/D.3/D.4/D.5, G, B.4, E.1/E.3, H.5 y docs.

Resumen de los pases de desarrollo local (sin deploy). Se cerraron todos los frentes técnicos y
de producto pendientes; al final la suite queda en **113 tests OK** y `ruff` limpio.

## Qué se hizo

### 1. Arranque y verificación en vivo
- Python 3.12, venv, PostgreSQL 18 (`bd_mundial`/`mundial`), `migrate` y `load_data` (idempotente).
- Suite: **107 tests OK** (91 → 94 → 106 → 107 con los tests nuevos).

### 2. Robustez de `/api/prediccion/` (fix del 500)
- Bug: un `recepcion` parcial dejaba grupos vacíos → `model.predict([])` → `UnboundLocalError` en Keras 3.
- Fix: guard de lote vacío en `PredictorKeras.predecir` + validación en `fase_de_grupos` (→ 400 con mensaje claro).
- Test: `test_predecir_lote_vacio_no_llama_a_keras`.

### 3. H.2 (refactor) — avance grande
- `print` → `logging` en `simulacion.py`.
- Quitados los `import *` de `views.py` y `simulacion.py` (imports explícitos por módulo fuente).
- Imports muertos eliminados; las re-exportaciones de `modulo.py` marcadas con `# noqa`.
- **`ruff check web/fases` → All checks passed!** (antes 95 errores).

### 4. D.1 — ensamble de semillas (cerrado)
- `medir_semillas`: el ensamble de 5 LSTM mejora a la LSTM única (log-loss 0.8116 → 0.8044, significativo).
- Decisión: **no se promueve** (Elo/GLM siguen de referencia).

### 5. G — Montecarlo ampliado
- 1954 (aproximado: playoffs contados como partidos de grupo) y las fases knockout de 2026 en `montecarlo.py`.
- Verificado en vivo: 1954/Fase de Grupos, 2026/octavos y 2026/dieciseisavos → 200.
- Test: `test_1954_y_2026_knockout_registrados`.

### 6. B.4 — Bradley-Terry
- `PredictorBradleyTerry` (extensión de Davidson con empates, gradiente analítico) añadido y registrado.
- Test: `test_bradley_terry_contrato_y_roundtrip`.

### 7. D.3 — Keras Tuner (arnés)
- `entrenar_con_tuner` implementado (RandomSearch sobre `unidades`, `dim_pais` y `lr`) y validado con un
  barrido de humo (1 trial, 1 época): devuelve el mejor `PredictorKeras`.
- Nota: keras-tuner 1.4.8 + Keras 3 no siempre escribe `checkpoint.weights.h5`; hay fallback a
  reconstruir la arquitectura ganadora. `tensorboard` es opcional (aviso no fatal).

### 8. H.4 — UI (verificado)
- La plantilla ya trae `pintarCampeones` y `pintarLlave` (cuadro eliminatorio visual con
  probabilidades por partido). No requiere código nuevo.

### 9. E.1 / E.3 — estado
- **E.1**: los HTML de FBref (guardados a mano) ya se parsearon a `data/proxy_desempegno/fbref/fbref_*.csv`
  y la BD tiene los datos reales (2018/2022: `pk_real=1` y `sot_real=1`; 1998–2014: `pk_real=1`).
  Falta regenerar el dataset (está del 23/10, anterior al proxy del 15/11) y re-medir.
- **E.3**: `etl/busca_fifa_ranking.py` ya es ejecutable solo con el CSV de Kaggle
  (`python etl/busca_fifa_ranking.py <csv>` → `data/ranking_fifa.csv`), sin el `pais_id` de la app.

### 10. Docs
- README y `docs/05`: `/api/probabilidades/` 501 → 200; payload corregido; conteo de tests actualizado.
- `docs/09`: estado consolidado refrescado.

## Estado final

- `ruff` limpio, `check` sin issues, **107 tests OK**.
- API: `/api/prediccion/` → **400** ante payload parcial; `/api/probabilidades/` → **200**.
- Montecarlo cubre 1934–2026 (incluidos 1954 y el knockout de 2026).
- D.3 con arnés funcional; H.4 ya implementado.
- Sin tocar: deploy, Docker, CI, `.env`, datos, `git push`, `.spec-tmp/` ni `agents/`.

## Pase 19–25 nov (H.2 / D.4 / D.5 / E.3 / H.5)

- **H.2 cerrado**: `agno`, `score_0/1` y `Pts/PJ/PG/PE/PP/GF/GC` → `IntegerField` con índices
  (migraciones `0009` y `0010`); `formar_dataset_real` y los filtros ORM ajustados al tipo entero.
- **D.5**: calibración por temperatura en `PredictorKeras` (`temperatura` + `calibrar`).
- **E.3 cerrado**: ranking FIFA como extra (`ranking_por_agno` en dataset y app). Sobre el LSTM k=1
  mejora de forma **significativa** (log-loss −0.0086, IC 95 % [−0.0126, −0.0046], RPS −0.0022,
  Diebold-Mariano p < 0.001, bootstrap p = 0.001). Sobre el modelo completo k=8+Elo es redundante.
- **D.4 cerrado**: secuencias **k>1** en la app (historial partido a partido en simulación y
  Montecarlo) + flags `--ciclo` y `--configuracion` en `evaluar_modelos`. El LSTM **k=8 + Elo sobre
  internacionales** gana al Elo en el Mundial 2026 (acierto 67.3 % vs 66.3 %; log-loss 0.815 vs
  0.862; p_dm 0.048) y es el modelo en uso (`MODEL_DIR=lstm-2026-11-23`).
- **H.5**: bot de Telegram (`manage.py bot_telegram` con `/favoritos` y `/partido`), listo a falta del token.

## Pendiente

- **H.3**: deploy (Docker/gunicorn) — requiere el entorno Docker del usuario.
- **H.6**: verificar que el workflow de CI salió verde en GitHub Actions (se dispara con cada push).
- **H.7**: publicar en LinkedIn/Fiverr (textos en `docs/marketing/h7-linkedin-fiverr.md`).
- **H.5**: correr el bot con el token (`TELEGRAM_TOKEN` en `.env`).
