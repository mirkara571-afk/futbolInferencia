# 09 · Estado por fases (consolidado)

**Fecha:** 2026-11-04 · **Rama:** `main` · **Fuente:** guías v1/v2, `08-informe-impacto.md`, README y verificación en vivo.

Resumen ejecutivo: la infraestructura, la medición y la API están maduras; la calidad
predictiva **no supera al Elo/GLM** de forma consistente, así que ningún modelo se promovió
y la app sigue con el LSTM (`lstm-2026-10-25`). Este pase añadió la **Etapa G** (Montecarlo
de campeones), inició el refactor **H.2** y dejó listos los textos de **H.7**.

---

## Guía v1 · fases 0–6

| Fase | Objetivo | Estado (antes) | Estado (ahora) | Nota |
|---|---|---|---|---|
| 0 · Seguridad y arranque | Secretos a `.env`, `.gitignore`, arranque, `load_data` | 95 % | 95 % | ✅ verificado en vivo (runserver + 94 tests OK) |
| 1 · README y orden | requirements, carpetas, scraper único, README | 100 % | 100 % | — |
| 2 · Medir antes de cambiar | notebook, baselines, `resultados_baseline.json` | 100 % | 100 % | — |
| 3 · Propuestas | formatos (2026), métricas opcionales, scraping | 90 % | 90 % | E.1 real sigue bloqueado por sandbox |
| 4 · Refactor y tests | partir `modulo.py`, tipar, logging, ruff | 10 % | ~15 % | **H.2 iniciado:** `formatos.py` extraído (snapshot OK) |
| 5 · Docker, UI, CI | Dockerfile, deploy, rediseño UI, API, CI | 30 % | ~35 % | API ✅; UI: cuadro de favoritos añadido; deploy/CI sin ejecutar |
| 6 · Portafolio | LinkedIn y Fiverr | 0 % | ~20 % | Textos listos en `docs/marketing/h7-linkedin-fiverr.md` |

## Guía v2 · etapas A–H

| Etapa | Objetivo | Estado (antes) | Estado (ahora) | Nota |
|---|---|---|---|---|
| A · Vara de medir | log-loss/RPS/Brier/ECE, Elo, validación móvil | 97 % | 97 % | — |
| B · Clásicos | Poisson GLM, Dixon-Coles, ordinal | 100 % | 100 % | GLM 0,8366 vs Elo 0,8409 |
| C · Árboles | LightGBM, Random Forest, importancia | 100 % | 100 % | LightGBM 0,8382 |
| D · Exprimir la red | ensamble de semillas, multitarea, tuner, calibración | 40 % | 40 % | jerárquico cubre D.4/D.5 parcial |
| E · Datos nuevos | FBref real, StatsBomb xG, plantel | 60 % | 60 % | E.2 hecho (128 partidos); E.1 bloqueado |
| F · Ensamble y elección | ensamble, decisión de empates, elección | 92 % | 92 % | ensamble 0,8327; no se promueve |
| **G · 5 campeones** | Montecarlo + `/api/probabilidades/` + cuadro | **0 %** | **~85 %** | ✅ `montecarlo.py`, vista 200, cuadro en plantilla, tests |
| H · Producto | API, refactor, deploy, UI, bot, CI, portafolio | 30 % | ~40 % | H.1 ✅ · H.2 iniciado · H.4 parcial · H.7 textos |

**Totales:** v1 ~61 % → ~64 % · v2 ~65 % → ~70 % (estimación; la subida es por G y H.2/H.7).

---

## Qué se hizo en este pase (local, sin deploy)

1. **Etapa G**: `web/fases/montecarlo.py` (motor numpy puro: `simular_knockout`,
   `simular_grupo`, `probabilidades_campeon`); `POST /api/probabilidades/` deja de devolver
   501 y responde con los 5 favoritos; cuadro "Campeones más probables" en la plantilla;
   +3 tests. Alcance v1: fase de grupos de 1998–2022 y 2026.
2. **H.2 (inicio)**: reglas de formato extraídas a `web/fases/formatos.py` (`modulo.py`
   de 173 KB → 160 KB), verificadas con `snapshot_consulta --verificar` (OK).
3. **H.4 (parcial)**: cuadro de favoritos y tablas legibles en la plantilla.
4. **H.7 / Fase 6**: textos de LinkedIn y Fiverr en `docs/marketing/h7-linkedin-fiverr.md`.

## Qué sigue (después de este pase)

- **G**: extender el Montecarlo a los formatos históricos (1934–1994) y a fases knockout,
  y soportar reglamento de terceros personalizado.
- **H.2**: seguir partiendo `modulo.py` (datos, métricas, modelo, simulación), logging/ruff.
- **D.1–D.3/D.5**: ensamble de semillas, multitarea, Keras Tuner, calibración de la red.
- **H.4**: cuadro eliminatorio visual y probabilidades por partido.
- **H.3/H.5/H.6**: deploy, bot de Telegram y CI (fuera de este pase por reglas/entorno).
