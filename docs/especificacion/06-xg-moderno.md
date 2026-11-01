# Datos modernos y xG (E.2) — estado y plan

**Fecha:** 2026-11-01 · **Rama:** `feat/probabilidades-campeon`

## 1. Qué se descargó

`etl/busca_statsbomb.py` baja StatsBomb Open Data (competencia 43 = FIFA World Cup) y resume por partido:

- `xg_0/xg_1` — suma de `statsbomb_xg` por equipo;
- `tiros_0/tiros_1` — número de remates;
- `sot_0/sot_1` — tiros a puerta (outcome `Goal` o `Saved`);
- `score_0/score_1`, fecha y nombres de equipo (en inglés).

Resultado:

| Archivo | Partidos | Torneo |
|---|---|---|
| `data/partidos/statsbomb/43_3.csv` | 64 | 2018 |
| `data/partidos/statsbomb/43_106.csv` | 64 | 2022 |

Ejemplo (Brasil–Bélgica 2018): 1–2 con **2,61 xG vs 0,41 xG** — Brasil generó mucho más pero perdió. Es exactamente el tipo de señal que los goles no capturan.

## 2. Limitación importante (no es una crítica, es un hecho)

xG solo existe de forma homogénea en **2018 y 2022**. Son **128 partidos** y **dos torneos**. Por eso:

- Un xG “rolling” como feature tabular tiene **poca señal**: para 2022 solo hay 2018 como historia, y en 2018 no hay nada anterior.
- El xG **del partido** no se puede usar para predecir ese mismo partido (fuga). Sirve como **observación de calidad** a posteriori, no como input pre-partido.
- El camino correcto no es imputar xG de 1930 ni usarlo como columna más en el GLM: es un **modelo jerárquico** que trate el xG como una medición de la fuerza latente.

## 3. Dataset moderno (hecho)

`etl/unir_statsbomb.py` une StatsBomb a `partidos_features.csv` y escribe `data/datasets/partidos_features_xg.csv`:

- 9.833 filas (misma base), 63 columnas;
- `xg_0/xg_1`, `tiros_0/1`, `sot_0/1` y `xG_disponible`;
- xG unido en **64/64 partidos de 2018** y **64/64 de 2022** (solo fase mundial); el resto queda `xG_disponible=0`.

## 3.1 Diagnóstico: ¿el xG predice mejor?

Usar el xG del partido como `λ` (oráculo, no desplegable) en los 128 partidos de 2018+2022:

| Modelo | Log-loss | RPS | Acierto |
|---|---:|---:|---:|
| xG como λ | 1,0636 | 0,2142 | 52,3 % |
| Elo puro | 1,0201 | 0,2170 | 51,6 % |
| Poisson GLM | 1,0117 | 0,2147 | 53,1 % |

**Conclusión honesta:** el xG **solo** (como goles esperados) es **peor** que el Elo y el GLM para predecir el resultado. El xG captura calidad de ocasiones, pero no fuerza relativa ni empates. Su valor debe venir como **medición auxiliar en un modelo jerárquico**, no como `λ` directa. Además, con solo 128 partidos modernos, cualquier modelo que entrene con xG tiene muy poca muestra.

## 4. Cómo se usará

1. **Traducir nombres** de StatsBomb (inglés) a la BD (español) con `data/partidos/nombres_en_es.csv` + excepciones (`IR Iran`→Irán, `Korea Republic`→Corea del Sur, etc.).
2. **Unir** a `partidos_features.csv` por `(agno, home, away, marcador)`.
3. **Añadir** `xg_0/xg_1`, `tiros_0/1`, `sot_0/1` y el flag `xG_disponible` (1 solo 2018/2022).
4. **Prototipo jerárquico**: goles ~ Poisson(ataque+defensa+localía+época); xG ~ Normal(ataque+defensa+…, σ_xg) solo cuando existe; ataque/defensa con partial pooling y random walk. En partidos sin xG, solo entra la verosimilitud de goles.
5. **Backtest A/B/C** solo como diagnóstico del imputador, no como modelo final:
   - A: núcleo sin xG;
   - B: núcleo + xG (rolling u observación);
   - C: núcleo + xG imputado;
   con GLM/LightGBM, test 2022, entrenamiento 2018/historia.

## 4. Licencia

StatsBomb Open Data es gratuito para investigación **no comercial**. Antes de usarlo en un producto comercial hay que revisar la licencia y, si no aplica, optar por un proveedor con licencia (Opta, Stats Perform, Sportmonks, API-Football).

## 6. Pendiente

- ~~Traducción + unión de nombres~~ (hecho con `nombres_en_es.csv`).
- ~~Dataset moderno con `xg_0/xg_1` y `xG_disponible`~~ (hecho: `partidos_features_xg.csv`).
- Backtest A/B/C (diagnóstico del imputador).
- Prototipo jerárquico (partial pooling + random walk) con `statsmodels`/PyMC.
