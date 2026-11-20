# E.3 — Fuentes de datos externos (ranking FIFA, valor de plantel, edad)

## Tabla fuente × dato × años × licencia

| Dato | Fuente (URL) | Años | Licencia/uso | Formato | Notas |
|---|---|---|---|---|---|
| Ranking FIFA (oficial) | https://www.fifa.com/fifa-world-ranking/men | dic-1992 → hoy | Sin API pública; JSON scrapeable sujeto a ToS | HTML/JSON | Rango+puntos por selección. |
| Ranking FIFA (espejo CSV) | https://www.kaggle.com/datasets/cashncarry/fifaworldranking | 1992-08 → 2024 | Gratis; ver ficha (típicamente permisiva) | CSV | rank_date, country, total_points, rank… El más limpio para ML. |
| Rating Elo | https://www.eloratings.net/ | 1872 → hoy | Gratis personal/no comercial, atribución, sin API | HTML; espejos GitHub/Kaggle | Elo por selección/fecha; alternativa al ranking FIFA. |
| Resultados internacionales (Elo propio) | https://github.com/martj42/international_results | 1872 → hoy | Repo permisivo (Kaggle CC0/atribución) | CSV | Backbone para Elo/features. |
| Valor plantilla (origen) | https://www.transfermarkt.com/ | Actual; histórico escaso | ToS prohíben scraping masivo; sin API | HTML | Valor total y edad media por plantel. |
| Valor plantilla (wrapper) | https://pypi.org/project/transfermarkt-api/ | Mayormente actual | No oficial, scraping (puede romperse) | Librería Python | market value sin escribir scraper. |
| Valor plantilla (gestionado) | https://apify.com/haketa/transfermarkt-scraper | Actual | Pago (free tier), token | API JSON (Apify) | Alternativa gestionada. |
| Edad media (computable) | https://github.com/jfjelstul/worldcup | Mundiales 1930 → 2022 | Gratis, requiere atribución (verificar) | CSV / paquete R | DOB por jugador → edad media por torneo. |

## Recomendación

Incorporar **primero el Ranking FIFA en CSV** (Kaggle `cashncarry/fifaworldranking`), opcionalmente con `eloratings.net` o `martj42/international_results` como respaldo/feature. Justificación: cobertura exacta 1992→hoy, licencia gratuita, un solo CSV, integración trivial (`pandas.read_csv` → normalizar equipo+fecha → bulk insert en Postgres).

**Valor de plantilla** y **edad media** en segunda fase: requieren scraping de Transfermarkt (ToS, sin API, histórico escaso) o cómputo desde DOB (`jfjelstul/worldcup`).

> Nota: las licencias exactas de los datasets de Kaggle deben confirmarse en sus fichas (cargan por JS); URLs, años y formato sí están verificados.
