# Informe de impacto — calidad predictiva hacia 2030+

**Fecha:** 2026-11-01 · **Rama:** `feat/probabilidades-campeon` · **Alcance:** E.1/E.2, ablation, backtest xG y modelo jerárquico/state-space (D/E/F).

---

## 1. Resumen ejecutivo

Se ejecutó el plan de calidad predictiva priorizando el **futuro (2030+)**, con datos nuevos (xG) y un modelo jerárquico/state-space. **Resultado honesto: no se encontró un modelo nuevo que supere al Elo/GLM de forma consistente.** El xG real (rolling) apunta a ayudar, pero la muestra (128 partidos) es demasiado pequeña para confirmarlo; el xG imputado no aporta. El modelo jerárquico con pooling + random walk + xG queda como **prototipo funcional**, no como modelo a desplegar.

---

## 2. Qué se hizo y qué se aprendió

### 2.1 Datos (E.1/E.2)

| Tarea | Estado | Resultado |
|---|---|---|
| E.1 · FBref real (penales 1998+, SOT 2018+) | **Bloqueado por sandbox** | Firefox y red existen, pero Selenium no puede iniciar el navegador (caché del driver denegado). El código está listo; se ejecuta fuera de DSH. |
| E.2 · StatsBomb Open Data (xG, tiros, SOT) | **Hecho** | 128 partidos (2018 + 2022) descargados y unidos a `partidos_features_xg.csv` con `xg_0/xg_1`, `tiros_*/sot_*` y `xG_disponible`. |

**Diagnóstico xG (128 partidos 2018+2022):** usar el xG del partido como `λ` da log-loss 1,0636, peor que Elo (1,0201) y GLM (1,0117). El xG **solo** no predice mejor el resultado.

### 2.2 Ablation de variables (GLM y LightGBM Poisson)

`experimentos/resultados_ablacion.json`. Media de validación móvil (2006/2010/2014):

| Bloque apagado | GLM Δ log-loss | LightGBM Δ log-loss |
|---|---:|---:|
| sin `gkps` | −0,0001 | −0,0012 |
| sin `mos` | +0,0005 | −0,0005 |
| sin `gkps`+`mos` | +0,0006 | **−0,0020** |
| sin `rate_GF/rate_GC` | **+0,0049** | **+0,0077** |
| sin contexto (Elo/local/neutral/fase) | **+0,1847** | **+0,1440** |

**Conclusión:** `gkps` y `mos` no aportan en estos modelos (candidatas a retirar). `rate_GF/rate_GC_rival` y el contexto (sobre todo Elo) sí aportan. `mds`/`mms` no están en GLM/LightGBM, solo en el LSTM; su ablation se pospone (ver §5).

### 2.3 Backtest xG A/B/C (diagnóstico, n=64 por ciclo)

`experimentos/resultados_xg_backtest.json`. Entrenar 2018 → probar 2022:

| Brazo | LightGBM log-loss | GLM log-loss |
|---|---:|---:|
| A · núcleo | 1,0380 | 1,1132 |
| B · +xG real rolling | **1,0085** | 1,1078 |
| C · +xG imputado | 1,0319 | 1,0936 |
| O · oráculo (xG del partido, no desplegable) | 1,0183 | 1,1013 |

**Conclusión:** el xG **real** (rolling) ayuda direccionalmente en LightGBM, pero con `p_dm > 0,28` (sin potencia). El xG **imputado** (R²_log1p ≈ 0,18) no aporta suficiente. No hay señal para imputar xG histórico.

### 2.4 Modelo jerárquico/state-space (JH-01 a JH-04)

`web/fases/modelos_jerarquicos.py` (PredictorJerarquico): estados de ataque/defensa por selección y ciclo, partial pooling (priors Gaussianas) + random walk temporal + verosimilitud xG (log-normal) con omisión. MAP con `scipy.optimize` y gradiente analítico (~3 s, 4.994 parámetros).

**Validación móvil (media de 2006/2010/2014):**

| Modelo | Log-loss | RPS | Acierto | ECE |
|---|---:|---:|---:|---:|
| Elo puro | **0,8409** | 0,1624 | 62,4 % | 0,015 |
| Poisson GLM | **0,8366** | 0,1607 | 62,5 % | 0,019 |
| Jerárquico (solo goles) | 0,8526 | 0,1648 | 61,5 % | 0,022 |

**Split diagnóstico ≤2018 → 2022 (64 partidos):**

| Modelo | Log-loss | Acierto |
|---|---:|---:|
| Jerárquico + xG | **1,0220** | 50,0 % |
| Jerárquico sin xG | 1,0277 | 48,4 % |
| Elo puro | 1,0417 | 48,4 % |
| Poisson GLM | 1,0524 | 51,6 % |

**Conclusión:** en validación el jerárquico **pierde** contra Elo/GLM. En el split de 2022 el jerárquico+xG es nominalmente el mejor, pero **sin significancia** (n=64). El modelo jerárquico es un prototipo correcto (partial pooling + random walk + xG con omisión), no un ganador.

---

## 3. Estado por etapas (guía v2)

| Etapa | Antes | Ahora | Nota |
|---|---:|---:|---|
| A · Medición | 95 % | 97 % | Ablation y calibración documentadas |
| B · Clásicos | 100 % | 100 % | — |
| C · Árboles | 100 % | 100 % | — |
| D · Red a fondo | 10 % | 40 % | JH-02/JH-03 cubren el state-space; D.1–D.5 y D.4 siguen |
| E · Datos nuevos | 10 % | 60 % | E.2 hecho; E.1 bloqueado por sandbox; E.3 parcial |
| F · Ensamble/elección | 90 % | 92 % | Jerárquico evaluado, no promovido |
| G · 5 campeones | 0 % | 0 % | No es prioridad predictiva |
| H · Producto | 30 % | 30 % | No es prioridad predictiva |
| **Total A–H** | **~54 %** | **~65 %** | Subida por E y D |

## 4. Estado por fases (guía v1)

| Fase | Antes | Ahora |
|---|---:|---:|
| 0 · Seguridad y arranque | 95 % | 95 % |
| 1 · README y orden | 100 % | 100 % |
| 2 · Medir antes de cambiar | 100 % | 100 % |
| 3 · Propuestas / datos | 85 % | 90 % |
| 4 · Refactor modular | 10 % | 10 % |
| 5 · Docker, UI, CI, despliegue | 30 % | 30 % |
| 6 · Portafolio | 0 % | 0 % |
| **Total fases 0–6** | **~60 %** | **~61 %** |

---

## 5. Qué queda pendiente y recomendación

1. **E.1 real**: ejecutar `python etl\busca_metricas.py --todos` fuera de DSH y reconstruir el proxy (`--verificar` `pk_real/sot_real`).
2. **E.3 parcial**: ranking FIFA (1992+), valor de plantilla, lesiones (según licencia).
3. **Ablation `mds/mms` en LSTM** (segundo paso): pospuesta — `mds`/`mms` solo existen en el LSTM (que ya no es el mejor modelo) y derivan de `SOT` imputado, consistente con que `gkps`/`mos` no aporten en GLM/LightGBM. Se puede hacer neutralizando esas columnas en un pliegue, pero su valor marginal es bajo.
4. **Decisión de features**: retirar `gkps`/`mos` del modelo por defecto (el ablation lo respalda); mantener `rate_GF/rate_GC` y contexto.
5. **Datos modernos antes que más modelos**: si se consiguen más torneos con xG (Euro 2020/24, Copa 2024, AFCON 2023 de StatsBomb), re-evaluar el jerárquico+xG y el backtest B/C.
6. **G/H.2/H.4/H.7** siguen pendientes como trabajo de producto, no de predicción.

**Conclusión final:** la calidad predictiva no mejora por ahora con xG ni con el modelo jerárquico; el **Elo y el Poisson GLM siguen siendo la referencia**. El trabajo sí deja el pipeline de datos modernos, el ablation, el backtest y un prototipo jerárquico listos para re-evaluar cuando haya más datos.
