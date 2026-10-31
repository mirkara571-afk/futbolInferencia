# Etapas C y F: árboles, ensamble y decisión final

**Proyecto:** futbolInferencia · **Fecha:** 2026-10-30 · **Rama:** `exp/etapa-a-medicion`
**Reproducir:** `python manage.py evaluar_c`, `python manage.py evaluar_f`, `python manage.py evaluar_final`

## 1. Qué se construyó

| Tarea | Modelo | Archivo | Estado |
|---|---|---|---|
| C-01 | `PredictorLightGBMPoisson` | `web/fases/modelos_arboles.py` | Registrado, probado |
| C-01 | `PredictorLightGBMMulticlase` | `web/fases/modelos_arboles.py` | Registrado, probado |
| C-02 | Importancia de variables | `experimentos/importancia_c.json` | Calculada |
| C-03 | `PredictorRandomForest` | `web/fases/modelos_arboles.py` | Registrado, probado |
| C-04 | Comparación contra Elo/GLM | `web/fases/management/commands/evaluar_c.py` | Ejecutada |
| F-01 | `PredictorEnsamble` | `web/fases/ensamble.py` | Registrado, probado |
| F-02 | `decision_rps` / `decision_acierto` | `web/fases/evaluacion.py` | Implementadas y probadas |
| F-03 | Mirada final al hold-out | `web/fases/management/commands/evaluar_final.py` | Ejecutada una vez |

## 2. Etapa C: resultados en validación móvil

| Modelo | Log-loss | RPS | Acierto | ECE |
|---|---:|---:|---:|---:|
| Poisson GLM | **0,8366** | **0,1607** | 62,5 % | 0,019 |
| LightGBM Poisson | 0,8382 | 0,1614 | **62,6 %** | 0,017 |
| Elo puro | 0,8409 | 0,1624 | 62,4 % | **0,015** |
| Random Forest | 0,8637 | 0,1670 | 60,4 % | 0,026 |
| LightGBM multiclase | 0,8946 | 0,1703 | 60,4 % | 0,055 |

LightGBM Poisson queda **muy cerca del GLM** y un poco por delante del Elo, pero sin una ventaja concluyente. La importancia de variables es inequívoca:

1. `elo_diff`
2. `rate_GC_rival`
3. `es_local`
4. `mos`
5. `gkps`
6. `rate_GF`

El resto (neutral, categoría, fase final) aporta mucho menos. Los modelos que predicen directamente el resultado (multiclase y Random Forest) son peores y están peor calibrados: con 14 métricas y ~6.000 partidos, la señal tabular no da para tanto.

## 3. Etapa F: ensamble y decisión

### Pesos

El ensamble combina Elo + Poisson GLM + LightGBM Poisson. Los pesos se ajustan en una **validación interna** (el ciclo anterior dentro del entrenamiento), nunca en la validación externa ni en la prueba.

| Pliegue | Elo | GLM | LightGBM |
|---|---:|---:|---:|
| 2006 | 0,000 | 0,616 | 0,384 |
| 2010 | 0,145 | 0,429 | 0,425 |
| 2014 | 0,172 | 0,405 | 0,422 |

Pesos finales usados en la prueba: **Elo 0,106 · GLM 0,484 · LightGBM 0,411**.

### Validación móvil

| Modelo | Log-loss | RPS | Acierto | ECE |
|---|---:|---:|---:|---:|
| **Ensamble** | **0,8327** | **0,1601** | **62,8 %** | 0,020 |
| Poisson GLM | 0,8366 | 0,1607 | 62,5 % | 0,019 |
| LightGBM Poisson | 0,8382 | 0,1614 | 62,6 % | 0,017 |
| Elo puro | 0,8409 | 0,1624 | 62,4 % | **0,015** |

La mejora del ensamble sobre el Elo es de **0,0082 de log-loss** y **0,0023 de RPS**. La comparación pareada da `p` de Diebold-Mariano 0,095 / 0,047 / 0,028: dos de tres pliegues por debajo de 0,05. Es la primera señal algo consistente, aunque modesta.

### La mirada final (2018+2022 y 2026)

| Modelo | 2018+2022 log-loss | Acierto | 2026 log-loss | Acierto |
|---|---:|---:|---:|---:|
| Elo puro | 1,0201 | 51,6 % | **0,8541** | **66,3 %** |
| Poisson GLM | 1,0117 | **53,1 %** | 0,8584 | 64,4 % |
| LightGBM Poisson | 1,0122 | 52,3 % | 0,8677 | 64,4 % |
| **Ensamble** | **1,0073** | 51,6 % | 0,8581 | 64,4 % |

- **2018+2022:** el ensamble tiene el mejor log-loss (1,0073), pero el acierto no mejora al Elo; la diferencia no es significativa (`p_dm=0,391`).
- **2026:** el Elo gana en log-loss, RPS y acierto; el ensamble queda a 0,004 y la diferencia no es significativa (`p_dm=0,754`).
- **Conclusión:** el ensamble no generaliza de forma consistente. **No se promueve**; `MODEL_DIR` sigue en el LSTM y el Elo sigue siendo la referencia.

### Decisión de empates

`decision_acierto` es el argmax: maximiza el acierto esperado, pero casi nunca predice empate (0 % en la validación). `decision_rps` minimiza el RPS esperado:

| Modelo | Empates argmax | Empates RPS | Acierto argmax | Acierto RPS |
|---|---:|---:|---:|---:|
| Elo | 0,0 % | 29,2 % | 62,4 % | 58,7 % |
| GLM | 0,0 % | 30,4 % | 62,5 % | 59,1 % |
| LightGBM | 0,0 % | 26,0 % | 62,6 % | 59,6 % |
| Ensamble | 0,0 % | 27,8 % | 62,8 % | 59,5 % |

La tasa real de empates ronda el 22 %. La regla RPS se acerca más, pero **cuesta 2–4 puntos de acierto**. No es una mejora gratuita: es un intercambio entre acertar más y representar mejor los empates. No se activó en la app.

## 4. Definiciones

### LightGBM

Es un modelo de **árboles de decisión potenciados** (*gradient boosting*): entrena muchos árboles pequeños, cada uno corrigiendo los errores de los anteriores. Se usa con dos objetivos:

- **Poisson:** predice goles esperados; una fila por equipo y partido, igual que el GLM.
- **Multiclase:** predice directamente local/empate/visitante; una fila por partido.

Ventajas: captura interacciones no lineales y da importancia de variables. Riesgo: con pocos datos y muchas variables, sobreajusta; por eso se compara siempre contra el Elo y el GLM.

### Random Forest

Muchos árboles entrenados en paralelo sobre muestras aleatorias; la predicción es el voto mayoritario. Es un contraste simple y robusto, pero en estos datos quedó por detrás del boosting.

### Ensamble

Combina varios modelos con pesos que suman 1. La idea es que se equivoquen en partidos distintos. Los pesos se ajustan minimizando el log-loss en una validación; no se eligen a ojo.

### Log-loss, RPS, ECE

- **Log-loss:** castigo por dar poca probabilidad a lo que pasó; menor es mejor.
- **RPS:** como el log-loss pero respeta el orden pierde < empata < gana; menor es mejor.
- **ECE:** cuánto se desvía la probabilidad dicha de la frecuencia real; `0,02` ≈ 2 puntos porcentuales.
- **Diebold-Mariano:** compara dos modelos en los mismos partidos; `p < 0,05` indica que la diferencia es difícil de explicar por azar, no que sea grande.

### Decisión RPS

Para cada acción (local, empate, visitante) se calcula el RPS esperado con las probabilidades del modelo y se elige la de menor valor. A diferencia del argmax, puede elegir empate cuando las probabilidades están parejas; por eso sube la tasa de empates, pero también puede bajar el acierto.

## 5. Nivel de efectividad

| Pregunta | Respuesta |
|---|---|
| ¿El ensamble es mejor que el Elo? | En validación sí, por 0,0082 de log-loss; en 2026 no |
| ¿Es una mejora grande? | No: ~1 % de log-loss y 0,4 puntos de acierto en validación |
| ¿Es significativa? | En validación, dos de tres pliegues con `p < 0,05`; en la prueba, no |
| ¿Se despliega? | **No.** No hay evidencia consistente; `MODEL_DIR` sigue en el LSTM |
| ¿La regla RPS resuelve los empates? | Sube los empates al 26–30 %, pero baja 2–4 puntos el acierto |
| ¿Qué variable manda? | Elo, seguido de la defensa del rival, la localía y las métricas de portería/ataque |

## 5.1 Estado y % restante

### Etapas de la spec v2

| Etapa | Hecho | Falta |
|---|---:|---:|
| A · Medición | 100 % | 0 % |
| B · Modelos clásicos | 100 % | 0 % |
| C · Árboles | 100 % | 0 % |
| D · Red a fondo | 10 % | 90 % |
| E · Datos nuevos | 10 % | 90 % |
| F · Ensamble y elección | 100 % | 0 % |
| G · 5 campeones | 0 % | 100 % |
| H · Producto | 30 % | 70 % |

**A–H: ~56 % hecho, ~44 % pendiente.** Por esfuerzo, lo que queda está sobre todo en H (UI, demo, portafolio), G (Montecarlo) y E.1/E.2 (datos reales); D.4/D.5 es opcional.

### Fases de la guía v1

| Fase | Hecho | Falta |
|---|---:|---:|
| 0 · Seguridad y arranque | 95 % | 5 % |
| 1 · README, dependencias y orden | 100 % | 0 % |
| 2 · Medir antes de cambiar | 100 % | 0 % |
| 3 · Propuestas / datos | 85 % | 15 % |
| 4 · Refactor modular | 0 % | 100 % |
| 5 · Docker, UI, CI y despliegue | 0 % | 100 % |
| 6 · Portafolio | 0 % | 100 % |

**Base acumulada (fases 0–3): ~90 %.**

## 6. Visión y siguientes pasos

La conclusión de C/F refuerza la de B: **el fútbol es ruidoso y el Elo es muy difícil de batir**. Las mejoras son pequeñas y no siempre se sostienen fuera de la validación. Por eso:

1. **Datos antes que modelos.** xG, tiros, posesión, plantilla y disponibilidad son la palanca más grande; sin eso, cambiar de modelo produce décimas.
2. **Jerarquía bayesiana / state-space.** Un modelo con *partial pooling* por selección y un Elo dinámico pueden capturar mejor la incertidumbre y la forma.
3. **Producto.** G (Montecarlo de campeones) y H (API/demo) dan más valor visible que perseguir 0,5 puntos de log-loss.
4. **No reutilizar la prueba.** La mirada de F.3 ya se hizo; cualquier modelo nuevo necesita una validación nueva y una mirada pre-registrada.

**Siguiente orden recomendado:** G (cuadro de favoritos) → H.1/H.2 (API y refactor con la foto de A-06) → E.1/E.2 si hay datos → D.4/D.5 solo si hay tiempo y datos.
