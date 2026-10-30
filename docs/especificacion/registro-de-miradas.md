# Registro de miradas al examen (hold-out)

**Proyecto:** futbolInferencia · **Etapa:** A (vara de medición) · **Última actualización:** 2026-10-29

## Para qué sirve

La guía v2 pide que 2018+2022 y 2026 se miren **una sola vez** al final (Etapa F.3) y que la selección de modelos, métricas, hiperparámetros y umbrales se haga **solo con validación móvil**. Este registro existe para que ese acuerdo sea verificable: si alguien mira el examen antes, queda anotado y la cifra deja de ser "virgen".

## Regla operativa

1. **A–E usan solo validación móvil** (ciclos 2006, 2010 y 2014, cada uno entrenando con lo anterior). No se usan las métricas de 2018+2022 ni de 2026 para elegir nada.
2. **La mirada confirmatoria es una sola**, en F.3, después de fijar el modelo/ensamble y la regla de empates. Se registra antes de verla (fecha, modelo, métrica primaria, criterio de éxito).
3. **Cualquier mirada extraordinaria** (depuración, demo, prensa, comparación con un modelo nuevo) se anota aquí con fecha, motivo, quién y artefacto. No se usa para reoptimizar y luego presentar el resultado como si fuera la primera mirada.
4. **`--solo-validacion`** ejecuta la validación y escribe `experimentos/resultados_validacion.json`; no toca `resultados_baseline.json`, no entrena la prueba y no cambia el modelo de la app.
5. La métrica primaria es **log-loss**; RPS es secundaria; el acierto no decide por sí solo.

## Miradas ya realizadas (histórico, no virgen)

| Fecha | Qué se miró | Resultado publicado | Artefacto | Estado |
|---|---|---|---|---|
| 2026-10-23/24 | 2026, modelo `lstm-2026-10-23` | 66,35 % de acierto en 2026 | `data/modelos/lstm-2026-10-23/esquema.json` | Histórico: no usar como prueba limpia |
| 2026-10-25 | 2018+2022 y 2026, LSTM con/sin umbral + baselines + Elo | 51,56 % con umbral y 63,46 % en 2026 (LSTM); Elo 51,56 % y 66,35 % | `experimentos/resultados_baseline.json` | Histórico: es la evidencia de la auditoría, no una mirada confirmatoria nueva |
| 2026-10-25 | README / guía v2 | 53,9 % (LSTM sin umbral) y 63,8 % en 2026 | `README.md`, PDF v2 | Histórico: cifras narrativas, no comparables entre sí |
| 2026-10-30 | Validación móvil completa, **sin prueba** | Elo 0,8409 vs LSTM 0,8406 de log-loss; acierto 62,4 % vs 61,6 %; ECE 0,011 vs 0,024 | `experimentos/resultados_validacion.json` | Correcta: no toca 2018/2022/2026; sirve para elegir |

> **2026-10-30:** se lanzó un `evaluar_modelos` completo y se detuvo **antes** de llegar a la prueba al comprobar que iba a mirar el hold-out; no se usó ningún resultado de test de esa corrida. La corrida válida de ese día es `--solo-validacion`.

> Las tres cifras de 2026 (66,35 / 63,46 / 63,8) no son tres modelos distintos necesariamente: son tres formas de contar el mismo periodo. Hasta F.3 no se debe afirmar "mejoramos en 2026" con ninguna de ellas.

## Próxima mirada permitida

| Campo | Valor |
|---|---|
| Cuándo | Etapa F.3, después de fijar el ganador por validación |
| Qué se medirá | El modelo/ensamble elegido y el Elo puro en 2018+2022 y 2026 |
| Métrica primaria | log-loss (con intervalo bootstrap y Diebold-Mariano) |
| Métrica secundaria | RPS; acierto como guardrail |
| Criterio | Ventaja en log-loss y RPS con IC 95 % sin 0 y comparación pareada; empates predichos dentro de ±5 pp del ~22 % real |
| Quién | Dueño del proyecto / orquestador |
| Artefacto | `experimentos/resultados_finales.json` (nuevo) y actualización de README/notebook |

## Cómo se registrará una mirada extraordinaria

Añadir una fila a la tabla anterior con: fecha, qué se miró (año y modelo), motivo, resultado observado, artefacto y si se usó para tomar alguna decisión. Si se usó para decidir, la mirada deja de ser confirmatoria y hay que decirlo en el README.
