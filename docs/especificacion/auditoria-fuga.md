# Auditoría de fuga de datos (leakage)

**Proyecto:** futbolInferencia · **Etapa:** A (A-05) · **Fecha:** 2026-10-29

## Qué se revisó

La regla es: **cada fila de entrenamiento y cada predicción solo pueden usar información anterior al partido**. Se revisaron los cuatro caminos por los que el futuro podría colarse:

| Mecanismo | Dónde vive | Qué se comprobó | Test |
|---|---|---|---|
| Métricas acumuladas del partido | `dataset.construir_dataset` | La fila N usa el acumulado de los partidos 1..N-1; no incluye el marcador de N ni posteriores | `ConstruirDatasetTests.test_metricas_previas_al_partido` |
| Histórico de métricas abstractas | `modulo.historico_abstractas` | Solo suma Mundiales **anteriores** al año de la fila; el Mundial actual no entra | `HistoricoAbstractasTests.test_solo_mundiales_anteriores` |
| Elo | `contexto.calcular_elo` | La predicción del partido usa el Elo **antes** del partido; el resultado del partido solo actualiza el Elo del siguiente | `ContextoTests.test_elo_anterior_al_partido` |
| Partición temporal | `evaluacion.particion` | Entrenamiento `agno <= hasta`; validación es el ciclo siguiente; prueba son solo los Mundiales indicados; no se solapan | `EvaluacionTests.test_particion_temporal` |
| Secuencias por equipo | `dataset.construir_secuencias` | Relleno a la izquierda con máscara; el paso del partido N no incluye su resultado (`y` es el objetivo, no una entrada) | `ConstruirSecuenciasTests.test_relleno_y_orden` |
| Valor neutro | `modulo.VALOR_NEUTRO` | Se calculó con `agno <= 2014` (6.983 partidos), el corte de entrenamiento de las pruebas 2018/2022, sin usar ningún año de prueba | `ValorNeutroTests.test_todas_las_metricas_tienen_valor_neutro` |
| Simulación de la app | `modulo.contexto_simulacion` / `calculo_metricas_0` | Usa el estado acumulado antes del partido simulado y actualiza después; **no tiene test de integración todavía** | Pendiente (A-06/A-07) |
| Datos externos | `etl/construye_proxy.py` | `pk_real`/`sot_real` marcan dato real; el proxy imputado no usa resultados del partido que predice | Revisión manual; sin test dedicado |

## Hallazgos

- **No se encontró una fuga activa** en dataset, Elo, histórico abstracto ni partición.
- El riesgo residual está en la **simulación de la app** (`consulta_general`): comparte funciones con el dataset, pero no hay un test de integración que congele su comportamiento. Se cubre con el test de foto de A-06 (2018 y 1986) antes de tocar `modulo.py`.
- `VALOR_NEUTRO` se calculó con `agno <= 2014`; si en el futuro se entrena con otro corte, hay que recalcularlo y subir la fecha del comentario.
- Los datos de 2026 no entran en el entrenamiento que predice 2026 (`particion` usa `hasta=2022`), pero el **hold-out ya se miró** varias veces: eso no es fuga de entrenamiento, es contaminación del examen (ver `registro-de-miradas.md`).

## Cómo se comprueba

```powershell
cd web
..\.venv\Scripts\python.exe manage.py test fases -v 2
```

Los tests citados deben pasar. Cualquier cambio en `dataset.py`, `contexto.py`, `evaluacion.particion` o `modulo.historico_abstractas` obliga a revisar esta tabla.

## Pendiente

- Test de integración de `consulta_general` para 2018/1986 (A-06), que también sirve de red para el refactor H.2.
- Test de que la simulación no usa el marcador real del partido que está prediciendo (hoy la evidencia es lectura de código).
