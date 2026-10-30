# Etapa B: modelos estadísticos clásicos

**Proyecto:** futbolInferencia · **Fecha:** 2026-10-30 · **Rama:** `exp/etapa-a-medicion` (los modelos B se añaden aquí)
**Reproducir:** `python manage.py evaluar_b` (validación) y `python manage.py entrenar_b` (artefactos)

## 1. Qué se construyó

| Tarea | Modelo | Archivo | Estado |
|---|---|---|---|
| B-01 | `PredictorPoissonGLM` | `web/fases/modelos_estadisticos.py` | Registrado, probado, guardado en `data/modelos/poisson_glm/` |
| B-02 | `PredictorOrdinal` | `web/fases/modelos_estadisticos.py` | Registrado, probado, guardado en `data/modelos/ordinal/` |
| B-03 | `PredictorDixonColes` con `rho` y `xi` | `web/fases/modelos_estadisticos.py` | Registrado, probado, guardado en `data/modelos/dixon_coles/` |
| B-04 | Comparación contra el Elo en validación móvil | `web/fases/management/commands/evaluar_b.py` | Ejecutado; JSON en `experimentos/resultados_b.json` |

Los tres cumplen el contrato `Predictor`: `predecir(Lote) -> (n, 2)` y, cuando tienen probabilidades propias, `probabilidades(Lote) -> (n, 3)` en el orden local / empate / visitante. Se guardan con `guardar_predictor` y se cargan con `cargar_predictor`; no hay pickles ni código duplicado de Elo/Poisson.

**No se tocó 2018/2022/2026.** Toda la comparación es validación móvil (2006, 2010, 2014): cada ciclo se entrena con todos los anteriores y se valida en el ciclo completo.

## 2. Resultados (validación móvil)

Media de los tres pliegues; menor es mejor en log-loss, RPS y ECE.

| Modelo | Log-loss | RPS | Acierto | ECE |
|---|---:|---:|---:|---:|
| **Poisson GLM** | **0,8366** | **0,1607** | **62,5 %** | 0,019 |
| **Elo puro** | 0,8409 | 0,1624 | 62,4 % | **0,015** |
| Dixon-Coles, `xi=0,05` | 0,8520 | 0,1657 | 61,8 % | 0,023 |
| Dixon-Coles, `xi=0,10` | 0,8534 | 0,1655 | 61,5 % | 0,021 |
| Ordinal logit | 0,8927 | 0,1693 | 61,3 % | 0,065 |

Detalle por pliegue (log-loss):

| Pliegue | Elo | GLM | Ordinal | Dixon-Coles `xi=0,05` |
|---|---:|---:|---:|---:|
| 2006 | 0,8152 | **0,8105** | 0,8679 | 0,8220 |
| 2010 | 0,8427 | **0,8407** | 0,8894 | 0,8568 |
| 2014 | 0,8648 | **0,8586** | 0,9209 | 0,8771 |

**Lectura honesta:** el GLM gana al Elo por 0,0043 de log-loss de media, pero la comparación pareada **no es significativa** en ningún pliegue (`p` de Diebold-Mariano 0,198 / 0,747 / 0,073). La mejora de RPS también es pequeña; un pliegue llega a `p=0,009`, los otros no. Es un candidato, no un campeón. Dixon-Coles y el ordinal no superan al Elo con estos datos y estas features.

**Empates:** en el pliegue 2014 la tasa real es 22,5 %. La probabilidad media de empate que dan los modelos ronda el 20–22 %, pero el resultado *más probable* casi nunca es empate: 0 % en Elo/GLM/ordinal y 1,6 % en Dixon-Coles. Ese es el problema que debe resolver F.2; no lo resuelven estos modelos.

## 3. Definiciones, sin jerga

### Elo

Un número que resume la fuerza de una selección a partir de sus resultados. Ganas al que tenía más Elo y subes; pierdes y bajas. La diferencia de Elo entre dos equipos se usa como una variable más. **No predice goles directamente**: hay que convertirla en probabilidades o en goles esperados.

### Poisson GLM (B-01)

Un **GLM** (modelo lineal generalizado) de Poisson modela **cuántos goles marca un equipo** como una media `λ` (lambda). La media se escribe:

```
log(λ) = constante + β1·(Elo propio − Elo rival)/100 + β2·local + β3·neutro
         + β4·rate_GF propio + β5·rate_GC rival + β6·gkps + β7·mos + β8·fase_final + ...
```

- `λ` es el número esperado de goles; el modelo no dice “gana este”, dice “este equipo marcará 1,4 goles de media”.
- Los `β` se ajustan con **máxima verosimilitud**: se eligen los valores que hacen más probables los goles que realmente ocurrieron.
- Se ajusta **una fila por equipo y partido**: el local y el visitante aportan cada uno su `λ`.
- La ventaja de local, el tipo de partido y las métricas propias/del rival entran como variables; no hay una “caja negra”.

Coeficientes del modelo guardado (entrenado hasta 2022; `coeficientes.json`):

| Variable | Coeficiente | Lectura aproximada |
|---|---:|---|
| `elo_diff` | 0,1769 | +100 de Elo ≈ +19 % goles esperados (`exp(0,1769)`) |
| `es_local` | 0,4733 | jugar en casa ≈ +60 % goles esperados |
| `neutral` | 0,3186 | partido en cancha neutral (asociación, no causalidad) |
| `rate_GF` | 0,0575 | +10 en la tasa ofensiva ≈ +5,9 % |
| `rate_GC_rival` | 0,0899 | +10 en la tasa defensiva del rival ≈ +9,4 % |
| `gkps` | −0,0910 | más puntaje de portería propia se asocia con menos goles |
| `mos` | −0,2543 | asociación negativa; cuidado, hay colinealidad con otras métricas |
| `fase_final` | 0,1220 | eliminatoria vs fase de grupos |
| `cat_mundial` | 0,0516 | Mundial vs eliminatoria (referencia) |

> Los coeficientes son **asociaciones**, no relaciones causales. `gkps` y `mos` están correlacionados con `rate_GF`/`rate_GC`; por eso algunos signos pueden sorprender. Para una lectura causal haría falta un diseño experimental que aquí no existe.

### Regresión ordinal (B-02)

En vez de modelar goles, modela directamente el **resultado ordenado**: pierde < empata < gana. Es una regresión logística con dos “cortes”:

```
P(visitante) = sigmoide(corte1 − η)
P(empate)    = sigmoide(corte2 − η) − sigmoide(corte1 − η)
P(local)     = 1 − sigmoide(corte2 − η)
η = β·(Elo, localía, diferencias de rate_GF, rate_GC y mms)
```

La ventaja es que puede dar más probabilidad al empate que un Poisson, porque no tiene que “elegir” un marcador. En nuestros datos **no está bien calibrado** (ECE 0,065): dice 60 % cuando acierta bastante menos. Para el marcador usa los goles esperados del GLM.

### Dixon-Coles (B-03)

Es un Poisson para los dos equipos, pero con **ataque y defensa propios de cada selección** y una corrección para marcadores bajos. La media de goles es:

```
λ_local  = exp(ataque_local + defensa_visitante + ventaja_local)
λ_visita = exp(ataque_visita + defensa_local)
```

- `ataque_i` y `defensa_i` se estiman por máxima verosimilitud, con la restricción de que su media sea 0 (si no, el modelo no es identificable: sumar 1 al ataque y restar 1 a la defensa da lo mismo).
- `rho` corrige la dependencia en marcadores bajos: 0-0, 1-0, 0-1 y 1-1. Con `rho` negativo (≈ −0,05) sube la probabilidad de 0-0 y 1-1 y baja la de 1-0/0-1; eso se acerca a la idea de que los partidos cerrados no son independientes.
- `xi` es el **decaimiento temporal**: un partido de hace `t` años pesa `exp(−xi·t)`. `xi=0` es “todos los partidos pesan igual”; `xi=0,05` da algo más de peso a lo reciente. Se elige en validación, no en la prueba.
- `rho` y `xi` se estiman/eligen con `scipy.optimize` y gradiente analítico; el modelo guarda ataque, defensa, ventaja, `rho` y `xi` por equipo.

**¿Por qué no ganó?** Porque con ~6.000 partidos de entrenamiento y 212 selecciones, los efectos por equipo son ruidosos; el Elo ya resume la fuerza con menos parámetros. Dixon-Coles suele lucir más cuando hay muchos partidos por equipo y datos de calidad (xG, tiros, posesión), no solo resultados.

### Log-loss, RPS, Brier y ECE

- **Log-loss:** castigo por dar poca probabilidad a lo que pasó. Si dices 90 % y ocurre, pierdes poco; si dices 10 % y ocurre, pierdes mucho. Menor es mejor. Repartir 1/3 a cada resultado da `log(3) ≈ 1,099`.
- **RPS (ranked probability score):** como el log-loss, pero respeta el orden pierde < empata < gana. Equivocarse por empate cuesta menos que equivocarse por el contrario. Menor es mejor.
- **Brier:** suma de errores cuadráticos de las tres probabilidades. Menor es mejor.
- **ECE (expected calibration error):** promedio de “probabilidad dicha menos frecuencia real” en 10 tramos. `ECE=0,02` significa que, en promedio, el modelo se desvía 2 puntos porcentuales de la realidad. Un “70 %” debería acertar 7 de cada 10; el ECE mide cuánto se aleja de eso.

### Incertidumbre

- **Bootstrap:** remuestrear partidos con reposición para estimar el margen de error de una métrica. Con 868 partidos de validación, un intervalo del 95 % típico es ±2–3 puntos de acierto; con 128 partidos de prueba, ±9.
- **Diebold-Mariano:** compara dos modelos en los mismos partidos; un `p < 0,05` indica que la diferencia media es difícil de explicar por azar. No es una garantía: con muchos modelos y muchas comparaciones, algún `p` pequeño aparece por casualidad.
- **McNemar:** compara el acierto de dos modelos contando solo los partidos en que uno acierta y el otro no.

## 4. Nivel de efectividad

| Pregunta | Respuesta |
|---|---|
| ¿El GLM es mejor que el Elo? | En media sí, por 0,0043 de log-loss y 0,0017 de RPS; **no es estadísticamente concluyente** |
| ¿Cuánto mejora en acierto? | +0,1 puntos porcentuales (62,5 % vs 62,4 %): práctico empate |
| ¿Y la calibración? | El Elo está mejor calibrado (ECE 0,015 vs 0,019) |
| ¿Dixon-Coles aporta? | No con estas features; es un benchmark explicable, no un ganador |
| ¿El ordinal aporta? | No: peor log-loss y mala calibración |
| ¿Se puede afirmar “modelo B listo para la app”? | No; B-04 dice que hay que seguir con C/F y elegir en F.3 |

**Conclusión:** la Etapa B está **terminada como experimento**, no como modelo final. El GLM es el mejor candidato de B y el único que roza al Elo, pero la ventaja es demasiado pequeña para venderla. Lo honesto es decir: “empatados; el Elo sigue siendo la referencia”.

## 5. Mi visión: otras métricas y métodos

### Lo que probablemente más mejoraría

1. **Datos de calidad antes que modelos exóticos.** xG, tiros, tiros a puerta, posesión, faltas, córners y minutos de los jugadores explican más que cambiar de familia de modelo. El proxy actual imputa penales y tiros; E.1 (FBref real) y E.2 (StatsBomb) son la mayor palanca.
2. **Fuerza del plantel y disponibilidad.** Ranking FIFA, valor de mercado, edad media, minutos de titulares, lesiones y convocatorias. Un 10 % más de información real suele valer más que un 10 % más de hiperparámetros.
3. **Jerarquía bayesiana.** Un modelo con **partial pooling** por selección (en vez de 212 ataques/defensas independientes) controla mejor el ruido y da intervalos honestos. `PyMC`/`Stan` o un GLMM con efectos aleatorios serían el siguiente paso natural tras Dixon-Coles.
4. **Estado dinámico del Elo.** El Elo clásico es estático entre partidos; un **Kalman filter / state-space** o un Elo con `K` variable por competición y recencia puede capturar mejor el momento de forma.
5. **Modelos de árboles y ensambles.** LightGBM/CatBoost con las mismas features suelen batir al GLM en datos tabulares; el ensamble con Elo/GLM es la apuesta de F. Si C no mejora, no hay que forzarlo.
6. **Modelar el empate como decisión, no como argmax.** El problema de los empates no es solo de probabilidad: es de **regla de decisión**. F.2 (maximizar acierto esperado o minimizar RPS) puede aportar más que un modelo nuevo.
7. **Cuotas de casas de apuestas como benchmark.** Si se consiguen con licencia, son un competidor muy fuerte y una fuente de calibración; también revelan qué información pública ya está incorporada.

### Lo que probaría con cautela

- **Poisson bivariado/cópula** en vez de la corrección `rho`: más flexible, pero más parámetros y riesgo de sobreajuste.
- **Zero-inflated Poisson**: los 0-0 son frecuentes; puede ayudar, pero confunde “pocos goles” con “mala puntería”.
- **Modelos de jugadores y simulación de eventos**: muy atractivos para una demo, pero necesitan datos de alineaciones y eventos que hoy no están.
- **Redes con atención/secuencias largas**: pueden capturar forma, pero con 6.000–9.000 partidos y 212 equipos el riesgo de sobreajuste es alto. D.4 lo deja como opcional por eso.
- **Odds/mercados**: potentes, pero su licencia y disponibilidad son un problema; no las metería sin resolver eso.

### Lo que no recomendaría

- Añadir una 15.ª métrica “porque suena bien” sin medir su efecto en validación. El esquema fijo de 14 existe para que los modelos guardados sigan siendo comparables.
- Elegir por acierto en un solo Mundial. 104 partidos dan ±9 puntos; es ruido.
- Optimizar contra 2018/2022/2026. El hold-out ya está contaminado; cada mirada extra lo hace menos útil.
- Vender “predice el Mundial”. Con estos datos, la frase honesta es “da probabilidades calibradas y las compara con el Elo”.

### Visión de producto

El valor no está en decir “Brasil gana” con 55 % en vez de 54 %, sino en:
- probabilidades **calibradas** y con intervalos;
- una explicación clara de qué pesa (Elo, localía, ataque, defensa);
- un cuadro de favoritos con Montecarlo;
- una API y una demo que enseñen el “¿qué hubiera pasado si…?”.

Eso es más defendible que perseguir 0,5 puntos de log-loss con un modelo más complejo.

## 6. Próximos pasos

1. **C** (LightGBM y Random Forest) con las mismas features y validación móvil; comparar contra GLM y Elo.
2. **F** (ensamble y regla de empates) y **una sola** mirada final a 2018/2022/2026.
3. **E.1/E.2** si los datos reales están disponibles: es la palanca más grande.
4. **H.2** (refactor con la foto de A-06) antes de tocar `modulo.py`.

---

**Archivos:** `web/fases/modelos_estadisticos.py`, `web/fases/management/commands/evaluar_b.py`, `web/fases/management/commands/entrenar_b.py`, `experimentos/resultados_b.json`, `data/modelos/poisson_glm/`, `data/modelos/ordinal/`, `data/modelos/dixon_coles/`.
