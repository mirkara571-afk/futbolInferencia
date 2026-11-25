# 13 · Metodología matemática y estadística

**Fecha:** 2026-11-25 · **Alcance:** sustento científico del motor de predicción (`web/fases/`).

Este documento describe, fórmula por fórmula, qué calcula el código, de dónde salen sus datos y
cómo lo ejecuta NumPy/SciPy por debajo. La notación sigue los nombres reales de las variables del
programa para que cada ecuación sea trazable a una línea de código.

---

## 1. Encuadre y notación

Un **partido** es una fila con dos equipos, `home` (local) y `away` (visitante). El predictor entrega
los **goles esperados** $\lambda = (\lambda_0, \lambda_1)$ (una media de Poisson por lado), y de ahí se
deriva la probabilidad del resultado (local / empate / visitante).

| Símbolo | Significado | Fuente en el código |
|---|---|---|
| $\lambda_0, \lambda_1$ | goles esperados de home y away | salida de `predecir(lote)` → `(n, 2)` |
| $g_0, g_1$ | goles reales | columnas `goles_0`, `goles_1` del dataset |
| $r$ | resultado real: $+1$ local, $0$ empate, $-1$ visitante | `np.sign(g_0 - g_1)` |
| $p = (p_L, p_E, p_V)$ | probabilidad de local/empate/visitante | `(n, 3)` |

El resultado real se codifica como `1 - r` para indexar `[local, empate, visitante]` ($1\to0$, $0\to1$, $-1\to2$).

---

## 2. Elo (vara de medir) — `contexto.calcular_elo`

Recorre los partidos en orden de fecha y actualiza un rating por selección.

**Resultado esperado del local** (modelo logístico con la ventaja de campo):

$$
E = \frac{1}{1 + 10^{-(r_0 - r_1 + h)/400}}
$$

donde $r_0, r_1$ son los ratings de home y away **antes** del partido, y $h$ es `VENTAJA_LOCAL`
(solo si no es campo neutral). El `400` es la constante clásica de Elo.

**Actualización** tras el partido:

$$
R' = R + K \cdot G(\Delta g) \cdot (W - E)
$$

- $W \in \{1, 0.5, 0\}$ es el resultado real (ganó, empató, perdió).
- $K$ es `K_POR_CATEGORIA[categoria]` (por defecto 30): los mundiales pesan más que los amistosos.
- $G(\Delta g)$ es `multiplicador_goles(g_0 - g_1)`: amplifica el cambio cuando la diferencia de goles es grande.
- El home suma $K G (W-E)$ y el away **resta** lo mismo (juego de suma cero).

**Implementación NumPy:** `np.argsort(fecha, kind="stable")` ordena cronológicamente; el bucle acumula
el rating en un `defaultdict`; la exponencial `10**(-x/400)` es la función logística con base 10.

---

## 3. Predictor Elo — `predictores.PredictorElo`

Línea base fuerte: todo sale de una variable, la diferencia de Elo escalada:

$$
x = \frac{r_0 - r_1 + h \cdot (\ell_0 - \ell_1)}{100}
$$

donde $\ell_i \in \{0,1\}$ es jugar de local (`local_0`/`local_1`) y $h$ la ventaja local.

### 3.1 Resultado — regresión logística ordinal

Modelo de probabilidades acumuladas (visitante < empate < local):

$$
P(\text{resultado} \le j) = \sigma(c_j - \beta\, x), \qquad \sigma(z) = \frac{1}{1+e^{-z}}
$$

con dos cortes $c_{\text{visitante}} < c_{\text{empate}}$. Las probabilidades de los tres resultados:

$$
p_L = 1 - \sigma(c_E - \beta x),\quad
p_E = \sigma(c_E - \beta x) - \sigma(c_V - \beta x),\quad
p_V = \sigma(c_V - \beta x).
$$

`_ordinal` parametriza $c_E = c_V + \exp(\cdot)$ para imponer el orden. Se ajusta por máxima
verosimilitud con `scipy.optimize.minimize` (BFGS) minimizando $-\sum_i \log p_{y_i}$.

### 3.2 Goles — regresión de Poisson

$$
\log \lambda_0 = b + \beta_g\, x, \qquad \log \lambda_1 = b - \beta_g\, x
$$

($x$ se duplica con signo opuesto: `np.concatenate([x, -x])`). La pérdida es la deviance de Poisson
(véase §7). `predecir` devuelve $\lambda = \exp(b \pm \beta_g x)$.

**Implementación NumPy:** `expit` (`scipy.special`) es $\sigma$; `np.exp`, `np.stack` arman la salida.

---

## 4. Bradley–Terry con empates (extensión de Davidson) — `PredictorBradleyTerry`

Cada equipo tiene una fuerza $f_i$ (con la restricción $\sum_i f_i = 0$). Para un partido:

$$
t_h = f_{\text{home}} + \eta\,\ell_0, \qquad t_a = f_{\text{away}} + \eta\,\ell_1
$$

donde $\eta$ es la ventaja local. Se definen las "intensidades":

$$
s_h = e^{t_h}, \quad s_a = e^{t_a}, \quad e = \delta\, e^{(t_h + t_a)/2}
$$

El término $e$ (media geométrica amortiguada por $\delta>0$) captura el empate. Las probabilidades:

$$
p_L = \frac{s_h}{s_h + s_a + e}, \quad
p_E = \frac{e}{s_h + s_a + e}, \quad
p_V = \frac{s_a}{s_h + s_a + e}.
$$

**Goles:** $\lambda_0 = \exp(\operatorname{clip}(t_h, -20, 20))$, $\lambda_1 = \exp(\operatorname{clip}(t_a, -20, 20))$.

**Ajuste:** máxima verosimilitud con gradiente analítico. La verosimilitud es
$\sum_i \log p_{y_i}$, y el gradiente se acumula por equipo con `np.add.at(g_f, home, g_h)` (suma
dispersa sobre índices). `scipy.optimize.minimize(..., method="L-BFGS-B", jac=True)`.

---

## 5. LSTM (modelo de la app) — `predictores.PredictorKeras`

Red de series temporales por equipo. Para cada equipo, su **secuencia** de los últimos $k$ partidos
(§6) entra a:

$$
h = \operatorname{LSTM}\big(\operatorname{Masking}(S)\big)
$$

donde $S \in \mathbb{R}^{k \times f}$ son los pasos (métricas abstractas + contexto + extras) y
`Masking(0.0)` ignora los pasos rellenos con cero. En paralelo, el id del equipo se incrusta:

$$
u = \operatorname{Flatten}\big(\operatorname{Embedding}(id)\big).
$$

Ambos se concatenan ($[h_0, u_0]$ y $[h_1, u_1]$) y pasan por una cabeza de goles:

$$
\lambda = \operatorname{softplus}\!\big(\operatorname{Dense}_{32,\text{ReLU}}(\,[h_{\text{atac}}, u_{\text{atac}}, h_{\text{def}}, u_{\text{def}}]\,)\big).
$$

La salida `lambdas` = $[\lambda_{\text{home}}, \lambda_{\text{away}}]$ se entrena minimizando la
**pérdida de Poisson** $\lambda - g\log\lambda$ (§7). El id desconocido va al slot `n_paises`.

`entradas` normaliza cada paso con la media/desviación del entrenamiento:
$S_{\text{norm}} = (S - \mu)/\sigma$, anulando los pasos enmascarados.

---

## 6. Construcción de secuencias — `dataset.construir_secuencias`

Para cada partido y equipo, el paso es el vector de sus **métricas previas** del lado que jugó:

$$
\text{paso} = [\underbrace{\text{ALL\_METRICS}_{\text{lado}}}_{\text{métricas}}, \underbrace{[\ell, \nu], \text{one-hot}(\text{categoría}), [\text{pk\_real}, \text{sot\_real}]}_{\text{contexto}}, \underbrace{\text{extras}}_{\text{elo, fase, …}}]
$$

La secuencia de un equipo en el partido $i$ son sus últimos $k$ pasos; si tiene menos de $k$ se
rellena al principio con ceros y se marca en `mascara` (1 = paso real, 0 = relleno). Esto es lo que
consume el LSTM, y es exactamente lo que la app reconstruye en `secuencias_desde_tabla` (D.4).

---

## 7. De $\lambda$ a probabilidad de resultado — `evaluacion.probabilidades_resultado`

Los goles de cada lado son dos **Poisson independientes**:

$$
P(g=i) = \frac{e^{-\lambda}\lambda^i}{i!}, \qquad i \in \{0,\dots,10\}.
$$

La **distribución conjunta** es el producto exterior $C = p_0 \otimes p_1$ ($C_{ij} = P(g_0=i,\,g_1=j)$):

$$
p_L = \sum_{i>j} C_{ij}, \qquad p_E = \sum_{i=j} C_{ij}, \qquad p_V = \sum_{i<j} C_{ij}.
$$

**Implementación NumPy:** `poisson.pmf` calcula la PMF; `p_0[:, :, None] * p_1[:, None, :]` es el
producto exterior; `np.tril(C, -1).sum`, `np.trace(C)`, `np.triu(C, 1).sum` extraen las tres regiones.

El **marcador** predicho es el par $(i,j)$ más probable dentro del resultado elegido (`resultado_y_marcador`),
y el resultado sale del argmax de $p$ o del **umbral de empate** ($|p_L - p_V| < u \Rightarrow$ empate).

---

## 8. Métricas de evaluación — `evaluacion.metricas_probabilidad`

Sea $o \in \{0,1\}^3$ el one-hot del resultado real ($o_j = 1$ si pasó $j$). Con $p$ normalizado
(`np.clip` + división por la suma):

| Métrica | Fórmula | Rango | Implementación |
|---|---|---|---|
| **Log-loss** | $-\sum_j o_j \log p_j$ | $[0,\infty)$; $1/3$ c/u → $\log 3$ | `-np.log((probs*ocurrio).sum(1))` |
| **Brier** | $\sum_j (p_j - o_j)^2$ | $[0,2]$ | `((probs-ocurrio)**2).sum(1)` |
| **RPS** | $\frac12 \sum_{m=1}^{2} \big(\sum_{j\le m}(p_j-o_j)\big)^2$ | $[0,1]$ | `np.cumsum(probs-ocurrio)[:, :2]` → cuadrado/suma |
| **Deviance** | $\lambda - g\log\lambda$ (por lado) | $\ge 0$ | `(l - goles*np.log(l)).mean(1)` |
| **MAE goles** | $|\lambda - g|$ | $\ge 0$ | `np.abs(lambdas-goles).mean(1)` |
| **ECE** | $\sum_t \frac{n_t}{n}\,|\bar p_t - \bar o_t|$ | $[0,1]$ | `tabla_calibracion` + media ponderada |

`ocurrio = np.eye(3)[1 - r]` construye el one-hot. El RPS usa probabilidades **acumuladas** en el
orden local < empate < visitante, de modo que predecir empate ante una victoria local castiga menos
que predecir visitante.

---

## 9. Inferencia sobre la diferencia entre modelos — `evaluacion.comparar`

Para dos predictores medidos en los **mismos** partidos, sea $d_i = m^a_i - m^b_i$ la diferencia de la
métrica $m$ en el partido $i$.

- **Bootstrap pareado:** se remuestrean $n$ partidos con reposición ($\bar d^*$) y el p-valor es
  $2\min(P(\bar d^*\le 0), P(\bar d^*\ge 0))$. El IC 95 % son los percentiles 2.5 y 97.5 de $\bar d^*$.
  → `np.random.default_rng(seed).integers(0, n, (B, n))` + `np.percentile(medias, [2.5, 97.5])`.
- **Diebold–Mariano:** $DM = \bar d / \operatorname{se}(\bar d)$ contra una normal; p = $2\,\Phi(-|DM|)$.
  → `np.std(ddof=1)/np.sqrt(n)` y `scipy.stats.norm.sf`.
- **McNemar exacta:** con $b$ = partidos que solo acierta $a$ y $c$ = los que solo acierta $b$, si fueran
  iguales $b \sim \operatorname{Binomial}(b+c, 1/2)$. → `scipy.stats.binomtest`.

---

## 10. Montecarlo — `montecarlo.simular_knockout`

Fijadas las medias $\lambda$ de cada pareja, cada partido se simula muestreando:

$$
g_0 \sim \operatorname{Poisson}(\lambda_0), \qquad g_1 \sim \operatorname{Poisson}(\lambda_1),
$$

y avanza el que tenga más goles (empate → desempate por probabilidad). Se repite $n$ veces para
estimar $P(\text{campeón}) = \frac{\#\text{títulos}}{n}$.

---

## 11. Flujo de datos (qué consume cada fórmula)

| Fórmula | Datos de entrada | Extracción/transformación |
|---|---|---|
| Elo | `internacionales` (fecha, score, categoría, neutral) | `formar_dataset_real("internacionales")` → df; `np.argsort(fecha)` |
| PredictorElo | `elo_0/1`, `local_0/1`, `goles_0/1` | columnas de `partidos_features.csv` (agregadas por `agregar_contexto`) |
| Bradley-Terry | `home_id/away_id`, `local_0/1`, `goles_0/1` | ids de `mapa_paises`; `np.where` sobre goles |
| LSTM | secuencias (`seq_0/1`, `mascara_0/1`, `id_0/1`) | `construir_secuencias` sobre `partidos_features.csv` |
| Métricas | $\lambda$ (n,2) vs `goles` (n,2) | `predecir(lote)` + columnas `goles_0/1` |

El dataset de entrenamiento (`partidos_features.csv` para eliminatorias, `..._internacionales.csv`
para D.4) se genera con `construir_dataset`, que acumula las métricas partido a partido
(`calculo_metricas_0` / `calculo_metricas_1`) y añade el contexto con `agregar_contexto` (Elo, fase,
continental, descanso, grupo y ranking FIFA).

---

## 12. Mapa de funciones NumPy/SciPy

| Función | Operación matemática | Rol en el algoritmo |
|---|---|---|
| `np.argsort` | ordenar por clave | orden cronológico del Elo |
| `np.exp`, `np.log`, `np.clip`, `np.maximum` | elementales | log-verosimilitud, exponenciales, cotas numéricas |
| `scipy.stats.poisson.pmf` | $e^{-\lambda}\lambda^i/i!$ | distribución de goles |
| `np.tril` / `np.triu` / `np.trace` | suma triangular/diagonal | $p_L, p_E, p_V$ de la conjunta |
| `np.eye` | matriz identidad → one-hot | codificar el resultado real |
| `np.cumsum` | suma acumulada | probabilidades acumuladas del RPS |
| `scipy.special.expit` | $\sigma(z)$ | regresión ordinal y Elo |
| `np.add.at` | suma dispersa sobre índices | gradiente del Bradley-Terry |
| `scipy.optimize.minimize` | BFGS / L-BFGS-B | ajustar Elo, ordinal y Bradley-Terry |
| `np.random.default_rng().integers` | remuestreo con reposición | bootstrap pareado |
| `np.percentile` | cuantiles | intervalo de confianza del 95 % |
| `scipy.stats.norm.sf` | cola normal $\Phi(-|z|)$ | p-valor de Diebold-Mariano |
| `scipy.stats.binomtest` | binomial exacta | McNemar |
