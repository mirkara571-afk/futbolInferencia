"""
Evaluacion de predictores (Fase 2): las metricas de la guia sobre una particion temporal del dataset.

Particion (sin mirar el futuro):
  - entrenamiento: todo lo jugado en los ciclos hasta (hasta) inclusive (eliminatorias y mundiales)
  - validacion: el ciclo siguiente completo (sus eliminatorias y su mundial), para elegir configuracion
  - prueba: los partidos de mundial de los años de prueba (2018 y 2022), que solo se miran al final

Metricas, con los goles esperados (lambda) que devuelve cada predictor:
  - mae_goles: error absoluto medio entre lambda y los goles reales (de los dos equipos)
  - acierto_resultado: % de partidos donde el resultado predicho coincide con el real: el mas probable (local / empate /
    visitante, con dos Poisson independientes), salvo que las probabilidades de ganar de los dos esten a menos de
    (umbral_empate) de distancia: entonces empate (con dos Poisson el empate casi nunca es el resultado mas probable)
  - marcador_exacto: % de partidos donde el marcador predicho (el mas probable dentro del resultado predicho) es el real
  - deviance: perdida de Poisson media (la que minimiza el LSTM; menor es mejor)

Metricas de las probabilidades del resultado (local / empate / visitante), menor es mejor en las tres:
  - log_loss: media de -log(probabilidad dada al resultado que paso); castiga fuerte estar muy seguro y equivocarse
    (repartir 1/3 a cada uno da log(3) = 1,099)
  - brier: media de la suma de (probabilidad - 1 si paso / 0 si no)^2 sobre los tres resultados (entre 0 y 2)
  - rps: ranked probability score, como el Brier pero con las probabilidades acumuladas en el orden local < empate <
    visitante, asi dar empate cuando gana el local se castiga menos que dar visitante (entre 0 y 1)
  - calibracion: 10 tramos de probabilidad predicha (de los tres resultados juntos) con su media y la frecuencia real
  - ece: expected calibration error, media ponderada por numero de casos de |probabilidad media - frecuencia real| en esos tramos (0 = perfectamente calibrado)

Incertidumbre (con 128 o 104 partidos de prueba una diferencia chica puede ser suerte):
  - intervalos_bootstrap: intervalo del 95 % de cada metrica remuestreando partidos con reposicion
  - comparar: dos predictores sobre los mismos partidos; diferencia de log-loss (u otra metrica) con bootstrap pareado y
    prueba de Diebold-Mariano, y McNemar exacta para el acierto (solo cuentan los partidos donde uno acierta y el otro no)

Las lineas base deterministas (siempre 1-1, gana el de mas puntos) declaran su marcador con el atributo
"determinista": su resultado y su marcador son los que dicen, no los de pasar sus numeros por Poisson; sus
probabilidades son 1 al resultado que dicen y 0 a los otros, suavizadas con (EPSILON) para que el log-loss sea finito.
"""
import numpy as np
from scipy.stats import binomtest, norm, poisson

from .dataset import FASE_MUNDIAL
from .predictores import Lote

MAX_GOLES = 10
EPSILON = 0.05      # probabilidad repartida entre los tres resultados en las predicciones deterministas
TRAMOS = 10
REMUESTREOS = 2000
# metricas que se promedian por partido (las que llevan intervalo)
POR_PARTIDO = ("mae_goles", "acierto_resultado", "marcador_exacto", "deviance", "log_loss", "brier", "rps")


def probabilidades_resultado(lambdas):
  """(n, 2) goles esperados -> (n, 3) probabilidades de gana local, empate y gana visitante con dos Poisson independientes"""
  goles = np.arange(MAX_GOLES + 1)
  p_0 = poisson.pmf(goles[None, :], np.maximum(lambdas[:, [0]], 1e-9))
  p_1 = poisson.pmf(goles[None, :], np.maximum(lambdas[:, [1]], 1e-9))
  conjunta = p_0[:, :, None] * p_1[:, None, :]
  return np.stack([np.tril(conjunta, -1).sum(axis=(1, 2)), np.trace(conjunta, axis1=1, axis2=2),
                   np.triu(conjunta, 1).sum(axis=(1, 2))], axis=1)


def resultado_y_marcador(lambdas, umbral_empate=0.0, probabilidades=None):
  """
  ¿QUE HACE?
  Resultado y marcador predichos desde los goles esperados (n, 2); lo usan la evaluacion y la simulacion de la app

  ¿COMO LO HACE?
  0-) Probabilidades de gana local, empate y gana visitante con "probabilidades_resultado", salvo que vengan en (probabilidades) (n, 3) de un predictor que las calcula por su cuenta (p. ej. PredictorElo)

  1-) Empate si es lo mas probable o si |P(local) - P(visitante)| < (umbral_empate); si no, gana el mas probable de los dos

  2-) Marcador: el par de goles mas probable de la distribucion conjunta entre los que dan ese resultado (1-0, 1-1, ...)

  Returns:

    - (resultado, marcador, probabilidades): (n,) con 1 local / 0 empate / -1 visitante, (n, 2) enteros y (n, 3)
  """
  lambdas = np.asarray(lambdas, dtype=float)
  goles = np.arange(MAX_GOLES + 1)
  p_0 = poisson.pmf(goles[None, :], np.maximum(lambdas[:, [0]], 1e-9))
  p_1 = poisson.pmf(goles[None, :], np.maximum(lambdas[:, [1]], 1e-9))
  conjunta = p_0[:, :, None] * p_1[:, None, :]
  if probabilidades is None:
    probs = np.stack([np.tril(conjunta, -1).sum(axis=(1, 2)), np.trace(conjunta, axis1=1, axis2=2),
                      np.triu(conjunta, 1).sum(axis=(1, 2))], axis=1)
  else:
    probs = np.asarray(probabilidades, dtype=float)
  empate = (probs.argmax(axis=1) == 1) | (np.abs(probs[:, 0] - probs[:, 2]) < umbral_empate)
  resultado = np.where(empate, 0, np.where(probs[:, 0] >= probs[:, 2], 1, -1))
  diferencia = goles[:, None] - goles[None, :]
  marcador = np.zeros((len(lambdas), 2), dtype=int)
  for r in (1, 0, -1):
    filas = resultado == r
    if filas.any():
      region = np.where(np.sign(diferencia) == r, conjunta[filas], -1)
      planos = region.reshape(filas.sum(), -1).argmax(axis=1)
      marcador[filas] = np.stack(np.unravel_index(planos, diferencia.shape), axis=1)
  return resultado, marcador, probs


def mejor_umbral(goles, lambdas, umbrales=np.arange(0, 0.41, 0.01)):
  """Umbral de empate que maximiza el acierto de resultado en (goles) reales con (lambdas); con varias semillas se pasa la lista de (lambdas) y se promedia"""
  lista = lambdas if isinstance(lambdas, list) else [lambdas]
  real = np.sign(np.asarray(goles)[:, 0] - np.asarray(goles)[:, 1])
  aciertos = [np.mean([(resultado_y_marcador(l, u)[0] == real).mean() for l in lista]) for u in umbrales]
  return float(umbrales[int(np.argmax(aciertos))]), [float(a) for a in aciertos]


def probabilidades_deterministas(resultado, epsilon=EPSILON):
  """(n,) resultados 1 / 0 / -1 -> (n, 3) probabilidades: 1 - epsilon + epsilon / 3 al resultado dicho y epsilon / 3 a los otros"""
  uno = np.eye(3)[1 - np.asarray(resultado, dtype=int)]
  return uno * (1 - epsilon) + epsilon / 3


def metricas_probabilidad(real, probs, medias=True):
  """
  ¿QUE HACE?
  Log-loss, Brier y RPS de las probabilidades (probs) (n, 3) de local / empate / visitante contra el resultado (real) (n,) con 1 / 0 / -1

  Returns:

    - diccionario con log_loss, brier y rps: medias, o arrays (n,) con el valor de cada partido si no (medias)
  """
  probs = np.clip(np.asarray(probs, dtype=float), 1e-12, 1)
  probs = probs / probs.sum(axis=1, keepdims=True)
  ocurrio = np.eye(3)[1 - np.asarray(real, dtype=int)]                # columnas local, empate, visitante
  acumulada = np.cumsum(probs - ocurrio, axis=1)[:, :2]
  por_partido = {
    "log_loss": -np.log((probs * ocurrio).sum(axis=1)),
    "brier": ((probs - ocurrio) ** 2).sum(axis=1),
    "rps": (acumulada ** 2).sum(axis=1) / 2,
  }
  return {m: float(v.mean()) for m, v in por_partido.items()} if medias else por_partido


def tabla_calibracion(real, probs, tramos=TRAMOS):
  """
  ¿QUE HACE?
  Tabla de calibracion: junta las 3 probabilidades de cada partido con si ese resultado paso (1) o no (0), las reparte en (tramos) tramos iguales de probabilidad predicha y en cada uno compara la probabilidad media con la frecuencia real; bien calibrado es que se parezcan

  Returns:

    - lista de {desde, hasta, n, probabilidad, frecuencia} (los tramos vacios llevan probabilidad y frecuencia None)
  """
  p = np.asarray(probs, dtype=float).ravel()
  o = np.eye(3)[1 - np.asarray(real, dtype=int)].ravel()
  tramo = np.minimum((p * tramos).astype(int), tramos - 1)
  tabla = []
  for t in range(tramos):
    filas = tramo == t
    tabla.append({"desde": t / tramos, "hasta": (t + 1) / tramos, "n": int(filas.sum()),
                  "probabilidad": float(p[filas].mean()) if filas.any() else None,
                  "frecuencia": float(o[filas].mean()) if filas.any() else None})
  return tabla


def ece(real, probs, tramos=TRAMOS):
  """
  ¿QUE HACE?
  Expected calibration error (ECE) de las probabilidades (probs) (n, 3) contra el resultado (real) (n,) con 1 / 0 / -1: la tabla de calibracion, pero resumida en un solo numero; 0 es calibracion perfecta y cuanto mas grande, peor

  ¿COMO LO HACE?
  Junta las 3 probabilidades de cada partido (igual que "tabla_calibracion"), y promedia |probabilidad media - frecuencia real| de cada tramo ponderando por cuantos casos tiene; los tramos vacios no cuentan

  Returns:

    - float entre 0 y 1
  """
  tabla = tabla_calibracion(real, probs, tramos)
  total = sum(t["n"] for t in tabla)
  if not total:
    return 0.0
  return sum(t["n"] * abs(t["probabilidad"] - t["frecuencia"]) for t in tabla if t["n"]) / total


def metricas_por_partido(goles, lambdas, determinista=False, umbral_empate=0.0, probabilidades=None):
  """
  ¿QUE HACE?
  El valor de cada metrica de POR_PARTIDO en cada partido, para (lambdas) (n, 2) contra los (goles) reales (n, 2); si (determinista), (lambdas) es directamente el marcador predicho; si no, el resultado y el marcador salen de "resultado_y_marcador" con (umbral_empate) y las (probabilidades) propias del predictor si las tiene. Su media es "metricas"; el bootstrap y las comparaciones remuestrean estos arrays

  Returns:

    - (por_partido, real, probs): diccionario metrica -> array (n,) (acierto y marcador exacto en 0 / 100), el resultado real (n,) y las probabilidades (n, 3)
  """
  goles, lambdas = np.asarray(goles, dtype=float), np.asarray(lambdas, dtype=float)
  real = np.sign(goles[:, 0] - goles[:, 1])                        # 1 local, 0 empate, -1 visitante
  if determinista:
    predicho, moda = np.sign(lambdas[:, 0] - lambdas[:, 1]), np.round(lambdas)
    probs = probabilidades_deterministas(predicho)
  else:
    predicho, moda, probs = resultado_y_marcador(lambdas, umbral_empate, probabilidades)
  l = np.maximum(lambdas, 1e-9)
  return {
    "mae_goles": np.abs(lambdas - goles).mean(axis=1),
    "acierto_resultado": (predicho == real) * 100.0,
    "marcador_exacto": (moda == goles).all(axis=1) * 100.0,
    "deviance": (l - goles * np.log(l)).mean(axis=1),
    **metricas_probabilidad(real, probs, medias=False),
  }, real, probs


def metricas(goles, lambdas, determinista=False, umbral_empate=0.0, probabilidades=None):
  """
  ¿QUE HACE?
  Calcula las metricas de la guia: la media por partido de "metricas_por_partido" (mismos argumentos) y la tabla de calibracion

  Returns:

    - diccionario con partidos, mae_goles, acierto_resultado, marcador_exacto (en %), deviance, log_loss, brier, rps, calibracion y ece
  """
  por_partido, real, probs = metricas_por_partido(goles, lambdas, determinista, umbral_empate, probabilidades)
  return {"partidos": int(len(real)), **{m: float(v.mean()) for m, v in por_partido.items()},
          "calibracion": tabla_calibracion(real, probs), "ece": ece(real, probs)}


def intervalos_bootstrap(por_partido, remuestreos=REMUESTREOS, nivel=0.95, semilla=0):
  """
  ¿QUE HACE?
  Intervalo de confianza de la media de cada metrica de (por_partido) ("metricas_por_partido"): repite (remuestreos) veces sortear n partidos con reposicion y calcular la media; el intervalo son los percentiles del (nivel) central de esas medias

  Returns:

    - diccionario metrica -> [desde, hasta]
  """
  n = len(next(iter(por_partido.values())))
  indices = np.random.default_rng(semilla).integers(0, n, (remuestreos, n))
  colas = [(1 - nivel) / 2 * 100, (1 + nivel) / 2 * 100]
  return {m: [float(x) for x in np.percentile(np.asarray(v)[indices].mean(axis=1), colas)] for m, v in por_partido.items()}


def comparar(por_partido_a, por_partido_b, metrica="log_loss", remuestreos=REMUESTREOS, semilla=0):
  """
  ¿QUE HACE?
  Compara dos predictores medidos en los mismos partidos ("metricas_por_partido" de cada uno), en pares: cada partido aporta la diferencia a - b, asi se descuenta que unos partidos son mas dificiles que otros para los dos

  ¿COMO LO HACE?
  0-) Diferencia media de (metrica) (negativa: a es mejor si la metrica es una perdida) con su intervalo del 95 % por bootstrap pareado (se remuestrean partidos, no modelos) y su valor p (dos colas: cuantas veces el remuestreo cruza el 0)

  1-) Prueba de Diebold-Mariano: media de las diferencias sobre su error estandar, contra una normal (los partidos se tratan como independientes)

  2-) McNemar exacta del acierto: con b partidos que acierta solo a y c que acierta solo b, si los dos fueran igual de buenos b seguiria una binomial(b + c, 1/2)

  Returns:

    - diccionario con diferencia, intervalo, p_bootstrap, dm, p_dm, solo_a, solo_b y p_mcnemar
  """
  d = np.asarray(por_partido_a[metrica]) - np.asarray(por_partido_b[metrica])
  medias = d[np.random.default_rng(semilla).integers(0, len(d), (remuestreos, len(d)))].mean(axis=1)
  p_bootstrap = min(1.0, 2 * min((medias <= 0).mean(), (medias >= 0).mean()))
  error = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
  dm = float(d.mean() / error) if error > 0 else 0.0
  acierta_a, acierta_b = np.asarray(por_partido_a["acierto_resultado"]) > 0, np.asarray(por_partido_b["acierto_resultado"]) > 0
  solo_a, solo_b = int((acierta_a & ~acierta_b).sum()), int((acierta_b & ~acierta_a).sum())
  return {
    "metrica": metrica, "diferencia": float(d.mean()), "intervalo": [float(x) for x in np.percentile(medias, [2.5, 97.5])],
    "p_bootstrap": float(p_bootstrap), "dm": dm, "p_dm": float(2 * norm.sf(abs(dm))),
    "solo_a": solo_a, "solo_b": solo_b,
    "p_mcnemar": float(binomtest(solo_a, solo_a + solo_b, 0.5).pvalue) if solo_a + solo_b else 1.0,
  }


def particion(dataset, hasta, validacion=None, prueba=None):
  """
  ¿QUE HACE?
  Filas booleanas de entrenamiento, validacion y prueba sobre (dataset) segun la particion del modulo

  Args:

    - hasta: ultimo mundial (ciclo) de entrenamiento

    - validacion: mundial cuyo ciclo completo valida (None: ninguno)

    - prueba: mundiales cuyos partidos de mundial se usan de prueba (None: ninguno)

  Returns:

    - (entrenamiento, validacion, prueba): arrays booleanos
  """
  agno = dataset["agno"].to_numpy()
  vacio = np.zeros(len(dataset), dtype=bool)
  return (
    agno <= hasta,
    agno == validacion if validacion is not None else vacio,
    np.isin(agno, prueba) & (dataset["fase"].to_numpy() == FASE_MUNDIAL) if prueba is not None else vacio,
  )


def lote(dataset, secuencias, filas):
  """Lote de (filas) de (dataset) con sus secuencias, para cualquier predictor"""
  return Lote(dataset[filas].reset_index(drop=True), {c: v[filas] for c, v in secuencias.items()} if secuencias else None)


def evaluar(predictor, dataset, secuencias, filas, por_partido=False):
  """Metricas de (predictor) en las (filas) de (dataset); con (por_partido) devuelve (metricas, valores por partido) para intervalos y comparaciones"""
  entrada = lote(dataset, secuencias, filas)
  argumentos = (dataset.loc[filas, ["goles_0", "goles_1"]].to_numpy(), predictor.predecir(entrada),
                getattr(predictor, "determinista", False), getattr(predictor, "config", {}).get("umbral_empate", 0.0),
                predictor.probabilidades(entrada) if hasattr(predictor, "probabilidades") else None)
  resultado = metricas(*argumentos)
  return (resultado, metricas_por_partido(*argumentos)[0]) if por_partido else resultado
