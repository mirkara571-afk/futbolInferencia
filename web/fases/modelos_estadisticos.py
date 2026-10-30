"""
Modelos estadisticos clasicos de la Etapa B: Poisson GLM, ordinal y Dixon-Coles.

Todos cumplen el contrato de "predictores.py": predecir devuelve (n, 2) goles esperados y, cuando el modelo tiene
probabilidades propias del resultado, "probabilidades" devuelve (n, 3) en el orden local / empate / visitante.

La entrada es siempre un Lote con la tabla del dataset (entrenamiento) o la que arma "contexto_simulacion" en la app:
home_id, away_id, elo_0/1, local_0/1, neutral, categoria, rate_GF_0/1, rate_GC_0/1, gkps_0/1, mos_0/1, mms_0/1,
fase_final y goles_0/1 (solo en entrenamiento).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import poisson
from statsmodels.miscmodels.ordinal_model import OrderedModel

from .predictores import Lote, PredictorElo

MAX_GOLES = 10


def _columna(tabla, nombre, defecto=0.0):
  """Columna numerica de (tabla), o un vector con (defecto) si no existe (la app puede no traer un contexto)"""
  if nombre in tabla:
    return pd.to_numeric(tabla[nombre], errors="coerce").fillna(defecto).to_numpy(dtype=float)
  return np.full(len(tabla), float(defecto))


def _categorias(dataset):
  """Categorias de partido del entrenamiento, sin la primera (esa es la referencia del one-hot)"""
  if "categoria" not in dataset:
    return []
  return sorted(str(c) for c in dataset["categoria"].dropna().unique())[1:]


def _base_lado(tabla, i, categorias):
  """Features de un lado (i=0 home, i=1 away): Elo, localia, tipo de partido y metricas propias/del rival"""
  otro = 1 - i
  datos = {
    "elo_diff": (_columna(tabla, f"elo_{i}", 1500.0) - _columna(tabla, f"elo_{otro}", 1500.0)) / 100.0,
    "es_local": _columna(tabla, f"local_{i}"),
    "neutral": _columna(tabla, "neutral", 1.0),
    "rate_GF": _columna(tabla, f"rate_GF_{i}") / 10.0,
    "rate_GC_rival": _columna(tabla, f"rate_GC_{otro}") / 10.0,
    "gkps": _columna(tabla, f"gkps_{i}") / 100.0,
    "mos": _columna(tabla, f"mos_{i}") / 100.0,
    "fase_final": _columna(tabla, "fase_final"),
  }
  categoria = tabla["categoria"].astype(str) if "categoria" in tabla else pd.Series([""] * len(tabla), index=tabla.index)
  for cat in categorias:
    datos[f"cat_{cat}"] = (categoria == cat).to_numpy(dtype=float)
  return pd.DataFrame(datos, index=tabla.index)


def _matriz_glm(tabla, categorias):
  """Matriz de diseno con una fila por equipo y partido (home y away) e intercepto"""
  return sm.add_constant(pd.concat([_base_lado(tabla, 0, categorias), _base_lado(tabla, 1, categorias)],
                                   ignore_index=True), has_constant="add")


def _matriz_ordinal(tabla, categorias):
  """Matriz de diseno con una fila por partido; OrderedModel pone los cortes, no lleva constante"""
  datos = {
    "elo_diff": (_columna(tabla, "elo_0", 1500.0) - _columna(tabla, "elo_1", 1500.0)) / 100.0,
    "es_local": _columna(tabla, "local_0") - _columna(tabla, "local_1"),
    "dif_rate_GF": (_columna(tabla, "rate_GF_0") - _columna(tabla, "rate_GF_1")) / 10.0,
    "dif_rate_GC": (_columna(tabla, "rate_GC_0") - _columna(tabla, "rate_GC_1")) / 10.0,
    "dif_mms": (_columna(tabla, "mms_0") - _columna(tabla, "mms_1")) / 100.0,
    "fase_final": _columna(tabla, "fase_final"),
  }
  categoria = tabla["categoria"].astype(str) if "categoria" in tabla else pd.Series([""] * len(tabla), index=tabla.index)
  for cat in categorias:
    datos[f"cat_{cat}"] = (categoria == cat).to_numpy(dtype=float)
  return pd.DataFrame(datos, index=tabla.index)


class PredictorPoissonGLM:
  """
  ¿QUE HACE?
  Regresion de Poisson (GLM) de los goles de cada equipo: una fila por equipo y partido; log(lambda) es lineal en la
  diferencia de Elo, la localia, el tipo de partido y las metricas propias/del rival (rate_GF, rate_GC, gkps, mos)

  ¿COMO LO HACE?
  Ajusta con statsmodels un Poisson con intercepto y las features de "_base_lado"; guarda los coeficientes y las
  columnas en esquema.json para reconstruir el modelo sin pickles
  """
  nombre = "poisson_glm"

  def __init__(self, params, columnas, categorias):
    self.params = np.asarray(params, dtype=float)
    self.columnas = list(columnas)
    self.categorias = list(categorias)

  @classmethod
  def entrenar(cls, dataset):
    categorias = _categorias(dataset)
    X = _matriz_glm(dataset, categorias)
    y = np.concatenate([dataset["goles_0"].to_numpy(dtype=float), dataset["goles_1"].to_numpy(dtype=float)])
    ajuste = sm.GLM(y, X, family=sm.families.Poisson()).fit(maxiter=300)
    return cls(ajuste.params, X.columns, categorias)

  def _X(self, lote):
    X = _matriz_glm(lote.tabla, self.categorias)
    return X.reindex(columns=self.columnas, fill_value=0.0)

  def predecir(self, lote):
    X = self._X(lote).to_numpy(dtype=float)
    lambdas = np.exp(np.clip(X @ self.params, -20, 20))
    n = len(lote.tabla)
    return np.stack([lambdas[:n], lambdas[n:]], axis=1)

  def coeficientes(self):
    """Tabla de coeficientes del GLM (para el informe de B.1)"""
    return pd.DataFrame({"coeficiente": self.params}, index=self.columnas)

  def a_esquema(self):
    return {"backend": self.nombre, "columnas": self.columnas, "categorias": self.categorias,
            "coef": {c: float(v) for c, v in zip(self.columnas, self.params)}}

  @classmethod
  def desde_esquema(cls, esquema, carpeta):
    columnas = list(esquema["columnas"])
    return cls([esquema["coef"][c] for c in columnas], columnas, esquema["categorias"])


class PredictorOrdinal:
  """
  ¿QUE HACE?
  Regresion logistica ordinal del resultado (visitante < empate < local) con la diferencia de Elo, la localia y las
  diferencias de rate_GF, rate_GC y mms; para el marcador usa los goles esperados de un PredictorPoissonGLM

  ¿COMO LO HACE?
  Ajusta OrderedModel (logit) con una fila por partido; guarda los coeficientes y los dos cortes. Al predecir, el
  resultado sale de las probabilidades ordinales y el marcador de los goles del GLM
  """
  nombre = "ordinal"

  def __init__(self, params, columnas, categorias, poisson):
    self.params = np.asarray(params, dtype=float)
    self.columnas = list(columnas)
    self.categorias = list(categorias)
    self.poisson = poisson

  @classmethod
  def entrenar(cls, dataset):
    poisson = PredictorPoissonGLM.entrenar(dataset)
    categorias = _categorias(dataset)
    X = _matriz_ordinal(dataset, categorias)
    # OrderedModel rechaza columnas constantes (p. ej. una categoria que no aparece en el pliegue)
    X = X.loc[:, X.nunique() > 1]
    goles_0, goles_1 = dataset["goles_0"].to_numpy(), dataset["goles_1"].to_numpy()
    y = np.where(goles_0 > goles_1, 2, np.where(goles_0 == goles_1, 1, 0))
    if len(np.unique(y)) < 3:
      raise ValueError("El ordinal necesita partidos con los tres resultados (local, empate, visitante)")
    ajuste = OrderedModel(y, X, distr="logit").fit(method="bfgs", maxiter=300, disp=False)
    return cls(ajuste.params, X.columns, categorias, poisson)

  def _X(self, lote):
    X = _matriz_ordinal(lote.tabla, self.categorias)
    return X.reindex(columns=self.columnas, fill_value=0.0)

  def probabilidades(self, lote):
    X = self._X(lote).to_numpy(dtype=float)
    k = len(self.columnas)
    beta, corte_0, corte_1 = self.params[:k], self.params[k], self.params[k + 1]
    eta = X @ beta
    p_visitante = expit(corte_0 - eta)
    p_local = 1 - expit(corte_1 - eta)
    p_empate = 1 - p_visitante - p_local
    probs = np.stack([p_local, p_empate, p_visitante], axis=1)
    probs = np.clip(probs, 0, None)
    return probs / probs.sum(axis=1, keepdims=True)

  def predecir(self, lote):
    return self.poisson.predecir(lote)

  def a_esquema(self):
    k = len(self.columnas)
    return {"backend": self.nombre, "columnas": self.columnas, "categorias": self.categorias,
            "coef": {c: float(v) for c, v in zip(self.columnas, self.params[:k])},
            "cortes": [float(self.params[k]), float(self.params[k + 1])],
            "poisson": self.poisson.a_esquema()}

  @classmethod
  def desde_esquema(cls, esquema, carpeta):
    columnas = list(esquema["columnas"])
    params = [esquema["coef"][c] for c in columnas] + list(esquema["cortes"])
    return cls(params, columnas, esquema["categorias"], PredictorPoissonGLM.desde_esquema(esquema["poisson"], carpeta))


def _tau_dixon(y_0, y_1, lambda_0, lambda_1, rho):
  """
  Correccion de Dixon-Coles para marcadores bajos y sus derivadas respecto a lambda_0, lambda_1 y rho.
  Devuelve tau, d log(tau)/d lambda_0, d log(tau)/d lambda_1 y d log(tau)/d rho.
  """
  tau = np.ones_like(lambda_0)
  d_0, d_1, d_rho = np.zeros_like(lambda_0), np.zeros_like(lambda_0), np.zeros_like(lambda_0)

  m = (y_0 == 0) & (y_1 == 0)
  tau[m] = 1 - lambda_0[m] * lambda_1[m] * rho
  d_0[m], d_1[m], d_rho[m] = -lambda_1[m] * rho / tau[m], -lambda_0[m] * rho / tau[m], -lambda_0[m] * lambda_1[m] / tau[m]

  m = (y_0 == 0) & (y_1 == 1)
  tau[m] = 1 + lambda_0[m] * rho
  d_0[m], d_rho[m] = rho / tau[m], lambda_0[m] / tau[m]

  m = (y_0 == 1) & (y_1 == 0)
  tau[m] = 1 + lambda_1[m] * rho
  d_1[m], d_rho[m] = rho / tau[m], lambda_1[m] / tau[m]

  m = (y_0 == 1) & (y_1 == 1)
  tau[m] = 1 - rho
  d_rho[m] = -1 / tau[m]

  return tau, d_0, d_1, d_rho


class PredictorDixonColes:
  """
  ¿QUE HACE?
  Dixon-Coles: Poisson para los dos equipos con ataque y defensa por seleccion, ventaja de local y correccion rho
  para los marcadores bajos (0-0, 1-0, 0-1, 1-1); opcionalmente decaimiento temporal xi (los partidos viejos pesan menos)

  ¿COMO LO HACE?
  Maximiza la verosimilitud con scipy.optimize (L-BFGS-B y gradiente analitico). Ataque y defensa se centran en 0 para
  que el modelo sea identificable. Guarda los parametros por equipo en esquema.json; un equipo desconocido usa la media
  """
  nombre = "dixon_coles"

  def __init__(self, equipos, ataque, defensa, ventaja_local, rho, xi=0.0, usar_elo=False, beta_elo=0.0):
    self.equipos = [int(e) for e in equipos]
    self.ataque = np.asarray(ataque, dtype=float)
    self.defensa = np.asarray(defensa, dtype=float)
    self.ventaja_local = float(ventaja_local)
    self.rho = float(rho)
    self.xi = float(xi)
    self.usar_elo = bool(usar_elo)
    self.beta_elo = float(beta_elo)
    self.indice = {equipo: i for i, equipo in enumerate(self.equipos)}
    self.n = len(self.equipos)

  @staticmethod
  def _parametros(theta, n, usar_elo):
    ataque = np.concatenate([theta[:n - 1], [-theta[:n - 1].sum()]])
    defensa = np.concatenate([theta[n - 1:2 * (n - 1)], [-theta[n - 1:2 * (n - 1)].sum()]])
    ventaja, rho = theta[2 * (n - 1)], theta[2 * (n - 1) + 1]
    beta = theta[-1] if usar_elo else 0.0
    return ataque, defensa, ventaja, rho, beta

  @classmethod
  def entrenar(cls, dataset, xi=0.0, usar_elo=False, maxiter=250):
    equipos = sorted(set(dataset["home_id"].astype(int)) | set(dataset["away_id"].astype(int)))
    n = len(equipos)
    if n < 4:
      raise ValueError("Dixon-Coles necesita al menos 4 equipos")
    indice = {equipo: i for i, equipo in enumerate(equipos)}
    home = dataset["home_id"].map(indice).to_numpy(dtype=int)
    away = dataset["away_id"].map(indice).to_numpy(dtype=int)
    y_0, y_1 = dataset["goles_0"].to_numpy(dtype=float), dataset["goles_1"].to_numpy(dtype=float)
    local_0, local_1 = dataset["local_0"].to_numpy(dtype=float), dataset["local_1"].to_numpy(dtype=float)
    peso = np.exp(-xi * (dataset["agno"].max() - dataset["agno"].to_numpy(dtype=float)))
    elo_diff = (dataset["elo_0"].to_numpy(dtype=float) - dataset["elo_1"].to_numpy(dtype=float)) / 100.0

    def objetivo(theta):
      ataque, defensa, ventaja, rho, beta = cls._parametros(theta, n, usar_elo)
      log_0 = ataque[home] + defensa[away] + ventaja * local_0 + (beta * elo_diff if usar_elo else 0.0)
      log_1 = ataque[away] + defensa[home] + ventaja * local_1 - (beta * elo_diff if usar_elo else 0.0)
      lambda_0, lambda_1 = np.exp(np.clip(log_0, -20, 20)), np.exp(np.clip(log_1, -20, 20))
      tau, d_0, d_1, d_rho = _tau_dixon(y_0, y_1, lambda_0, lambda_1, rho)
      if np.any(tau <= 1e-8):
        return 1e12, np.zeros_like(theta)
      log_verosimilitud = peso * (y_0 * np.log(lambda_0) - lambda_0 + y_1 * np.log(lambda_1) - lambda_1 + np.log(tau))
      g_0 = peso * (y_0 - lambda_0) + d_0
      g_1 = peso * (y_1 - lambda_1) + d_1
      g_a, g_d = np.zeros(n), np.zeros(n)
      np.add.at(g_a, home, g_0 * lambda_0)
      np.add.at(g_d, away, g_0 * lambda_0)
      np.add.at(g_a, away, g_1 * lambda_1)
      np.add.at(g_d, home, g_1 * lambda_1)
      g_ventaja = float(np.sum(g_0 * lambda_0 * local_0 + g_1 * lambda_1 * local_1))
      g_rho = float(np.sum(d_rho))
      grad = np.empty_like(theta)
      grad[:n - 1] = g_a[:n - 1] - g_a[-1]
      grad[n - 1:2 * (n - 1)] = g_d[:n - 1] - g_d[-1]
      grad[2 * (n - 1)] = g_ventaja
      grad[2 * (n - 1) + 1] = g_rho
      if usar_elo:
        grad[-1] = float(np.sum(g_0 * lambda_0 * elo_diff - g_1 * lambda_1 * elo_diff))
      return -float(np.sum(log_verosimilitud)), -grad

    theta_0 = np.zeros(2 * (n - 1) + 2 + (1 if usar_elo else 0))
    theta_0[2 * (n - 1)] = 0.2
    limites = [(-5, 5)] * (2 * (n - 1)) + [(-1, 1), (-0.2, 0.2)] + ([(-1, 1)] if usar_elo else [])
    ajuste = minimize(objetivo, theta_0, method="L-BFGS-B", jac=True, bounds=limites,
                      options={"maxiter": maxiter, "ftol": 1e-9})
    ataque, defensa, ventaja, rho, beta = cls._parametros(ajuste.x, n, usar_elo)
    predictor = cls(equipos, ataque, defensa, ventaja, rho, xi=xi, usar_elo=usar_elo, beta_elo=beta)
    predictor.convergencia = {"ok": bool(ajuste.success), "iteraciones": int(ajuste.nit), "mensaje": str(ajuste.message)}
    return predictor

  def _indices(self, lote):
    home = np.array([self.indice.get(int(t), self.n) for t in lote.tabla["home_id"]], dtype=int)
    away = np.array([self.indice.get(int(t), self.n) for t in lote.tabla["away_id"]], dtype=int)
    ataque = np.append(self.ataque, 0.0)
    defensa = np.append(self.defensa, 0.0)
    return home, away, ataque, defensa

  def _lambdas(self, lote):
    home, away, ataque, defensa = self._indices(lote)
    local_0, local_1 = _columna(lote.tabla, "local_0"), _columna(lote.tabla, "local_1")
    elo_diff = (_columna(lote.tabla, "elo_0", 1500.0) - _columna(lote.tabla, "elo_1", 1500.0)) / 100.0
    log_0 = ataque[home] + defensa[away] + self.ventaja_local * local_0 + (self.beta_elo * elo_diff if self.usar_elo else 0.0)
    log_1 = ataque[away] + defensa[home] + self.ventaja_local * local_1 - (self.beta_elo * elo_diff if self.usar_elo else 0.0)
    return np.exp(np.clip(log_0, -20, 20)), np.exp(np.clip(log_1, -20, 20))

  def predecir(self, lote):
    lambda_0, lambda_1 = self._lambdas(lote)
    return np.stack([lambda_0, lambda_1], axis=1)

  def probabilidades(self, lote):
    lambda_0, lambda_1 = self._lambdas(lote)
    goles = np.arange(MAX_GOLES + 1)
    p_0 = poisson.pmf(goles[None, :], np.maximum(lambda_0[:, None], 1e-9))
    p_1 = poisson.pmf(goles[None, :], np.maximum(lambda_1[:, None], 1e-9))
    conjunta = p_0[:, :, None] * p_1[:, None, :]
    conjunta[:, 0, 0] *= 1 - lambda_0 * lambda_1 * self.rho
    conjunta[:, 0, 1] *= 1 + lambda_0 * self.rho
    conjunta[:, 1, 0] *= 1 + lambda_1 * self.rho
    conjunta[:, 1, 1] *= 1 - self.rho
    conjunta = np.clip(conjunta, 0, None)
    conjunta = conjunta / conjunta.sum(axis=(1, 2), keepdims=True)
    return np.stack([np.tril(conjunta, -1).sum(axis=(1, 2)), np.trace(conjunta, axis1=1, axis2=2),
                     np.triu(conjunta, 1).sum(axis=(1, 2))], axis=1)

  def a_esquema(self):
    return {"backend": self.nombre, "equipos": self.equipos, "ataque": self.ataque.tolist(),
            "defensa": self.defensa.tolist(), "ventaja_local": self.ventaja_local, "rho": self.rho,
            "xi": self.xi, "usar_elo": self.usar_elo, "beta_elo": self.beta_elo,
            "convergencia": getattr(self, "convergencia", {})}

  @classmethod
  def desde_esquema(cls, esquema, carpeta):
    return cls(esquema["equipos"], esquema["ataque"], esquema["defensa"], esquema["ventaja_local"], esquema["rho"],
               xi=esquema.get("xi", 0.0), usar_elo=esquema.get("usar_elo", False), beta_elo=esquema.get("beta_elo", 0.0))


PREDICTORES = {
  PredictorPoissonGLM.nombre: PredictorPoissonGLM.desde_esquema,
  PredictorOrdinal.nombre: PredictorOrdinal.desde_esquema,
  PredictorDixonColes.nombre: PredictorDixonColes.desde_esquema,
}

# al importar este modulo se registran sus backends en el contrato de predictores; si el modulo se importa desde
# "cargar_predictor" (import perezoso), predictores ya esta cargado y no hay ciclo
from .predictores import REGISTRO  # noqa: E402

REGISTRO.update(PREDICTORES)
