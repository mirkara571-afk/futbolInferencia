"""
Modelo jerarquico estatico de goles (JH-01): partial pooling de ataque y defensa por seleccion.

PredictorJerarquico estima por maxima a posteriori (MAP) los ataques (alpha) y las defensas (delta) de cada seleccion
con priors Gaussianas centradas en 0. Los equipos con poca historia se encogen hacia la media del pool (0) y los de
mucha historia conservan su estimacion. Solo usa goles: cada partido aporta dos Poisson independientes.

Convencion de signo (fijada en JH-01): delta entra ADITIVO en el log-rate (`alpha_atacante + delta_defensor`), por lo que
un delta POSITIVO significa "concede mas goles" (debilidad defensiva). Es equivalente a la convencion `-delta` del
plan 07 (seccion 4.1) bajo priors simetricos: basta definir delta_plan = -delta_impl. Esta convencion coincide con la de
PredictorDixonColes.

El MAP se resuelve con scipy.optimize.minimize (L-BFGS-B) y gradiente analitico, el mismo patron de PredictorDixonColes.
Se registra en el contrato de predictores con nombre "jerarquico" a traves de PREDICTORES (ver predictores.cargar_predictor).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from .modelos_estadisticos import _columna


def _neg_log_posterior(theta, home, away, y_0, y_1, local_0, local_1, sigma_attack, sigma_defense):
  """
  Valor y gradiente de la negativa de la log-posterior del modelo jerarquico de goles.

  theta = (beta0, gamma, alpha_0..alpha_{n-1}, delta_0..delta_{n-1}); (home) y (away) son los indices de cada partido
  dentro del mapa id->indice. La verosimilitud es la de dos Poisson independientes por partido; los priors Gaussianos
  regularizan beta0, gamma, alpha y delta hacia 0. El termino -log(goles!) es constante respecto a los parametros y no
  cambia el optimo; se incluye con gammaln para que el valor sea la log-posterior completa (salvo normalizaciones).
  """
  n = (len(theta) - 2) // 2
  beta0, gamma = theta[0], theta[1]
  alpha = theta[2:2 + n]
  delta = theta[2 + n:2 + 2 * n]
  sa2, sd2 = sigma_attack ** 2, sigma_defense ** 2

  log_0 = beta0 + alpha[home] + delta[away] + gamma * local_0
  log_1 = beta0 + alpha[away] + delta[home] + gamma * local_1
  lambda_0 = np.exp(np.clip(log_0, -20, 20))
  lambda_1 = np.exp(np.clip(log_1, -20, 20))

  ll = float(np.sum(y_0 * np.log(lambda_0) - lambda_0 - gammaln(y_0 + 1)
                    + y_1 * np.log(lambda_1) - lambda_1 - gammaln(y_1 + 1)))
  log_prior = -0.5 * (beta0 ** 2 / 9.0 + gamma ** 2 + float(np.sum(alpha ** 2)) / sa2 + float(np.sum(delta ** 2)) / sd2)

  g0 = y_0 - lambda_0
  g1 = y_1 - lambda_1
  g_alpha = np.zeros(n)
  g_delta = np.zeros(n)
  np.add.at(g_alpha, home, g0)
  np.add.at(g_alpha, away, g1)
  np.add.at(g_delta, away, g0)
  np.add.at(g_delta, home, g1)

  grad = np.empty_like(theta)
  grad[0] = -(g0.sum() + g1.sum()) + beta0 / 9.0
  grad[1] = -float(np.sum(g0 * local_0 + g1 * local_1)) + gamma
  grad[2:2 + n] = -g_alpha + alpha / sa2
  grad[2 + n:2 + 2 * n] = -g_delta + delta / sd2
  return -(ll + log_prior), grad


class PredictorJerarquico:
  """
  ¿QUE HACE?
  Modelo de goles jerarquico estatico: para cada seleccion estima un ataque (alpha) y una defensa (delta) con priors
  Gaussianas centradas en 0 (partial pooling), un intercepto beta0 y una ventaja de local gamma. Predice los goles
  esperados de home y away como dos Poisson independientes

  ¿COMO LO HACE?
  Maximiza la posterior (MAP) con scipy.optimize (L-BFGS-B) y gradiente analitico. Ataque y defensa se regularizan hacia
  0: los equipos con pocos partidos se encogen a la media del pool. Guarda los estados por equipo en esquema.json; un
  equipo desconocido usa la media del prior (alpha=delta=0). No usa xG ni features de metricas

  Verosimilitud (solo goles):
    lambda_home = exp(beta0 + alpha_home + delta_away + gamma * local_0)
    lambda_away = exp(beta0 + alpha_away + delta_home + gamma * local_1)

  Priors: alpha_i ~ Normal(0, sigma_attack), delta_i ~ Normal(0, sigma_defense), beta0 ~ Normal(0, 3), gamma ~ Normal(0, 1).
  Para JH-01 sigma_attack y sigma_defense se fijan (por defecto 1.0); se pueden pasar otros valores a entrenar para
  estudiar el shrinkage sin cambiar la estructura del modelo.
  """
  nombre = "jerarquico"

  def __init__(self, equipos, beta0, gamma, alpha, delta, sigma_attack=1.0, sigma_defense=1.0):
    self.equipos = [int(e) for e in equipos]
    self.beta0 = float(beta0)
    self.gamma = float(gamma)
    self.alpha = np.asarray(alpha, dtype=float)
    self.delta = np.asarray(delta, dtype=float)
    self.sigma_attack = float(sigma_attack)
    self.sigma_defense = float(sigma_defense)
    self.indice = {equipo: i for i, equipo in enumerate(self.equipos)}
    self.n = len(self.equipos)

  @classmethod
  def entrenar(cls, dataset, sigma_attack=1.0, sigma_defense=1.0, maxiter=500):
    if sigma_attack <= 0 or sigma_defense <= 0:
      raise ValueError("sigma_attack y sigma_defense deben ser positivos")
    equipos = sorted(set(dataset["home_id"].astype(int)) | set(dataset["away_id"].astype(int)))
    n = len(equipos)
    if n < 2:
      raise ValueError("El modelo jerarquico necesita al menos 2 equipos")
    indice = {equipo: i for i, equipo in enumerate(equipos)}
    home = dataset["home_id"].map(indice).to_numpy(dtype=int)
    away = dataset["away_id"].map(indice).to_numpy(dtype=int)
    y_0 = _columna(dataset, "goles_0")
    y_1 = _columna(dataset, "goles_1")
    local_0 = _columna(dataset, "local_0")
    local_1 = _columna(dataset, "local_1")

    # arranque determinista (L-BFGS-B no tiene estocasticidad): intercepto en la media de goles y una leve ventaja local
    media = float((y_0.sum() + y_1.sum()) / max(2 * len(dataset), 1))
    theta_0 = np.zeros(2 + 2 * n)
    theta_0[0] = np.log(max(media, 1e-3))
    theta_0[1] = 0.2
    limites = [(-5.0, 5.0), (-2.0, 2.0)] + [(-5.0, 5.0)] * (2 * n)

    def objetivo(theta):
      return _neg_log_posterior(theta, home, away, y_0, y_1, local_0, local_1, sigma_attack, sigma_defense)

    ajuste = minimize(objetivo, theta_0, method="L-BFGS-B", jac=True, bounds=limites,
                      options={"maxiter": maxiter, "ftol": 1e-9})
    predictor = cls(equipos, ajuste.x[0], ajuste.x[1], ajuste.x[2:2 + n], ajuste.x[2 + n:2 + 2 * n],
                    sigma_attack, sigma_defense)
    predictor.convergencia = {"ok": bool(ajuste.success), "iteraciones": int(ajuste.nit), "mensaje": str(ajuste.message)}
    return predictor

  def _indices(self, lote):
    home = np.array([self.indice.get(int(t), self.n) for t in lote.tabla["home_id"]], dtype=int)
    away = np.array([self.indice.get(int(t), self.n) for t in lote.tabla["away_id"]], dtype=int)
    # el indice self.n lleva alpha=delta=0 (la media del prior) para los equipos nunca vistos en entrenamiento
    alpha = np.append(self.alpha, 0.0)
    delta = np.append(self.delta, 0.0)
    return home, away, alpha, delta

  def predecir(self, lote):
    home, away, alpha, delta = self._indices(lote)
    local_0 = _columna(lote.tabla, "local_0")
    local_1 = _columna(lote.tabla, "local_1")
    log_0 = self.beta0 + alpha[home] + delta[away] + self.gamma * local_0
    log_1 = self.beta0 + alpha[away] + delta[home] + self.gamma * local_1
    return np.stack([np.exp(np.clip(log_0, -20, 20)), np.exp(np.clip(log_1, -20, 20))], axis=1)

  def a_esquema(self):
    return {"backend": self.nombre, "equipos": self.equipos, "beta0": self.beta0, "gamma": self.gamma,
            "sigma_attack": self.sigma_attack, "sigma_defense": self.sigma_defense,
            "alpha": {str(e): float(a) for e, a in zip(self.equipos, self.alpha)},
            "delta": {str(e): float(d) for e, d in zip(self.equipos, self.delta)},
            "convergencia": getattr(self, "convergencia", {})}

  @classmethod
  def desde_esquema(cls, esquema, carpeta):
    equipos = [int(e) for e in esquema["equipos"]]
    alpha = [float(esquema["alpha"][str(e)]) for e in equipos]
    delta = [float(esquema["delta"][str(e)]) for e in equipos]
    return cls(equipos, esquema["beta0"], esquema["gamma"], alpha, delta,
               esquema.get("sigma_attack", 1.0), esquema.get("sigma_defense", 1.0))


PREDICTORES = {
  PredictorJerarquico.nombre: PredictorJerarquico.desde_esquema,
}
