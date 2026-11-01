"""
Modelo jerarquico temporal de goles (JH-02): estados por (seleccion, ciclo) con random walk.

PredictorJerarquico estima por maxima a posteriori (MAP) el ataque (alpha) y la defensa (delta) de cada seleccion EN
CADA CICLO de Mundial (agno), no un unico estado por equipo. Un equipo solo tiene estado en los ciclos donde aparece
(el par (id, agno) esta presente en el dataset). Los priors y el random walk suavizan la estimacion:

  - partial pooling: el PRIMER estado observado de cada equipo se ancla a Normal(0, sigma_attack) para alpha y
    Normal(0, sigma_defense) para delta; los equipos con poca historia se encogen hacia la media del pool (0).
  - random walk: entre ciclos consecutivos observados de un mismo equipo se penaliza
    (estado_siguiente - estado_anterior)^2 / (2 * sigma_rw^2), tanto para alpha como para delta.

Convencion de signo (fijada en JH-01): delta entra ADITIVO en el log-rate (`alpha_atacante + delta_defensor`), por lo
que un delta POSITIVO significa "concede mas goles" (debilidad defensiva). Es la misma convencion de PredictorDixonColes.

Verosimilitud (solo goles), para un partido del ciclo t:
  log_lambda_home = beta0 + alpha_{home,t} + delta_{away,t} + gamma * local_0
  log_lambda_away = beta0 + alpha_{away,t} + delta_{home,t} + gamma * local_1
  goles ~ Poisson(exp(clip(log_lambda, -20, 20)))

El MAP se resuelve con scipy.optimize.minimize (L-BFGS-B) y gradiente analitico, igual que JH-01. Al predecir, el estado
de un equipo en su ciclo se extrapola SIN deriva: se usa el ultimo ciclo observado en entrenamiento que sea <= al ciclo
pedido. Un equipo desconocido usa la media del prior (alpha=delta=0). La version JH-01 (estatica) se recupera con
sigma_rw muy pequeno (ver tests). Se registra con nombre "jerarquico" a traves de PREDICTORES.
"""
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from .modelos_estadisticos import _columna


def _neg_log_posterior_temporal(theta, ia_home, ia_away, id_home, id_away, y_0, y_1, local_0, local_1,
                                sigma_attack, sigma_defense, sigma_rw, primeros, rw_previos, rw_siguientes):
  """
  Valor y gradiente de la negativa de la log-posterior del modelo jerarquico temporal.

  theta = (beta0, gamma, alpha_0..alpha_{P-1}, delta_0..delta_{P-1}) con P = numero de pares (equipo, ciclo).
  (ia_home)/(ia_away) son los indices del par (equipo, ciclo) que aporta el ataque de home/away, y (id_home)/(id_away)
  los que aportan la defensa de home/away (misma estructura: el defensor de home es el away). (primeros) contiene los
  indices de los estados que son el primer ciclo observado de su equipo (anclados a Normal(0, sigma)); (rw_previos) y
  (rw_siguientes) son los bordes del random walk (ciclos consecutivos observados de cada equipo). Devuelve (valor, grad).
  """
  P = (len(theta) - 2) // 2
  beta0, gamma = theta[0], theta[1]
  alpha = theta[2:2 + P]
  delta = theta[2 + P:2 + 2 * P]
  sa2, sd2, srw2 = sigma_attack ** 2, sigma_defense ** 2, sigma_rw ** 2

  log_0 = beta0 + alpha[ia_home] + delta[id_away] + gamma * local_0
  log_1 = beta0 + alpha[ia_away] + delta[id_home] + gamma * local_1
  lambda_0 = np.exp(np.clip(log_0, -20, 20))
  lambda_1 = np.exp(np.clip(log_1, -20, 20))

  ll = float(np.sum(y_0 * np.log(lambda_0) - lambda_0 - gammaln(y_0 + 1)
                    + y_1 * np.log(lambda_1) - lambda_1 - gammaln(y_1 + 1)))

  # priors: intercepto, ventaja de local y ancla del primer estado de cada equipo (partial pooling)
  log_prior = -0.5 * (beta0 ** 2 / 9.0 + gamma ** 2
                      + float(np.sum(alpha[primeros] ** 2)) / sa2
                      + float(np.sum(delta[primeros] ** 2)) / sd2)
  # random walk: suavizado temporal entre ciclos consecutivos de un mismo equipo
  if len(rw_previos) > 0:
    da = alpha[rw_siguientes] - alpha[rw_previos]
    dd = delta[rw_siguientes] - delta[rw_previos]
    log_prior += -0.5 * (float(np.sum(da ** 2)) + float(np.sum(dd ** 2))) / srw2

  # gradiente de la verosimilitud: d(log_l)/d(alpha_j) acumula (y - lambda) de los lados que usan ese estado
  g0 = y_0 - lambda_0
  g1 = y_1 - lambda_1
  g_alpha = np.zeros(P)
  g_delta = np.zeros(P)
  np.add.at(g_alpha, ia_home, g0)
  np.add.at(g_alpha, ia_away, g1)
  np.add.at(g_delta, id_away, g0)
  np.add.at(g_delta, id_home, g1)

  grad = np.empty_like(theta)
  grad[0] = -(g0.sum() + g1.sum()) + beta0 / 9.0
  grad[1] = -float(np.sum(g0 * local_0 + g1 * local_1)) + gamma
  g_a = -g_alpha
  g_d = -g_delta
  # ancla del primer estado: d(-0.5 * a^2 / sa2)/da = -a / sa2
  g_a[primeros] += alpha[primeros] / sa2
  g_d[primeros] += delta[primeros] / sd2
  # random walk: d(sum(da^2)/(2*srw2))/d(siguiente) = da/srw2 y d(...)/d(anterior) = -da/srw2
  if len(rw_previos) > 0:
    da = alpha[rw_siguientes] - alpha[rw_previos]
    dd = delta[rw_siguientes] - delta[rw_previos]
    np.add.at(g_a, rw_siguientes, da / srw2)
    np.add.at(g_a, rw_previos, -da / srw2)
    np.add.at(g_d, rw_siguientes, dd / srw2)
    np.add.at(g_d, rw_previos, -dd / srw2)
  grad[2:2 + P] = g_a
  grad[2 + P:2 + 2 * P] = g_d
  return -(ll + log_prior), grad


def _neg_log_posterior(theta, home, away, y_0, y_1, local_0, local_1, sigma_attack, sigma_defense):
  """
  Version estatica (JH-01) de la negativa de la log-posterior: un unico estado por equipo y sin random walk.

  Se conserva como caso particular del modelo temporal (una unica observacion por equipo y sin bordes de random walk) y
  por compatibilidad con el test de gradiente de JH-01. (home)/(away) son los indices de equipo dentro del mapa
  id->indice y valen tanto para el ataque como para la defensa.
  """
  P = (len(theta) - 2) // 2
  primeros = np.arange(P, dtype=int)
  rw_previos = np.array([], dtype=int)
  rw_siguientes = np.array([], dtype=int)
  return _neg_log_posterior_temporal(theta, home, away, home, away, y_0, y_1, local_0, local_1,
                                     sigma_attack, sigma_defense, 1.0, primeros, rw_previos, rw_siguientes)


class PredictorJerarquico:
  """
  ¿QUE HACE?
  Modelo de goles jerarquico temporal (state-space): para cada seleccion y cada ciclo de Mundial estima un ataque
  (alpha_{i,t}) y una defensa (delta_{i,t}) con partial pooling (el primer estado se ancla a Normal(0, sigma)) y
  random walk entre ciclos consecutivos (suavizado temporal con sigma_rw). Predice los goles esperados de home y away
  como dos Poisson independientes.

  ¿COMO LO HACE?
  Maximiza la posterior (MAP) con scipy.optimize (L-BFGS-B) y gradiente analitico. Al predecir, el estado de un equipo
  en un ciclo se extrapola sin deriva: se usa el ultimo ciclo observado en entrenamiento que sea <= al ciclo pedido.
  Un equipo desconocido usa la media del prior (alpha=delta=0). No usa xG ni features de metricas.

  Verosimilitud (solo goles):
    lambda_home = exp(beta0 + alpha_{home,t} + delta_{away,t} + gamma * local_0)
    lambda_away = exp(beta0 + alpha_{away,t} + delta_{home,t} + gamma * local_1)

  Priors: beta0 ~ Normal(0, 3), gamma ~ Normal(0, 1); el primer estado de cada equipo ~ Normal(0, sigma_attack) para
  alpha y Normal(0, sigma_defense) para delta; entre ciclos consecutivos de un equipo se penaliza el salto con
  sigma_rw. Con sigma_rw muy pequeno se recupera el modelo estatico de JH-01 (estados casi constantes por equipo).
  """
  nombre = "jerarquico"

  def __init__(self, estados, beta0, gamma, sigma_attack=1.0, sigma_defense=1.0, sigma_rw=0.5):
    self.estados = {}
    for equipo, v in estados.items():
      ciclos = [int(c) for c in v["ciclos"]]
      orden = np.argsort(ciclos)
      self.estados[int(equipo)] = {
        "ciclos": [ciclos[i] for i in orden],
        "alpha": np.asarray(v["alpha"], dtype=float)[orden],
        "delta": np.asarray(v["delta"], dtype=float)[orden],
      }
    self.beta0 = float(beta0)
    self.gamma = float(gamma)
    self.sigma_attack = float(sigma_attack)
    self.sigma_defense = float(sigma_defense)
    self.sigma_rw = float(sigma_rw)
    self.equipos = sorted(self.estados.keys())
    self.n = len(self.equipos)
    self.indice = {equipo: i for i, equipo in enumerate(self.equipos)}
    # compatibilidad JH-01: exponer el estado del primer ciclo observado de cada equipo como alpha/delta por equipo
    self.alpha = np.array([self.estados[e]["alpha"][0] for e in self.equipos])
    self.delta = np.array([self.estados[e]["delta"][0] for e in self.equipos])
    self._ciclos = {e: np.asarray(self.estados[e]["ciclos"], dtype=int) for e in self.equipos}

  @classmethod
  def entrenar(cls, dataset, sigma_attack=1.0, sigma_defense=1.0, sigma_rw=0.5, maxiter=2000):
    if sigma_attack <= 0 or sigma_defense <= 0:
      raise ValueError("sigma_attack y sigma_defense deben ser positivos")
    if sigma_rw <= 0:
      raise ValueError("sigma_rw debe ser positivo")
    home_id = dataset["home_id"].astype(int).to_numpy()
    away_id = dataset["away_id"].astype(int).to_numpy()
    agno = _columna(dataset, "agno", 0.0).astype(int)
    y_0 = _columna(dataset, "goles_0")
    y_1 = _columna(dataset, "goles_1")
    local_0 = _columna(dataset, "local_0")
    local_1 = _columna(dataset, "local_1")

    # pares (equipo, ciclo) presentes: P estados en el bloque de ataque y otros P en el de defensa
    pares = sorted(set(zip(home_id.tolist(), agno.tolist())) | set(zip(away_id.tolist(), agno.tolist())))
    indice_par = {par: i for i, par in enumerate(pares)}
    P = len(pares)
    equipos = sorted(set(home_id.tolist()) | set(away_id.tolist()))
    if len(equipos) < 2:
      raise ValueError("El modelo jerarquico necesita al menos 2 equipos")

    n = len(dataset)
    ia_home = np.empty(n, dtype=int)
    ia_away = np.empty(n, dtype=int)
    for k in range(n):
      ia_home[k] = indice_par[(int(home_id[k]), int(agno[k]))]
      ia_away[k] = indice_par[(int(away_id[k]), int(agno[k]))]
    id_home = ia_home.copy()
    id_away = ia_away.copy()

    # por equipo: sus ciclos observados ordenados; el primero se ancla y los consecutivos forman el random walk
    ciclos = defaultdict(list)
    for par, i in indice_par.items():
      ciclos[par[0]].append((par[1], i))
    primeros = []
    rw_previos = []
    rw_siguientes = []
    for equipo in equipos:
      items = sorted(ciclos[equipo])  # ordena por (ciclo, indice)
      primeros.append(items[0][1])
      for (_, i_antes), (_, i_despues) in zip(items, items[1:]):
        rw_previos.append(i_antes)
        rw_siguientes.append(i_despues)
    primeros = np.array(primeros, dtype=int)
    rw_previos = np.array(rw_previos, dtype=int)
    rw_siguientes = np.array(rw_siguientes, dtype=int)

    # arranque determinista: intercepto en la media de goles y una leve ventaja local (L-BFGS-B no es estocastico)
    media = float((y_0.sum() + y_1.sum()) / max(2 * n, 1))
    theta_0 = np.zeros(2 + 2 * P)
    theta_0[0] = np.log(max(media, 1e-3))
    theta_0[1] = 0.2
    limites = [(-5.0, 5.0), (-2.0, 2.0)] + [(-5.0, 5.0)] * (2 * P)

    def objetivo(theta):
      return _neg_log_posterior_temporal(theta, ia_home, ia_away, id_home, id_away, y_0, y_1, local_0, local_1,
                                         sigma_attack, sigma_defense, sigma_rw, primeros, rw_previos, rw_siguientes)

    ajuste = minimize(objetivo, theta_0, method="L-BFGS-B", jac=True, bounds=limites,
                      options={"maxiter": maxiter, "ftol": 1e-9})

    estados = {}
    for equipo in equipos:
      items = sorted(ciclos[equipo])
      idx = np.array([i for _, i in items], dtype=int)
      estados[equipo] = {"ciclos": [c for c, _ in items],
                         "alpha": ajuste.x[2 + idx].tolist(),
                         "delta": ajuste.x[2 + P + idx].tolist()}
    predictor = cls(estados, ajuste.x[0], ajuste.x[1], sigma_attack, sigma_defense, sigma_rw)
    predictor.convergencia = {"ok": bool(ajuste.success), "iteraciones": int(ajuste.nit), "mensaje": str(ajuste.message)}
    return predictor

  def _estado_equipo(self, equipo, agno, cual):
    """Estado (cual = 'alpha' o 'delta') del equipo en el ultimo ciclo observado <= agno; 0 si no hay ninguno"""
    ciclos = self._ciclos.get(int(equipo))
    if ciclos is None:
      return 0.0
    pos = int(np.searchsorted(ciclos, int(agno), side="right")) - 1
    if pos < 0:
      return 0.0
    return float(self.estados[int(equipo)][cual][pos])

  def _estado_final(self, equipo, cual):
    """Estado del equipo en su ultimo ciclo observado (extrapolacion sin deriva); 0 si el equipo es desconocido"""
    ciclos = self._ciclos.get(int(equipo))
    if ciclos is None or len(ciclos) == 0:
      return 0.0
    return float(self.estados[int(equipo)][cual][-1])

  def predecir(self, lote):
    home_id = lote.tabla["home_id"].astype(int).to_numpy()
    away_id = lote.tabla["away_id"].astype(int).to_numpy()
    local_0 = _columna(lote.tabla, "local_0")
    local_1 = _columna(lote.tabla, "local_1")
    n = len(lote.tabla)
    if "agno" in lote.tabla.columns:
      agno = _columna(lote.tabla, "agno", 0.0).astype(int)
      alpha_home = np.array([self._estado_equipo(home_id[k], agno[k], "alpha") for k in range(n)])
      delta_away = np.array([self._estado_equipo(away_id[k], agno[k], "delta") for k in range(n)])
      alpha_away = np.array([self._estado_equipo(away_id[k], agno[k], "alpha") for k in range(n)])
      delta_home = np.array([self._estado_equipo(home_id[k], agno[k], "delta") for k in range(n)])
    else:
      # el Lote de runtime no trae "agno": usar el ultimo ciclo observado de cada equipo (estado final), no el prior 0
      alpha_home = np.array([self._estado_final(home_id[k], "alpha") for k in range(n)])
      delta_away = np.array([self._estado_final(away_id[k], "delta") for k in range(n)])
      alpha_away = np.array([self._estado_final(away_id[k], "alpha") for k in range(n)])
      delta_home = np.array([self._estado_final(home_id[k], "delta") for k in range(n)])
    log_0 = self.beta0 + alpha_home + delta_away + self.gamma * local_0
    log_1 = self.beta0 + alpha_away + delta_home + self.gamma * local_1
    return np.stack([np.exp(np.clip(log_0, -20, 20)), np.exp(np.clip(log_1, -20, 20))], axis=1)

  def a_esquema(self):
    return {"backend": self.nombre, "beta0": self.beta0, "gamma": self.gamma,
            "sigma_attack": self.sigma_attack, "sigma_defense": self.sigma_defense, "sigma_rw": self.sigma_rw,
            "estados": {str(e): {"ciclos": [int(c) for c in self.estados[e]["ciclos"]],
                                 "alpha": [float(a) for a in self.estados[e]["alpha"]],
                                 "delta": [float(d) for d in self.estados[e]["delta"]]}
                        for e in self.equipos},
            "convergencia": getattr(self, "convergencia", {})}

  @classmethod
  def desde_esquema(cls, esquema, carpeta):
    estados = {int(e): {"ciclos": [int(c) for c in v["ciclos"]],
                        "alpha": [float(a) for a in v["alpha"]],
                        "delta": [float(d) for d in v["delta"]]}
               for e, v in esquema["estados"].items()}
    return cls(estados, esquema["beta0"], esquema["gamma"],
               esquema.get("sigma_attack", 1.0), esquema.get("sigma_defense", 1.0), esquema.get("sigma_rw", 0.5))


PREDICTORES = {
  PredictorJerarquico.nombre: PredictorJerarquico.desde_esquema,
}
