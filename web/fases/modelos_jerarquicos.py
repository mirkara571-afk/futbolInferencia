"""
Modelo jerarquico temporal de goles y xG (JH-02 + JH-03): estados por (seleccion, ciclo) con random walk y verosimilitud
xG con omision.

PredictorJerarquico estima por maxima a posteriori (MAP) el ataque (alpha) y la defensa (delta) de cada seleccion EN
CADA CICLO de Mundial (agno), no un unico estado por equipo. Un equipo solo tiene estado en los ciclos donde aparece
(el par (id, agno) esta presente en el dataset). Los priors y el random walk suavizan la estimacion:

  - partial pooling: el PRIMER estado observado de cada equipo se ancla a Normal(0, sigma_attack) para alpha y
    Normal(0, sigma_defense) para delta; los equipos con poca historia se encogen hacia la media del pool (0).
  - random walk: entre ciclos consecutivos observados de un mismo equipo se penaliza
    (estado_siguiente - estado_anterior)^2 / (2 * sigma_rw^2), tanto para alpha como para delta.

Convencion de signo (fijada en JH-01): delta entra ADITIVO en el log-rate (`alpha_atacante + delta_defensor`), por lo
que un delta POSITIVO significa "concede mas goles" (debilidad defensiva). Es la misma convencion de PredictorDixonColes.

Verosimilitud de goles, para un partido del ciclo t:
  log_lambda_home = beta0 + alpha_{home,t} + delta_{away,t} + gamma * local_0
  log_lambda_away = beta0 + alpha_{away,t} + delta_{home,t} + gamma * local_1
  goles ~ Poisson(exp(clip(log_lambda, -20, 20)))

Verosimilitud de xG (JH-03), solo donde xG_disponible == 1 y el valor no es NaN, por lado:
  log xg ~ Normal(log_lambda + eta, sigma_xg)
Es la MISMA media latente (mismos estados alpha/delta/ciclo y beta0/gamma) que los goles, con un sesgo de escala eta y
ruido sigma_xg = exp(log_sigma_xg). Donde no hay xG (toda la historia pre-2018 y los no-mundiales) el termino xG se
OMITE de la log-verosimilitud: solo contribuyen los goles. Con usar_xg=False (o sin columnas xg_*) el resultado coincide
con el de JH-02.

El MAP se resuelve con scipy.optimize.minimize (L-BFGS-B) y gradiente analitico, igual que JH-01/JH-02. Al predecir, el
estado de un equipo en su ciclo se extrapola SIN deriva: se usa el ultimo ciclo observado en entrenamiento que sea <= al
ciclo pedido. Un equipo desconocido usa la media del prior (alpha=delta=0). predecir NUNCA lee xg_* (el xG del partido
objetivo no se usa). La version JH-01 (estatica) se recupera con sigma_rw muy pequeno (ver tests). Se registra con
nombre "jerarquico" a traves de PREDICTORES.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from .modelos_estadisticos import _columna


def _neg_log_posterior_temporal(theta, ia_home, ia_away, id_home, id_away, y_0, y_1, local_0, local_1,
                                sigma_attack, sigma_defense, sigma_rw, primeros, rw_previos, rw_siguientes,
                                xg_0=None, xg_1=None):
  """
  Valor y gradiente de la negativa de la log-posterior del modelo jerarquico temporal (JH-02 + verosimilitud xG JH-03).

  theta = (beta0, gamma, eta, log_sigma_xg, alpha_0..alpha_{P-1}, delta_0..delta_{P-1}) con P = numero de pares
  (equipo, ciclo). (eta) es el sesgo de escala del xG y (log_sigma_xg) parametriza sigma_xg = exp(log_sigma_xg) para
  mantenerla positiva. (ia_home)/(ia_away) son los indices del par (equipo, ciclo) que aporta el ataque de home/away, y
  (id_home)/(id_away) los que aportan la defensa de home/away (misma estructura: el defensor de home es el away).
  (primeros) contiene los indices de los estados que son el primer ciclo observado de su equipo (anclados a
  Normal(0, sigma)); (rw_previos) y (rw_siguientes) son los bordes del random walk (ciclos consecutivos observados de
  cada equipo). (xg_0)/(xg_1) son arrays de la misma longitud que los goles con NaN donde no hay xG; si son None (o no
  tienen ningun valor finito) el termino xG se omite y la funcion degenera en la de JH-02. Devuelve (valor, grad).
  """
  P = (len(theta) - 4) // 2
  beta0, gamma, eta, log_sigma_xg = theta[0], theta[1], theta[2], theta[3]
  alpha = theta[4:4 + P]
  delta = theta[4 + P:4 + 2 * P]
  sigma_xg = np.exp(log_sigma_xg)
  sa2, sd2, srw2 = sigma_attack ** 2, sigma_defense ** 2, sigma_rw ** 2

  log_0 = beta0 + alpha[ia_home] + delta[id_away] + gamma * local_0
  log_1 = beta0 + alpha[ia_away] + delta[id_home] + gamma * local_1
  lambda_0 = np.exp(np.clip(log_0, -20, 20))
  lambda_1 = np.exp(np.clip(log_1, -20, 20))

  ll = float(np.sum(y_0 * np.log(lambda_0) - lambda_0 - gammaln(y_0 + 1)
                    + y_1 * np.log(lambda_1) - lambda_1 - gammaln(y_1 + 1)))

  # priors: intercepto, ventaja de local, sesgo/ruido del xG y ancla del primer estado de cada equipo (partial pooling)
  log_prior = -0.5 * (beta0 ** 2 / 9.0 + gamma ** 2 + eta ** 2 + log_sigma_xg ** 2
                      + float(np.sum(alpha[primeros] ** 2)) / sa2
                      + float(np.sum(delta[primeros] ** 2)) / sd2)
  # random walk: suavizado temporal entre ciclos consecutivos de un mismo equipo
  if len(rw_previos) > 0:
    da = alpha[rw_siguientes] - alpha[rw_previos]
    dd = delta[rw_siguientes] - delta[rw_previos]
    log_prior += -0.5 * (float(np.sum(da ** 2)) + float(np.sum(dd ** 2))) / srw2

  # gradiente de la verosimilitud de goles: d(log_l)/d(alpha_j) acumula (y - lambda) de los lados que usan ese estado
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
  grad[2] = eta
  grad[3] = log_sigma_xg
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
  grad[4:4 + P] = g_a
  grad[4 + P:4 + 2 * P] = g_d

  # termino xG: contribuye solo donde hay observacion disponible (xG_disponible=1 y valor finito), por lado
  if xg_0 is not None and xg_1 is not None:
    ok_0 = np.isfinite(xg_0) & (xg_0 > 0)
    ok_1 = np.isfinite(xg_1) & (xg_1 > 0)
    r_0 = np.log(xg_0[ok_0]) - (log_0[ok_0] + eta)
    r_1 = np.log(xg_1[ok_1]) - (log_1[ok_1] + eta)
    # contribucion a la negativa de la log-posterior: 0.5*(r/sigma_xg)^2 + log(sigma_xg) por observacion
    xg_contrib = float(0.5 * (np.sum(r_0 ** 2) + np.sum(r_1 ** 2)) / sigma_xg ** 2
                       + (ok_0.sum() + ok_1.sum()) * np.log(sigma_xg))
    # h = -r/sigma^2 = d(contribucion)/d(log_lambda) = d(contribucion)/d(eta)
    h_0 = -r_0 / sigma_xg ** 2
    h_1 = -r_1 / sigma_xg ** 2
    grad[0] += float(np.sum(h_0) + np.sum(h_1))
    grad[1] += float(np.sum(h_0 * local_0[ok_0]) + np.sum(h_1 * local_1[ok_1]))
    grad[2] += float(np.sum(h_0) + np.sum(h_1))
    grad[3] += float((ok_0.sum() + ok_1.sum()) - (np.sum(r_0 ** 2) + np.sum(r_1 ** 2)) / sigma_xg ** 2)
    g_xg_alpha = np.zeros(P)
    g_xg_delta = np.zeros(P)
    np.add.at(g_xg_alpha, ia_home[ok_0], h_0)
    np.add.at(g_xg_alpha, ia_away[ok_1], h_1)
    np.add.at(g_xg_delta, id_away[ok_0], h_0)
    np.add.at(g_xg_delta, id_home[ok_1], h_1)
    grad[4:4 + P] += g_xg_alpha
    grad[4 + P:4 + 2 * P] += g_xg_delta
    return -(ll + log_prior) + xg_contrib, grad

  return -(ll + log_prior), grad


def _neg_log_posterior(theta, home, away, y_0, y_1, local_0, local_1, sigma_attack, sigma_defense):
  """
  Version estatica (JH-01) de la negativa de la log-posterior: un unico estado por equipo y sin random walk ni xG.

  Se conserva como caso particular del modelo temporal (una unica observacion por equipo y sin bordes de random walk) y
  por compatibilidad con el test de gradiente de JH-01. (home)/(away) son los indices de equipo dentro del mapa
  id->indice y valen tanto para el ataque como para la defensa. Recibe el layout estatico
  theta = (beta0, gamma, alpha..., delta...) y lo re-empaqueta al layout temporal con eta = log_sigma_xg = 0 (su prior),
  descartando despues esas dos componentes del gradiente.
  """
  P = (len(theta) - 2) // 2
  primeros = np.arange(P, dtype=int)
  rw_previos = np.array([], dtype=int)
  rw_siguientes = np.array([], dtype=int)
  theta_ext = np.concatenate([theta[:2], np.array([0.0, 0.0]), theta[2:]])
  valor, grad_ext = _neg_log_posterior_temporal(theta_ext, home, away, home, away, y_0, y_1, local_0, local_1,
                                                sigma_attack, sigma_defense, 1.0, primeros, rw_previos, rw_siguientes)
  return valor, np.concatenate([grad_ext[:2], grad_ext[4:]])


class PredictorJerarquico:
  """
  ¿QUE HACE?
  Modelo de goles jerarquico temporal (state-space): para cada seleccion y cada ciclo de Mundial estima un ataque
  (alpha_{i,t}) y una defensa (delta_{i,t}) con partial pooling (el primer estado se ancla a Normal(0, sigma)) y
  random walk entre ciclos consecutivos (suavizado temporal con sigma_rw). Ademas incorpora el xG (2018/2022) como
  observacion log-normal de la MISMA media latente, con sesgo eta y ruido sigma_xg, solo donde xG_disponible == 1.
  Predice los goles esperados de home y away como dos Poisson independientes.

  ¿COMO LO HACE?
  Maximiza la posterior (MAP) con scipy.optimize (L-BFGS-B) y gradiente analitico. Al predecir, el estado de un equipo
  en un ciclo se extrapola sin deriva: se usa el ultimo ciclo observado en entrenamiento que sea <= al ciclo pedido.
  Un equipo desconocido usa la media del prior (alpha=delta=0). predecir NUNCA lee xg_* (el xG del partido objetivo no
  se usa) ni features de metricas.

  Verosimilitud:
    lambda_home = exp(beta0 + alpha_{home,t} + delta_{away,t} + gamma * local_0)
    lambda_away = exp(beta0 + alpha_{away,t} + delta_{home,t} + gamma * local_1)
    goles ~ Poisson(lambda); log xg ~ Normal(log(lambda) + eta, sigma_xg) solo donde xG_disponible == 1.

  Priors: beta0 ~ Normal(0, 3), gamma ~ Normal(0, 1), eta ~ Normal(0, 1), log_sigma_xg ~ Normal(0, 1); el primer estado
  de cada equipo ~ Normal(0, sigma_attack) para alpha y Normal(0, sigma_defense) para delta; entre ciclos consecutivos
  de un equipo se penaliza el salto con sigma_rw. Con sigma_rw muy pequeno se recupera el modelo estatico de JH-01
  (estados casi constantes por equipo).
  """
  nombre = "jerarquico"

  def __init__(self, estados, beta0, gamma, sigma_attack=1.0, sigma_defense=1.0, sigma_rw=0.5,
               eta=0.0, sigma_xg=1.0, usar_xg=False):
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
    self.eta = float(eta)
    self.sigma_xg = float(sigma_xg)
    self.usar_xg = bool(usar_xg)
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
  def entrenar(cls, dataset, sigma_attack=1.0, sigma_defense=1.0, sigma_rw=0.5, usar_xg=True, maxiter=2000):
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

    # verosimilitud xG (JH-03): solo si usar_xg=True y el dataset trae xg_0/xg_1/xG_disponible con observaciones reales;
    # donde xG_disponible != 1 (o el valor es NaN) se omite el termino. Sin observaciones el resultado coincide con JH-02.
    xg_0 = None
    xg_1 = None
    if usar_xg and all(c in dataset.columns for c in ("xg_0", "xg_1", "xG_disponible")):
      xg_0 = pd.to_numeric(dataset["xg_0"], errors="coerce").to_numpy(dtype=float)
      xg_1 = pd.to_numeric(dataset["xg_1"], errors="coerce").to_numpy(dtype=float)
      disp = pd.to_numeric(dataset["xG_disponible"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
      xg_0 = np.where((disp == 1.0) & (xg_0 > 0), xg_0, np.nan)
      xg_1 = np.where((disp == 1.0) & (xg_1 > 0), xg_1, np.nan)
      if not (np.isfinite(xg_0).any() or np.isfinite(xg_1).any()):
        xg_0 = None
        xg_1 = None

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
    theta_0 = np.zeros(4 + 2 * P)
    theta_0[0] = np.log(max(media, 1e-3))
    theta_0[1] = 0.2
    theta_0[2] = 0.0   # eta
    theta_0[3] = 0.0   # log_sigma_xg (sigma_xg = 1)
    limites = [(-5.0, 5.0), (-2.0, 2.0), (-5.0, 5.0), (-5.0, 5.0)] + [(-5.0, 5.0)] * (2 * P)

    def objetivo(theta):
      return _neg_log_posterior_temporal(theta, ia_home, ia_away, id_home, id_away, y_0, y_1, local_0, local_1,
                                         sigma_attack, sigma_defense, sigma_rw, primeros, rw_previos, rw_siguientes,
                                         xg_0=xg_0, xg_1=xg_1)

    ajuste = minimize(objetivo, theta_0, method="L-BFGS-B", jac=True, bounds=limites,
                      options={"maxiter": maxiter, "ftol": 1e-9})

    estados = {}
    for equipo in equipos:
      items = sorted(ciclos[equipo])
      idx = np.array([i for _, i in items], dtype=int)
      estados[equipo] = {"ciclos": [c for c, _ in items],
                         "alpha": ajuste.x[4 + idx].tolist(),
                         "delta": ajuste.x[4 + P + idx].tolist()}
    predictor = cls(estados, ajuste.x[0], ajuste.x[1], sigma_attack, sigma_defense, sigma_rw,
                    eta=float(ajuste.x[2]), sigma_xg=float(np.exp(ajuste.x[3])), usar_xg=bool(xg_0 is not None))
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
            "eta": self.eta, "sigma_xg": self.sigma_xg, "usar_xg": self.usar_xg,
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
               esquema.get("sigma_attack", 1.0), esquema.get("sigma_defense", 1.0), esquema.get("sigma_rw", 0.5),
               eta=esquema.get("eta", 0.0), sigma_xg=esquema.get("sigma_xg", 1.0),
               usar_xg=esquema.get("usar_xg", False))


PREDICTORES = {
  PredictorJerarquico.nombre: PredictorJerarquico.desde_esquema,
}
