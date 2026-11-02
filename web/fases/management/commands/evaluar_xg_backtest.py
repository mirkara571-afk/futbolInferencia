"""
Backtest A/B/C de xG + oraculo (plan 07, tarea XG-03).

¿QUE HACE?
"python manage.py evaluar_xg_backtest" es un diagnostico, no un modelo de produccion: mide si el xG historico
(rolling) recupera la senal que el xG del partido no aporta como lambda directa. Compara cuatro brazos con el MISMO
nucleo (Poisson GLM y LightGBM Poisson):

  - A = nucleo sin xG (referencia);
  - B = nucleo + rolling de xG REAL (los partidos previos con xG_disponible == 1);
  - C = nucleo + rolling de xG IMPUTADO (el ImputadorXG rellena donde no hay real);
  - O = oraculo (techo NO desplegable): nucleo + xg_0/xg_1 del partido objetivo.

¿COMO LO HACE?
Particion interna de diagnostico: entrena con los 64 partidos de mundial de 2018 y evalua en los 64 de 2022. Los cuatro
brazos se entrenan con las MISMAS 64 filas de 2018 (la unica diferencia es el conjunto de columnas del diseno). Para B y
C el rolling se construye sobre las columnas xg_* en orden (agno, orden) sin usar nunca el xG del partido objetivo; O usa
el xG del partido y por eso se marca no_desplegable. Cada brazo se mide con log-loss, RPS, acierto y ECE, intervalos
bootstrap y, para B/C/O, la comparacion pareada (Diebold-Mariano) contra A en log-loss y RPS (mas B-C para C). Vuelca
experimentos/resultados_xg_backtest.json.

No toca 2026, no modifica modelos existentes y no cambia el esquema de las 14 metricas.
"""
import json
import warnings
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from statsmodels.tools.sm_exceptions import SingularMatrixWarning

from fases.dataset import FASE_MUNDIAL
from fases.evaluacion import comparar, intervalos_bootstrap, metricas, metricas_por_partido
from fases.imputador_xg import ImputadorXG
from fases.modelos_estadisticos import _categorias, _columna, _matriz_glm
from fases.predictores import Lote
from fases.xg_rolling import construir_rolling

RAIZ = Path(settings.BASE_DIR).parent
DATASET = RAIZ / "data" / "datasets" / "partidos_features_xg.csv"
RESULTADOS = RAIZ / "experimentos" / "resultados_xg_backtest.json"

# columnas extra por lado: el sufijo 0 es el local y el 1 el visitante en la fila-equipo del diseno
COLS_ROLLING = ["xg_roll_for", "xg_roll_against", "xg_roll_n", "xg_roll_has"]
COLS_ORACULO = ["xg"]
# brazos -> columnas extra (sin sufijo) que agrega cada uno sobre el nucleo de "_matriz_glm"
BRAZOS = {"A": [], "B": COLS_ROLLING, "C": COLS_ROLLING, "O": COLS_ORACULO}
# el oraculo usa el xG del partido objetivo: techo no desplegable
NO_DESPLEGABLE = {"O"}

FAMILIAS = ("glm", "lightgbm")
# configuracion del LightGBM Poisson del backtest (semilla fija, la misma para A/B/C/O)
PARAMS_LGB = {"n_estimators": 200, "learning_rate": 0.05, "random_state": 0, "n_jobs": 1, "verbose": -1}


def particion_interna(dataset):
  """
  ¿QUE HACE?
  Particion interna de diagnostico: entrenamiento = fase mundial de 2018 y prueba = fase mundial de 2022 (64 y 64
  partidos). Es la particion compartida por los cuatro brazos A/B/C/O.

  Returns:
    - (filas_entrenamiento, filas_prueba): arrays booleanos sobre (dataset)
  """
  agno = dataset["agno"].to_numpy()
  fase = dataset["fase"].to_numpy()
  entrenamiento = (agno == 2018) & (fase == FASE_MUNDIAL)
  prueba = (agno == 2022) & (fase == FASE_MUNDIAL)
  return entrenamiento, prueba


def media_global_xg(ds_entrenamiento):
  """Media global de xG (los dos lados juntos) del entrenamiento, para rellenar el rolling sin historia previa"""
  valores = np.concatenate([
    pd.to_numeric(ds_entrenamiento["xg_0"], errors="coerce").to_numpy(dtype=float),
    pd.to_numeric(ds_entrenamiento["xg_1"], errors="coerce").to_numpy(dtype=float),
  ])
  return float(np.nanmean(valores))


def construir_nucleo(dataset, columnas_extra=None, categorias=None):
  """
  ¿QUE HACE?
  Devuelve el diseno del nucleo para entrenar: la matriz de "_matriz_glm" (una fila por equipo y partido, home primero y
  away despues, con intercepto "const") mas, para cada lado, las columnas de (columnas_extra) con el sufijo del lado
  (0 home, 1 away). No toca las columnas xg_* salvo que (columnas_extra) las pida explicitamente (brazo O).

  Returns:
    - (X, y, categorias): matriz de diseno (2n, m), goles apilados (2n,) y las categorias usadas para el one-hot
  """
  if categorias is None:
    categorias = _categorias(dataset)
  X = _matriz_glm(dataset, categorias)
  if columnas_extra:
    # una columna por concepto y lado: la fila-equipo del local activa el sufijo 0 y la del visitante el 1; el lado
    # inactivo queda en 0 (encoding ancho y por lado, con 8 columnas rolling para B/C y 2 para el oraculo)
    por_lado = [pd.DataFrame({f"{c}_{lado}": _columna(dataset, f"{c}_{lado}") for c in columnas_extra})
                for lado in (0, 1)]
    extra = pd.concat(por_lado, ignore_index=True).fillna(0.0)
    X = pd.concat([X.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
  y = np.concatenate([dataset["goles_0"].to_numpy(dtype=float), dataset["goles_1"].to_numpy(dtype=float)])
  return X, y, categorias


def datos_brazo(ds, filas_tr, filas_pr, brazo, media_global):
  """
  ¿QUE HACE?
  Devuelve (entrenamiento, prueba) con las columnas listas para (brazo), sin tocar (ds):

    - A y O: las mismas filas de (ds) (O ya trae xg_0/xg_1 del partido);
    - B: rolling REAL sobre entrenamiento+prueba en orden (agno, orden);
    - C: rolling sobre el dataset completo con xg imputado por un ImputadorXG entrenado solo en 2018 (filas reales).

  Los cuatro brazos comparten las mismas 64 filas de 2018 como entrenamiento y las mismas 64 de 2022 como prueba.
  """
  if brazo == "B":
    rango = pd.concat([ds[filas_tr], ds[filas_pr]], ignore_index=True).sort_values(
      ["agno", "orden"], kind="stable").reset_index(drop=True)
    rango = construir_rolling(rango, ventana=None, media_global=media_global)
    return rango[rango["agno"] == 2018].reset_index(drop=True), rango[rango["agno"] == 2022].reset_index(drop=True)
  if brazo == "C":
    imputador = ImputadorXG.entrenar(ds[filas_tr], semilla=0)
    pred = imputador.predecir(ds)
    ds_c = ds.copy()
    ds_c["xg_0"] = pred["xg_imputado_0"].to_numpy()
    ds_c["xg_1"] = pred["xg_imputado_1"].to_numpy()
    ds_c["xG_disponible"] = 1
    ds_c = construir_rolling(ds_c, ventana=None, media_global=media_global)
    return ds_c[filas_tr].reset_index(drop=True), ds_c[filas_pr].reset_index(drop=True)
  return ds[filas_tr].reset_index(drop=True), ds[filas_pr].reset_index(drop=True)


class _ModeloNucleo:
  """Predictor minimo del nucleo (GLM o LightGBM Poisson) con predecir(lote) -> (n, 2); solo para el backtest"""

  def __init__(self, columnas, categorias, columnas_extra, params=None, booster=None):
    self.columnas = list(columnas)
    self.categorias = list(categorias)
    self.columnas_extra = list(columnas_extra)
    self.params = np.asarray(params, dtype=float) if params is not None else None
    self.booster = booster

  def _X(self, tabla):
    X, _, _ = construir_nucleo(tabla, self.columnas_extra, self.categorias)
    return X.reindex(columns=self.columnas, fill_value=0.0)

  def predecir(self, lote):
    X = self._X(lote.tabla).to_numpy(dtype=float)
    lambdas = self.booster.predict(X) if self.booster is not None else np.exp(np.clip(X @ self.params, -20, 20))
    n = len(lote.tabla)
    return np.stack([lambdas[:n], lambdas[n:]], axis=1)


def entrenar_nucleo(familia, dataset, columnas_extra, categorias):
  """Ajusta el nucleo de (familia) sobre (dataset) con las columnas extra del brazo y devuelve un predictor"""
  X, y, _ = construir_nucleo(dataset, columnas_extra, categorias)
  if familia == "glm":
    # el GLM puede avisar de rango deficiente si alguna columna queda constante (p. ej. los primeros partidos de un
    # equipo sin xG previo); las predicciones siguen bien definidas, asi que solo se silencia esa advertencia
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", SingularMatrixWarning)
      ajuste = sm.GLM(y, X, family=sm.families.Poisson()).fit(maxiter=300)
    return _ModeloNucleo(X.columns, categorias, columnas_extra, params=ajuste.params)
  ajuste = lgb.LGBMRegressor(objective="poisson", **PARAMS_LGB).fit(X, y)
  return _ModeloNucleo(X.columns, categorias, columnas_extra, booster=ajuste.booster_)


def resumen_brazo(resultado, por_partido, por_partido_a=None):
  """
  ¿QUE HACE?
  Resumen de un brazo para el JSON: log-loss, RPS, acierto y ECE, intervalos bootstrap por metrica y, si hay referencia
  (por_partido_a), la comparacion pareada contra A en log-loss y RPS (Diebold-Mariano).
  """
  res = {
    "log_loss": round(float(resultado["log_loss"]), 6),
    "rps": round(float(resultado["rps"]), 6),
    "acierto": round(float(resultado["acierto_resultado"]), 4),
    "ece": round(float(resultado["ece"]), 6),
    "intervalos": intervalos_bootstrap(por_partido),
  }
  if por_partido_a is not None:
    res["p_vs_A"] = {
      "log_loss": comparar(por_partido, por_partido_a, "log_loss"),
      "rps": comparar(por_partido, por_partido_a, "rps"),
    }
  else:
    res["p_vs_A"] = None
  return res


def estructura_json(fecha, resultados):
  """Estructura del JSON de resultados con los campos principales y la marca de no desplegable del oraculo"""
  return {
    "fecha": fecha,
    "dataset": DATASET.name,
    "particion": {
      "entrenamiento": "mundial 2018 (fase mundial, 64 partidos)",
      "prueba": "mundial 2022 (fase mundial, 64 partidos)",
    },
    "familias": list(FAMILIAS),
    "brazos": resultados,
    "oraculo_no_desplegable": True,
    "nota": ("Diagnostico con n=64 por ciclo; lectura direccional, no evidencia de promocion. "
             "El brazo O (oraculo, xG del partido) es un techo NO desplegable."),
  }


class Command(BaseCommand):
  """Backtest A/B/C de xG + oraculo sobre la particion 2018 -> 2022; resultados en experimentos/resultados_xg_backtest.json."""
  help = "Backtest xG A/B/C + oraculo (GLM y LightGBM Poisson) con intervalos y Diebold-Mariano; resultados en experimentos/resultados_xg_backtest.json"
  requires_system_checks = []

  def avisar(self, texto):
    self.stdout.write(texto)
    self.stdout.flush()

  def handle(self, *args, **opciones):
    if not DATASET.exists():
      raise CommandError(f"No existe {DATASET}: el backtest xG necesita el dataset partidos_features_xg.csv")
    ds = pd.read_csv(DATASET)
    filas_tr, filas_pr = particion_interna(ds)
    if filas_tr.sum() != 64 or filas_pr.sum() != 64:
      raise CommandError(f"La particion interna esperaba 64/64 partidos y obtuvo {filas_tr.sum()}/{filas_pr.sum()}")
    self.avisar(f"Backtest xG (diagnostico): entrenamiento {filas_tr.sum()} partidos de 2018, prueba {filas_pr.sum()} de 2022")

    categorias = _categorias(ds[filas_tr])
    media_global = media_global_xg(ds[filas_tr])

    # las features de cada brazo no dependen de la familia: se construyen una vez y se reutilizan para GLM y LightGBM
    datasets = {brazo: datos_brazo(ds, filas_tr, filas_pr, brazo, media_global) for brazo in BRAZOS}

    resultados = {}
    for familia in FAMILIAS:
      por_familia = {}
      por_partido_a = None
      por_partido_por_brazo = {}
      for brazo in BRAZOS:
        ds_tr, ds_pr = datasets[brazo]
        predictor = entrenar_nucleo(familia, ds_tr, BRAZOS[brazo], categorias)
        goles = ds_pr[["goles_0", "goles_1"]].to_numpy(dtype=float)
        lambdas = predictor.predecir(Lote(ds_pr))
        resultado = metricas(goles, lambdas)
        por_partido, _, _ = metricas_por_partido(goles, lambdas)
        por_partido_por_brazo[brazo] = por_partido
        if brazo == "A":
          por_partido_a = por_partido
        res = resumen_brazo(resultado, por_partido, por_partido_a if brazo != "A" else None)
        if brazo == "C":
          # comparacion adicional del plan (XG-03, paso 4): B vs C para sostener la lectura B ≻ C o C ≻ B
          res["p_vs_B"] = {
            "log_loss": comparar(por_partido, por_partido_por_brazo["B"], "log_loss"),
            "rps": comparar(por_partido, por_partido_por_brazo["B"], "rps"),
          }
        if brazo in NO_DESPLEGABLE:
          res["no_desplegable"] = True
        por_familia[brazo] = res
        self.avisar(f"  {familia:<10} {brazo}: log-loss {res['log_loss']:.4f}  RPS {res['rps']:.4f}  "
                    f"acierto {res['acierto']:5.1f}%  ECE {res['ece']:.4f}")
      resultados[familia] = por_familia

    estructura = estructura_json(date.today().isoformat(), resultados)
    RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RESULTADOS.write_text(json.dumps(estructura, indent=2, ensure_ascii=False), encoding="utf-8")

    self.avisar("")
    self.avisar(f"{'familia':<10} {'brazo':<6} {'log-loss':>9} {'RPS':>8} {'acierto':>8} {'ECE':>8} {'p_dm vs A':>10}")
    for familia in FAMILIAS:
      for brazo in BRAZOS:
        r = resultados[familia][brazo]
        p_dm = r["p_vs_A"]["log_loss"]["p_dm"] if r["p_vs_A"] else None
        p_dm_txt = "-" if p_dm is None else f"{p_dm:.3f}"
        marca = " (NO DESPLEGABLE)" if brazo in NO_DESPLEGABLE else ""
        self.avisar(f"{familia:<10} {brazo:<6} {r['log_loss']:>9.4f} {r['rps']:>8.4f} {r['acierto']:>8.1f} "
                    f"{r['ece']:>8.4f} {p_dm_txt:>10}{marca}")
    self.avisar("")
    self.avisar("Advertencia: n=64 por ciclo; la lectura es direccional (Diebold-Mariano sin potencia en n=64).")
    self.avisar("El brazo O (oraculo, xG del partido) es un techo NO desplegable y no debe usarse en produccion.")
    self.stdout.write(self.style.SUCCESS(f"Resultados -> {RESULTADOS}"))
