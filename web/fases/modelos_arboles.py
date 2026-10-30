"""
Modelos de arboles de la Etapa C: LightGBM (Poisson y multiclase) y Random Forest.

Comparten con los modelos clasicos las features de "modelos_estadisticos" y el contrato de "predictores.py".
LightGBM Poisson predice goles por equipo; LightGBM multiclase y Random Forest predicen el resultado y usan un
PredictorPoissonGLM para el marcador.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier

from .modelos_estadisticos import PredictorPoissonGLM, _categorias, _columna, _matriz_glm
from .modulo import ALL_METRICS

ARCHIVO_GOLES = "modelo_goles.txt"
ARCHIVO_RESULTADO = "modelo_resultado.txt"
ARCHIVO_RF = "modelo_rf.joblib"


def _matriz_partido(tabla, categorias):
  """Una fila por partido con las 14 metricas de los dos lados, Elo, localia, contexto y categoria"""
  datos = {
    "elo_0": _columna(tabla, "elo_0", 1500.0), "elo_1": _columna(tabla, "elo_1", 1500.0),
    "elo_diff": (_columna(tabla, "elo_0", 1500.0) - _columna(tabla, "elo_1", 1500.0)) / 100.0,
    "local_0": _columna(tabla, "local_0"), "local_1": _columna(tabla, "local_1"),
    "neutral": _columna(tabla, "neutral", 1.0), "fase_final": _columna(tabla, "fase_final"),
  }
  for metrica in ALL_METRICS:
    for i in (0, 1):
      datos[f"{metrica}_{i}"] = _columna(tabla, f"{metrica}_{i}")
  categoria = tabla["categoria"].astype(str) if "categoria" in tabla else pd.Series([""] * len(tabla), index=tabla.index)
  for cat in categorias:
    datos[f"cat_{cat}"] = (categoria == cat).to_numpy(dtype=float)
  return pd.DataFrame(datos, index=tabla.index)


class PredictorLightGBMPoisson:
  """
  ¿QUE HACE?
  LightGBM con objetivo Poisson sobre una fila por equipo y partido; predice goles esperados a partir de Elo, localia,
  tipo de partido y las metricas propias/del rival

  ¿COMO LO HACE?
  Entrena con la API sklearn de LightGBM, guarda el modelo como texto (modelo_goles.txt) y las columnas en esquema.json
  """
  nombre = "lightgbm_poisson"

  def __init__(self, booster, columnas, categorias, params):
    self.booster = booster
    self.columnas = list(columnas)
    self.categorias = list(categorias)
    self.params = dict(params)

  @classmethod
  def entrenar(cls, dataset, params=None, validacion=None):
    categorias = _categorias(dataset)
    X = _matriz_glm(dataset, categorias)
    y = np.concatenate([dataset["goles_0"].to_numpy(dtype=float), dataset["goles_1"].to_numpy(dtype=float)])
    parametros = {"objective": "poisson", "n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31,
                  "min_child_samples": 20, "subsample": 0.9, "subsample_freq": 1, "colsample_bytree": 0.8,
                  "random_state": 0, "n_jobs": -1, "verbose": -1}
    parametros.update(params or {})
    llamadas = []
    if validacion is not None:
      Xv = _matriz_glm(validacion, categorias).reindex(columns=X.columns, fill_value=0.0)
      yv = np.concatenate([validacion["goles_0"].to_numpy(dtype=float), validacion["goles_1"].to_numpy(dtype=float)])
      parametros["early_stopping_rounds"] = 50
      llamadas = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
      ajuste = lgb.LGBMRegressor(**parametros).fit(X, y, eval_set=[(Xv, yv)], callbacks=llamadas)
    else:
      ajuste = lgb.LGBMRegressor(**parametros).fit(X, y)
    return cls(ajuste.booster_, X.columns, categorias, parametros)

  def _X(self, lote):
    X = _matriz_glm(lote.tabla, self.categorias)
    return X.reindex(columns=self.columnas, fill_value=0.0)

  def predecir(self, lote):
    lambdas = self.booster.predict(self._X(lote).to_numpy(dtype=float))
    n = len(lote.tabla)
    return np.stack([lambdas[:n], lambdas[n:]], axis=1)

  def importancia(self, tipo="gain"):
    """Importancia de variables (ganancia por defecto) como Serie ordenada"""
    return pd.Series(self.booster.feature_importance(importance_type=tipo), index=self.columnas).sort_values(ascending=False)

  def guardar_pesos(self, carpeta):
    from pathlib import Path
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    self.booster.save_model(str(Path(carpeta) / ARCHIVO_GOLES))

  def a_esquema(self):
    return {"backend": self.nombre, "columnas": self.columnas, "categorias": self.categorias, "params": self.params}

  @classmethod
  def desde_esquema(cls, esquema, carpeta):
    from pathlib import Path
    return cls(lgb.Booster(model_file=str(Path(carpeta) / ARCHIVO_GOLES)), esquema["columnas"], esquema["categorias"],
               esquema.get("params", {}))


class PredictorLightGBMMulticlase:
  """
  ¿QUE HACE?
  LightGBM multiclase para el resultado (local / empate / visitante) y un Poisson GLM para los goles esperados

  ¿COMO LO HACE?
  Entrena el clasificador con una fila por partido y las 14 metricas de los dos lados; guarda el clasificador como
  modelo_resultado.txt y el GLM en el esquema
  """
  nombre = "lightgbm_multiclase"

  def __init__(self, clasificador, poisson, columnas, categorias, params):
    self.clasificador = clasificador
    self.poisson = poisson
    self.columnas = list(columnas)
    self.categorias = list(categorias)
    self.params = dict(params)

  @classmethod
  def entrenar(cls, dataset, params=None, validacion=None):
    poisson = PredictorPoissonGLM.entrenar(dataset)
    categorias = _categorias(dataset)
    X = _matriz_partido(dataset, categorias)
    goles_0, goles_1 = dataset["goles_0"].to_numpy(), dataset["goles_1"].to_numpy()
    y = np.where(goles_0 > goles_1, 2, np.where(goles_0 == goles_1, 1, 0))
    parametros = {"objective": "multiclass", "num_class": 3, "n_estimators": 400, "learning_rate": 0.03,
                  "num_leaves": 31, "min_child_samples": 20, "subsample": 0.9, "subsample_freq": 1,
                  "colsample_bytree": 0.8, "random_state": 0, "n_jobs": -1, "verbose": -1}
    parametros.update(params or {})
    if validacion is not None:
      Xv = _matriz_partido(validacion, categorias).reindex(columns=X.columns, fill_value=0.0)
      gv0, gv1 = validacion["goles_0"].to_numpy(), validacion["goles_1"].to_numpy()
      yv = np.where(gv0 > gv1, 2, np.where(gv0 == gv1, 1, 0))
      parametros["early_stopping_rounds"] = 50
      ajuste = lgb.LGBMClassifier(**parametros).fit(X, y, eval_set=[(Xv, yv)],
                                                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    else:
      ajuste = lgb.LGBMClassifier(**parametros).fit(X, y)
    return cls(ajuste.booster_, poisson, X.columns, categorias, parametros)

  def _X(self, lote):
    X = _matriz_partido(lote.tabla, self.categorias)
    return X.reindex(columns=self.columnas, fill_value=0.0)

  def probabilidades(self, lote):
    crudas = self.clasificador.predict(self._X(lote).to_numpy(dtype=float))
    # LightGBM devuelve columnas 0=visitante, 1=empate, 2=local; la evaluacion espera local, empate, visitante
    return crudas[:, [2, 1, 0]]

  def predecir(self, lote):
    return self.poisson.predecir(lote)

  def importancia(self, tipo="gain"):
    valores = self.clasificador.feature_importance(importance_type=tipo)
    if valores.ndim > 1:
      valores = valores.mean(axis=1)
    return pd.Series(valores, index=self.columnas).sort_values(ascending=False)

  def guardar_pesos(self, carpeta):
    from pathlib import Path
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    self.clasificador.save_model(str(Path(carpeta) / ARCHIVO_RESULTADO))

  def a_esquema(self):
    return {"backend": self.nombre, "columnas": self.columnas, "categorias": self.categorias, "params": self.params,
            "poisson": self.poisson.a_esquema()}

  @classmethod
  def desde_esquema(cls, esquema, carpeta):
    from pathlib import Path
    clasificador = lgb.Booster(model_file=str(Path(carpeta) / ARCHIVO_RESULTADO))
    return cls(clasificador, PredictorPoissonGLM.desde_esquema(esquema["poisson"], carpeta), esquema["columnas"],
               esquema["categorias"], esquema.get("params", {}))


class PredictorRandomForest:
  """
  ¿QUE HACE?
  Random Forest del resultado (local / empate / visitante) con las mismas variables que LightGBM multiclase y el
  Poisson GLM para el marcador; sirve de contraste simple

  ¿COMO LO HACE?
  Entrena sklearn RandomForestClassifier y lo guarda con joblib; el GLM va en el esquema
  """
  nombre = "random_forest"

  def __init__(self, clasificador, poisson, columnas, categorias, params):
    self.clasificador = clasificador
    self.poisson = poisson
    self.columnas = list(columnas)
    self.categorias = list(categorias)
    self.params = dict(params)

  @classmethod
  def entrenar(cls, dataset, params=None):
    poisson = PredictorPoissonGLM.entrenar(dataset)
    categorias = _categorias(dataset)
    X = _matriz_partido(dataset, categorias)
    goles_0, goles_1 = dataset["goles_0"].to_numpy(), dataset["goles_1"].to_numpy()
    y = np.where(goles_0 > goles_1, 2, np.where(goles_0 == goles_1, 1, 0))
    parametros = {"n_estimators": 400, "min_samples_leaf": 5, "max_features": "sqrt", "n_jobs": 1,
                  "random_state": 0, "class_weight": "balanced_subsample"}
    parametros.update(params or {})
    ajuste = RandomForestClassifier(**parametros).fit(X, y)
    return cls(ajuste, poisson, X.columns, categorias, parametros)

  def _X(self, lote):
    X = _matriz_partido(lote.tabla, self.categorias)
    return X.reindex(columns=self.columnas, fill_value=0.0)

  def probabilidades(self, lote):
    crudas = self.clasificador.predict_proba(self._X(lote).to_numpy(dtype=float))
    # sklearn ordena las clases 0=visitante, 1=empate, 2=local
    return crudas[:, [2, 1, 0]]

  def predecir(self, lote):
    return self.poisson.predecir(lote)

  def importancia(self):
    return pd.Series(self.clasificador.feature_importances_, index=self.columnas).sort_values(ascending=False)

  def guardar_pesos(self, carpeta):
    from pathlib import Path
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    dump(self.clasificador, Path(carpeta) / ARCHIVO_RF)

  def a_esquema(self):
    return {"backend": self.nombre, "columnas": self.columnas, "categorias": self.categorias, "params": self.params,
            "poisson": self.poisson.a_esquema()}

  @classmethod
  def desde_esquema(cls, esquema, carpeta):
    from pathlib import Path
    clasificador = load(Path(carpeta) / ARCHIVO_RF)
    return cls(clasificador, PredictorPoissonGLM.desde_esquema(esquema["poisson"], carpeta), esquema["columnas"],
               esquema["categorias"], esquema.get("params", {}))


PREDICTORES = {
  PredictorLightGBMPoisson.nombre: PredictorLightGBMPoisson.desde_esquema,
  PredictorLightGBMMulticlase.nombre: PredictorLightGBMMulticlase.desde_esquema,
  PredictorRandomForest.nombre: PredictorRandomForest.desde_esquema,
}
