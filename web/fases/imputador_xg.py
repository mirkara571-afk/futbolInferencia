"""
Imputador de xG historico (plan 07, tarea XG-02).

El xG real solo existe en 128 partidos de mundial (2018 y 2022). Para el brazo C del backtest hace falta un xG en toda
la historia; este modulo aprende a predecir log1p(xg) a partir de las features pre-partido de "_matriz_glm" (sin
intercepto) y devuelve el xG imputado en las filas donde no hay real, conservando el real donde xG_disponible == 1.

Es un modelo de diagnostico (no de produccion): se entrena en partidos de mundial y se aplica a eliminatorias/amistosos,
por lo que el domain shift se documenta como limitacion en el plan 07.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

from .modelos_estadisticos import _categorias, _columna, _matriz_glm


class ImputadorXG:
  """
  ¿QUE HACE?
  Aprende un Ridge sobre log1p(xg) con las features pre-partido de "_base_lado"/"_matriz_glm" (sin intercepto) usando
  solo las filas con xG_disponible == 1, y luego imputa el xG de todo el dataset conservando el real donde existe.

  ¿COMO LO HACE?
  0-) Filtra las filas reales, arma la matriz de diseno (una fila por equipo y partido) y apila los dos lados como
      observaciones independientes; el target es log1p(xg_0) para el local y log1p(xg_1) para el visitante.

  1-) Elige alpha de Ridge con GridSearchCV (KFold barajado con semilla fija) y guarda el modelo, las columnas y las
      categorias para reconstruir la matriz en "predecir" exactamente con las mismas features.

  2-) "metricas_imputacion" valida con la particion mas honesta disponible: si hay xG real en 2018 y en 2022 entrena
      solo en 2018 y mide en 2022 (split temporal, sin fuga entre ciclos); si no, usa cross_val_predict de 5 pliegues
      sobre las filas reales disponibles.

  Atributos:
    - modelo: Ridge ajustado sobre log1p(xg)
    - columnas, categorias: features de "_matriz_glm" sin "const" y categorias para el one-hot
    - alpha_: alpha elegido por CV
    - X_entrenamiento_, y_entrenamiento_, y_original_entrenamiento_, agnos_entrenamiento_: datos reales usados (por lado)
  """
  nombre = "imputador_xg"
  ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

  def __init__(self, modelo, columnas, categorias, alpha,
               X_entrenamiento, y_entrenamiento, y_original_entrenamiento, agnos_entrenamiento):
    self.modelo = modelo
    self.columnas = list(columnas)
    self.categorias = list(categorias)
    self.alpha_ = float(alpha)
    self.X_entrenamiento_ = np.asarray(X_entrenamiento, dtype=float)
    self.y_entrenamiento_ = np.asarray(y_entrenamiento, dtype=float)
    self.y_original_entrenamiento_ = np.asarray(y_original_entrenamiento, dtype=float)
    self.agnos_entrenamiento_ = np.asarray(agnos_entrenamiento, dtype=float)

  @classmethod
  def _ajustar(cls, X, y, semilla=0):
    """Ajusta Ridge eligiendo alpha por GridSearchCV (KFold barajado, semilla fija) y devuelve (modelo, alpha)"""
    if len(y) < 5:
      return Ridge(alpha=1.0).fit(X, y), 1.0
    cv = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=int(semilla))
    busqueda = GridSearchCV(Ridge(), {"alpha": cls.ALPHAS}, cv=cv, scoring="neg_mean_squared_error")
    busqueda.fit(X, y)
    return busqueda.best_estimator_, float(busqueda.best_params_["alpha"])

  @classmethod
  def entrenar(cls, dataset, semilla=0):
    """
    ¿QUE HACE?
    Ajusta el Ridge sobre las 256 observaciones reales de xG (128 partidos x 2 lados) del dataset recibido.

    ¿COMO LO HACE?
    Filtra xG_disponible == 1, arma "_matriz_glm" sin la columna "const", apila los dos lados como observaciones y
    ajusta sobre log1p(xg). Guarda el modelo y los datos reales para poder validar despues sin re-leer el CSV.
    """
    disp = _columna(dataset, "xG_disponible") != 0
    real = dataset.iloc[np.flatnonzero(disp)]
    if len(real) == 0:
      raise ValueError("El dataset no tiene filas con xG_disponible == 1; no se puede entrenar el imputador")

    categorias = _categorias(dataset)
    X = _matriz_glm(real, categorias).drop(columns=["const"])
    columnas = list(X.columns)
    xg_0 = _columna(real, "xg_0")
    xg_1 = _columna(real, "xg_1")
    y_original = np.concatenate([xg_0, xg_1])
    y = np.log1p(y_original)
    agnos = np.concatenate([_columna(real, "agno"), _columna(real, "agno")])

    modelo, alpha = cls._ajustar(X.to_numpy(dtype=float), y, semilla)
    return cls(modelo, columnas, categorias, alpha, X.to_numpy(dtype=float), y, y_original, agnos)

  def _X(self, dataset):
    """Matriz de diseno de (dataset) con las mismas columnas del entrenamiento (sin intercepto)"""
    X = _matriz_glm(dataset, self.categorias).drop(columns=["const"])
    return X.reindex(columns=self.columnas, fill_value=0.0)

  def predecir(self, dataset):
    """
    ¿QUE HACE?
    Devuelve un DataFrame con xg_imputado_0 y xg_imputado_1 para todas las filas: el xG real donde xG_disponible == 1 y
    expm1(prediccion) en el resto. No toca tiros/sot.

    ¿COMO LO HACE?
    Aplica el Ridge a la matriz de diseno (una fila por equipo y partido) y deshace log1p con expm1; la mascara de
    xG_disponible decide que filas conservan el valor real.
    """
    X = self._X(dataset).to_numpy(dtype=float)
    pred_log = self.modelo.predict(X)
    n = len(dataset)
    disp = _columna(dataset, "xG_disponible") != 0
    xg_0 = _columna(dataset, "xg_0", np.nan)
    xg_1 = _columna(dataset, "xg_1", np.nan)
    # el xG no puede ser negativo: expm1 puede bajar de 0 con features fuera de la distribucion del entrenamiento
    out_0 = np.where(disp, xg_0, np.maximum(0.0, np.expm1(pred_log[:n])))
    out_1 = np.where(disp, xg_1, np.maximum(0.0, np.expm1(pred_log[n:])))
    return pd.DataFrame({"xg_imputado_0": out_0, "xg_imputado_1": out_1}, index=dataset.index)

  @staticmethod
  def _metricas(pred_log, y_log, y_original, metodo):
    """MAE, RMSE y R2 en escala log1p y en escala original (expm1) de una validacion"""
    pred_original = np.expm1(pred_log)
    return {
      "metodo": metodo,
      "n": int(len(y_log)),
      "mae_log1p": float(mean_absolute_error(y_log, pred_log)),
      "rmse_log1p": float(np.sqrt(mean_squared_error(y_log, pred_log))),
      "r2_log1p": float(r2_score(y_log, pred_log)),
      "mae_original": float(mean_absolute_error(y_original, pred_original)),
      "rmse_original": float(np.sqrt(mean_squared_error(y_original, pred_original))),
      "r2_original": float(r2_score(y_original, pred_original)),
    }

  def metricas_imputacion(self):
    """
    ¿QUE HACE?
    Reporta MAE, RMSE y R2 del imputador en escala log1p y original, con validacion interna sobre las filas reales.

    ¿COMO LO HACE?
    Elige la validacion mas honesta: si hay xG real en 2018 y en 2022, entrena solo en 2018 (re-elegiendo alpha por CV
    dentro de 2018) y mide en 2022; si no, usa cross_val_predict de 5 pliegues sobre las filas reales disponibles.
    """
    agnos = self.agnos_entrenamiento_
    X, y, y_original = self.X_entrenamiento_, self.y_entrenamiento_, self.y_original_entrenamiento_
    if np.any(agnos <= 2018) and np.any(agnos >= 2022):
      tr = agnos <= 2018
      te = agnos >= 2022
      modelo, _ = self._ajustar(X[tr], y[tr])
      pred = modelo.predict(X[te])
      return self._metricas(pred, y[te], y_original[te], "temporal_2018_a_2022")
    cv = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=0)
    pred = cross_val_predict(Ridge(alpha=self.alpha_), X, y, cv=cv)
    return self._metricas(pred, y, y_original, "cv_5_folds")
