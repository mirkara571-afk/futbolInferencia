"""
Ensamble de predictores (Etapa F).

Combina las probabilidades de resultado y los goles esperados de varios predictores con pesos no negativos que suman 1.
Los pesos se eligen minimizando el log-loss en una validacion (nunca en la prueba).
"""
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from .evaluacion import metricas_probabilidad, probabilidades_resultado
from .predictores import Lote, cargar_predictor, guardar_predictor


class PredictorEnsamble:
  """
  ¿QUE HACE?
  Promedia varios predictores registrados: goles esperados para el marcador y probabilidades de resultado para la
  decision. Los pesos se ajustan en validacion minimizando el log-loss.

  ¿COMO LO HACE?
  Guarda cada submodelo en una subcarpeta (sub_0, sub_1, ...) y los pesos en su propio esquema.json
  """
  nombre = "ensamble"

  def __init__(self, predictores, pesos, nombres=None):
    self.predictores = list(predictores)
    self.pesos = np.asarray(pesos, dtype=float)
    self.pesos = self.pesos / self.pesos.sum()
    self.nombres = list(nombres) if nombres is not None else [p.nombre for p in self.predictores]

  @classmethod
  def entrenar(cls, predictores, dataset, filas):
    """Ajusta los pesos con las (filas) de validacion; los predictores ya estan entrenados"""
    lote = Lote(dataset[filas].reset_index(drop=True))
    real = np.sign(lote.tabla["goles_0"].to_numpy() - lote.tabla["goles_1"].to_numpy())
    probs = [p.probabilidades(lote) if hasattr(p, "probabilidades") else probabilidades_resultado(p.predecir(lote))
             for p in predictores]

    def objetivo(theta):
      w = softmax(theta)
      combinadas = sum(wi * pi for wi, pi in zip(w, probs))
      return metricas_probabilidad(real, combinadas)["log_loss"]

    ajuste = minimize(objetivo, np.zeros(len(predictores)), method="BFGS", options={"maxiter": 300})
    pesos = softmax(ajuste.x)
    return cls(predictores, pesos)

  def _probs(self, lote):
    partes = [p.probabilidades(lote) if hasattr(p, "probabilidades") else probabilidades_resultado(p.predecir(lote))
              for p in self.predictores]
    probs = sum(w * p for w, p in zip(self.pesos, partes))
    return probs / probs.sum(axis=1, keepdims=True)

  def predecir(self, lote):
    return sum(w * p.predecir(lote) for w, p in zip(self.pesos, self.predictores))

  def probabilidades(self, lote):
    return self._probs(lote)

  def a_esquema(self):
    return {"backend": self.nombre, "pesos": [float(w) for w in self.pesos], "nombres": self.nombres,
            "predictores": [{"carpeta": f"sub_{i}", "nombre": p.nombre} for i, p in enumerate(self.predictores)]}

  def guardar_pesos(self, carpeta):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    for i, predictor in enumerate(self.predictores):
      guardar_predictor(predictor, carpeta / f"sub_{i}")

  @classmethod
  def desde_esquema(cls, esquema, carpeta):
    predictores = [cargar_predictor(Path(carpeta) / sub["carpeta"])[0] for sub in esquema["predictores"]]
    return cls(predictores, esquema["pesos"], esquema.get("nombres"))
