"""
D.5 · Calibración de probabilidades.

Una red (o cualquier predictor) que dice "70 %" debe ganar 7 de cada 10 veces. Si no, se
calibra. Aquí: calibración por temperatura (escala las log-odds de las 3 clases por un factor
T > 0 ajustado en validación minimizando el log-loss) y un envoltorio ``PredictorCalibrado``
que la aplica a cualquier predictor que exponga ``probabilidades``.

No se entrena ningún modelo nuevo: el calibrador se ajusta sobre las probabilidades ya
predichas en la validación (fuera de la prueba).
"""
import numpy as np
from scipy.optimize import minimize


def _logodds(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class CalibradorTemperatura:
    """Escala las probabilidades (n, 3) por una temperatura T para minimizar el log-loss."""

    def __init__(self, T=1.0):
        self.T = float(T)

    def ajustar(self, probs, y):
        """``y`` (n,) con 0=local, 1=empate, 2=visitante. Ajusta T minimizando el log-loss."""
        logits = _logodds(probs)
        objetivo = lambda t: -np.log(np.clip(_softmax(logits / t[0])[np.arange(len(y)), y], 1e-12, 1)).mean()
        self.T = float(minimize(objetivo, [1.0], bounds=[(0.05, 20.0)], method="L-BFGS-B").x[0])
        return self

    def calibrar(self, probs):
        return _softmax(_logodds(probs) / self.T)


class PredictorCalibrado:
    """Envuelve un predictor y calibra sus probabilidades (D.5)."""

    nombre = "calibrado"

    def __init__(self, base, calibrador):
        self.base, self.calibrador = base, calibrador

    def predecir(self, lote):
        return self.base.predecir(lote)

    def probabilidades(self, lote):
        return self.calibrador.calibrar(self.base.probabilidades(lote))

    def a_esquema(self):
        return {"backend": self.nombre, "base": self.base.a_esquema(), "T": self.calibrador.T}
