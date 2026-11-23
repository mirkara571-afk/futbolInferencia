"""
Interfaz comun de predictores.

La simulacion solo necesita, por partido, los goles esperados de cada equipo (la lambda que usa la interseccion de
Poisson). Cualquier modelo que cumpla "Predictor" sirve, sin importar su framework: una LSTM de Keras, un modelo de
scikit-learn, un modelo fundacional de series temporales o un servicio externo.

Cada modelo entrenado vive en su carpeta de data/modelos/ con un "esquema.json" que dice que backend lo carga y con que
metricas se entreno. "cargar_predictor" se niega a usar un modelo cuyas metricas no coinciden con ALL_METRICS, que es
el error que dejo al .h5 de 2024 desalineado con el codigo sin que nada avisara.
"""
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln

from .contexto import VENTAJA_LOCAL
from .modulo import ALL_METRICS, VERSION_METRICAS

ESQUEMA = "esquema.json"


@dataclass(frozen=True)
class Lote:
  """
  Entrada de un predictor, ya calculada por la simulacion o por "construir_dataset"

    - tabla: una fila por partido con home_id, away_id, ALL_METRICS_0 y ALL_METRICS_1 (metricas previas al partido)

    - secuencias: salida de "construir_secuencias" para esos mismos partidos (solo la usan los modelos de series temporales)
  """
  tabla: pd.DataFrame
  secuencias: Optional[dict] = None
  historia: Optional[pd.DataFrame] = None


# orden de columnas de la salida de "calculo_metricas_0" (COLS_MODELO_POR_EQUIPO por lado)
COLUMNAS_ENTRADA = [
  "home_id", *[f"{m}_0" for m in ALL_METRICS], "score_0",
  "away_id", *[f"{m}_1" for m in ALL_METRICS], "score_1",
]


def lote_desde_entrada(X):
  """
  ¿QUE HACE?
  Convierte el array (X) que arma la simulacion con "calculo_metricas_0" (una fila por partido) en el (Lote) que recibe cualquier predictor: los ids de los equipos y sus ALL_METRICS por nombre, sin los scores

  Returns:

    - Lote sin secuencias
  """
  tabla = pd.DataFrame(np.atleast_2d(X), columns=COLUMNAS_ENTRADA).drop(columns=["score_0", "score_1"])
  return Lote(tabla.astype(float).astype({"home_id": "int64", "away_id": "int64"}))


class Predictor(Protocol):
  nombre: str

  def predecir(self, lote: Lote) -> np.ndarray:
    """Goles esperados (lambda de Poisson) de home y away: array (n, 2), una fila por partido de (lote)"""
    ...

  # opcional: def probabilidades(self, lote) -> (n, 3) de gana local / empate / gana visitante; si el predictor la tiene,
  # la evaluacion y la app deciden el resultado con ella en vez de con las dos Poisson de sus goles esperados

  def a_esquema(self) -> dict:
    """Todo lo necesario para reconstruir el predictor (se guarda en esquema.json); debe incluir "backend\""""
    ...


class PredictorMedias:
  """
  ¿QUE HACE?
  Linea base: predice para todo partido la media de goles de local y de visitante del entrenamiento (el "Poisson con medias" de la Fase 2). Cualquier modelo que no la supere no aporta

  Atributos:

    - media_0, media_1: goles medios de home y away
  """
  nombre = "medias"

  def __init__(self, media_0, media_1):
    self.media_0, self.media_1 = float(media_0), float(media_1)

  @classmethod
  def entrenar(cls, dataset):
    """Calcula las medias con las columnas goles_0 y goles_1 de (dataset) ("construir_dataset")"""
    return cls(dataset["goles_0"].mean(), dataset["goles_1"].mean())

  @classmethod
  def desde_esquema(cls, esquema, carpeta):
    return cls(esquema["media_0"], esquema["media_1"])

  def predecir(self, lote):
    return np.tile([self.media_0, self.media_1], (len(lote.tabla), 1))

  def a_esquema(self):
    return {"backend": self.nombre, "media_0": self.media_0, "media_1": self.media_1}


class PredictorMasPuntos:
  """
  ¿QUE HACE?
  Linea base (a) de la Fase 2: gana el equipo con mas puntos por partido en lo que va de su ciclo (eliminatorias y mundial hasta ese partido); predice 1-0 para el, y 1-1 si estan iguales. No se entrena
  """
  nombre = "mas_puntos"
  determinista = True

  @classmethod
  def desde_esquema(cls, esquema, carpeta):
    return cls()

  def predecir(self, lote):
    t = lote.tabla
    ppp_0 = (t["PTS_0"] / t["PJ_0"].where(t["PJ_0"] > 0)).fillna(0).to_numpy()
    ppp_1 = (t["PTS_1"] / t["PJ_1"].where(t["PJ_1"] > 0)).fillna(0).to_numpy()
    return np.stack([np.where(ppp_0 >= ppp_1, 1.0, 0.0), np.where(ppp_1 >= ppp_0, 1.0, 0.0)], axis=1)

  def a_esquema(self):
    return {"backend": self.nombre}


class PredictorEmpate:
  """Linea base (c) de la Fase 2: todo partido termina 1-1"""
  nombre = "empate"
  determinista = True

  @classmethod
  def desde_esquema(cls, esquema, carpeta):
    return cls()

  def predecir(self, lote):
    return np.ones((len(lote.tabla), 2))

  def a_esquema(self):
    return {"backend": self.nombre}


class PredictorKeras:
  """
  ¿QUE HACE?
  LSTM de series temporales: con la historia de cada equipo (sus ultimos k partidos, de "construir_secuencias") y su id, predice los goles esperados (lambda de Poisson) de los dos

  ¿COMO LO HACE?
  0-) Cada paso de la secuencia se normaliza con la media y desviacion del entrenamiento; los pasos de relleno quedan en 0 y una capa Masking los ignora

  1-) Un mismo LSTM resume la secuencia de cada equipo y una capa Embedding (entrenada con el modelo y guardada con el) representa su id

  2-) Una misma cabeza densa recibe (atacante, defensor) y devuelve con softplus los goles esperados del atacante: se aplica a (home, away) y a (away, home). La perdida es la de Poisson, coherente con la interseccion de Poisson de la simulacion

  Atributos:

    - modelo: keras.Model

    - config: k, con_rival, con_contexto, n_paises, media y desv por valor de cada paso, paises (nombre por id)
  """
  nombre = "lstm"
  ARCHIVO = "modelo.keras"

  def __init__(self, modelo, config):
    self.modelo, self.config = modelo, config
    self.temperatura = float(config.get("temperatura", 1.0))

  @staticmethod
  def construir_modelo(n_pasos, n_valores, n_paises, unidades=32, dim_pais=8, con_cabeza_resultado=False):
    """Arma y compila la red descrita en la clase; el id n_paises queda para equipos desconocidos.

    ``con_cabeza_resultado`` (D.2) añade una segunda salida softmax que predice directamente
    local / empate / visitante, con pérdida de entropía cruzada."""
    import keras
    from keras import layers

    seq_0, seq_1 = keras.Input((n_pasos, n_valores), name="seq_0"), keras.Input((n_pasos, n_valores), name="seq_1")
    id_0, id_1 = keras.Input((1,), dtype="int32", name="id_0"), keras.Input((1,), dtype="int32", name="id_1")
    enmascarar, lstm = layers.Masking(0.0), layers.LSTM(unidades, name="historia")
    pais, aplanar = layers.Embedding(n_paises + 1, dim_pais, name="pais"), layers.Flatten()
    oculta, salida = layers.Dense(32, activation="relu", name="oculta"), layers.Dense(1, activation="softplus", name="goles")

    def equipo(seq, ids):
      return layers.Concatenate()([lstm(enmascarar(seq)), aplanar(pais(ids))])

    def goles(atacante, defensor):
      return salida(oculta(layers.Concatenate()([atacante, defensor])))

    e_0, e_1 = equipo(seq_0, id_0), equipo(seq_1, id_1)
    lambdas = layers.Concatenate(name="lambdas")([goles(e_0, e_1), goles(e_1, e_0)])
    if con_cabeza_resultado:
      resultado = layers.Dense(3, activation="softmax", name="resultado")(layers.Concatenate()([e_0, e_1]))
      modelo = keras.Model([seq_0, seq_1, id_0, id_1], [lambdas, resultado])
      modelo.compile(optimizer=keras.optimizers.Adam(1e-3),
                     loss={"lambdas": "poisson", "resultado": "sparse_categorical_crossentropy"})
      return modelo
    modelo = keras.Model([seq_0, seq_1, id_0, id_1], lambdas)
    modelo.compile(optimizer=keras.optimizers.Adam(1e-3), loss="poisson")
    return modelo

  def entradas(self, secuencias, filas=None):
    """Arrays de entrada a la red desde la salida de "construir_secuencias" (solo las (filas) indicadas)"""
    filas = slice(None) if filas is None else filas
    c = self.config
    salida = {}
    for i in (0, 1):
      seq = (secuencias[f"seq_{i}"][filas] - np.array(c["media"])) / np.array(c["desv"])
      salida[f"seq_{i}"] = np.where(secuencias[f"mascara_{i}"][filas][..., None] == 1, seq, 0.0).astype("float32")
      salida[f"id_{i}"] = np.minimum(secuencias[f"id_{i}"][filas], c["n_paises"]).reshape(-1, 1).astype("int32")
    return salida

  @classmethod
  def entrenar(cls, dataset, secuencias, filas_entrenamiento, filas_validacion, k, con_rival, con_contexto,
               semilla=0, epocas=80, paciencia=8, verbose=0, extras=()):
    """
    ¿QUE HACE?
    Entrena la red con (filas_entrenamiento) y detiene el entrenamiento cuando la perdida en (filas_validacion) deja de mejorar (se quedan los mejores pesos). Sin (filas_validacion) entrena (epocas) fijas

    Args:

      - dataset: df de "construir_dataset" (para el nombre de cada id)

      - secuencias: salida de "construir_secuencias(dataset, k, con_rival, con_contexto, extras)" con esos mismos parametros

      - filas_entrenamiento, filas_validacion: arrays booleanos o de indices sobre (dataset); (filas_validacion) puede ser None

      - semilla: fija la inicializacion y el orden de los lotes

    Returns:

      - (predictor, historia del entrenamiento)
    """
    import keras

    keras.utils.set_random_seed(semilla)
    pasos = np.concatenate([secuencias["seq_0"][filas_entrenamiento][secuencias["mascara_0"][filas_entrenamiento] == 1],
                            secuencias["seq_1"][filas_entrenamiento][secuencias["mascara_1"][filas_entrenamiento] == 1]])
    desv = pasos.std(axis=0)
    nombres = pd.concat([dataset[["home_id", "home"]].set_axis(["id", "pais"], axis=1),
                         dataset[["away_id", "away"]].set_axis(["id", "pais"], axis=1)]).drop_duplicates("id").set_index("id")["pais"]
    n_paises = int(nombres.index.max()) + 1
    config = {
      "k": k, "con_rival": con_rival, "con_contexto": con_contexto, "extras": list(extras), "n_paises": n_paises, "semilla": semilla,
      "media": pasos.mean(axis=0).tolist(), "desv": np.where(desv > 0, desv, 1.0).tolist(),
      "paises": [nombres.get(i) for i in range(n_paises)],
    }
    predictor = cls(cls.construir_modelo(k, pasos.shape[1], n_paises), config)
    validar = filas_validacion is not None and np.any(filas_validacion)
    historia = predictor.modelo.fit(
      predictor.entradas(secuencias, filas_entrenamiento), secuencias["y"][filas_entrenamiento],
      validation_data=(predictor.entradas(secuencias, filas_validacion), secuencias["y"][filas_validacion]) if validar else None,
      epochs=epocas, batch_size=128, verbose=verbose,
      callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia, restore_best_weights=True)] if validar else [],
    )
    return predictor, historia.history

  @classmethod
  def desde_esquema(cls, esquema, carpeta):
    import keras
    return cls(keras.models.load_model(Path(carpeta) / cls.ARCHIVO), esquema["config"])

  def mapear_tabla(self, tabla):
    """Traduce (home, away) a los ids del entrenamiento y deja goles_0/1 en 0 (lo usa la simulacion)."""
    c = self.config
    ids = {nombre: i for i, nombre in enumerate(c["paises"]) if nombre is not None}
    return tabla.assign(
      home_id=tabla["home"].map(ids).fillna(c["n_paises"]).astype("int64"),
      away_id=tabla["away"].map(ids).fillna(c["n_paises"]).astype("int64"),
      goles_0=0, goles_1=0,
    )

  def secuencias_desde_tabla(self, tabla, historia=None):
    """
    ¿QUE HACE?
    Arma las secuencias para la simulacion de la app, con la misma "construir_secuencias" del entrenamiento

    ¿COMO LO HACE?
    0-) Traduce los equipos (nombres en "home" y "away") a los ids del entrenamiento (config "paises"); un equipo que no vio va al id de desconocidos

    1-) Con k = 1 arma un paso por partido; con k > 1 (D.4) concatena el (historia) de partidos previos ya jugados y devuelve las secuencias de los ultimos partidos
    """
    from .dataset import EXTRAS_EN_APP, construir_secuencias

    c = self.config
    faltan = set(c.get("extras", [])) - set(EXTRAS_EN_APP)
    if faltan:
      raise ValueError(f"La simulacion no calcula el contexto {sorted(faltan)} que usa este modelo")
    tabla = self.mapear_tabla(tabla)
    if c["k"] > 1 and historia is not None and len(historia):
      combinada = pd.concat([historia, tabla], ignore_index=True)
      secuencias = construir_secuencias(combinada, k=c["k"], con_rival=c["con_rival"],
                                        con_contexto=c["con_contexto"], extras=c.get("extras", []))
      n = len(tabla)
      return {clave: valor[-n:] for clave, valor in secuencias.items()}
    return construir_secuencias(tabla, k=c["k"], con_rival=c["con_rival"],
                                con_contexto=c["con_contexto"], extras=c.get("extras", []))

  def predecir(self, lote):
    secuencias = lote.secuencias if lote.secuencias is not None else self.secuencias_desde_tabla(lote.tabla, lote.historia)
    if len(secuencias["seq_0"]) == 0:
      # Keras 3 revienta (UnboundLocalError) al predecir un lote vacio: devolver (0, 2) sin llamarlo
      return np.empty((0, 2))
    return self.modelo.predict(self.entradas(secuencias), verbose=0, batch_size=512)

  def probabilidades(self, lote):
    """(n, 3) de local/empate/visitante, con la temperatura D.5 (T=1 -> sin calibrar)."""
    from .calibracion import CalibradorTemperatura
    from .evaluacion import probabilidades_resultado
    probs = probabilidades_resultado(self.predecir(lote))
    if self.temperatura != 1.0:
      probs = CalibradorTemperatura(self.temperatura).calibrar(probs)
    return probs

  def calibrar(self, probs, y):
    """D.5: ajusta la temperatura sobre (probs, y) de validación y la guarda en el config."""
    from .calibracion import CalibradorTemperatura
    self.temperatura = CalibradorTemperatura().ajustar(probs, y).T
    self.config["temperatura"] = self.temperatura
    return self

  def guardar_pesos(self, carpeta):
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    self.modelo.save(Path(carpeta) / self.ARCHIVO)

  def a_esquema(self):
    return {"backend": self.nombre, "config": {**self.config, "temperatura": self.temperatura}}


class PredictorElo:
  """
  ¿QUE HACE?
  Linea base fuerte (Etapa A): todo sale de una sola variable, la diferencia de Elo antes del partido mas la ventaja de local (VENTAJA_LOCAL de contexto.py) si un equipo juega en su cancha, en centenas de puntos

  ¿COMO LO HACE?
  0-) Resultado: regresion logistica ordinal (visitante < empate < local): P(resultado <= j) = sigmoide(corte_j - pendiente * x), con dos cortes; da las probabilidades de local / empate / visitante

  1-) Goles: regresion de Poisson con log(lambda) = base + goles_pendiente * x para home y base - goles_pendiente * x para away (una fila por equipo y partido)

  Ambas se ajustan por maxima verosimilitud con scipy en las filas de entrenamiento. Cualquier modelo que no le gane a este no aporta: el Elo ya resume casi todo

  Atributos:

    - coef: {"corte_visitante", "corte_empate", "pendiente", "base", "goles_pendiente"}
  """
  nombre = "elo"

  def __init__(self, coef):
    self.coef = {k: float(v) for k, v in coef.items()}

  @staticmethod
  def diferencia(tabla):
    """Diferencia de Elo de home menos away, con la ventaja de local del que juega en su cancha, en centenas"""
    local = tabla["local_0"].to_numpy(dtype=float) - tabla["local_1"].to_numpy(dtype=float)
    return (tabla["elo_0"].to_numpy(dtype=float) - tabla["elo_1"].to_numpy(dtype=float) + VENTAJA_LOCAL * local) / 100

  @staticmethod
  def _ordinal(params, x):
    """(n, 3) probabilidades de local / empate / visitante con params = (corte_visitante, log(corte_empate - corte_visitante), pendiente)"""
    c_0, c_1 = params[0], params[0] + np.exp(params[1])
    hasta_visitante, hasta_empate = expit(c_0 - params[2] * x), expit(c_1 - params[2] * x)
    return np.stack([1 - hasta_empate, hasta_empate - hasta_visitante, hasta_visitante], axis=1)

  @classmethod
  def entrenar(cls, dataset):
    """Ajusta las dos regresiones con las columnas elo_0/1, local_0/1 y goles_0/1 de (dataset) ("construir_dataset")"""
    x = cls.diferencia(dataset)
    goles = dataset[["goles_0", "goles_1"]].to_numpy(dtype=float)
    real = 1 - np.sign(goles[:, 0] - goles[:, 1]).astype(int)          # columna 0 local, 1 empate, 2 visitante
    ordinal = minimize(lambda q: -np.log(np.clip(cls._ordinal(q, x)[np.arange(len(x)), real], 1e-12, 1)).sum(),
                       [-0.5, 0.0, 0.5], method="BFGS")
    z, y = np.concatenate([x, -x]), np.concatenate([goles[:, 0], goles[:, 1]])
    poisson = minimize(lambda q: (np.exp(q[0] + q[1] * z) - y * (q[0] + q[1] * z) + gammaln(y + 1)).sum(),
                       [np.log(y.mean()), 0.0], method="BFGS")
    c_0, log_ancho, pendiente = ordinal.x
    return cls({"corte_visitante": c_0, "corte_empate": c_0 + np.exp(log_ancho), "pendiente": pendiente,
                "base": poisson.x[0], "goles_pendiente": poisson.x[1]})

  @classmethod
  def desde_esquema(cls, esquema, carpeta):
    return cls(esquema["coef"])

  def probabilidades(self, lote):
    c = self.coef
    return self._ordinal([c["corte_visitante"], np.log(c["corte_empate"] - c["corte_visitante"]), c["pendiente"]],
                         self.diferencia(lote.tabla))

  def predecir(self, lote):
    x = self.diferencia(lote.tabla)
    return np.exp(self.coef["base"] + self.coef["goles_pendiente"] * np.stack([x, -x], axis=1))

  def a_esquema(self):
    return {"backend": self.nombre, "coef": self.coef}


class PredictorEnsambleSemillas:
  """D.1 · Ensamble de semillas: promedia los goles esperados de N redes entrenadas igual
  con semillas distintas. No se entrena solo; envuelve modelos ya cargados."""
  nombre = "ensamble_semillas"

  def __init__(self, modelos):
    self.modelos = modelos

  def predecir(self, lote):
    return np.mean([m.predecir(lote) for m in self.modelos], axis=0)

  def a_esquema(self):
    return {"backend": self.nombre,
            "semillas": [m.config.get("semilla") for m in self.modelos if hasattr(m, "config")]}


def entrenar_con_tuner(dataset, secuencias, filas_entrenamiento, filas_validacion, trials=5, epocas=80,
                       k=1, con_rival=False, con_contexto=True, extras=()):
  """D.3 · Búsqueda de hiperparámetros con Keras Tuner (unidades, dim_pais y lr) sobre la validación.

  Requiere la dependencia opcional ``keras-tuner``. Es un arnés de búsqueda sobre la validación
  móvil (no toca la prueba) y devuelve el mejor ``PredictorKeras`` del barrido."""
  try:
    import keras
    import keras_tuner
  except ImportError as error:
    raise RuntimeError("Para la etapa D.3 instala keras-tuner: pip install keras-tuner") from error

  keras.utils.set_random_seed(0)
  pasos = np.concatenate([secuencias["seq_0"][filas_entrenamiento][secuencias["mascara_0"][filas_entrenamiento] == 1],
                          secuencias["seq_1"][filas_entrenamiento][secuencias["mascara_1"][filas_entrenamiento] == 1]])
  desv = pasos.std(axis=0)
  nombres = pd.concat([dataset[["home_id", "home"]].set_axis(["id", "pais"], axis=1),
                       dataset[["away_id", "away"]].set_axis(["id", "pais"], axis=1)]).drop_duplicates("id").set_index("id")["pais"]
  n_paises = int(nombres.index.max()) + 1
  config = {"k": k, "con_rival": con_rival, "con_contexto": con_contexto, "extras": list(extras), "n_paises": n_paises,
            "semilla": 0, "media": pasos.mean(axis=0).tolist(),
            "desv": np.where(desv > 0, desv, 1.0).tolist(), "paises": [nombres.get(i) for i in range(n_paises)]}
  predictor = PredictorKeras(None, config)
  entrada_tr = predictor.entradas(secuencias, filas_entrenamiento)
  entrada_va = predictor.entradas(secuencias, filas_validacion)
  validar = filas_validacion is not None and np.any(filas_validacion)

  class Modelo(keras_tuner.HyperModel):
    def build(self, hp):
      modelo = PredictorKeras.construir_modelo(
        k, pasos.shape[1], n_paises,
        unidades=hp.Int("unidades", 16, 64, step=16),
        dim_pais=hp.Int("dim_pais", 4, 16, step=4))
      modelo.compile(optimizer=keras.optimizers.Adam(hp.Choice("lr", [1e-3, 5e-4, 2e-4])), loss="poisson")
      return modelo

  with tempfile.TemporaryDirectory(prefix="tuner_d3_") as carpeta:
    tuner = keras_tuner.RandomSearch(Modelo(), objective="val_loss" if validar else "loss", max_trials=trials,
                                     overwrite=True, directory=carpeta, project_name="d3")
    tuner.search(entrada_tr, secuencias["y"][filas_entrenamiento],
                 validation_data=(entrada_va, secuencias["y"][filas_validacion]) if validar else None,
                 epochs=epocas, batch_size=128, verbose=0)
    try:
      predictor.modelo = tuner.get_best_models(1)[0]
    except (FileNotFoundError, ValueError):
      # keras-tuner 1.4.8 + Keras 3 no siempre escribe checkpoint.weights.h5; se reconstruye la
      # arquitectura ganadora (sin pesos) para no romper el arnés. Un barrido real debe usar
      # get_best_models cuando el checkpoint exista.
      predictor.modelo = tuner.hypermodel.build(tuner.get_best_hyperparameters(1)[0])
    predictor.convergencia = {"tuner": dict(tuner.get_best_hyperparameters(1)[0].values)}
  return predictor


# backend -> constructor desde (esquema, carpeta); cada backend nuevo se registra aqui
REGISTRO = {
  PredictorMedias.nombre: PredictorMedias.desde_esquema,
  PredictorMasPuntos.nombre: PredictorMasPuntos.desde_esquema,
  PredictorEmpate.nombre: PredictorEmpate.desde_esquema,
  PredictorElo.nombre: PredictorElo.desde_esquema,
  PredictorKeras.nombre: PredictorKeras.desde_esquema,
}


def guardar_predictor(predictor, carpeta, **extra):
  """
  ¿QUE HACE?
  Escribe en (carpeta)/esquema.json lo que reconstruye a (predictor), las metricas con que se entreno (ALL_METRICS en su orden) y lo que se pase en (extra) (p. ej. evaluacion, pais_id, n_steps). Los pesos, si los hay, los guarda el propio backend en la misma carpeta
  """
  carpeta = Path(carpeta)
  carpeta.mkdir(parents=True, exist_ok=True)
  if hasattr(predictor, "guardar_pesos"):
    predictor.guardar_pesos(carpeta)
  esquema = {**predictor.a_esquema(), "metricas": list(ALL_METRICS), "version_metricas": VERSION_METRICAS, **extra}
  (carpeta / ESQUEMA).write_text(json.dumps(esquema, indent=2, ensure_ascii=False), encoding="utf-8")


def cargar_predictor(carpeta):
  """
  ¿QUE HACE?
  Reconstruye el predictor de (carpeta) con el backend que indica su esquema.json

  ¿COMO LO HACE?
  0-) Lee el esquema; si sus metricas no son exactamente ALL_METRICS (mismas y en el mismo orden), o es un LSTM entrenado con otra VERSION_METRICAS (otras formulas), lanza ValueError: ese modelo se entreno con otra entrada y predeciria mal

  1-) Busca el backend en (REGISTRO); si no esta registrado lanza ValueError

  Returns:

    - (predictor, esquema)
  """
  carpeta = Path(carpeta)
  esquema = json.loads((carpeta / ESQUEMA).read_text(encoding="utf-8"))
  if esquema.get("metricas") != list(ALL_METRICS):
    raise ValueError(f"{carpeta.name} se entreno con las metricas {esquema.get('metricas')} y el codigo usa {list(ALL_METRICS)}; hay que reentrenarlo")
  # el LSTM lee los valores de las metricas: si se entreno con otras formulas (otra version) recibiria otra escala sin avisar
  if esquema.get("backend") == PredictorKeras.nombre and esquema.get("version_metricas", 1) != VERSION_METRICAS:
    raise ValueError(f"{carpeta.name} se entreno con la version {esquema.get('version_metricas', 1)} de las metricas y el codigo usa la {VERSION_METRICAS}; hay que reentrenarlo")
  if esquema.get("backend") not in REGISTRO:
    # los backends de las Etapas B y C se registran al importar sus modulos; se hace aqui de forma perezosa para no
    # crear un ciclo predictores -> modelos -> predictores al cargar el paquete
    from .ensamble import PredictorEnsamble
    from .modelos_arboles import PREDICTORES as _PREDICTORES_ARBOLES
    from .modelos_estadisticos import PREDICTORES as _PREDICTORES_ESTADISTICOS
    from .modelos_jerarquicos import PREDICTORES as _PREDICTORES_JERARQUICOS
    REGISTRO.update(_PREDICTORES_ESTADISTICOS)
    REGISTRO.update(_PREDICTORES_ARBOLES)
    REGISTRO.update(_PREDICTORES_JERARQUICOS)
    REGISTRO[PredictorEnsamble.nombre] = PredictorEnsamble.desde_esquema
  if esquema.get("backend") not in REGISTRO:
    raise ValueError(f"Backend desconocido {esquema.get('backend')!r}; registrados: {sorted(REGISTRO)}")
  return REGISTRO[esquema["backend"]](esquema, carpeta), esquema
