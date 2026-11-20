from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from fases.dataset import construir_secuencias
from fases.predictores import PredictorKeras, guardar_predictor

RAIZ = Path(settings.BASE_DIR).parent
DATASET = RAIZ / "data" / "datasets" / "partidos_features.csv"


class Command(BaseCommand):
  """
  ¿QUE HACE?
  D.1: entrena N copias de la LSTM con distinta semilla sobre el mismo dataset (hasta --hasta, por
  defecto 2022, igual que el modelo en uso) y guarda cada una en data/modelos/lstm-semilla-<s>/.
  El ensamble de semillas es el promedio de sus lambdas: con las N carpetas guardadas se puede
  armar un PredictorEnsambleSemillas o promediar en la API.

  Split: entrena con agno <= hasta-4 y detiene con validacion en agno == hasta (el año previo al
  siguiente Mundial, nunca en la prueba 2026).

  Uso:
    python manage.py entrenar_semillas --semillas 5 --hasta 2022 --epocas 80
  """
  help = "Entrena N LSTM por semillas (D.1) en data/modelos/lstm-semilla-<s>/"
  requires_system_checks = []

  def add_arguments(self, parser):
    parser.add_argument("--hasta", type=int, default=2022, help="último año de entrenamiento (por defecto 2022)")
    parser.add_argument("--semillas", type=int, default=5, help="número de semillas (por defecto 5)")
    parser.add_argument("--epocas", type=int, default=80, help="épocas máximas (por defecto 80)")

  def handle(self, *args, **opciones):
    if not DATASET.exists():
      raise CommandError(f"No existe {DATASET}: ejecutar antes 'python manage.py construir_dataset'")
    ds = pd.read_csv(DATASET)
    ds = ds[ds["agno"] <= opciones["hasta"]].reset_index(drop=True)
    if len(ds) == 0:
      raise CommandError(f"No hay partidos hasta {opciones['hasta']}")

    k, con_rival, con_contexto, extras = 1, False, True, ("elo",)
    secuencias = construir_secuencias(ds, k=k, con_rival=con_rival, con_contexto=con_contexto, extras=extras)
    filas_train = (ds["agno"] <= opciones["hasta"] - 4).to_numpy()
    filas_valid = (ds["agno"] == opciones["hasta"]).to_numpy()

    for s in range(opciones["semillas"]):
      predictor, _ = PredictorKeras.entrenar(
        ds, secuencias, filas_train, filas_valid, k=k, con_rival=con_rival, con_contexto=con_contexto,
        semilla=s, epocas=opciones["epocas"], extras=extras)
      carpeta = settings.MODEL_PATH / f"lstm-semilla-{s}"
      guardar_predictor(predictor, carpeta, entrenado_hasta=opciones["hasta"], semilla=s)
      self.stdout.write(self.style.SUCCESS(f"semilla {s} -> {carpeta}"))
