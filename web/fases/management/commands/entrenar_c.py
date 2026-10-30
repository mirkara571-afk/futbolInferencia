from datetime import date
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from fases.modelos_arboles import PredictorLightGBMMulticlase, PredictorLightGBMPoisson, PredictorRandomForest
from fases.predictores import guardar_predictor

RAIZ = Path(settings.BASE_DIR).parent
DATASET = RAIZ / "data" / "datasets" / "partidos_features.csv"


class Command(BaseCommand):
  """
  ¿QUE HACE?
  Entrena y guarda los tres modelos de la Etapa C con todos los datos hasta (--hasta), por defecto 2022, en
  data/modelos/<backend>/ con su esquema.json. No cambia MODEL_DIR.

  Uso:
    python manage.py entrenar_c
  """
  help = "Entrena y guarda LightGBM Poisson, LightGBM multiclase y Random Forest hasta --hasta (por defecto 2022)"
  requires_system_checks = []

  def add_arguments(self, parser):
    parser.add_argument("--hasta", type=int, default=2022, help="Ultimo ano de entrenamiento (por defecto 2022)")

  def handle(self, *args, **opciones):
    if not DATASET.exists():
      raise CommandError(f"No existe {DATASET}: ejecutar antes 'python manage.py construir_dataset'")
    ds = pd.read_csv(DATASET)
    ds = ds[ds["agno"] <= opciones["hasta"]].reset_index(drop=True)
    if len(ds) == 0:
      raise CommandError(f"No hay partidos hasta {opciones['hasta']}")
    modelos = {
      "lightgbm_poisson": PredictorLightGBMPoisson.entrenar(ds, params={"n_estimators": 400}),
      "lightgbm_multiclase": PredictorLightGBMMulticlase.entrenar(ds, params={"n_estimators": 400}),
      "random_forest": PredictorRandomForest.entrenar(ds, params={"n_estimators": 400}),
    }
    for nombre, predictor in modelos.items():
      carpeta = settings.MODEL_PATH / nombre
      guardar_predictor(predictor, carpeta, fecha=date.today().isoformat(), entrenado_hasta=opciones["hasta"],
                        partidos_entrenamiento=int(len(ds)))
      self.stdout.write(self.style.SUCCESS(f"{nombre} -> {carpeta}"))
