import json
from datetime import date
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from fases.modelos_estadisticos import PredictorDixonColes, PredictorOrdinal, PredictorPoissonGLM
from fases.predictores import guardar_predictor

RAIZ = Path(settings.BASE_DIR).parent
DATASET = RAIZ / "data" / "datasets" / "partidos_features.csv"


class Command(BaseCommand):
  """
  ¿QUE HACE?
  Entrena y guarda los tres modelos de la Etapa B con todos los datos hasta (--hasta), por defecto 2022, en
  data/modelos/<backend>/ con su esquema.json. No los pone en la app: eso lo decide F.3.

  Uso:
    python manage.py entrenar_b
    python manage.py entrenar_b --hasta 2014
  """
  help = "Entrena y guarda Poisson GLM, ordinal y Dixon-Coles hasta --hasta (por defecto 2022)"
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
      "poisson_glm": PredictorPoissonGLM.entrenar(ds),
      "ordinal": PredictorOrdinal.entrenar(ds),
      "dixon_coles": PredictorDixonColes.entrenar(ds, xi=0.05),
    }
    for nombre, predictor in modelos.items():
      carpeta = settings.MODEL_PATH / nombre
      guardar_predictor(predictor, carpeta, fecha=date.today().isoformat(), entrenado_hasta=opciones["hasta"],
                        partidos_entrenamiento=int(len(ds)))
      self.stdout.write(self.style.SUCCESS(f"{nombre} -> {carpeta}"))
    # deja la tabla de coeficientes del GLM junto al modelo, para leerla sin abrir Python
    glm = modelos["poisson_glm"]
    (settings.MODEL_PATH / "poisson_glm" / "coeficientes.json").write_text(
      json.dumps(glm.coeficientes()["coeficiente"].to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
