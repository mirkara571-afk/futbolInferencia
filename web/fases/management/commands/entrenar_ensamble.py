import json
from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from fases.ensamble import PredictorEnsamble
from fases.predictores import cargar_predictor, guardar_predictor

RAIZ = Path(settings.BASE_DIR).parent
BASES = ["elo", "poisson_glm", "lightgbm_poisson"]


class Command(BaseCommand):
  """
  ¿QUE HACE?
  Construye y guarda un ensamble desplegable (Elo + Poisson GLM + LightGBM Poisson) en
  data/modelos/ensamble/, usando los pesos medios de los pliegues de validacion movil
  (2006/2010/2014) de experimentos/resultados_f.json. Los modelos base ya estan entrenados
  hasta 2022; aqui solo se combinan, sin tocar la prueba.

  Uso:
    python manage.py entrenar_ensamble
  """
  help = "Guarda el ensamble Elo+GLM+LightGBM como modelo desplegable (pesos medios de validacion)"
  requires_system_checks = []

  def handle(self, *args, **opciones):
    resultados = RAIZ / "experimentos" / "resultados_f.json"
    if not resultados.exists():
      raise CommandError(f"No existe {resultados}: ejecutar antes 'python manage.py evaluar_f'")
    pesos_pliegues = json.loads(resultados.read_text(encoding="utf-8"))["pesos"]
    pesos = np.mean([[pesos_pliegues[p][n] for n in BASES] for p in pesos_pliegues], axis=0)

    predictores = [cargar_predictor(settings.MODEL_PATH / n)[0] for n in BASES]
    ensamble = PredictorEnsamble(predictores, pesos, BASES)
    guardar_predictor(ensamble, settings.MODEL_PATH / "ensamble",
                      entrenado_hasta=2022, origen="pesos medios de validacion movil 2006/2010/2014")
    self.stdout.write(self.style.SUCCESS(
      f"Ensamble -> {settings.MODEL_PATH / 'ensamble'} | pesos " + ", ".join(f"{n}={w:.3f}" for n, w in zip(BASES, pesos))))
