import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from fases.ensamble import PredictorEnsamble
from fases.evaluacion import (comparar, decision_acierto, decision_rps, evaluar, intervalos_bootstrap, particion,
                              probabilidades_resultado)
from fases.modelos_arboles import PredictorLightGBMPoisson
from fases.modelos_estadisticos import PredictorPoissonGLM
from fases.predictores import Lote, PredictorElo

RAIZ = Path(settings.BASE_DIR).parent
DATASET = RAIZ / "data" / "datasets" / "partidos_features.csv"
VALIDACION = RAIZ / "experimentos" / "resultados_f.json"
RESULTADOS = RAIZ / "experimentos" / "resultados_finales.json"


def probabilidades(predictor, lote):
  return predictor.probabilidades(lote) if hasattr(predictor, "probabilidades") else probabilidades_resultado(predictor.predecir(lote))


def decisiones(goles, probs):
  real = np.sign(np.asarray(goles)[:, 0] - np.asarray(goles)[:, 1])
  salida = {}
  for nombre, decision in (("argmax", decision_acierto), ("rps", decision_rps)):
    pred = decision(probs)
    salida[nombre] = {"acierto": float((pred == real).mean() * 100), "empates_predichos": float((pred == 0).mean() * 100)}
  salida["empates_reales"] = float((real == 0).mean() * 100)
  return salida


class Command(BaseCommand):
  """
  ¿QUE HACE?
  La unica mirada confirmatoria al hold-out (2018+2022 y 2026) de la Etapa F: evalua Elo, Poisson GLM, LightGBM Poisson
  y el ensamble con los pesos fijados en validacion. No se usa para elegir ni para reoptimizar: se mira una vez.

  Uso:
    python manage.py evaluar_final
  """
  help = "Mirada final al hold-out 2018/2022 y 2026 con el ensamble elegido en validacion"
  requires_system_checks = []

  def avisar(self, texto):
    self.stdout.write(texto)
    self.stdout.flush()

  def pesos_validacion(self):
    """Media de los pesos de los tres pliegues de F, normalizada; no se reajusta con la prueba"""
    datos = json.loads(VALIDACION.read_text(encoding="utf-8"))["pesos"]
    nombres = list(datos["2006"])
    pesos = np.array([[datos[str(a)][n] for n in nombres] for a in (2006, 2010, 2014)]).mean(axis=0)
    return nombres, pesos / pesos.sum()

  def bases(self, dataset):
    return {
      "elo": PredictorElo.entrenar(dataset),
      "poisson_glm": PredictorPoissonGLM.entrenar(dataset),
      "lightgbm_poisson": PredictorLightGBMPoisson.entrenar(dataset, params={"n_estimators": 400}),
    }

  def evaluar_bloque(self, ds, tr, prueba, titulo):
    base = self.bases(ds[tr])
    nombres, pesos = self.pesos_validacion()
    ensamble = PredictorEnsamble([base[n] for n in nombres], pesos, nombres)
    resultados, referencia = {}, None
    for nombre, predictor in {**base, "ensamble": ensamble}.items():
      resultado, por_partido = evaluar(predictor, ds, None, prueba, por_partido=True)
      lote = Lote(ds[prueba].reset_index(drop=True))
      probs = probabilidades(predictor, lote)
      extra = {"intervalos": intervalos_bootstrap(por_partido), "decisiones": decisiones(ds.loc[prueba, ["goles_0", "goles_1"]].to_numpy(), probs)}
      if referencia is not None:
        extra["contra_elo"] = {m: comparar(por_partido, referencia, m) for m in ("log_loss", "rps")}
      if nombre == "elo":
        referencia = por_partido
      resultados[nombre] = {**resultado, **extra}
      self.avisar(f"  {nombre:<18} log-loss {resultado['log_loss']:.4f}  acierto {resultado['acierto_resultado']:5.1f}%  "
                  f"RPS {resultado['rps']:.4f}  ECE {resultado['ece']:.3f}")
    return resultados, [float(w) for w in pesos]

  def handle(self, *args, **opciones):
    if not DATASET.exists():
      raise CommandError(f"No existe {DATASET}: ejecutar antes 'python manage.py construir_dataset'")
    if not VALIDACION.exists():
      raise CommandError(f"No existe {VALIDACION}: ejecutar antes 'python manage.py evaluar_f'")
    ds = pd.read_csv(DATASET)
    tr_2018, _, prueba_2018 = particion(ds, hasta=2014, prueba=[2018, 2022])
    tr_2026, _, prueba_2026 = particion(ds, hasta=2022, prueba=[2026])
    self.avisar("Prueba 2018 y 2022 (entrenando hasta 2014)")
    r_2018, pesos = self.evaluar_bloque(ds, tr_2018, prueba_2018, "2018 y 2022")
    self.avisar("Prueba 2026 (entrenando hasta 2022)")
    r_2026, _ = self.evaluar_bloque(ds, tr_2026, prueba_2026, "2026")
    RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RESULTADOS.write_text(json.dumps({
      "fecha": date.today().isoformat(), "dataset": DATASET.name, "partidos": len(ds), "pesos_ensamble": pesos,
      "prueba": r_2018, "prueba_2026": r_2026,
      "nota": "Mirada confirmatoria unica de F.3; no se reoptimiza con estos resultados",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    self.stdout.write(self.style.SUCCESS(f"Resultados finales -> {RESULTADOS}"))
