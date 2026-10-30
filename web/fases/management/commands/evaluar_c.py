import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from fases.evaluacion import POR_PARTIDO, comparar, evaluar, intervalos_bootstrap, particion
from fases.modelos_arboles import PredictorLightGBMMulticlase, PredictorLightGBMPoisson, PredictorRandomForest
from fases.modelos_estadisticos import PredictorPoissonGLM
from fases.predictores import PredictorElo

RAIZ = Path(settings.BASE_DIR).parent
DATASET = RAIZ / "data" / "datasets" / "partidos_features.csv"
RESULTADOS = RAIZ / "experimentos" / "resultados_c.json"
IMPORTANCIA = RAIZ / "experimentos" / "importancia_c.json"
PLIEGUES = [2006, 2010, 2014]
METRICAS = list(POR_PARTIDO)


def resumen(corridas):
  return {**{m: {"media": float(np.mean([c[m] for c in corridas])), "desv": float(np.std([c[m] for c in corridas]))}
             for m in METRICAS},
          "ece": {"media": float(np.mean([c["ece"] for c in corridas])), "desv": float(np.std([c["ece"] for c in corridas]))}}


class Command(BaseCommand):
  """
  ¿QUE HACE?
  Evalua la Etapa C (LightGBM Poisson, LightGBM multiclase y Random Forest) con la validacion movil y la vara de A:
  log-loss, RPS, Brier, acierto, ECE, intervalos y comparacion pareada contra el Elo. No toca 2018/2022/2026.

  Uso:
    python manage.py evaluar_c
  """
  help = "Validacion movil de los modelos de arboles de la Etapa C; resultados en experimentos/resultados_c.json"
  requires_system_checks = []

  def avisar(self, texto):
    self.stdout.write(texto)
    self.stdout.flush()

  def modelos(self, dataset):
    return {
      "elo": PredictorElo.entrenar(dataset),
      "poisson_glm": PredictorPoissonGLM.entrenar(dataset),
      "lightgbm_poisson": PredictorLightGBMPoisson.entrenar(dataset, params={"n_estimators": 400}),
      "lightgbm_multiclase": PredictorLightGBMMulticlase.entrenar(dataset, params={"n_estimators": 400}),
      "random_forest": PredictorRandomForest.entrenar(dataset, params={"n_estimators": 400}),
    }

  def handle(self, *args, **opciones):
    if not DATASET.exists():
      raise CommandError(f"No existe {DATASET}: ejecutar antes 'python manage.py construir_dataset'")
    ds = pd.read_csv(DATASET)
    detalle, resumen_pliegues = {}, {}
    for agno in PLIEGUES:
      tr, va = particion(ds, hasta=agno - 4, validacion=agno)[:2]
      self.avisar(f"Pliegue {agno}: {tr.sum()} entrenamiento / {va.sum()} validacion")
      modelos = self.modelos(ds[tr])
      medidos = {}
      for nombre, predictor in modelos.items():
        resultado, por_partido = evaluar(predictor, ds, None, va, por_partido=True)
        extra = {"intervalos": intervalos_bootstrap(por_partido)}
        if nombre != "elo":
          extra["contra_elo"] = {m: comparar(por_partido, medidos["elo"][1], m) for m in ("log_loss", "rps")}
        medidos[nombre] = (resultado, por_partido)
        detalle.setdefault(nombre, {})[str(agno)] = {**resultado, **extra}
        self.avisar(f"  {nombre:<20} log-loss {resultado['log_loss']:.4f}  acierto {resultado['acierto_resultado']:5.1f}%  "
                    f"RPS {resultado['rps']:.4f}  ECE {resultado['ece']:.3f}")
      for nombre, (resultado, _) in medidos.items():
        resumen_pliegues.setdefault(nombre, []).append(resultado)

    resumen_final = {nombre: resumen(corridas) for nombre, corridas in resumen_pliegues.items()}
    comparaciones = {}
    for nombre, pliegues in detalle.items():
      if nombre == "elo":
        continue
      comparaciones[nombre] = {m: {"media": float(np.mean([p["contra_elo"][m]["diferencia"] for p in pliegues.values()])),
                                   "p_dm_pliegues": [p["contra_elo"][m]["p_dm"] for p in pliegues.values()]}
                              for m in ("log_loss", "rps")}
    mejor = min(resumen_final, key=lambda n: resumen_final[n]["log_loss"]["media"])

    # importancia de variables del LightGBM Poisson entrenado hasta 2022 (sin usar 2026, ano de prueba)
    completo = PredictorLightGBMPoisson.entrenar(ds[ds["agno"] <= 2022], params={"n_estimators": 400})
    importancia = completo.importancia("gain").head(25)
    IMPORTANCIA.write_text(json.dumps({k: float(v) for k, v in importancia.items()}, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RESULTADOS.write_text(json.dumps({
      "fecha": date.today().isoformat(), "dataset": DATASET.name, "partidos": len(ds), "pliegues": PLIEGUES,
      "resumen": resumen_final, "detalle": detalle, "comparaciones": comparaciones, "mejor_log_loss": mejor,
      "nota": "Solo validacion movil; 2018/2022/2026 no se tocaron (reservados a F.3)",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    self.avisar("Resumen de validacion movil (media de los pliegues):")
    for nombre, r in sorted(resumen_final.items(), key=lambda x: x[1]["log_loss"]["media"]):
      self.avisar(f"  {nombre:<20} log-loss {r['log_loss']['media']:.4f}  RPS {r['rps']['media']:.4f}  "
                  f"acierto {r['acierto_resultado']['media']:5.1f}%  ECE {r['ece']['media']:.3f}")
    self.avisar("Importancia (ganancia, top 10):")
    for nombre, valor in importancia.head(10).items():
      self.avisar(f"  {nombre:<22} {valor:.0f}")
    self.stdout.write(self.style.SUCCESS(f"Mejor por log-loss: {mejor} -> {RESULTADOS}"))
