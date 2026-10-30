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
RESULTADOS = RAIZ / "experimentos" / "resultados_f.json"
PLIEGUES = [2006, 2010, 2014]


def probabilidades(predictor, lote):
  return predictor.probabilidades(lote) if hasattr(predictor, "probabilidades") else probabilidades_resultado(predictor.predecir(lote))


def metricas_decision(goles, probs):
  """Acierto y tasa de empates predichos con dos reglas: argmax y RPS esperado"""
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
  Etapa F: elige los pesos de un ensamble Elo + Poisson GLM + LightGBM Poisson en una validacion interna, lo evalua en
  la validacion externa (2006/2010/2014) y compara la regla de decision argmax con la de RPS esperado.
  No toca 2018/2022/2026: la prueba sigue reservada a F.3.

  Uso:
    python manage.py evaluar_f
  """
  help = "Ensamble y regla de empates en validacion movil; resultados en experimentos/resultados_f.json"
  requires_system_checks = []

  def avisar(self, texto):
    self.stdout.write(texto)
    self.stdout.flush()

  def bases(self, dataset):
    return {
      "elo": PredictorElo.entrenar(dataset),
      "poisson_glm": PredictorPoissonGLM.entrenar(dataset),
      "lightgbm_poisson": PredictorLightGBMPoisson.entrenar(dataset, params={"n_estimators": 400}),
    }

  def handle(self, *args, **opciones):
    if not DATASET.exists():
      raise CommandError(f"No existe {DATASET}: ejecutar antes 'python manage.py construir_dataset'")
    ds = pd.read_csv(DATASET)
    detalle, pesos, referencia_pp = {}, {}, {}
    for agno in PLIEGUES:
      tr, va = particion(ds, hasta=agno - 4, validacion=agno)[:2]
      inner_tr = (ds["agno"] <= agno - 8).to_numpy()
      inner_va = (ds["agno"] == agno - 4).to_numpy()
      base_inner = self.bases(ds[inner_tr])
      ens_inner = PredictorEnsamble.entrenar(list(base_inner.values()), ds, inner_va)
      pesos[str(agno)] = {n: float(w) for n, w in zip(ens_inner.nombres, ens_inner.pesos)}
      self.avisar(f"Pliegue {agno}: pesos internos " + ", ".join(f"{n}={w:.3f}" for n, w in pesos[str(agno)].items()))

      base = self.bases(ds[tr])
      ensamble = PredictorEnsamble(list(base.values()), ens_inner.pesos)
      for nombre, predictor in {**base, "ensamble": ensamble}.items():
        resultado, por_partido = evaluar(predictor, ds, None, va, por_partido=True)
        lote = Lote(ds[va].reset_index(drop=True))
        probs = probabilidades(predictor, lote)
        detalle.setdefault(nombre, {})[str(agno)] = {
          **resultado, "intervalos": intervalos_bootstrap(por_partido), "decisiones": metricas_decision(
            ds.loc[va, ["goles_0", "goles_1"]].to_numpy(), probs)}
        if nombre == "elo":
          referencia_pp[str(agno)] = por_partido
        else:
          detalle[nombre][str(agno)]["contra_elo"] = {m: comparar(por_partido, referencia_pp[str(agno)], m)
                                                      for m in ("log_loss", "rps")}
        self.avisar(f"  {nombre:<18} log-loss {resultado['log_loss']:.4f}  acierto {resultado['acierto_resultado']:5.1f}%  "
                    f"empates argmax {detalle[nombre][str(agno)]['decisiones']['argmax']['empates_predichos']:.1f}%  "
                    f"rps {detalle[nombre][str(agno)]['decisiones']['rps']['empates_predichos']:.1f}%")

    # resumen de log-loss y decisiones (media de los pliegues)
    resumen = {}
    for nombre, pliegues in detalle.items():
      resumen[nombre] = {
        "log_loss": float(np.mean([p["log_loss"] for p in pliegues.values()])),
        "rps": float(np.mean([p["rps"] for p in pliegues.values()])),
        "acierto": float(np.mean([p["acierto_resultado"] for p in pliegues.values()])),
        "ece": float(np.mean([p["ece"] for p in pliegues.values()])),
        "empates_argmax": float(np.mean([p["decisiones"]["argmax"]["empates_predichos"] for p in pliegues.values()])),
        "empates_rps": float(np.mean([p["decisiones"]["rps"]["empates_predichos"] for p in pliegues.values()])),
      }
    RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RESULTADOS.write_text(json.dumps({
      "fecha": date.today().isoformat(), "dataset": DATASET.name, "partidos": len(ds), "pliegues": PLIEGUES,
      "pesos": pesos, "resumen": resumen, "detalle": detalle,
      "nota": "Solo validacion movil; 2018/2022/2026 no se tocaron (reservados a F.3)",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    self.avisar("Resumen (media de los pliegues):")
    for nombre, r in sorted(resumen.items(), key=lambda x: x[1]["log_loss"]):
      self.avisar(f"  {nombre:<18} log-loss {r['log_loss']:.4f}  RPS {r['rps']:.4f}  acierto {r['acierto']:5.1f}%  "
                  f"ECE {r['ece']:.3f}  empates argmax/RPS {r['empates_argmax']:.1f}%/{r['empates_rps']:.1f}%")
    self.stdout.write(self.style.SUCCESS(f"Resultados -> {RESULTADOS}"))
