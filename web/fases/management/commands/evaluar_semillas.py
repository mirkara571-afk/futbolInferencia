import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from fases.dataset import construir_secuencias
from fases.evaluacion import (comparar, decision_acierto, ece, lote, metricas_probabilidad,
                              particion, probabilidades_resultado)
from fases.predictores import PredictorKeras

RAIZ = Path(settings.BASE_DIR).parent
DATASET = RAIZ / "data" / "datasets" / "partidos_features.csv"
PLIEGUES = [2006, 2010, 2014]
RESULTADOS = RAIZ / "experimentos" / "resultados_semillas.json"

K, CON_RIVAL, CON_CONTEXTO, EXTRAS = 1, False, True, ("elo",)


def metricas(real, probs):
  """Resumen de un predictor sobre unos partidos: log-loss, Brier, RPS, ECE y acierto (argmax)."""
  m = metricas_probabilidad(real, probs)
  return {"log_loss": m["log_loss"], "brier": m["brier"], "rps": m["rps"],
          "ece": ece(real, probs), "acierto": float((decision_acierto(probs) == real).mean() * 100)}


def por_partido(real, probs):
  """Arrays por partido para el bootstrap pareado y McNemar."""
  pp = metricas_probabilidad(real, probs, medias=False)
  pp["acierto_resultado"] = (decision_acierto(probs) == real).astype(float)
  return pp


class Command(BaseCommand):
  """
  ¿QUE HACE?
  D.1 (validacion correcta): compara el promedio de N LSTM por semilla contra la LSTM unica (semilla 0)
  en la validacion movil 2006/2010/2014, **reentrenando por pliegue** para no mirar el futuro: en cada
  pliegue se entrena con los ciclos anteriores y se valida en el ciclo completo. No carga los modelos
  guardados (esos estan entrenados hasta 2018/2022 y contaminan los pliegues). No escribe en data/;
  guarda el detalle en experimentos/resultados_semillas.json.

  Uso:
    python manage.py evaluar_semillas --semillas 5 --epocas 80
  """
  help = "Valida el ensamble de semillas LSTM (D.1) contra la LSTM unica, reentrenando por pliegue"
  requires_system_checks = []

  def add_arguments(self, parser):
    parser.add_argument("--semillas", type=int, default=5, help="numero de semillas a promediar (por defecto 5)")
    parser.add_argument("--epocas", type=int, default=80, help="epocas maximas con parada temprana (por defecto 80)")

  def avisar(self, texto):
    self.stdout.write(texto)
    self.stdout.flush()

  def handle(self, *args, **opciones):
    if not DATASET.exists():
      raise CommandError(f"No existe {DATASET}: ejecutar antes 'python manage.py construir_dataset'")
    ds = pd.read_csv(DATASET)
    sec = construir_secuencias(ds, k=K, con_rival=CON_RIVAL, con_contexto=CON_CONTEXTO, extras=EXTRAS)

    pliegues = {}
    pp_base_cat = {m: [] for m in ("log_loss", "brier", "rps", "acierto_resultado")}
    pp_ens_cat = {m: [] for m in ("log_loss", "brier", "rps", "acierto_resultado")}

    for agno in PLIEGUES:
      tr, va = particion(ds, hasta=agno - 4, validacion=agno)[:2]
      self.avisar(f"Pliegue {agno}: entrenando {tr.sum()} filas (<= {agno - 4}) -> validando {va.sum()} filas")
      predictores = []
      for s in range(opciones["semillas"]):
        predictor, _ = PredictorKeras.entrenar(ds, sec, tr, va, k=K, con_rival=CON_RIVAL, con_contexto=CON_CONTEXTO,
                                               semilla=s, epocas=opciones["epocas"], extras=EXTRAS)
        predictores.append(predictor)
        self.avisar(f"  semilla {s} OK")

      entrada = lote(ds, sec, va)
      real = np.sign(ds.loc[va, "goles_0"].to_numpy() - ds.loc[va, "goles_1"].to_numpy())
      probs_semillas = [probabilidades_resultado(p.predecir(entrada)) for p in predictores]
      probs_base = probs_semillas[0]
      probs_ens = np.mean(probs_semillas, axis=0)

      res_base, res_ens = metricas(real, probs_base), metricas(real, probs_ens)
      pp_b, pp_e = por_partido(real, probs_base), por_partido(real, probs_ens)
      for m in pp_base_cat:
        pp_base_cat[m].append(pp_b[m])
        pp_ens_cat[m].append(pp_e[m])

      contra = {m: comparar(pp_e, pp_b, m) for m in ("log_loss", "rps")}
      pliegues[str(agno)] = {"partidos": int(va.sum()), "base": res_base, "ensamble": res_ens, "contra_base": contra}
      self.avisar(f"  base log-loss {res_base['log_loss']:.4f} | ensamble {res_ens['log_loss']:.4f} "
                  f"| diff {contra['log_loss']['diferencia']:+.4f} (p_dm {contra['log_loss']['p_dm']:.3f})")

    pp_b_cat = {m: np.concatenate(pp_base_cat[m]) for m in pp_base_cat}
    pp_e_cat = {m: np.concatenate(pp_ens_cat[m]) for m in pp_ens_cat}
    resumen_base = {m: float(np.mean([pliegues[str(a)]["base"][m] for a in PLIEGUES])) for m in
                    ("log_loss", "brier", "rps", "ece", "acierto")}
    resumen_ens = {m: float(np.mean([pliegues[str(a)]["ensamble"][m] for a in PLIEGUES])) for m in
                   ("log_loss", "brier", "rps", "ece", "acierto")}
    contra_global = {m: comparar(pp_e_cat, pp_b_cat, m) for m in ("log_loss", "rps")}

    self.avisar("\nPromedio de los 3 pliegues (menor es mejor, salvo acierto):")
    self.avisar(f"  {'metrica':<10}{'base':>10}{'ensamble':>10}")
    for m in ("log_loss", "brier", "rps", "ece"):
      self.avisar(f"  {m:<10}{resumen_base[m]:>10.4f}{resumen_ens[m]:>10.4f}")
    self.avisar(f"  {'acierto%':<10}{resumen_base['acierto']:>9.1f}{resumen_ens['acierto']:>10.1f}")
    for m in ("log_loss", "rps"):
      c = contra_global[m]
      self.avisar(f"  ensamble vs base ({m}): diferencia {c['diferencia']:+.4f} "
                  f"[{c['intervalo'][0]:+.4f}, {c['intervalo'][1]:+.4f}]  p_boot {c['p_bootstrap']:.3f}  "
                  f"DM {c['dm']:+.2f} (p {c['p_dm']:.3f})  mcnemar {c['p_mcnemar']:.3f}")

    RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RESULTADOS.write_text(json.dumps({
      "fecha": date.today().isoformat(), "dataset": DATASET.name, "partidos": int(len(ds)),
      "modo": "validacion movil por pliegue (reentrenando; sin fuga)", "semillas": opciones["semillas"],
      "epocas_max": opciones["epocas"], "k": K, "extras": list(EXTRAS),
      "pliegues": pliegues, "resumen": {"base": resumen_base, "ensamble": resumen_ens,
                                         "contra_base": contra_global},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    self.avisar(f"\nGuardado -> {RESULTADOS}")
