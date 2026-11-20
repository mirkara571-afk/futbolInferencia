from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from fases.dataset import construir_secuencias
from fases.evaluacion import comparar, ece, lote, metricas_probabilidad, probabilidades_resultado
from fases.predictores import cargar_predictor

RAIZ = Path(settings.BASE_DIR).parent
DATASET = RAIZ / "data" / "datasets" / "partidos_features.csv"
PLIEGUES = [2006, 2010, 2014]


class Command(BaseCommand):
  """
  ¿QUE HACE?
  D.1: compara el promedio de N LSTM por semilla (data/modelos/lstm-semilla-<i>) contra la LSTM unica
  (--base, por defecto lstm-2026-10-25) en la validacion movil 2006/2010/2014, sin reentrenar: carga los
  modelos guardados, predice cada pliegue y promedia las probabilidades de las semillas.

  Uso:
    python manage.py medir_semillas --semillas 5 --base lstm-2026-10-25
  """
  help = "Compara el promedio de LSTM semillas (D.1) contra la LSTM unica en validacion movil"
  requires_system_checks = []

  def add_arguments(self, parser):
    parser.add_argument("--semillas", type=int, default=5, help="numero de semillas a promediar (por defecto 5)")
    parser.add_argument("--base", default="lstm-2026-10-25", help="carpeta de la LSTM unica (por defecto lstm-2026-10-25)")

  def handle(self, *args, **opciones):
    if not DATASET.exists():
      raise CommandError(f"No existe {DATASET}: ejecutar antes 'python manage.py construir_dataset'")
    ds = pd.read_csv(DATASET)
    ds = ds[ds["agno"] <= 2022].reset_index(drop=True)
    k, con_rival, con_contexto, extras = 1, False, True, ("elo",)
    sec = construir_secuencias(ds, k=k, con_rival=con_rival, con_contexto=con_contexto, extras=extras)
    semillas = [cargar_predictor(settings.MODEL_PATH / f"lstm-semilla-{i}")[0] for i in range(opciones["semillas"])]
    base, _ = cargar_predictor(settings.MODEL_PATH / opciones["base"])
    reales = np.sign(ds["goles_0"].to_numpy() - ds["goles_1"].to_numpy())

    pp_base = {m: [] for m in ("log_loss", "brier", "rps", "acierto_resultado")}
    pp_ens = {m: [] for m in ("log_loss", "brier", "rps", "acierto_resultado")}
    res_base, res_ens = {"log_loss": [], "brier": [], "rps": [], "ece": [], "acierto": []}, \
                        {"log_loss": [], "brier": [], "rps": [], "ece": [], "acierto": []}
    for agno in PLIEGUES:
      filas = (ds["agno"] == agno).to_numpy()
      entrada = lote(ds, sec, filas)
      probs_ens = np.mean([probabilidades_resultado(s.predecir(entrada)) for s in semillas], axis=0)
      probs_base = probabilidades_resultado(base.predecir(entrada))
      real = reales[filas]
      for nombre, probs, pp, res in (("base", probs_base, pp_base, res_base), ("ensamble", probs_ens, pp_ens, res_ens)):
        m = metricas_probabilidad(real, probs, medias=False)
        for metrica in ("log_loss", "brier", "rps"):
          pp[metrica].append(m[metrica])
          res[metrica].append(float(m[metrica].mean()))
        res["ece"].append(float(ece(real, probs)))
        res["acierto"].append(float((probs.argmax(1) == (1 - real)).mean() * 100))
        pp["acierto_resultado"].append((probs.argmax(1) == (1 - real)).astype(float))
      self.stdout.write(f"  {agno}: base log-loss {res_base['log_loss'][-1]:.4f} | ensamble {res_ens['log_loss'][-1]:.4f}")

    self.stdout.write("\nPromedio de los 3 pliegues (menor es mejor):")
    self.stdout.write(f"  {'metrica':<10}{'base':>12}{'ensamble':>12}")
    for m in ("log_loss", "brier", "rps", "ece"):
      self.stdout.write(f"  {m:<10}{np.mean(res_base[m]):>12.4f}{np.mean(res_ens[m]):>12.4f}")
    self.stdout.write(f"  {'acierto%':<10}{np.mean(res_base['acierto']):>11.1f}{np.mean(res_ens['acierto']):>12.1f}")

    pp_ens_cat = {m: np.concatenate(pp_ens[m]) for m in pp_ens}
    pp_base_cat = {m: np.concatenate(pp_base[m]) for m in pp_base}
    for metrica in ("log_loss", "rps"):
      cmp = comparar(pp_ens_cat, pp_base_cat, metrica)
      self.stdout.write(f"\n  ensamble vs base ({metrica}): diferencia {cmp['diferencia']:+.4f} "
                        f"[{cmp['intervalo'][0]:+.4f}, {cmp['intervalo'][1]:+.4f}]  p_boot {cmp['p_bootstrap']:.3f}  "
                        f"DM {cmp['dm']:+.2f} (p {cmp['p_dm']:.3f})")
