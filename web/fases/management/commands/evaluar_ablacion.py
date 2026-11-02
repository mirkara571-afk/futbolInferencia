import json
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from fases.evaluacion import POR_PARTIDO, comparar, evaluar, intervalos_bootstrap, particion
from fases.modelos_estadisticos import _categorias

RAIZ = Path(settings.BASE_DIR).parent
DATASET = RAIZ / "data" / "datasets" / "partidos_features.csv"
RESULTADOS = RAIZ / "experimentos" / "resultados_ablacion.json"
PLIEGUES = [2006, 2010, 2014]

VARIANTES = {
  "base": set(),
  "sin_gkps": {"gkps"},
  "sin_mos": {"mos"},
  "sin_gkps_mos": {"gkps", "mos"},
  "sin_rate": {"rate_GF", "rate_GC_rival"},
  "sin_fase": {"fase_final"},
  "sin_contexto": {"elo_diff", "es_local", "neutral", "fase_final"},
}


def _columna(tabla, nombre, defecto=0.0):
  if nombre in tabla:
    return pd.to_numeric(tabla[nombre], errors="coerce").fillna(defecto).to_numpy(dtype=float)
  return np.full(len(tabla), float(defecto))


def _base_lado(tabla, i, categorias, excluir):
  """Igual que modelos_estadisticos._base_lado, pero sin las columnas de (excluir)"""
  otro = 1 - i
  datos = {}
  if "elo_diff" not in excluir:
    datos["elo_diff"] = (_columna(tabla, f"elo_{i}", 1500.0) - _columna(tabla, f"elo_{otro}", 1500.0)) / 100.0
  if "es_local" not in excluir:
    datos["es_local"] = _columna(tabla, f"local_{i}")
  if "neutral" not in excluir:
    datos["neutral"] = _columna(tabla, "neutral", 1.0)
  if "rate_GF" not in excluir:
    datos["rate_GF"] = _columna(tabla, f"rate_GF_{i}") / 10.0
  if "rate_GC_rival" not in excluir:
    datos["rate_GC_rival"] = _columna(tabla, f"rate_GC_{otro}") / 10.0
  if "gkps" not in excluir:
    datos["gkps"] = _columna(tabla, f"gkps_{i}") / 100.0
  if "mos" not in excluir:
    datos["mos"] = _columna(tabla, f"mos_{i}") / 100.0
  if "fase_final" not in excluir:
    datos["fase_final"] = _columna(tabla, "fase_final")
  categoria = tabla["categoria"].astype(str) if "categoria" in tabla else pd.Series([""] * len(tabla), index=tabla.index)
  for cat in categorias:
    if f"cat_{cat}" not in excluir:
      datos[f"cat_{cat}"] = (categoria == cat).to_numpy(dtype=float)
  return pd.DataFrame(datos, index=tabla.index)


def _matriz(tabla, categorias, excluir):
  X = pd.concat([_base_lado(tabla, 0, categorias, excluir), _base_lado(tabla, 1, categorias, excluir)],
                ignore_index=True)
  return sm.add_constant(X, has_constant="add")


class _PoissonModelo:
  """Predictor minimo con el contrato 'predecir(lote)' que usa la evaluacion; solo para el ablation"""
  def __init__(self, columnas, categorias, excluir, params=None, booster=None):
    self.columnas = list(columnas)
    self.categorias = list(categorias)
    self.excluir = set(excluir)
    self.params = params
    self.booster = booster

  def _X(self, tabla):
    X = _matriz(tabla, self.categorias, self.excluir)
    return X.reindex(columns=self.columnas, fill_value=0.0)

  def predecir(self, lote):
    X = self._X(lote.tabla).to_numpy(dtype=float)
    lambdas = self.booster.predict(X) if self.booster is not None else np.exp(np.clip(X @ self.params, -20, 20))
    n = len(lote.tabla)
    return np.stack([lambdas[:n], lambdas[n:]], axis=1)


def _entrenar_glm(dataset, categorias, excluir):
  X = _matriz(dataset, categorias, excluir)
  y = np.concatenate([dataset["goles_0"].to_numpy(dtype=float), dataset["goles_1"].to_numpy(dtype=float)])
  ajuste = sm.GLM(y, X, family=sm.families.Poisson()).fit(maxiter=300)
  return _PoissonModelo(X.columns, categorias, excluir, params=ajuste.params)


def _entrenar_lgb(dataset, categorias, excluir):
  X = _matriz(dataset, categorias, excluir)
  y = np.concatenate([dataset["goles_0"].to_numpy(dtype=float), dataset["goles_1"].to_numpy(dtype=float)])
  ajuste = lgb.LGBMRegressor(objective="poisson", n_estimators=200, learning_rate=0.05, num_leaves=31,
                             random_state=0, n_jobs=1, verbose=-1).fit(X, y)
  return _PoissonModelo(X.columns, categorias, excluir, booster=ajuste.booster_)


class Command(BaseCommand):
  """Ablation de features de GLM y LightGBM Poisson: se apaga un bloque y se mide el log-loss en validacion movil."""
  help = "Ablation de variables de GLM/LightGBM; resultados en experimentos/resultados_ablacion.json"
  requires_system_checks = []

  def avisar(self, texto):
    self.stdout.write(texto)
    self.stdout.flush()

  def handle(self, *args, **opciones):
    if not DATASET.exists():
      raise CommandError(f"No existe {DATASET}: ejecutar antes 'python manage.py construir_dataset'")
    ds = pd.read_csv(DATASET)
    categorias = _categorias(ds)
    familias = {"glm": _entrenar_glm, "lightgbm": _entrenar_lgb}
    resumen = {}
    for familia, entrenar in familias.items():
      resumen[familia] = {}
      base_por_pliegue = {}
      for nombre, excluir in VARIANTES.items():
        pliegues, contra_base = [], []
        for agno in PLIEGUES:
          tr, va = particion(ds, hasta=agno - 4, validacion=agno)[:2]
          predictor = entrenar(ds[tr], categorias, excluir)
          resultado, por_partido = evaluar(predictor, ds, None, va, por_partido=True)
          pliegues.append(resultado)
          if nombre == "base":
            base_por_pliegue[agno] = por_partido
            contra_base.append(None)
          else:
            contra_base.append(comparar(por_partido, base_por_pliegue[agno], "log_loss"))
        resumen[familia][nombre] = {
          "log_loss": float(np.mean([p["log_loss"] for p in pliegues])),
          "rps": float(np.mean([p["rps"] for p in pliegues])),
          "acierto": float(np.mean([p["acierto_resultado"] for p in pliegues])),
          "ece": float(np.mean([p["ece"] for p in pliegues])),
          "pliegues_log_loss": [round(p["log_loss"], 4) for p in pliegues],
          "p_dm_vs_base": [None if c is None else round(c["p_dm"], 3) for c in contra_base],
        }
        delta = resumen[familia][nombre]["log_loss"] - resumen[familia]["base"]["log_loss"]
        self.avisar(f"  {familia:<10} {nombre:<16} log-loss {resumen[familia][nombre]['log_loss']:.4f}  "
                    f"delta {delta:+.4f}  acierto {resumen[familia][nombre]['acierto']:5.1f}%  ECE {resumen[familia][nombre]['ece']:.3f}")
    RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RESULTADOS.write_text(json.dumps({
      "fecha": date.today().isoformat(), "dataset": DATASET.name, "pliegues": PLIEGUES, "variantes": resumen,
      "nota": "Solo validacion movil; mds/mms no estan en GLM/LightGBM Poisson (ablacion LSTM pendiente)",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    self.stdout.write(self.style.SUCCESS(f"Resultados -> {RESULTADOS}"))
