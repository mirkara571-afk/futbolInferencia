"""
Evaluacion del modelo jerarquico (plan 07, tarea JH-04).

¿QUE HACE?
"python manage.py evaluar_jerarquico" mide el modelo jerarquico/state-space (PredictorJerarquico, nombre "jerarquico")
contra el Elo puro y el Poisson GLM, sin tocar la prueba congelada 2018/2022/2026:

  1. Validacion movil (pliegues 2006/2010/2014, sin xG): cada pliegue entrena los tres predictores con los ciclos
     anteriores (hasta agno - 4) y valida en el ciclo completo. Es la vara desplegable real (no hay xG antes de 2018),
     asi que el jerarquico degenera a un "Poisson jerarquico de goles" y se compara contra Elo y GLM en log-loss y RPS
     con Diebold-Mariano pareado, ademas de intervalos bootstrap.

  2. Split diagnostico <=2018 -> 2022 (mirada extraordinaria, NO promocion): entrena el jerarquico sin xG, el jerarquico
     con xG (usar_xg=True, con el xG de 2018 como observacion de calidad latente), el Elo y el GLM sobre toda la
     historia hasta 2018 inclusive, y los evalua en los 64 partidos de mundial de 2022. El xG del partido objetivo nunca
     entra en predecir. Se registra como mirada diagnostica: sus numeros no se usan para elegir ni promover.

¿COMO LO HACE?
Reutiliza la vara de medir de evaluacion.py (evaluar, intervalos_bootstrap, comparar, particion) y los predictores ya
aprobados (PredictorJerarquico, PredictorElo, PredictorPoissonGLM). No modifica ningun modelo y no cambia el esquema de
las 14 metricas. Vuelca experimentos/resultados_jerarquico.json.

No toca 2026, no reoptimiza el modelo jerarquico y no re-mira la prueba 2018/2022 de F.3: el split diagnostico sobre
2022 queda registrado como mirada extraordinaria.
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from fases.dataset import FASE_MUNDIAL
from fases.evaluacion import comparar, evaluar, intervalos_bootstrap, particion
from fases.modelos_estadisticos import PredictorPoissonGLM
from fases.modelos_jerarquicos import PredictorJerarquico
from fases.predictores import PredictorElo

RAIZ = Path(settings.BASE_DIR).parent
DATASET = RAIZ / "data" / "datasets" / "partidos_features.csv"
DATASET_XG = RAIZ / "data" / "datasets" / "partidos_features_xg.csv"
RESULTADOS = RAIZ / "experimentos" / "resultados_jerarquico.json"

PLIEGUES = [2006, 2010, 2014]
# metricas resumen que se guardan por modelo/pliegue (acierto_resultado y ece salen de "resultado", no del bootstrap)
CLAVES_RESUMEN = ("log_loss", "rps", "acierto_resultado", "ece")


def particiones_jerarquico(ds):
  """
  ¿QUE HACE?
  Construye las particiones del comando sobre (ds):

    - pliegues: validacion movil 2006/2010/2014; cada pliegue entrena con los ciclos anteriores (hasta agno - 4) y
      valida el ciclo completo (sus eliminatorias y su mundial), con "particion".
    - split_diagnostico: entrenamiento = toda la historia hasta 2018 inclusive (agno <= 2018) y prueba = los partidos
      de mundial de 2022 (agno == 2022 y fase == "mundial"), como mirada diagnostica.

  Returns:
    - dict con "pliegues" {agno: (filas_entrenamiento, filas_validacion)} (arrays booleanos) y
      "split_diagnostico" (filas_entrenamiento, filas_prueba) (arrays booleanos)
  """
  pliegues = {agno: particion(ds, hasta=agno - 4, validacion=agno)[:2] for agno in PLIEGUES}
  agno = ds["agno"].to_numpy()
  fase = ds["fase"].to_numpy()
  filas_tr = agno <= 2018
  filas_pr = (agno == 2022) & (fase == FASE_MUNDIAL)
  return {"pliegues": pliegues, "split_diagnostico": (filas_tr, filas_pr)}


def resumen_modelo(resultado, por_partido, con_intervalos=True):
  """
  ¿QUE HACE?
  Resumen de un modelo para el JSON: las metricas de (CLAVES_RESUMEN) y, si (con_intervalos), el intervalo bootstrap
  del 95 % de cada metrica por partido. "acierto_resultado" va con mas decimales (es un porcentaje).

  Returns:
    - dict con log_loss, rps, acierto_resultado, ece y, opcionalmente, intervalos
  """
  res = {m: round(float(resultado[m]), 6) for m in ("log_loss", "rps", "ece")}
  res["acierto_resultado"] = round(float(resultado["acierto_resultado"]), 4)
  if con_intervalos and por_partido is not None:
    res["intervalos"] = intervalos_bootstrap(por_partido)
  return res


def comparaciones_jerarquico(pp_jerarquico, pp_elo, pp_glm):
  """
  ¿QUE HACE?
  Comparacion pareada del jerarquico contra el Elo y el GLM en log-loss y RPS (bootstrap pareado + Diebold-Mariano).

  Returns:
    - dict con "contra_elo" y "contra_glm", cada uno con las dos metricas y el resultado de "comparar"
  """
  return {
    "contra_elo": {m: comparar(pp_jerarquico, pp_elo, m) for m in ("log_loss", "rps")},
    "contra_glm": {m: comparar(pp_jerarquico, pp_glm, m) for m in ("log_loss", "rps")},
  }


def media_pliegues(detalle):
  """
  ¿QUE HACE?
  Media simple (sin ponderar por numero de partidos) de cada metrica resumen de (CLAVES_RESUMEN) sobre los pliegues de
  (detalle) {agno: {modelo: resumen}}. Es la tabla de validacion movil media.

  Returns:
    - dict {modelo: {log_loss, rps, acierto_resultado, ece}} promediado sobre los pliegues
  """
  modelos = sorted({nombre for pliegue in detalle.values() for nombre in pliegue})
  return {
    nombre: {m: float(np.mean([detalle[a][nombre][m] for a in detalle if nombre in detalle[a]]))
             for m in CLAVES_RESUMEN}
    for nombre in modelos
  }


def detalle_split_diagnostico(resultados, comparaciones, n_entrenamiento, n_prueba):
  """
  ¿QUE HACE?
  Arma el dict del split diagnostico con la marca explicita "diagnostico": true (mirada extraordinaria, no promocion) y
  los tamanos de la particion.

  Returns:
    - dict con n_entrenamiento, n_prueba, diagnostico=True, los (resultados) por modelo y las (comparaciones) pareadas
  """
  return {
    "n_entrenamiento": n_entrenamiento,
    "n_prueba": n_prueba,
    "diagnostico": True,
    **resultados,
    "comparaciones": comparaciones,
  }


def estructura_json(fecha, pliegues, media_validacion, split_diagnostico):
  """
  ¿QUE HACE?
  Estructura final del JSON de resultados con los campos principales del comando y la nota de que el split usa 2022 como
  mirada diagnostica (no promocion) y que 2026 no se toca.
  """
  return {
    "fecha": fecha,
    "dataset": DATASET.name,
    "dataset_xg": DATASET_XG.name,
    "pliegues": pliegues,
    "validacion_movil_media": media_validacion,
    "split_diagnostico": split_diagnostico,
    "nota": ("Validacion movil 2006/2010/2014 (sin xG) para comparar el jerarquico contra Elo y GLM; el split "
             "<=2018 -> 2022 (mundial) es una mirada DIAGNOSTICA (no promocion): se registra como mirada "
             "extraordinaria y no se usa para elegir ni promover. 2026 no se toca."),
  }


class Command(BaseCommand):
  """Evaluacion del modelo jerarquico: validacion movil + split diagnostico xG; resultados en experimentos/resultados_jerarquico.json."""
  help = "Validacion movil del jerarquico contra Elo y GLM + split diagnostico xG (mirada, no promocion); resultados en experimentos/resultados_jerarquico.json"
  requires_system_checks = []

  def avisar(self, texto):
    self.stdout.write(texto)
    self.stdout.flush()

  def handle(self, *args, **opciones):
    if not DATASET.exists():
      raise CommandError(f"No existe {DATASET}: ejecutar antes 'python manage.py construir_dataset'")
    if not DATASET_XG.exists():
      raise CommandError(f"No existe {DATASET_XG}: el split diagnostico necesita el dataset partidos_features_xg.csv")
    ds = pd.read_csv(DATASET)
    ds_xg = pd.read_csv(DATASET_XG)

    partes = particiones_jerarquico(ds)
    pliegues = partes["pliegues"]
    filas_tr_diag, filas_pr_diag = partes["split_diagnostico"]

    # 1) validacion movil (sin xG): jerarquico, Elo y GLM sobre el mismo tr/va de cada pliegue
    detalle = {}
    for agno in PLIEGUES:
      tr, va = pliegues[agno]
      self.avisar(f"Pliegue {agno}: {int(tr.sum())} entrenamiento / {int(va.sum())} validacion")
      modelos = {
        "jerarquico": PredictorJerarquico.entrenar(ds[tr].reset_index(drop=True)),
        "elo": PredictorElo.entrenar(ds[tr].reset_index(drop=True)),
        "glm": PredictorPoissonGLM.entrenar(ds[tr].reset_index(drop=True)),
      }
      por_partido = {}
      detalle[agno] = {}
      for nombre, predictor in modelos.items():
        resultado, pp = evaluar(predictor, ds, None, va, por_partido=True)
        por_partido[nombre] = pp
        detalle[agno][nombre] = resumen_modelo(resultado, pp)
        self.avisar(f"  {nombre:<12} log-loss {resultado['log_loss']:.4f}  RPS {resultado['rps']:.4f}  "
                    f"acierto {resultado['acierto_resultado']:5.1f}%  ECE {resultado['ece']:.4f}")
      detalle[agno]["jerarquico"].update(
        comparaciones_jerarquico(por_partido["jerarquico"], por_partido["elo"], por_partido["glm"]))

    media_validacion = media_pliegues(detalle)

    # 2) split diagnostico <=2018 -> 2022 (mirada extraordinaria): jerarquico sin/con xG, Elo y GLM
    self.avisar(f"Split diagnostico <=2018 -> 2022: {int(filas_tr_diag.sum())} entrenamiento / "
                f"{int(filas_pr_diag.sum())} prueba (mundial 2022)")
    jer = PredictorJerarquico.entrenar(ds[filas_tr_diag].reset_index(drop=True))
    jer_xg = PredictorJerarquico.entrenar(ds_xg[filas_tr_diag].reset_index(drop=True), usar_xg=True)
    elo = PredictorElo.entrenar(ds[filas_tr_diag].reset_index(drop=True))
    glm = PredictorPoissonGLM.entrenar(ds[filas_tr_diag].reset_index(drop=True))

    r_jer, pp_jer = evaluar(jer, ds, None, filas_pr_diag, por_partido=True)
    r_jer_xg, pp_jer_xg = evaluar(jer_xg, ds_xg, None, filas_pr_diag, por_partido=True)
    r_elo, pp_elo = evaluar(elo, ds, None, filas_pr_diag, por_partido=True)
    r_glm, pp_glm = evaluar(glm, ds, None, filas_pr_diag, por_partido=True)

    resultados_split = {
      "jerarquico": resumen_modelo(r_jer, pp_jer),
      "jerarquico_xg": resumen_modelo(r_jer_xg, pp_jer_xg),
      "elo": resumen_modelo(r_elo, pp_elo),
      "glm": resumen_modelo(r_glm, pp_glm),
    }
    comparaciones_split = {
      "jerarquico_vs_elo": {m: comparar(pp_jer, pp_elo, m) for m in ("log_loss", "rps")},
      "jerarquico_vs_glm": {m: comparar(pp_jer, pp_glm, m) for m in ("log_loss", "rps")},
      "jerarquico_xg_vs_jerarquico": {m: comparar(pp_jer_xg, pp_jer, m) for m in ("log_loss", "rps")},
    }
    split_diagnostico = detalle_split_diagnostico(resultados_split, comparaciones_split,
                                                  int(filas_tr_diag.sum()), int(filas_pr_diag.sum()))

    # 3) volcar JSON
    estructura = estructura_json(date.today().isoformat(), {str(a): detalle[a] for a in PLIEGUES},
                                 media_validacion, split_diagnostico)
    RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RESULTADOS.write_text(json.dumps(estructura, indent=2, ensure_ascii=False), encoding="utf-8")

    # 4) tabla de consola
    self.avisar("")
    self.avisar("Validacion movil (media de los pliegues):")
    self.avisar(f"{'modelo':<12} {'log-loss':>9} {'RPS':>8} {'acierto':>8} {'ECE':>8}")
    for nombre in ("jerarquico", "elo", "glm"):
      m = media_validacion[nombre]
      self.avisar(f"{nombre:<12} {m['log_loss']:>9.4f} {m['rps']:>8.4f} {m['acierto_resultado']:>7.1f}% {m['ece']:>8.4f}")

    self.avisar("")
    self.avisar("Split diagnostico <=2018 -> 2022 (mirada diagnostica, no promocion):")
    self.avisar(f"{'modelo':<20} {'log-loss':>9} {'RPS':>8} {'acierto':>8} {'ECE':>8}")
    for nombre in ("jerarquico", "jerarquico_xg", "elo", "glm"):
      r = resultados_split[nombre]
      self.avisar(f"{nombre:<20} {r['log_loss']:>9.4f} {r['rps']:>8.4f} {r['acierto_resultado']:>7.1f}% {r['ece']:>8.4f}")
    self.avisar("Comparaciones pareadas (p de Diebold-Mariano):")
    for nombre, comp in comparaciones_split.items():
      self.avisar(f"  {nombre:<32} log-loss p_dm {comp['log_loss']['p_dm']:.3f}  RPS p_dm {comp['rps']['p_dm']:.3f}")

    self.avisar("")
    self.avisar("El split diagnostico usa 2022 como mirada extraordinaria: no se usa para promover ni para reoptimizar.")
    self.avisar("2026 no se toco.")
    self.stdout.write(self.style.SUCCESS(f"Resultados -> {RESULTADOS}"))
