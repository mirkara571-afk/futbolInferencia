import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from fases.dataset import EXTRAS, EXTRAS_EN_APP, construir_secuencias
from fases.evaluacion import POR_PARTIDO, comparar, evaluar, intervalos_bootstrap, lote, mejor_umbral, particion
from fases.predictores import PredictorElo, PredictorEmpate, PredictorKeras, PredictorMasPuntos, PredictorMedias, guardar_predictor

RAIZ = Path(settings.BASE_DIR).parent
DATASET = RAIZ / "data" / "datasets" / "partidos_features.csv"
RESULTADOS = RAIZ / "experimentos" / "resultados_baseline.json"
VALIDACION = RAIZ / "experimentos" / "resultados_validacion.json"

KS = [1, 3, 5, 8, 12]
SEMILLAS = [0, 1, 2]
METRICAS = list(POR_PARTIDO)
# validacion por origen movil: cada ciclo se valida entrenando con todos los anteriores
PLIEGUES = [2006, 2010, 2014]
CRITERIO = "log_loss"
REFERENCIA = "elo"          # todos los modelos se comparan en pares contra el Elo puro


def resumen(corridas):
  """Media y desviacion de cada metrica sobre varias semillas"""
  return {m: {"media": float(np.mean([c[m] for c in corridas])), "desv": float(np.std([c[m] for c in corridas]))} for m in METRICAS}


def promedio_pliegues(por_pliegue):
  """Resumen de la validacion movil: media de las medias de cada pliegue y media de sus desviaciones entre semillas (la variacion entre ciclos no cuenta como ruido de semilla)"""
  return {m: {"media": float(np.mean([r[m]["media"] for r in por_pliegue.values()])),
              "desv": float(np.mean([r[m]["desv"] for r in por_pliegue.values()]))} for m in METRICAS}


def criterio(seleccion):
  return seleccion["validacion"][CRITERIO]["media"], seleccion["validacion"][CRITERIO]["desv"]


def con_incertidumbre(resultado, por_partido, referencia=None):
  """(resultado) de prueba con el intervalo bootstrap de cada metrica y, si hay (referencia) (valores por partido de otro modelo), la comparacion pareada contra ella"""
  extra = {"intervalos": intervalos_bootstrap(por_partido)}
  if referencia is not None:
    extra["contra_" + REFERENCIA] = {m: comparar(por_partido, referencia, m) for m in ("log_loss", "rps")}
  return {**resultado, **extra}


def detalle_pliegue(resultado, por_partido):
  """Metricas de un pliegue de validacion con su intervalo bootstrap; "calibracion" y "ece" ya vienen en (resultado)"""
  return {**resultado, "intervalos": intervalos_bootstrap(por_partido)}


class Command(BaseCommand):
  """
  ¿QUE HACE?
  "python manage.py evaluar_modelos" es la evaluacion completa: elige la configuracion del LSTM con validacion por origen movil, la mide una vez en los mundiales 2018 y 2022 y en 2026 contra las lineas base y el Elo puro, con intervalos y comparaciones pareadas, y guarda todo en experimentos/resultados_baseline.json (y el modelo en data/modelos/lstm-<fecha>/)

  ¿COMO LO HACE?
  0-) Seleccion por origen movil: cada configuracion se entrena y valida en tres pliegues (valida el ciclo 2006 entrenando con los ciclos hasta 2002, el 2010 con hasta 2006 y el 2014 con hasta 2010), 3 semillas por pliegue; la medida es el log-loss del resultado promediado en los pliegues, y la tolerancia la desviacion media entre semillas

    0.0) Largo de la secuencia K en KS (con rival y con contexto); gana el K mas chico cuyo log-loss medio este a menos de una desviacion del mejor

    0.1) Con ese K, se prueba quitar las metricas del rival, el contexto o ambos; se queda la variante mas simple que no empeora mas de una desviacion

    0.2) Contexto adicional (EXTRAS): se prueba agregar cada uno; se agrega el que mas mejora si mejora mas que la desviacion de las semillas, y se repite con los que quedan. Solo pueden entrar los que la app sabe calcular (EXTRAS_EN_APP); los demas quedan medidos en el json

    0.3) Umbral de empate: el que maximiza el acierto de resultado en el ultimo pliegue (ciclo 2014), con sus modelos

    0.4) Las lineas base y el Elo puro se miden en los mismos pliegues, con intervalos bootstrap, calibracion, ECE y comparacion pareada contra el Elo; eso queda guardado en cada pliegue, no solo la media

  0.5) Con --solo-validacion se detiene aqui: escribe experimentos/resultados_validacion.json con todo el detalle de la validacion y no ejecuta la prueba 2018/2022/2026 ni guarda el modelo de la app

  1-) Prueba: reentrena la configuracion elegida con los ciclos hasta 2014 (epocas fijas = media de las mejores epocas de la seleccion) y la mide en los partidos de mundial de 2018 y 2022, igual que las lineas base (medias y Elo entrenados hasta 2014, mas puntos y siempre 1-1). Cada metrica lleva su intervalo del 95 % por bootstrap, y cada modelo su comparacion pareada de log-loss y RPS (bootstrap pareado y Diebold-Mariano) y McNemar del acierto contra el Elo puro; el LSTM con la semilla 0, la que se guarda

  2-) Segunda prueba, el mundial 2026: lo mismo entrenando con los ciclos hasta 2022 y midiendo en sus 104 partidos, que ningun modelo vio

  3-) Guarda el modelo de la semilla 0 entrenado hasta 2022 (con su umbral de empate), que es el que usa la app, y el json de resultados

  Tarda alrededor de una hora en una CPU de 4 nucleos (tres pliegues por configuracion); imprime el avance de cada entrenamiento
  """
  help = "Seleccion del LSTM por validacion movil, prueba contra lineas base y Elo puro con intervalos; resultados en experimentos/resultados_baseline.json"
  requires_system_checks = []

  def add_arguments(self, parser):
    parser.add_argument("--solo-validacion", action="store_true",
                        help="Corre solo la validacion movil y guarda experimentos/resultados_validacion.json; no toca la prueba 2018/2022/2026 ni el modelo de la app")

  def avisar(self, texto):
    self.stdout.write(texto)
    self.stdout.flush()

  def imprimir_validacion(self, validacion_lineas_base, base):
    """Tabla de la validacion movil (media de los pliegues) para la consola"""
    self.avisar("Validacion movil (media de los pliegues):")
    filas_validacion = {**{n: r["validacion"] for n, r in validacion_lineas_base.items()}, "lstm": base["validacion"]}
    for nombre, v in filas_validacion.items():
      fuente = base if nombre == "lstm" else validacion_lineas_base[nombre]
      calibracion = fuente["pliegues"][PLIEGUES[-1]]["detalle_semilla_0"]["ece"]
      self.avisar(f"  {nombre:<16} acierto {v['acierto_resultado']['media']:5.1f}%  log-loss {v['log_loss']['media']:.3f}  "
                  f"Brier {v['brier']['media']:.3f}  RPS {v['rps']['media']:.3f}  ECE {calibracion:.3f}")

  def entrenar_varias(self, ds, k, con_rival, con_contexto, filas_tr, filas_va, filas_ev, epocas=None, extras=(), sec=None):
    """Entrena una configuracion con cada semilla; devuelve (metricas por semilla, mejores epocas, predictores, secuencias y detalle de la semilla 0)"""
    if sec is None:
      sec = construir_secuencias(ds, k=k, con_rival=con_rival, con_contexto=con_contexto, extras=extras)
    corridas, epocas_mejores, predictores, detalle = [], [], [], None
    for semilla in SEMILLAS:
      inicio = time.time()
      validacion = filas_va if epocas is None else None
      predictor, historia = PredictorKeras.entrenar(ds, sec, filas_tr, validacion, k, con_rival, con_contexto, semilla=semilla,
                                                    epocas=epocas or 80, extras=extras)
      if epocas is None:
        epocas_mejores.append(int(np.argmin(historia["val_loss"])) + 1)
      resultado, por_partido = evaluar(predictor, ds, sec, filas_ev, por_partido=True)
      corridas.append(resultado)
      predictores.append(predictor)
      if detalle is None:                       # la semilla 0 es la que se guarda y la que resume el pliegue
        detalle = detalle_pliegue(resultado, por_partido)
      self.avisar(f"  k={k:<2} rival={con_rival!s:<5} contexto={con_contexto!s:<5} extras={','.join(extras) or '-':<26} semilla={semilla}: "
                  f"log-loss {resultado['log_loss']:.4f}, deviance {resultado['deviance']:.4f}, acierto {resultado['acierto_resultado']:.1f}% "
                  f"({time.time() - inicio:.0f}s)")
    return corridas, epocas_mejores, predictores, sec, detalle

  def configuracion(self, ds, pliegues, k, con_rival, con_contexto, extras=()):
    """Una configuracion en todos los (pliegues); devuelve (resumen, predictores y secuencias del ultimo pliegue)"""
    sec = construir_secuencias(ds, k=k, con_rival=con_rival, con_contexto=con_contexto, extras=extras)
    por_pliegue, epocas = {}, []
    for agno, (tr, va) in pliegues.items():
      corridas, mejores, predictores, _, detalle = self.entrenar_varias(ds, k, con_rival, con_contexto, tr, va, va, extras=extras, sec=sec)
      por_pliegue[agno] = {**resumen(corridas), "detalle_semilla_0": detalle}
      epocas += mejores
    return {"k": k, "con_rival": con_rival, "con_contexto": con_contexto, "extras": list(extras),
            "validacion": promedio_pliegues(por_pliegue), "pliegues": por_pliegue, "epocas": epocas}, predictores, sec

  def probar(self, ds, filas_tr, filas_prueba, titulo):
    """Lineas base y Elo puro entrenados en (filas_tr) y medidos en (filas_prueba); devuelve (resultados, valores por partido del Elo)"""
    modelos = {"medias": PredictorMedias.entrenar(ds[filas_tr]), "mas_puntos": PredictorMasPuntos(), "empate": PredictorEmpate(),
               REFERENCIA: PredictorElo.entrenar(ds[filas_tr])}
    medidos = {nombre: evaluar(p, ds, None, filas_prueba, por_partido=True) for nombre, p in modelos.items()}
    referencia = medidos[REFERENCIA][1]
    self.avisar(f"Prueba {titulo}: lineas base y Elo puro medidos")
    return {nombre: con_incertidumbre(r, pp, None if nombre == REFERENCIA else referencia) for nombre, (r, pp) in medidos.items()}, referencia

  def handle(self, *args, **opciones):
    if not DATASET.exists():
      raise CommandError(f"No existe {DATASET}: ejecutar antes 'python manage.py construir_dataset'")
    ds = pd.read_csv(DATASET)
    faltan = [c.format(i=0) for e in EXTRAS for c in EXTRAS[e] if c.format(i=0) not in ds]
    if faltan:
      raise CommandError(f"El dataset no tiene el contexto {faltan}: regenerarlo con 'python manage.py construir_dataset'")
    pliegues = {agno: particion(ds, hasta=agno - 4, validacion=agno)[:2] for agno in PLIEGUES}
    va = pliegues[PLIEGUES[-1]][1]
    tr_final, _, prueba = particion(ds, hasta=2014, prueba=[2018, 2022])
    self.avisar(f"Dataset: {len(ds)} partidos; validacion movil en " +
                ", ".join(f"{a} ({tr.sum()} entrenamiento / {v.sum()} validacion)" for a, (tr, v) in pliegues.items()) +
                f"; prueba {prueba.sum()} partidos de mundial")

    # 0.4) lineas base y Elo puro en la validacion movil (con intervalos y calibracion por pliegue)
    validacion_lineas_base, referencia_pp, base_pp = {}, {}, {}
    for nombre, crear in (("medias", PredictorMedias.entrenar), ("mas_puntos", lambda _: PredictorMasPuntos()),
                          ("empate", lambda _: PredictorEmpate()), (REFERENCIA, PredictorElo.entrenar)):
      por_pliegue = {}
      for a, (tr, v) in pliegues.items():
        resultado, por_partido = evaluar(crear(ds[tr]), ds, None, v, por_partido=True)
        por_pliegue[a] = {**resumen([resultado]), "detalle_semilla_0": detalle_pliegue(resultado, por_partido)}
        base_pp.setdefault(nombre, {})[a] = por_partido
        if nombre == REFERENCIA:
          referencia_pp[a] = por_partido
      validacion_lineas_base[nombre] = {"validacion": promedio_pliegues(por_pliegue), "pliegues": por_pliegue}
      self.avisar(f"  {nombre:<11} validacion movil: log-loss {validacion_lineas_base[nombre]['validacion']['log_loss']['media']:.4f}")
    # comparacion pareada de cada linea base contra el Elo en cada pliegue (la tabla de validacion tambien lleva p-valor)
    for nombre, datos in validacion_lineas_base.items():
      if nombre == REFERENCIA:
        continue
      for a in pliegues:
        datos["pliegues"][a]["detalle_semilla_0"]["contra_elo"] = {
          m: comparar(base_pp[nombre][a], referencia_pp[a], m) for m in ("log_loss", "rps")}

    # 0.0) largo de la secuencia
    self.avisar("Seleccion de K (con rival y contexto)")
    seleccion = [self.configuracion(ds, pliegues, k, True, True)[0] for k in KS]
    mejor, desv = min(criterio(s) for s in seleccion)
    k = min(s["k"] for s in seleccion if criterio(s)[0] <= mejor + desv)
    self.avisar(f"K elegido: {k}")

    # 0.1) que aporta cada bloque de la entrada
    self.avisar("Variantes con ese K")
    variantes = [s for s in seleccion if s["k"] == k]
    for con_rival, con_contexto in ((False, True), (True, False), (False, False)):
      variantes.append(self.configuracion(ds, pliegues, k, con_rival, con_contexto)[0])
    seleccion += variantes[1:]
    mejor_var = min(variantes, key=lambda v: criterio(v)[0])
    umbral = sum(criterio(mejor_var))
    elegida = min((v for v in variantes if criterio(v)[0] <= umbral), key=lambda v: (v["con_rival"] + v["con_contexto"], criterio(v)[0]))
    self.avisar(f"Variante elegida: rival={elegida['con_rival']}, contexto={elegida['con_contexto']}")

    # 0.2) contexto adicional, de a uno
    self.avisar("Contexto adicional")
    base = dict(elegida)
    _, predictores_base, sec_base = self.configuracion(ds, pliegues, k, base["con_rival"], base["con_contexto"])
    pruebas_extras = []
    while True:
      candidatos = [e for e in EXTRAS if e not in base["extras"] and (not pruebas_extras or e in EXTRAS_EN_APP)]
      ronda = []
      for e in candidatos:
        resultado, predictores, sec = self.configuracion(ds, pliegues, k, base["con_rival"], base["con_contexto"], tuple(base["extras"] + [e]))
        ronda.append((e, resultado, predictores, sec))
        pruebas_extras.append(resultado)
      seleccion += [r for _, r, _, _ in ronda]
      en_app = [x for x in ronda if x[0] in EXTRAS_EN_APP]
      if not en_app:
        break
      e, resultado, predictores, sec = min(en_app, key=lambda x: criterio(x[1])[0])
      mejora = criterio(base)[0] - criterio(resultado)[0]
      if mejora <= max(criterio(base)[1], criterio(resultado)[1]):
        break
      self.avisar(f"  se agrega {e}: log-loss {criterio(base)[0]:.4f} -> {criterio(resultado)[0]:.4f}")
      base, predictores_base, sec_base = resultado, predictores, sec
    epocas_final = int(round(np.mean(base["epocas"])))
    self.avisar(f"Configuracion elegida: k={k}, rival={base['con_rival']}, contexto={base['con_contexto']}, extras={base['extras']}, {epocas_final} epocas")

    # 0.3) umbral de empate con los modelos del ultimo pliegue (entrenados hasta 2010) sobre el ciclo 2014
    goles_va = ds.loc[va, ["goles_0", "goles_1"]].to_numpy()
    umbral_empate, curva = mejor_umbral(goles_va, [p.predecir(lote(ds, sec_base, va)) for p in predictores_base])
    self.avisar(f"Umbral de empate: {umbral_empate:.2f} (acierto en validacion {curva[0] * 100:.1f}% sin umbral -> {max(curva) * 100:.1f}%)")
    # comparacion del LSTM elegido contra el Elo en el ultimo pliegue (es el que decide el umbral)
    _, pp_lstm_va = evaluar(predictores_base[0], ds, sec_base, va, por_partido=True)
    base["pliegues"][PLIEGUES[-1]]["detalle_semilla_0"]["contra_elo"] = {
      m: comparar(pp_lstm_va, referencia_pp[PLIEGUES[-1]], m) for m in ("log_loss", "rps")}

    if opciones.get("solo_validacion"):
      VALIDACION.parent.mkdir(parents=True, exist_ok=True)
      VALIDACION.write_text(json.dumps({
        "fecha": date.today().isoformat(), "dataset": DATASET.name, "partidos": len(ds), "modo": "solo-validacion",
        "particion": {"seleccion": f"origen movil: valida cada ciclo de {PLIEGUES} entrenando con los anteriores; criterio {CRITERIO}",
                      "prueba": "no ejecutada (--solo-validacion)"},
        "validacion_lineas_base": validacion_lineas_base,
        "seleccion": seleccion,
        "lstm": {"validacion": base["validacion"], "pliegues": base["pliegues"], "epocas": epocas_final,
                 "umbral_empate": umbral_empate},
        "umbral_empate": {"umbrales": [round(u, 2) for u in np.arange(0, 0.41, 0.01)], "acierto_validacion": curva},
      }, indent=2, ensure_ascii=False), encoding="utf-8")
      self.imprimir_validacion(validacion_lineas_base, base)
      self.stdout.write(self.style.SUCCESS(f"Validacion -> {VALIDACION}; la prueba 2018/2022/2026 y el modelo de la app no se tocaron"))
      return

    # 1-) prueba
    self.avisar("Prueba en los mundiales 2018 y 2022")
    prueba_resultados, referencia = self.probar(ds, tr_final, prueba, "2018 y 2022")
    sec = construir_secuencias(ds, k=k, con_rival=base["con_rival"], con_contexto=base["con_contexto"], extras=base["extras"])
    corridas, sin_umbral, predictores = [], [], []
    for semilla in SEMILLAS:
      predictor, _ = PredictorKeras.entrenar(ds, sec, tr_final, None, k, base["con_rival"], base["con_contexto"], semilla=semilla,
                                             epocas=epocas_final, extras=base["extras"])
      sin_umbral.append(evaluar(predictor, ds, sec, prueba, por_partido=True))
      predictor.config["umbral_empate"] = umbral_empate
      corridas.append(evaluar(predictor, ds, sec, prueba, por_partido=True))
      predictores.append(predictor)
      self.avisar(f"  semilla={semilla}: acierto {sin_umbral[-1][0]['acierto_resultado']:.1f}% sin umbral, {corridas[-1][0]['acierto_resultado']:.1f}% con umbral")
    for nombre, tanda in (("lstm", corridas), ("lstm_sin_umbral", sin_umbral)):
      prueba_resultados[nombre] = {**resumen([r for r, _ in tanda]), "semillas": [r for r, _ in tanda],
                                   "semilla_0": con_incertidumbre(*tanda[0], referencia)}

    # 2-) mundial 2026, entrenando con todo hasta 2022
    tr_2026, _, prueba_2026 = particion(ds, hasta=2022, prueba=[2026])
    # Elo puro entrenado hasta 2022: queda como vara oficial reutilizable (A-04), no solo como fila de una tabla
    carpeta_elo = settings.MODEL_PATH / "elo"
    guardar_predictor(PredictorElo.entrenar(ds[tr_2026]), carpeta_elo, fecha=date.today().isoformat(), entrenado_hasta=2022)
    self.avisar(f"Elo puro (vara oficial) -> {carpeta_elo}")
    resultados_2026 = {}
    if prueba_2026.any():
      self.avisar(f"Prueba en el mundial 2026 ({prueba_2026.sum()} partidos), entrenando hasta 2022")
      resultados_2026, referencia_2026 = self.probar(ds, tr_2026, prueba_2026, "2026")
      corridas_2026, predictores = [], []
      for semilla in SEMILLAS:
        predictor, _ = PredictorKeras.entrenar(ds, sec, tr_2026, None, k, base["con_rival"], base["con_contexto"], semilla=semilla,
                                               epocas=epocas_final, extras=base["extras"])
        predictor.config["umbral_empate"] = umbral_empate
        corridas_2026.append(evaluar(predictor, ds, sec, prueba_2026, por_partido=True))
        predictores.append(predictor)
        self.avisar(f"  semilla={semilla}: acierto {corridas_2026[-1][0]['acierto_resultado']:.1f}%")
      resultados_2026["lstm"] = {**resumen([r for r, _ in corridas_2026]), "semillas": [r for r, _ in corridas_2026],
                                 "semilla_0": con_incertidumbre(*corridas_2026[0], referencia_2026)}

    # 3-) guardar
    carpeta = settings.MODEL_PATH / f"lstm-{date.today().isoformat()}"
    guardar_predictor(predictores[0], carpeta, fecha=date.today().isoformat(), entrenado_hasta=2022 if resultados_2026 else 2014,
                      prueba=(resultados_2026 or prueba_resultados)["lstm"]["semillas"][0])
    RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RESULTADOS.write_text(json.dumps({
      "fecha": date.today().isoformat(), "dataset": DATASET.name, "partidos": len(ds),
      "particion": {"seleccion": f"origen movil: valida cada ciclo de {PLIEGUES} entrenando con los anteriores; criterio {CRITERIO}",
                    "prueba": "entrena ciclos <= 2014, prueba mundiales 2018 y 2022 (fase mundial)"},
      "validacion_lineas_base": validacion_lineas_base,
      "validacion_lstm": {"validacion": base["validacion"], "pliegues": base["pliegues"], "epocas": epocas_final,
                          "umbral_empate": umbral_empate},
      "seleccion": seleccion,
      "elegida": {"k": k, "con_rival": base["con_rival"], "con_contexto": base["con_contexto"], "extras": base["extras"],
                  "epocas": epocas_final, "umbral_empate": umbral_empate},
      "umbral_empate": {"umbrales": [round(u, 2) for u in np.arange(0, 0.41, 0.01)], "acierto_validacion": curva},
      "prueba": prueba_resultados, "prueba_2026": resultados_2026, "modelo": carpeta.name, "modelo_elo": "elo",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    self.imprimir_validacion(validacion_lineas_base, base)
    for titulo, tabla in (("2018 y 2022", prueba_resultados), ("2026", resultados_2026)):
      self.avisar(f"Prueba {titulo} (intervalo del 95 %; contra el Elo puro: diferencia de log-loss, p de Diebold-Mariano, p de McNemar):")
      for nombre, r in tabla.items():
        r = r.get("semilla_0", r)
        i = r["intervalos"]
        linea = (f"  {nombre:<16} acierto {r['acierto_resultado']:5.1f}% [{i['acierto_resultado'][0]:.1f}, {i['acierto_resultado'][1]:.1f}]  "
                 f"log-loss {r['log_loss']:.3f} [{i['log_loss'][0]:.3f}, {i['log_loss'][1]:.3f}]  "
                 f"Brier {r['brier']:.3f}  RPS {r['rps']:.3f} [{i['rps'][0]:.3f}, {i['rps'][1]:.3f}]")
        if "contra_" + REFERENCIA in r:
          c = r["contra_" + REFERENCIA]["log_loss"]
          linea += f"  vs elo {c['diferencia']:+.3f} (p_dm {c['p_dm']:.3f}, p_mcnemar {c['p_mcnemar']:.3f})"
        self.avisar(linea)
    self.stdout.write(self.style.SUCCESS(f"Resultados -> {RESULTADOS}; modelo -> {carpeta}"))
