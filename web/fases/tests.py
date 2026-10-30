import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from .dataset import CATEGORIAS, CICLO_INTERNACIONALES, N_CONTEXTO, construir_dataset, construir_secuencias, mapa_paises
from .modulo import ALL_METRICS, BASE_METRICS, COLS_MODELO_POR_EQUIPO, METRICAS_ABSTRACTAS, OPTIONAL_METRICS, historico_abstractas
from .evaluacion import comparar, ece, intervalos_bootstrap, metricas, metricas_por_partido, metricas_probabilidad, particion, probabilidades_deterministas, tabla_calibracion
from .predictores import Lote, PredictorElo, PredictorEmpate, PredictorKeras, PredictorMasPuntos, PredictorMedias, cargar_predictor, guardar_predictor, lote_desde_entrada


def partidos_df(filas):
  return pd.DataFrame(filas, columns=["home", "away", "score_0", "score_1", "agno"])


PROXY_VACIO = pd.DataFrame(columns=["pais", "agno", *METRICAS_ABSTRACTAS])


class ConstruirDatasetTests(SimpleTestCase):
  """Torneo inventado: eliminatorias A-B 2:0 y B-C 1:1, mundial A-C 1:0 (ids: A=0, B=1, C=2)"""

  def setUp(self):
    self.clasificaciones = partidos_df([["A", "B", 2, 0, 1934], ["B", "C", 1, 1, 1934]])
    self.partidos = partidos_df([["A", "C", 1, 0, 1934]])
    self.ds = construir_dataset(self.partidos, self.clasificaciones, PROXY_VACIO, agnos=[1934])

  def test_una_fila_por_partido_en_orden(self):
    self.assertEqual(self.ds["fase"].tolist(), ["eliminatoria", "eliminatoria", "mundial"])
    self.assertEqual(self.ds[["goles_0", "goles_1"]].values.tolist(), [[2, 0], [1, 1], [1, 0]])

  def test_metricas_previas_al_partido(self):
    # 1er partido: nadie tiene historia, ni siquiera su propio resultado
    self.assertEqual(self.ds.loc[0, ["PJ_0", "GF_0", "PJ_1", "GF_1"]].tolist(), [0, 0, 0, 0])
    # B llega al 2do partido con la derrota 0:2
    self.assertEqual(self.ds.loc[1, ["PTS_0", "PJ_0", "PP_0", "GF_0", "GC_0", "D_0"]].tolist(), [0, 1, 1, 0, 2, -2])
    # al mundial: A con la victoria 2:0 y C con el empate 1:1
    self.assertEqual(self.ds.loc[2, ["PTS_0", "PJ_0", "PG_0", "GF_0", "GC_0", "D_0"]].tolist(), [3, 1, 1, 2, 0, 2])
    self.assertEqual(self.ds.loc[2, ["PTS_1", "PJ_1", "PE_1", "GF_1", "GC_1", "D_1"]].tolist(), [1, 1, 1, 1, 1, 0])

  def test_metricas_derivadas(self):
    # rate_GF = GF / PJ * 10
    self.assertAlmostEqual(self.ds.loc[2, "rate_GF_0"], 20.0)
    self.assertAlmostEqual(self.ds.loc[2, "rate_GF_1"], 10.0)
    # el progreso abstracto vuelve al historico al empezar el mundial (como en la app): sin historico se usan las tasas de
    # referencia (VI 0,30 y penales 0,12 por partido); A llega con 1 partido y 0 goles en contra
    # gkps = 60 * 0,30 + 40 - 10 * max(0, 0/1 - 0,75 * 0,12) = 58
    self.assertAlmostEqual(self.ds.loc[2, "gkps_0"], 58.0)

  def test_equipo_sin_eliminatorias_hereda_su_ultimo_ciclo(self):
    clasif = partidos_df([["A", "B", 2, 0, 1934], ["B", "C", 1, 1, 1938]])
    partidos = partidos_df([["A", "C", 1, 0, 1934], ["A", "B", 0, 0, 1938]])
    ds = construir_dataset(partidos, clasif, PROXY_VACIO, agnos=[1934, 1938])
    # en 1938 A no jugo eliminatorias: arranca con lo de sus eliminatorias de 1934 (victoria 2:0), no con el mundial
    fila = ds[(ds["agno"] == 1938) & (ds["fase"] == "mundial")].iloc[0]
    self.assertEqual([fila["PJ_0"], fila["GF_0"]], [1, 2])


class HistoricoAbstractasTests(SimpleTestCase):

  def test_solo_mundiales_anteriores(self):
    proxy = pd.DataFrame(
      [["A", 1930, 2, 1, 0, 0], ["A", 1938, 5, 1, 0, 0], ["B", 1934, 1, 1, 0, 0]],
      columns=["pais", "agno", *METRICAS_ABSTRACTAS],
    )
    hist = historico_abstractas(proxy, {"A": 0, "B": 1}, agno=1934)
    # A solo suma 1930; B (1934) no entra porque 1934 no es anterior a 1934
    self.assertEqual(hist["pais"].tolist(), [0])
    self.assertEqual(hist.loc[0, "VI"], 2)


class ConstruirSecuenciasTests(SimpleTestCase):

  def test_relleno_y_orden(self):
    clasif = partidos_df([["A", "B", 2, 0, 1934], ["B", "C", 1, 1, 1934]])
    partidos = partidos_df([["A", "C", 1, 0, 1934]])
    ds = construir_dataset(partidos, clasif, PROXY_VACIO, agnos=[1934])
    sec = construir_secuencias(ds, k=3)
    pj = 1  # posicion de PJ dentro de ALL_METRICS
    # A en el partido del mundial: 1 paso de relleno, luego su eliminatoria (PJ 0) y el actual (PJ 1)
    self.assertEqual(sec["mascara_0"][2].tolist(), [0, 1, 1])
    self.assertEqual(sec["seq_0"][2, :, pj].tolist(), [0, 0, 1])
    self.assertEqual(sec["y"][2].tolist(), [1, 0])
    self.assertEqual(sec["id_0"][2], mapa_paises(partidos, clasif)["A"])

  def test_metricas_del_rival(self):
    clasif = partidos_df([["A", "B", 2, 0, 1934], ["B", "C", 1, 1, 1934]])
    partidos = partidos_df([["A", "C", 1, 0, 1934]])
    ds = construir_dataset(partidos, clasif, PROXY_VACIO, agnos=[1934])
    f = len(ALL_METRICS)
    sec = construir_secuencias(ds, k=2, con_contexto=False)
    self.assertEqual(sec["seq_0"].shape, (3, 2, 2 * f))
    # en el partido del mundial, el ultimo paso de A trae sus metricas y despues las de C (su rival ese dia)
    self.assertEqual(sec["seq_0"][2, -1, :f].tolist(), ds.loc[2, [f"{m}_0" for m in ALL_METRICS]].tolist())
    self.assertEqual(sec["seq_0"][2, -1, f:].tolist(), ds.loc[2, [f"{m}_1" for m in ALL_METRICS]].tolist())
    # y el de C, al reves
    self.assertEqual(sec["seq_1"][2, -1, f:].tolist(), ds.loc[2, [f"{m}_0" for m in ALL_METRICS]].tolist())
    self.assertEqual(construir_secuencias(ds, k=2, con_rival=False, con_contexto=False)["seq_0"].shape, (3, 2, f))

  def test_contexto(self):
    clasif = partidos_df([["A", "B", 2, 0, 1934]])
    partidos = partidos_df([["A", "B", 1, 0, 1934]])
    ds = construir_dataset(partidos, clasif, PROXY_VACIO, agnos=[1934])
    sec = construir_secuencias(ds, k=1)
    contexto = sec["seq_0"][:, -1, 2 * len(ALL_METRICS):]
    mundial = [0.0] * len(CATEGORIAS)
    mundial[CATEGORIAS.index("mundial")] = 1.0
    eliminatoria = [0.0] * len(CATEGORIAS)
    eliminatoria[CATEGORIAS.index("eliminatoria_mundial")] = 1.0
    # eliminatoria en casa: es_local 1, neutral 0; mundial sin dato de sede: neutral 1, nadie es local
    self.assertEqual(contexto.shape[1], N_CONTEXTO)
    self.assertEqual(contexto[0].tolist(), [1.0, 0.0, *eliminatoria, 0.0, 0.0])
    self.assertEqual(contexto[1].tolist(), [0.0, 1.0, *mundial, 0.0, 0.0])
    # el visitante nunca es local
    self.assertEqual(sec["seq_1"][0, -1, 2 * len(ALL_METRICS)], 0.0)


class PredictoresTests(SimpleTestCase):

  def setUp(self):
    clasif = partidos_df([["A", "B", 2, 0, 1934], ["B", "C", 1, 1, 1934]])
    partidos = partidos_df([["A", "C", 1, 0, 1934]])
    self.ds = construir_dataset(partidos, clasif, PROXY_VACIO, agnos=[1934])

  def test_medias(self):
    predictor = PredictorMedias.entrenar(self.ds)
    salida = predictor.predecir(Lote(self.ds))
    # medias de (2, 1, 1) y (0, 1, 0)
    self.assertEqual(salida.shape, (3, 2))
    self.assertAlmostEqual(salida[0, 0], 4 / 3)
    self.assertAlmostEqual(salida[0, 1], 1 / 3)

  def test_guardar_y_cargar(self):
    with tempfile.TemporaryDirectory() as carpeta:
      guardar_predictor(PredictorMedias(1.5, 1.0), carpeta, fecha="2026-10-15")
      predictor, esquema = cargar_predictor(carpeta)
      self.assertEqual(predictor.predecir(Lote(self.ds)).tolist()[0], [1.5, 1.0])
      self.assertEqual(esquema["fecha"], "2026-10-15")

  def test_rechaza_metricas_distintas(self):
    with tempfile.TemporaryDirectory() as carpeta:
      guardar_predictor(PredictorMedias(1.5, 1.0), carpeta)
      ruta = Path(carpeta) / "esquema.json"
      esquema = json.loads(ruta.read_text(encoding="utf-8"))
      esquema["metricas"] = esquema["metricas"][:-1]
      ruta.write_text(json.dumps(esquema), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "reentrenarlo"):
        cargar_predictor(carpeta)

  def test_lote_desde_entrada(self):
    # dos partidos con la forma de "calculo_metricas_0": [home, ALL_METRICS_0, score_0, away, ALL_METRICS_1, score_1]
    lado = lambda equipo, base: [equipo, *[base + j for j in range(len(ALL_METRICS))], 9]
    X = [lado(3, 0) + lado(7, 100), lado(4, 0) + lado(8, 100)]
    self.assertEqual(len(X[0]), 2 * COLS_MODELO_POR_EQUIPO)
    tabla = lote_desde_entrada(X).tabla
    self.assertEqual(tabla["home_id"].tolist(), [3, 4])
    self.assertEqual(tabla["away_id"].tolist(), [7, 8])
    self.assertEqual(tabla[f"{ALL_METRICS[1]}_1"].tolist(), [101, 101])
    self.assertNotIn("score_0", tabla.columns)
    # la app la puede pasar tal cual a un predictor
    self.assertEqual(PredictorMedias(1.5, 1.0).predecir(lote_desde_entrada(X)).shape, (2, 2))


class ValorNeutroTests(SimpleTestCase):
  """A-01: apagar una metrica no puede romper la prediccion; el hueco se rellena con la media del entrenamiento"""

  def features(self):
    return {m: pd.Series([1.0, 2.0]) for m in ALL_METRICS}

  def test_todas_las_metricas_tienen_valor_neutro(self):
    from .modulo import VALOR_NEUTRO
    self.assertEqual(list(VALOR_NEUTRO), ALL_METRICS)
    self.assertTrue(all(isinstance(VALOR_NEUTRO[m], float) for m in ALL_METRICS))

  def test_validar_no_falla_con_metricas_apagadas(self):
    from .modulo import validar_valores_neutros
    # todas las base descartadas y todas las opcionales rechazadas: antes lanzaba ValueError por VALOR_NEUTRO=None
    validar_valores_neutros(BASE_METRICS, [])

  def test_neutralizar_solo_lo_apagado(self):
    from .modulo import VALOR_NEUTRO, neutralizar
    salida = neutralizar(self.features(), ["GF"], OPTIONAL_METRICS)
    self.assertEqual(salida["GF"].tolist(), [VALOR_NEUTRO["GF"]] * 2)
    self.assertEqual(salida["GC"].tolist(), [1.0, 2.0])          # no se toca
    self.assertEqual(salida["PTS"].tolist(), [1.0, 2.0])

  def test_neutralizar_opcionales_no_aceptadas(self):
    from .modulo import VALOR_NEUTRO, neutralizar
    salida = neutralizar(self.features(), [], ["gkps"])
    self.assertEqual(salida["gkps"].tolist(), [1.0, 2.0])        # aceptada: se conserva
    for m in ("mds", "mos", "mms", "rate_GC", "rate_GF"):
      self.assertEqual(salida[m].tolist(), [VALOR_NEUTRO[m]] * 2)


class CicloInternacionalesTests(SimpleTestCase):

  def test_ciclo_por_fechas(self):
    internacionales = pd.DataFrame([
      ["1930-07-13", 1930, "A", "B", 1, 0, "mundial", 1],    # mundial 1930: marca el inicio del ciclo de 1934
      ["1931-05-01", 1931, "A", "C", 3, 0, "amistoso", 0],
      ["1933-06-01", 1933, "B", "C", 2, 2, "eliminatoria_mundial", 0],
      ["1934-05-27", 1934, "A", "B", 2, 1, "mundial", 0],    # anfitrion: no neutral
      ["1934-08-01", 1934, "A", "C", 9, 0, "amistoso", 0],   # despues del mundial: no entra en su ciclo
    ], columns=["fecha", "agno", "home", "away", "score_0", "score_1", "categoria", "neutral"])
    partidos = partidos_df([["A", "B", 2, 1, 1934]])
    ds = construir_dataset(partidos, partidos_df([]), PROXY_VACIO, agnos=[1930, 1934],
                           internacionales=internacionales, ciclo=CICLO_INTERNACIONALES)
    ciclo_1934 = ds[(ds["agno"] == 1934) & (ds["fase"] == "eliminatoria")]
    self.assertEqual(ciclo_1934["categoria"].tolist(), ["amistoso", "eliminatoria_mundial"])
    mundial_1934 = ds[(ds["agno"] == 1934) & (ds["fase"] == "mundial")].iloc[0]
    self.assertEqual(mundial_1934["neutral"], 0)
    # A llega al mundial con su amistoso ganado 3-0
    self.assertEqual(mundial_1934["GF_0"], 3)

  def test_marca_de_datos_reales(self):
    # A tiene penales reales en 1930: en 1934 su historico es real; B solo tiene datos imputados
    proxy = pd.DataFrame([["A", 1930, 1.0, 5.0, 1.0, 0.0, 1, 0], ["B", 1930, 0.0, 2.0, 0.0, 0.0, 0, 0]],
                         columns=["pais", "agno", *METRICAS_ABSTRACTAS, "pk_real", "sot_real"])
    ds = construir_dataset(partidos_df([["A", "B", 1, 0, 1934]]), partidos_df([]), proxy, agnos=[1934])
    fila = ds.iloc[0]
    self.assertEqual((fila["pk_real_0"], fila["pk_real_1"], fila["sot_real_0"]), (1, 0, 0))


class EvaluacionTests(SimpleTestCase):

  def test_metricas_deterministas(self):
    goles = [[2, 1], [1, 1], [0, 3]]
    r = metricas(goles, [[1, 0], [1, 1], [1, 0]], determinista=True)
    # acierta el ganador del 1ro y el empate del 2do; marcador exacto solo el 1-1
    self.assertAlmostEqual(r["acierto_resultado"], 200 / 3)
    self.assertAlmostEqual(r["marcador_exacto"], 100 / 3)
    self.assertAlmostEqual(r["mae_goles"], (1 + 1 + 0 + 0 + 1 + 3) / 6)

  def test_resultado_mas_probable(self):
    # 2,5 contra 0,3 goles esperados: lo mas probable es que gane el local
    self.assertEqual(metricas([[3, 0]], [[2.5, 0.3]])["acierto_resultado"], 100.0)

  def test_metricas_probabilidad_a_mano(self):
    # partido 1: gana el local con probabilidades (0,5 0,3 0,2); partido 2: empate con (0,2 0,5 0,3)
    r = metricas_probabilidad([1, 0], [[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])
    self.assertAlmostEqual(r["log_loss"], -np.log(0.5))
    # Brier: (0,5^2 + 0,3^2 + 0,2^2) = 0,38 en los dos partidos
    self.assertAlmostEqual(r["brier"], 0.38)
    # RPS: acumuladas (0,5 0,8) contra (1 1) -> (0,25 + 0,04) / 2 = 0,145; (0,2 0,7) contra (0 1) -> (0,04 + 0,09) / 2 = 0,065
    self.assertAlmostEqual(r["rps"], (0.145 + 0.065) / 2)

  def test_rps_respeta_el_orden(self):
    # gana el local: equivocarse por empate cuesta menos que por visitante; log-loss y Brier no lo distinguen
    por_empate = metricas_probabilidad([1], [[0.2, 0.6, 0.2]])
    por_visitante = metricas_probabilidad([1], [[0.2, 0.2, 0.6]])
    self.assertLess(por_empate["rps"], por_visitante["rps"])
    self.assertAlmostEqual(por_empate["log_loss"], por_visitante["log_loss"])
    self.assertAlmostEqual(por_empate["brier"], por_visitante["brier"])

  def test_deterministas_suavizadas(self):
    np.testing.assert_allclose(probabilidades_deterministas([1, -1], epsilon=0.3), [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
    # siempre 1-1 en un 2-1: log-loss finito, -log(epsilon / 3)
    r = metricas([[2, 1]], [[1, 1]], determinista=True)
    self.assertAlmostEqual(r["log_loss"], -np.log(0.05 / 3))

  def test_tabla_calibracion(self):
    tabla = tabla_calibracion([1, 1], [[0.75, 0.15, 0.10], [0.75, 0.05, 0.20]])
    self.assertEqual(len(tabla), 10)
    self.assertEqual(sum(t["n"] for t in tabla), 6)
    self.assertEqual((tabla[7]["n"], tabla[7]["probabilidad"], tabla[7]["frecuencia"]), (2, 0.75, 1.0))
    # 0,10 y 0,15 caen en el tramo [0,1 0,2): ninguno de los dos paso
    self.assertEqual((tabla[1]["n"], tabla[1]["frecuencia"]), (2, 0.0))
    self.assertIsNone(tabla[9]["probabilidad"])

  def test_ece_a_mano(self):
    # un partido que gana el local con (0,5 0,3 0,2): cada probabilidad cae en un tramo distinto y solo la primera paso
    # ECE = (|0,5-1| + |0,3-0| + |0,2-0|) / 3 = 1/3
    self.assertAlmostEqual(ece([1], [[0.5, 0.3, 0.2]]), 1 / 3)
    # probabilidades perfectas (1 al resultado que paso): ECE = 0
    self.assertAlmostEqual(ece([1, 0, -1], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 0.0)
    # y "metricas" lo incluye para que viaje en el JSON de evaluacion
    self.assertIn("ece", metricas([[1, 0]], [[1.2, 0.8]]))

  def test_metricas_es_la_media_por_partido(self):
    goles, lambdas = [[2, 1], [0, 0], [1, 3]], [[1.5, 0.8], [1.0, 1.2], [0.9, 1.6]]
    r = metricas(goles, lambdas)
    pp, _, _ = metricas_por_partido(goles, lambdas)
    for m, v in pp.items():
      self.assertAlmostEqual(r[m], v.mean())

  def test_intervalos_bootstrap(self):
    pp = {"log_loss": np.random.default_rng(0).exponential(1, 300)}
    desde, hasta = intervalos_bootstrap(pp)["log_loss"]
    self.assertLess(desde, pp["log_loss"].mean())
    self.assertGreater(hasta, pp["log_loss"].mean())
    # sin variacion el intervalo es un punto
    np.testing.assert_allclose(intervalos_bootstrap({"rps": np.full(50, 0.2)})["rps"], [0.2, 0.2])

  def test_comparar_pareado(self):
    a = {"log_loss": np.array([0.5, 0.6, 0.7, 0.8]), "acierto_resultado": np.array([100, 100, 100, 100.0])}
    igual = comparar(a, a)
    self.assertEqual((igual["diferencia"], igual["p_dm"], igual["p_mcnemar"]), (0.0, 1.0, 1.0))
    # b pierde 0,1 en cada partido y falla los 4 que a acierta: McNemar exacta 2 * 0,5^4 = 0,125
    b = {"log_loss": a["log_loss"] + 0.1, "acierto_resultado": np.zeros(4)}
    c = comparar(a, b)
    self.assertAlmostEqual(c["diferencia"], -0.1)
    self.assertEqual((c["solo_a"], c["solo_b"]), (4, 0))
    self.assertAlmostEqual(c["p_mcnemar"], 0.125)
    self.assertEqual(c["p_bootstrap"], 0.0)

  def test_particion_temporal(self):
    ds = pd.DataFrame({"agno": [2010, 2014, 2014, 2018], "fase": ["mundial", "eliminatoria", "mundial", "mundial"]})
    tr, va, pr = particion(ds, hasta=2010, validacion=2014, prueba=[2018])
    self.assertEqual((tr.tolist(), va.tolist(), pr.tolist()), ([True, False, False, False], [False, True, True, False], [False, False, False, True]))

  def test_lineas_base(self):
    tabla = pd.DataFrame({"PTS_0": [6, 1], "PJ_0": [2, 1], "PTS_1": [3, 3], "PJ_1": [2, 1]})
    self.assertEqual(PredictorMasPuntos().predecir(Lote(tabla)).tolist(), [[1, 0], [0, 1]])
    self.assertEqual(PredictorEmpate().predecir(Lote(tabla)).tolist(), [[1, 1], [1, 1]])


class PredictorEloTests(SimpleTestCase):

  def tabla(self, n=600, semilla=0):
    # partidos inventados donde el de mas Elo gana mas seguido y hace mas goles
    rng = np.random.default_rng(semilla)
    dif = rng.normal(0, 200, n)
    goles_0, goles_1 = rng.poisson(np.exp(0.2 + dif / 400)), rng.poisson(np.exp(0.2 - dif / 400))
    return pd.DataFrame({"elo_0": 1500 + dif, "elo_1": 1500.0, "local_0": 0, "local_1": 0, "goles_0": goles_0, "goles_1": goles_1})

  def test_aprende_la_direccion_del_elo(self):
    p = PredictorElo.entrenar(self.tabla())
    self.assertGreater(p.coef["pendiente"], 0)
    # la Poisson inventada tiene 0,25 por centena de Elo
    self.assertAlmostEqual(p.coef["goles_pendiente"], 0.25, delta=0.05)
    probs = p.probabilidades(Lote(pd.DataFrame({"elo_0": [1800.0, 1500.0], "elo_1": [1500.0, 1800.0], "local_0": 0, "local_1": 0})))
    np.testing.assert_allclose(probs.sum(axis=1), 1)
    self.assertGreater(probs[0, 0], probs[0, 2])
    self.assertGreater(probs[1, 2], probs[1, 0])

  def test_ventaja_de_local(self):
    p = PredictorElo.entrenar(self.tabla())
    neutral, local = (Lote(pd.DataFrame({"elo_0": [1500.0], "elo_1": [1500.0], "local_0": [l], "local_1": [0]})) for l in (0, 1))
    self.assertGreater(p.probabilidades(local)[0, 0], p.probabilidades(neutral)[0, 0])
    self.assertGreater(p.predecir(local)[0, 0], p.predecir(neutral)[0, 0])

  def test_guardar_y_cargar(self):
    p = PredictorElo.entrenar(self.tabla())
    with tempfile.TemporaryDirectory() as carpeta:
      guardar_predictor(p, carpeta)
      cargado, _ = cargar_predictor(carpeta)
    lote = Lote(self.tabla(5, semilla=1))
    np.testing.assert_allclose(cargado.probabilidades(lote), p.probabilidades(lote))
    np.testing.assert_allclose(cargado.predecir(lote), p.predecir(lote))


class PredictorKerasTests(SimpleTestCase):

  def test_entrenar_guardar_y_cargar(self):
    clasif = partidos_df([["A", "B", 2, 0, 1934], ["B", "C", 1, 1, 1934], ["C", "A", 0, 1, 1934]] * 4)
    ds = construir_dataset(partidos_df([["A", "C", 1, 0, 1934]]), clasif, PROXY_VACIO, agnos=[1934])
    sec = construir_secuencias(ds, k=2)
    filas = np.ones(len(ds), dtype=bool)
    predictor, _ = PredictorKeras.entrenar(ds, sec, filas, filas, 2, True, True, epocas=2)
    salida = predictor.predecir(Lote(ds, sec))
    self.assertEqual(salida.shape, (len(ds), 2))
    self.assertTrue((salida > 0).all())
    with tempfile.TemporaryDirectory() as carpeta:
      guardar_predictor(predictor, carpeta)
      cargado, esquema = cargar_predictor(carpeta)
      self.assertEqual(esquema["config"]["k"], 2)
      np.testing.assert_allclose(cargado.predecir(Lote(ds, sec)), salida, rtol=1e-5)

  def test_anfitrion_listado_de_visitante(self):
    internacionales = pd.DataFrame([["1934-05-27", 1934, "B", "A", 2, 1, "mundial", 0]],
                                   columns=["fecha", "agno", "home", "away", "score_0", "score_1", "categoria", "neutral"])
    ds = construir_dataset(partidos_df([["A", "B", 1, 2, 1934]]), partidos_df([]), PROXY_VACIO, agnos=[1934], internacionales=internacionales)
    fila = ds.iloc[0]
    # B es el anfitrion aunque la tabla de mundiales lo liste de visitante
    self.assertEqual((fila["neutral"], fila["local_0"], fila["local_1"]), (0, 0, 1))
    sec = construir_secuencias(ds, k=1, con_rival=False)
    self.assertEqual(sec["seq_1"][0, -1, len(ALL_METRICS)], 1.0)
    self.assertEqual(sec["seq_0"][0, -1, len(ALL_METRICS)], 0.0)


class PuntajesPosicionTests(SimpleTestCase):

  def serie(self, *valores):
    return pd.Series(valores, dtype=float)

  def test_historico_dividido_por_sus_propios_partidos(self):
    from .modulo import calc_mms
    # mismo ciclo (6 de 6 puntos) y misma tasa historica de tiros (5 por partido) con historicos de largo distinto:
    # antes el equipo con mas mundiales sacaba mas; ahora los dos dan 100
    mms = calc_mms(self.serie(2, 2), self.serie(6, 6), self.serie(2, 2), self.serie(10, 100), self.serie(2, 20))
    self.assertEqual(mms.tolist(), [100.0, 100.0])

  def test_rango_y_sin_partidos(self):
    from .modulo import calc_gkps, calc_mds, calc_mos
    PJ, goles, PJ_hist = self.serie(0, 1, 2), self.serie(0, 9, 0), self.serie(0, 3, 3)
    for puntaje in (calc_gkps(PJ, goles, self.serie(0, 0, 3), self.serie(0, 0, 0), PJ_hist),
                    calc_mds(PJ, goles, self.serie(0, 0, 3), self.serie(0, 0, 0), PJ_hist),
                    calc_mos(PJ, goles, self.serie(0, 30, 0), self.serie(0, 0, 0), PJ_hist)):
      self.assertEqual(puntaje[0], 0)                       # sin partidos en el ciclo
      self.assertTrue(((puntaje >= 0) & (puntaje <= 100)).all())
    # 3 vallas invictas en 3 partidos de historico y 0 goles en contra: porteria perfecta
    self.assertEqual(calc_gkps(PJ, goles, self.serie(0, 0, 3), self.serie(0, 0, 0), PJ_hist)[2], 100.0)

  def test_version_de_metricas(self):
    from .modulo import VERSION_METRICAS
    with tempfile.TemporaryDirectory() as carpeta:
      ruta = Path(carpeta) / "esquema.json"
      ruta.write_text(json.dumps({"backend": "lstm", "metricas": list(ALL_METRICS), "config": {}}), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, f"version 1 de las metricas y el codigo usa la {VERSION_METRICAS}"):
        cargar_predictor(carpeta)


class DesempatesRealesTests(SimpleTestCase):

  def test_partido_repetido_recibe_cada_marcador(self):
    from .modulo import marcador_real, resultados_reales
    reales = resultados_reales(pd.DataFrame({"home": [7, 7], "away": [6, 6], "score_0": [1, 1], "score_1": [1, 0]}))
    # Italia - España 1934: el original 1:1 y la repeticion 1:0; con los equipos invertidos se gira
    self.assertEqual(marcador_real(reales, 7, 6), (1, 1))
    self.assertEqual(marcador_real(reales, 6, 7), (0, 1))

  def test_repeticion_decide_el_partido_original(self):
    from .modulo import func_prediccion_orden
    objs = func_prediccion_orden.__new__(func_prediccion_orden)
    objs.dic_matches = {f"Match {i}": {"ganador": None, "perdedor": None} for i in (12, 13)}
    objs.registrar_partido("Match 12", 6, 7, empate=True)    # 1:1: provisoriamente pasa el visitante
    objs.registrar_partido("Match 13", 7, 6, empate=False)   # la repeticion la gana Italia
    self.assertEqual(objs.dic_matches["Match 12"], {"ganador": 7, "perdedor": 6})


class ContextoTests(SimpleTestCase):

  def internacionales(self, filas):
    return pd.DataFrame(filas, columns=["fecha", "agno", "home", "away", "score_0", "score_1", "categoria", "neutral"])

  def test_elo(self):
    from .contexto import calcular_elo
    # amistoso en casa entre dos de 1500: esperado del local 1/(1+10^(-100/400)) = 0,640; K 20; gana el local
    antes_0, antes_1, despues_0, despues_1 = calcular_elo(self.internacionales([["2000-01-01", 2000, "A", "B", 1, 0, "amistoso", 0]]))
    self.assertEqual((antes_0[0], antes_1[0]), (1500, 1500))
    self.assertAlmostEqual(despues_0[0], 1500 + 20 * (1 - 1 / (1 + 10 ** (-100 / 400))))
    self.assertAlmostEqual(despues_0[0] + despues_1[0], 3000)

  def test_elo_anterior_al_partido(self):
    from .contexto import calcular_elo
    # el Elo con el que se predice el 2do partido es el de despues del 1ro, nunca el de despues del 2do
    antes_0, antes_1, despues_0, despues_1 = calcular_elo(self.internacionales([
      ["2000-01-01", 2000, "A", "B", 1, 0, "amistoso", 0],
      ["2000-02-01", 2000, "B", "A", 2, 2, "amistoso", 0],
    ]))
    self.assertEqual(antes_0[0], 1500)                       # el futuro no toca la primera fila
    # A es visitante en el 2do partido: su Elo previo es el que tenia tras el 1ro; B es local y lo mismo
    self.assertAlmostEqual(antes_1[1], despues_0[0])
    self.assertAlmostEqual(antes_0[1], despues_1[0])

  def test_contexto_del_mundial(self):
    from .contexto import contexto_de_mundiales
    filas = [["1999-05-01", 1999, "A", "B", 3, 0, "amistoso", 0],
             ["2000-06-01", 2000, "A", "B", 0, 1, "mundial", 0],     # A es el anfitrion (local, no neutral)
             ["2000-06-05", 2000, "B", "C", 1, 1, "mundial", 1]]
    ctx = contexto_de_mundiales(self.internacionales(filas))[2000]
    self.assertEqual(ctx["anfitriones"], {"A"})
    # Elo al inicio del mundial: despues del amistoso de 1999, antes de los partidos del mundial
    self.assertGreater(ctx["elo"]["A"], 1500)
    self.assertNotIn("C", ctx["elo"])

  def test_umbral_de_empate_y_marcador(self):
    from .evaluacion import resultado_y_marcador
    resultado, marcador, _ = resultado_y_marcador(np.array([[1.8, 1.1], [1.2, 1.15], [0.6, 2.4]]))
    self.assertEqual(resultado.tolist(), [1, 1, -1])
    self.assertEqual(marcador.tolist(), [[1, 0], [1, 0], [0, 2]])
    # con umbral 0,1 el partido parejo pasa a empate, con el marcador de empate mas probable
    resultado, marcador, _ = resultado_y_marcador(np.array([[1.2, 1.15]]), umbral_empate=0.1)
    self.assertEqual((resultado.tolist(), marcador.tolist()), ([0], [[1, 1]]))


class FormatoTorneoTests(SimpleTestCase):

  TABLAS = {f"Grupo {g}": pd.DataFrame({"pais": [f"{g}{i}" for i in range(1, 5)], "Pts": [6, 4, p3, 0]})
            for g, p3 in zip("ABCDEF", [3, 1, 3, 2, 3, 1])}

  def test_llaves_1986_1994_no_cambian(self):
    from .modulo import alinear_llave_terceros
    # resultado guardado antes de pasar los datos a FormatoTorneo (los 4 mejores terceros: A, C, E, D)
    esperado = {
      1986: [("Winner Grupo A", "Third-p Grupo C"), ("Winner Grupo F", "Runners-up Grupo E"), ("Runners-up Grupo A", "Runners-up Grupo C"),
             ("Winner Grupo B", "Third-p Grupo D"), ("Winner Grupo C", "Third-p Grupo A"), ("Runners-up Grupo F", "Runners-up Grupo B"),
             ("Winner Grupo E", "Runners-up Grupo D"), ("Winner Grupo D", "Third-p Grupo E")],
      1990: [("Winner Grupo A", "Third-p Grupo C"), ("Winner Grupo E", "Runners-up Grupo D"), ("Runners-up Grupo F", "Runners-up Grupo B"),
             ("Winner Grupo B", "Third-p Grupo D"), ("Runners-up Grupo A", "Runners-up Grupo C"), ("Winner Grupo C", "Third-p Grupo A"),
             ("Winner Grupo D", "Third-p Grupo E"), ("Winner Grupo F", "Runners-up Grupo E")],
      1994: [("Runners-up Grupo A", "Runners-up Grupo C"), ("Winner Grupo A", "Third-p Grupo C"), ("Winner Grupo B", "Third-p Grupo D"),
             ("Winner Grupo F", "Runners-up Grupo E"), ("Winner Grupo C", "Third-p Grupo A"), ("Winner Grupo E", "Runners-up Grupo D"),
             ("Winner Grupo D", "Third-p Grupo E"), ("Runners-up Grupo F", "Runners-up Grupo B")],
    }
    for agno, partidos in esperado.items():
      llave = alinear_llave_terceros(self.TABLAS, agno)
      self.assertEqual(list(zip(llave.home, llave.away)), partidos)

  def test_2026_reglamento_oficial(self):
    from .modulo import FORMATOS, ReglamentoTerceros, alinear_llave_terceros
    formato = FORMATOS[2026]
    self.assertEqual((len(formato.llave), len(formato.cruces), formato.n_terceros), (16, 495, 8))
    ReglamentoTerceros.oficial(2026)   # se valida al crearse
    # pasan los terceros de B, D, E, F, I, J, K y L (como en 2026): el 1E enfrenta al 3D, el 1A al 3E
    tablas = {f"Grupo {g}": pd.DataFrame({"pais": [f"{g}{i}" for i in range(1, 5)], "Pts": [7, 5, 3 if g in "BDEFIJKL" else 2, 0]})
              for g in "ABCDEFGHIJKL"}
    llave = alinear_llave_terceros(tablas, 2026)
    cruces = dict(zip(llave.home, llave.away))
    self.assertEqual(len(llave), 16)
    self.assertEqual((cruces["Winner Grupo E"], cruces["Winner Grupo A"]), ("Third-p Grupo D", "Third-p Grupo E"))

  def test_reglamento_2026_invalido(self):
    from .modulo import ReglamentoInvalido, ReglamentoTerceros
    oficial = ReglamentoTerceros.oficial(2026).a_json()
    oficial["llave"] = oficial["llave"][:-1]
    with self.assertRaisesRegex(ReglamentoInvalido, "16 partidos"):
      ReglamentoTerceros.desde_json(2026, oficial)
