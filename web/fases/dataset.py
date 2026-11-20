"""
Constructor del dataset de entrenamiento.

Genera, con el mismo codigo que usa la app para simular (calculo_metricas_0 y calculo_metricas_1), una fila por partido
real con las metricas de los dos equipos ANTES del partido y los goles que hicieron. Encima de esa tabla arma las
secuencias por equipo (sus ultimos K partidos) que consume un modelo de series temporales.

Ningun partido ve su propio resultado ni datos de mundiales posteriores:
  - las metricas base se acumulan partido a partido y cada fila guarda el estado previo al partido
  - el historico de metricas abstractas de un mundial solo suma mundiales anteriores ("historico_abstractas")
"""
import numpy as np
import pandas as pd

from . import modulo
from .contexto import DESCANSO_MAXIMO, ELO_INICIAL, calcular_elo, contexto_de_mundiales, descansos
from .modulo import BASE_METRICS, ALL_METRICS, calculo_metricas_0, calculo_metricas_1, historico_abstractas

FASE_ELIMINATORIA = "eliminatoria"
FASE_MUNDIAL = "mundial"

# ciclo previo a cada mundial: solo sus eliminatorias (lo que usa la app hoy) o todos los partidos de selecciones
CICLO_ELIMINATORIAS = "eliminatorias"
CICLO_INTERNACIONALES = "internacionales"

# tipo de partido; mismo orden que etl/busca_internacionales.py (el modelo la recibe como one-hot)
CATEGORIAS = ["mundial", "eliminatoria_mundial", "continental", "eliminatoria_continental", "liga_naciones", "amistoso", "regional"]
# contexto por paso en "construir_secuencias": es_local, neutral, one-hot de categoria, pk_real, sot_real
N_CONTEXTO = 2 + len(CATEGORIAS) + 2

# contexto adicional opcional (lo agrega "agregar_contexto"); "{i}" es el lado (0 home, 1 away)
EXTRAS = {
  "elo": ["elo_{i}"],                                  # Elo antes del partido (al inicio del torneo en un mundial)
  "fase": ["fase_final"],                              # 1 despues de la primera fase del mundial
  "continental": ["local_continental_{i}"],            # juega en su propia confederacion
  "descanso": ["descanso_{i}"],                        # dias desde su partido anterior
  "grupo": ["pts_torneo_{i}", "pj_torneo_{i}"],        # puntos y partidos que lleva en este mundial
  "ranking": ["ranking_{i}"],                          # posicion FIFA mas reciente previa al torneo (200 si falta)
}
# los que la simulacion de la app sabe calcular igual que el entrenamiento ("contexto_simulacion")
EXTRAS_EN_APP = ("elo", "fase", "continental", "ranking")


def ranking_por_agno():
  """{(agno, pais): posición FIFA más reciente con fecha <= 1 de julio de ese año}. 200 si no hay dato.

  Lee el modelo ``rankingfifa``; si la tabla no existe (tests sin BD) devuelve {} y el cálculo
  usa el valor neutro 200."""
  try:
    from .models import rankingfifa
    filas = list(rankingfifa.objects.order_by("fecha").values("fecha", "pais", "posicion"))
  except Exception:
    return {}
  if not filas:
    return {}
  df = pd.DataFrame(filas)
  salida = {}
  for agno in range(1992, 2027):
    vigente = df[df["fecha"] <= pd.Timestamp(year=agno, month=7, day=1)]
    for pais, g in vigente.groupby("pais"):
      salida[(agno, pais)] = int(g.iloc[-1]["posicion"])
  return salida


def mapa_paises(*dfs):
  """
  ¿QUE HACE?
  Asigna a cada pais un id estable: el orden alfabetico de todos los nombres que aparecen como home o away en (dfs). El mismo mapa debe guardarse junto al modelo entrenado, para que el id de un pais signifique lo mismo al predecir

  Returns:

    - diccionario nombre del pais -> id
  """
  nombres = sorted(set().union(*(set(df["home"]) | set(df["away"]) for df in dfs)))
  return {nombre: i for i, nombre in enumerate(nombres)}


def separar_score(df):
  """Pasa la columna "score" ("3:2") de la tabla partidos a enteros score_0 y score_1"""
  goles = df["score"].str.split(":", expand=True).astype("int64")
  return df.drop(columns="score").assign(score_0=goles[0], score_1=goles[1])


def cargar_fuentes():
  """
  ¿QUE HACE?
  Lee de la BD las tablas que necesita "construir_dataset": partidos de mundiales, partidos de eliminatorias, metricas abstractas por pais y mundial, y todos los partidos internacionales

  Returns:

    - (partidos, clasificaciones, proxy, internacionales, fixtures): df con home, away, score_0, score_1, agno (los dos primeros), df de "formar_dataset_real('proxy_desempegno')", df de "formar_dataset_real('internacionales')" (vacio si no se cargo) y df con agno, home, away del fixture de cada mundial
  """
  partidos = modulo.formar_dataset_real("partidos")
  # el marcador es el de los 90 o 120 minutos (la tanda queda aparte); los partidos no jugados (jugado = 0) no entran
  partidos = separar_score(partidos[partidos["jugado"] == 1].drop(columns=["penales", "jugado"]))
  clasificaciones = modulo.formar_dataset_real("clasificaciones")
  proxy = modulo.formar_dataset_real("proxy_desempegno")
  internacionales = modulo.formar_dataset_real("internacionales")
  fixtures = pd.DataFrame(list(modulo.fixtures.objects.values("agno", "home", "away"))).astype({"agno": "int64"})
  return partidos, clasificaciones, proxy, internacionales, fixtures


def partidos_primera_fase(fixtures):
  """Partidos de la primera fase de cada mundial: las filas del fixture con los dos equipos ya definidos (como "create_features")"""
  definidos = ~fixtures["away"].str.contains("Match|Grupo")
  return fixtures[definidos].groupby("agno").size().to_dict()


def agregar_contexto(ds, internacionales, fixtures=None):
  """
  ¿QUE HACE?
  Agrega a (ds) (salida de "construir_dataset") las columnas de EXTRAS, con los mismos datos que usara la simulacion de la app

  ¿COMO LO HACE?
  0-) Cada fila que no es de mundial se ubica en (internacionales) por (agno, home, away, goles); de ahi su fecha y su Elo antes del partido. Si no se encuentra, Elo del inicio del mundial anterior (sin mirar ese partido) y descanso maximo

  1-) En los partidos de mundial el Elo es el del inicio del torneo (la app no conoce los resultados simulados de antes), la fecha sale del mismo partido en (internacionales) y la fase final empieza al terminar los partidos de la primera fase del fixture

  2-) local_continental: en un mundial, si la confederacion del equipo es la del anfitrion; fuera de los mundiales, si los dos equipos son de la misma confederacion o el equipo juega en su cancha

  3-) pts_torneo / pj_torneo: lo que lleva el equipo en ese mundial antes del partido (2 puntos por victoria hasta 1990)
  """
  ds = ds.copy()
  mundiales = contexto_de_mundiales(internacionales)
  elo_0, elo_1, _, _ = calcular_elo(internacionales)
  descanso = descansos(internacionales)
  primera_fase = partidos_primera_fase(fixtures) if fixtures is not None else {}

  no_mundial = (internacionales["categoria"] != "mundial").to_numpy()
  por_partido = {}
  for fecha, a, h, w, g0, g1, e0, e1 in zip(internacionales["fecha"].to_numpy()[no_mundial], internacionales["agno"].to_numpy()[no_mundial],
                                            internacionales["home"].to_numpy()[no_mundial], internacionales["away"].to_numpy()[no_mundial],
                                            internacionales["score_0"].to_numpy()[no_mundial], internacionales["score_1"].to_numpy()[no_mundial],
                                            elo_0[no_mundial], elo_1[no_mundial]):
    por_partido.setdefault((a, h, w, g0, g1), []).append((fecha, e0, e1))
  fecha_mundial = {(r.agno, r.home, r.away): r.fecha for r in internacionales[~no_mundial].itertuples()}
  agnos = sorted(mundiales)

  columnas = {c.format(i=i): [] for cs in EXTRAS.values() for c in cs for i in (0, 1)}
  torneo = {}  # (agno, pais) -> [puntos, partidos] en el mundial
  ranking = ranking_por_agno()

  for r in ds.itertuples(index=False):
    ctx = mundiales.get(r.agno, {})
    confed = ctx.get("confederacion", {})
    if r.fase == FASE_MUNDIAL:
      fecha = fecha_mundial.get((r.agno, r.home, r.away), fecha_mundial.get((r.agno, r.away, r.home)))
      elos = [ctx.get("elo", {}).get(p, ELO_INICIAL) for p in (r.home, r.away)]
      continental = [int(confed.get(p) in ctx.get("confederacion_anfitrion", set())) for p in (r.home, r.away)]
      fase_final = int(r.orden >= primera_fase.get(r.agno, np.inf))
      acumulado = [torneo.setdefault((r.agno, p), [0, 0]) for p in (r.home, r.away)]
      grupo = [list(x) for x in acumulado]
      victoria = 2 if r.agno < 1994 else 3
      puntos = (victoria, 0) if r.goles_0 > r.goles_1 else (0, victoria) if r.goles_1 > r.goles_0 else (1, 1)
      for x, pts in zip(acumulado, puntos):
        x[0] += pts
        x[1] += 1
    else:
      encontrados = por_partido.get((r.agno, r.home, r.away, r.goles_0, r.goles_1))
      if encontrados:
        fecha, e0, e1 = encontrados.pop(0) if len(encontrados) > 1 else encontrados[0]
        elos = [e0, e1]
      else:
        anterior = max((a for a in agnos if a < r.agno), default=None)
        fecha, elos = None, [mundiales.get(anterior, {}).get("elo", {}).get(p, ELO_INICIAL) for p in (r.home, r.away)]
      misma = confed.get(r.home) is not None and confed.get(r.home) == confed.get(r.away)
      continental = [int(misma or r.local_0), int(misma or r.local_1)]
      fase_final, grupo = 0, [[0, 0], [0, 0]]
    for i, p in enumerate((r.home, r.away)):
      columnas[f"elo_{i}"].append(elos[i])
      columnas[f"local_continental_{i}"].append(continental[i])
      columnas[f"descanso_{i}"].append(descanso.get((fecha, p), DESCANSO_MAXIMO) if fecha else DESCANSO_MAXIMO)
      columnas[f"pts_torneo_{i}"].append(grupo[i][0])
      columnas[f"pj_torneo_{i}"].append(grupo[i][1])
      columnas[f"ranking_{i}"].append(ranking.get((r.agno, p), 200))
    columnas["fase_final"].append(fase_final)
  return ds.assign(**columnas)


def _jugar(estado, home, away, goles_0, goles_1):
  """
  ¿QUE HACE?
  Procesa un partido real: devuelve las metricas de entrada al modelo de ambos equipos antes del partido y deja en (estado) sus metricas base despues del partido

  ¿COMO LO HACE?
  0-) Arma un df de fase de una fila con las metricas base previas de cada equipo (0 si es su primer partido del ciclo) y el resultado real

  1-) "calculo_metricas_0" deriva las ALL_METRICS de ambos lados con el estado previo (y el progreso abstracto en "modulo.opcionales")

  2-) "calculo_metricas_1" acumula el resultado: la fila pasa a tener el estado posterior y "modulo.opcionales" avanza el progreso abstracto

  3-) Guarda en (estado) el estado posterior de ambos equipos

  Args:

    - estado: diccionario id -> {metrica base: valor}; se actualiza in-place

    - home, away: ids de los equipos

    - goles_0, goles_1: goles reales de home y away

  Returns:

    - Serie con ALL_METRICS_0 y ALL_METRICS_1 previas al partido
  """
  ceros = {m: 0.0 for m in BASE_METRICS}
  previo_0, previo_1 = estado.get(home, ceros), estado.get(away, ceros)
  fila = pd.DataFrame([{
    "home": home, **{f"{m}_0": float(previo_0[m]) for m in BASE_METRICS}, "score_0": goles_0,
    "away": away, **{f"{m}_1": float(previo_1[m]) for m in BASE_METRICS}, "score_1": goles_1,
  }])

  entrada = calculo_metricas_0(fila).iloc[0]
  calculo_metricas_1(fila)

  estado[home] = {m: fila.at[0, f"{m}_0"] for m in BASE_METRICS}
  estado[away] = {m: fila.at[0, f"{m}_1"] for m in BASE_METRICS}
  return entrada[[f"{m}_{i}" for i in (0, 1) for m in ALL_METRICS]]


def construir_dataset(partidos, clasificaciones, proxy, pais_id=None, agnos=None, internacionales=None, ciclo=CICLO_ELIMINATORIAS,
                      fixtures=None):
  """
  ¿QUE HACE?
  Arma la tabla de entrenamiento: una fila por partido real (ciclo previo y mundial) en orden cronologico, con las metricas de los dos equipos antes del partido, sus goles, el tipo de partido y si se jugo en campo neutral. Reproduce el ciclo que simula la app: cada mundial arranca con lo acumulado en su ciclo previo

  ¿COMO LO HACE?
  0-) Si no se pasa (pais_id), lo arma con "mapa_paises" (incluye los equipos de (internacionales) si se usan para el ciclo)

  1-) Para cada mundial Y de (agnos), en orden:

    1.0) Ciclo previo: las metricas base arrancan en 0 para todos (como "obtener_dic_agnos") y el progreso abstracto desde "historico_abstractas" hasta Y. Cada partido se procesa con "_jugar"

      - (ciclo) "eliminatorias": las eliminatorias con agno en (mundial anterior, Y]

      - (ciclo) "internacionales": todos los partidos de (internacionales) que no son de mundial, con fecha desde el inicio del mundial anterior hasta antes del inicio de Y, en orden de fecha

    1.1) Se guarda el estado al cerrar el ciclo

    1.2) Mundial Y: el progreso abstracto vuelve al historico previo a Y (como "reiniciar_opcionales" en la app). Un equipo sin partidos en este ciclo (anfitrion, campeon vigente) arranca con su estado al cierre del ciclo mas reciente en que jugo, o en 0 (como "agregar_features"). (neutral) sale de (internacionales); si no esta, vale 1. local_0 / local_1 marcan al anfitrion, este listado de local o de visitante

  2-) Restaura "modulo.opcionales" al terminar, para no dejar estado global de entrenamiento en el proceso

  Args:

    - partidos: df de mundiales con home, away, score_0, score_1 y agno (en orden de juego dentro de cada año)

    - clasificaciones: df de eliminatorias con home, away, score_0, score_1 y agno (en orden de juego)

    - proxy: df con pais, agno y METRICAS_ABSTRACTAS (puede estar vacio)

    - pais_id: diccionario nombre del país -> id; None lo arma con todos los equipos de las fuentes usadas

    - agnos: mundiales a procesar; None usa "modulo.agnos" (1934 - 2026)

    - internacionales: df de "formar_dataset_real('internacionales')" (fecha, agno, home, away, score_0, score_1, categoria, neutral) o None

    - ciclo: CICLO_ELIMINATORIAS o CICLO_INTERNACIONALES (este exige (internacionales))

    - fixtures: df con agno, home, away del fixture de cada mundial (para saber donde termina la primera fase); con (internacionales) se agregan las columnas de EXTRAS ("agregar_contexto")

  Returns:

    - df con columnas agno, fase, orden, home, away, home_id, away_id, categoria, neutral, local_0/1 (1 si ese equipo juega en su cancha), pk_real_0/1 y sot_real_0/1 (1 si el historico de metricas abstractas del equipo incluye datos reales de fbref), ALL_METRICS_0, ALL_METRICS_1, goles_0, goles_1
  """
  hay_internacionales = internacionales is not None and len(internacionales) > 0
  if ciclo == CICLO_INTERNACIONALES and not hay_internacionales:
    raise ValueError("el ciclo 'internacionales' necesita la tabla internacionales (etl/busca_internacionales.py y load_data)")
  fuentes = [partidos, clasificaciones] + ([internacionales] if ciclo == CICLO_INTERNACIONALES else [])
  pais_id = pais_id or mapa_paises(*fuentes)
  agnos = sorted(agnos or modulo.agnos)
  opcionales_previos = modulo.opcionales
  filas, cierre_ciclo = [], {}

  if hay_internacionales:
    en_mundial = internacionales["categoria"] == "mundial"
    inicio_mundial = internacionales[en_mundial].groupby("agno")["fecha"].min().to_dict()
    neutral_mundial = {(r.agno, r.home, r.away): r.neutral for r in internacionales[en_mundial].itertuples()}
    resto = internacionales[~en_mundial]
  else:
    inicio_mundial, neutral_mundial = {}, {}

  # mundiales anteriores con penales / tiros a puerta reales de cada pais (columnas pk_real y sot_real de proxy)
  reales = {marca: proxy.loc[proxy[marca] == 1].groupby("pais")["agno"].apply(list).to_dict() if marca in proxy else {}
            for marca in ("pk_real", "sot_real")}

  def con_dato_real(marca, pais, agno):
    """1 si el historico de (pais) que usa el mundial (agno) incluye algun mundial anterior con (marca) real"""
    return int(any(a < agno for a in reales[marca].get(pais, [])))

  def sede_de(agno, home, away):
    """(neutral, local_0, local_1) de un partido de mundial: el anfitrion juega de local aunque la tabla lo liste de visitante"""
    if neutral_mundial.get((agno, home, away)) == 0:
      return 0, 1, 0
    if neutral_mundial.get((agno, away, home)) == 0:
      return 0, 0, 1
    return 1, 0, 0

  def procesar(df_partidos, agno, fase, estado):
    for orden, p in enumerate(df_partidos.itertuples(index=False)):
      home, away = pais_id[p.home], pais_id[p.away]
      entrada = _jugar(estado, home, away, int(p.score_0), int(p.score_1))
      filas.append({
        "agno": agno, "fase": fase, "orden": orden, "home": p.home, "away": p.away, "home_id": home, "away_id": away,
        "categoria": p.categoria, "neutral": int(p.neutral), "local_0": int(p.local_0), "local_1": int(p.local_1),
        **{f"{marca}_{i}": con_dato_real(marca, nombre, agno) for marca in ("pk_real", "sot_real") for i, nombre in ((0, p.home), (1, p.away))},
        **entrada.to_dict(), "goles_0": int(p.score_0), "goles_1": int(p.score_1),
      })

  try:
    for i, agno in enumerate(agnos):
      desde = agnos[i - 1] if i > 0 else -np.inf

      # 1.0) ciclo previo
      modulo.opcionales = historico_abstractas(proxy, pais_id, agno)
      estado = {}
      if ciclo == CICLO_INTERNACIONALES:
        fecha_desde = inicio_mundial[desde] if i > 0 else ""
        previos = resto[(resto["fecha"] >= fecha_desde) & (resto["fecha"] < inicio_mundial[agno])]
      else:
        previos = clasificaciones[(clasificaciones["agno"] > desde) & (clasificaciones["agno"] <= agno)]
        previos = previos.assign(categoria="eliminatoria_mundial", neutral=0)
      # fuera de los mundiales el local juega en su cancha salvo campo neutral
      previos = previos.assign(local_0=1 - previos["neutral"].astype(int), local_1=0)
      procesar(previos, agno, FASE_ELIMINATORIA, estado)

      # 1.1) estado al cerrar el ciclo
      cierre_ciclo[agno] = dict(estado)

      # 1.2) mundial
      modulo.opcionales = historico_abstractas(proxy, pais_id, agno)
      mundial = partidos[partidos["agno"] == agno]
      sedes = np.array([sede_de(agno, h, a) for h, a in zip(mundial["home"], mundial["away"])], dtype=int).reshape(-1, 3)
      mundial = mundial.assign(categoria="mundial", neutral=sedes[:, 0], local_0=sedes[:, 1], local_1=sedes[:, 2])
      estado_mundial = dict(estado)
      for nombre in set(mundial["home"]) | set(mundial["away"]):
        equipo = pais_id[nombre]
        if equipo not in estado_mundial:
          anteriores = [cierre_ciclo[a][equipo] for a in reversed(agnos[:i]) if equipo in cierre_ciclo[a]]
          if anteriores:
            estado_mundial[equipo] = anteriores[0]
      procesar(mundial, agno, FASE_MUNDIAL, estado_mundial)
  finally:
    # 2-) sin estado global residual
    modulo.opcionales = opcionales_previos

  salida = pd.DataFrame(filas)
  return agregar_contexto(salida, internacionales, fixtures) if hay_internacionales and len(salida) else salida


def construir_secuencias(dataset, k=5, con_rival=True, con_contexto=True, extras=()):
  """
  ¿QUE HACE?
  Convierte la tabla de "construir_dataset" en muestras de serie temporal: para cada partido, la historia de cada equipo como la secuencia de sus ultimos (k) partidos, terminando en el partido a predecir. Cada paso lleva las ALL_METRICS previas del equipo y, si (con_rival), tambien las de su rival en ese partido: los goles que hace un equipo dependen de a quien enfrenta. Si (con_contexto), cada paso suma el contexto del partido: si el equipo jugo en su cancha (local_0 / local_1), si fue en campo neutral, su categoria en one-hot (CATEGORIAS) y si los penales y tiros a puerta de su historico son reales (pk_real, sot_real) o imputados

  ¿COMO LO HACE?
  0-) Recorre la tabla en orden y guarda, por equipo, la lista de sus pasos: sus metricas (las de su lado: _0 si jugo de local, _1 si de visitante) seguidas de las del otro lado

  1-) Para cada partido, toma de cada equipo sus ultimos (k) pasos incluido el actual (que solo contiene informacion previa al partido); si tiene menos de (k) rellena al principio con ceros y lo marca en la mascara

  Args:

    - dataset: df de "construir_dataset" (orden cronologico)

    - k: largo de la secuencia (N_STEPS del modelo)

    - con_rival: True agrega a cada paso las ALL_METRICS del rival (2 * len(ALL_METRICS) por paso)

    - con_contexto: True agrega a cada paso N_CONTEXTO valores (es_local, neutral, one-hot de categoria, pk_real, sot_real)

    - extras: nombres de EXTRAS (elo, fase, continental, descanso, grupo) cuyas columnas se agregan al final de cada paso

  Returns:

    - diccionario con arrays: "id_0", "id_1" (n,), "seq_0", "seq_1" (n, k, len(ALL_METRICS), el doble con rival, mas N_CONTEXTO con contexto), "mascara_0", "mascara_1" (n, k) con 1 en pasos reales, "y" (n, 2) goles de home y away
  """
  cols_extra = {i: [c.format(i=i) for e in extras for c in EXTRAS[e]] for i in (0, 1)}
  n, f = len(dataset), len(ALL_METRICS) * (2 if con_rival else 1) + (N_CONTEXTO if con_contexto else 0) + len(cols_extra[0])
  salida = {
    "id_0": dataset["home_id"].to_numpy(), "id_1": dataset["away_id"].to_numpy(),
    "seq_0": np.zeros((n, k, f)), "seq_1": np.zeros((n, k, f)),
    "mascara_0": np.zeros((n, k)), "mascara_1": np.zeros((n, k)),
    "y": dataset[["goles_0", "goles_1"]].to_numpy(dtype=float),
  }
  historia = {}
  cols = {i: [f"{m}_{i}" for m in ALL_METRICS] for i in (0, 1)}
  matriz = {i: dataset[cols[i]].to_numpy(dtype=float) for i in (0, 1)}
  if con_contexto:
    neutral = dataset["neutral"].to_numpy(dtype=float)
    # tablas sin local_0/1 (anteriores a esa columna): el local juega en su cancha salvo campo neutral
    local = {i: dataset[f"local_{i}"].to_numpy(dtype=float) if f"local_{i}" in dataset else (1 - neutral) * (i == 0) for i in (0, 1)}
    categoria = np.array([[c == cat for cat in CATEGORIAS] for c in dataset["categoria"]], dtype=float)
    marcas = {i: dataset[[f"pk_real_{i}", f"sot_real_{i}"]].to_numpy(dtype=float) for i in (0, 1)}
  valores_extra = {i: dataset[cols_extra[i]].to_numpy(dtype=float) for i in (0, 1)}

  for fila, (id_0, id_1) in enumerate(zip(salida["id_0"], salida["id_1"])):
    for lado, equipo in ((0, id_0), (1, id_1)):
      pasos = historia.setdefault(equipo, [])
      paso = matriz[lado][fila]
      if con_rival:
        paso = np.concatenate([paso, matriz[1 - lado][fila]])
      if con_contexto:
        paso = np.concatenate([paso, [local[lado][fila], neutral[fila]], categoria[fila], marcas[lado][fila]])
      if extras:
        paso = np.concatenate([paso, valores_extra[lado][fila]])
      pasos.append(paso)
      ultimos = pasos[-k:]
      salida[f"seq_{lado}"][fila, k - len(ultimos):] = ultimos
      salida[f"mascara_{lado}"][fila, k - len(ultimos):] = 1

  return salida
