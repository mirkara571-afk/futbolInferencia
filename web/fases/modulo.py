import re
import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.stats import poisson
from decimal import Decimal
from dataclasses import dataclass
from functools import wraps
from itertools import combinations
from types import MappingProxyType
from django.conf import settings
from .models import partidos, fixtures, grupos, clasificaciones, internacionales, proxy_desempegno

from numpy.random import seed
seed(1)
tf.random.set_seed(2)

agnos = ["1934", "1938", "1950", "1954", "1958", "1962", "1966", "1970", "1974", "1978", "1982", "1986", "1990", "1994", "1998", "2002", "2006", "2010", "2014", "2018", "2026", "2022"]

# métricas que se acumulan fase a fase en los df de fase (funcion_tabla_desempegno); el usuario puede descartarlas
BASE_METRICS = ["PTS", "PJ", "PG", "PP", "PE", "GF", "GC", "D"]
# catálogo de métricas derivadas (calculo_metricas_0); el usuario acepta todas, algunas o ninguna
OPTIONAL_METRICS = ["gkps", "mds", "mos", "mms", "rate_GC", "rate_GF"]
# por defecto no se descarta ninguna base; cada petición trae las suyas como parámetro (nunca se escribe esta global)
DROP_METRICS = ()
# esquema fijo de entrada al modelo por equipo: no cambia con lo que elija el usuario
ALL_METRICS = BASE_METRICS + OPTIONAL_METRICS

# insumos internos de gkps, mds, mos y mms; viven en (opcionales) y el usuario no los elige
METRICAS_ABSTRACTAS = ["VI", "SOT", "PKATT", "PKATTALLOW"]
# version de las formulas de las metricas derivadas: un modelo entrenado con otra version recibiria valores en otra escala
# 2: gkps, mds, mos y mms dividen el historico abstracto por sus propios partidos (PJ del historico) y van de 0 a 100
VERSION_METRICAS = 2

# valor que recibe una métrica descartada o no aceptada: la media de esa columna en el set de entrenamiento.
# Calculado el 2026-10-29 sobre data/datasets/partidos_features.csv con agno <= 2014 (6.983 partidos, ambos lados):
# es el corte con el que se entrenan las pruebas 2018/2022, así el valor neutro no ve ningún año de prueba.
# Para recalcularlo, promedia <metrica>_0 y <metrica>_1 del dataset filtrado (ver docs/especificacion/02-plan-tecnico.md, A-01).
VALOR_NEUTRO = {
  "PTS": 8.232923, "PJ": 4.787126, "PG": 2.389589, "PP": 1.333381, "PE": 1.064156,
  "GF": 8.278963, "GC": 5.251826, "D": 3.027137,
  "gkps": 41.044991, "mds": 44.974489, "mos": 45.788291, "mms": 43.036899,
  "rate_GC": 10.437478, "rate_GF": 14.971177,
}

# progreso por país (id) de METRICAS_ABSTRACTAS; lo carga (reiniciar_opcionales) y lo actualiza (calculo_metricas_1)
opcionales = None

EMBEDDING_DIM = 3
COLS_FASE_POR_EQUIPO = 1 + len(BASE_METRICS) + 1    # df de fase: equipo + base + score
COLS_MODELO_POR_EQUIPO = 1 + len(ALL_METRICS) + 1   # entrada al modelo: equipo + base + derivadas + score
C_S_DELETE_INDEX = EMBEDDING_DIM + len(ALL_METRICS) # posición del score tras reemplazar el id por su embedding
C_S_Y_INDEX = C_S_DELETE_INDEX
N_STEPS = 1 # partidos por ventana; el .h5 actual se entreno con 1 (el reentrenamiento lo leera de su esquema)


def resultados_reales(df_partidos):
  """
  Diccionario (home, away) -> lista de (score_0, score_1) en orden de juego con los marcadores reales de (df_partidos) (columnas home, away, score_0 y score_1, con ids de pais). Es una lista porque en 1934 y 1938 un empate se repetia (Italia - España 1:1 y luego 1:0). Lo usa "intsc_prob_goles" (alterno=1) para jugar las fases anteriores a la objetivo
  """
  goles = df_partidos[["score_0", "score_1"]].apply(pd.to_numeric).to_numpy()
  salida = {}
  for h, a, (g0, g1) in zip(df_partidos["home"], df_partidos["away"], goles):
    salida.setdefault((h, a), []).append((g0, g1))
  return salida

def marcador_real(resultados, home, away):
  """
  Marcador real de (home) contra (away) en (resultados) ("resultados_reales"); si el partido figura con los equipos invertidos se devuelve girado. Si la pareja se enfrento mas de una vez (partido de desempate) se consume el primero que queda, asi el original y su repeticion reciben cada uno su marcador. Lanza ValueError si ese partido no se jugo en el mundial
  """
  def siguiente(lista):
    return lista.pop(0) if len(lista) > 1 else lista[0]
  if (home, away) in resultados:
    return siguiente(resultados[(home, away)])
  if (away, home) in resultados:
    g0, g1 = siguiente(resultados[(away, home)])
    return g1, g0
  raise ValueError(f"El partido {home} - {away} no figura entre los resultados reales del mundial")

def desempates_reales(df_partidos):
  """
  Diccionario (home, away) -> "home" o "away" con quien paso en los empates reales de (df_partidos) (columnas home, away, penales y jugado): el ganador de la tanda ("4:2") o, en un partido no jugado (jugado = 0), el local, que es como se guarda el que paso sin jugar
  """
  salida = {}
  for h, a, tanda, jugado in zip(df_partidos["home"], df_partidos["away"], df_partidos["penales"], df_partidos["jugado"]):
    if tanda:
      p0, p1 = (int(x) for x in tanda.split(":"))
      salida[(h, a)] = "home" if p0 > p1 else "away"
    elif int(jugado) == 0:
      salida[(h, a)] = "home"
  return salida

def gana_local_desempate(desempates, home, away):
  """True si en el empate real de (home) contra (away) paso (home), segun "desempates_reales" (tambien con los equipos invertidos)"""
  if (home, away) in desempates:
    return desempates[(home, away)] == "home"
  if (away, home) in desempates:
    return desempates[(away, home)] == "away"
  return False

# predictor cargado por carpeta: cargar un LSTM de Keras tarda segundos y no conviene repetirlo en cada consulta
_PREDICTORES = {}

def predictor_en_uso(carpeta):
  """Predictor de (carpeta) con "cargar_predictor", guardado en memoria; se vuelve a cargar si cambia su esquema.json"""
  from .predictores import ESQUEMA, cargar_predictor
  version = (carpeta / ESQUEMA).stat().st_mtime
  if _PREDICTORES.get(carpeta, (None,))[0] != version:
    _PREDICTORES[carpeta] = (version, cargar_predictor(carpeta)[0])
  return _PREDICTORES[carpeta][1]

def formar_dataset_real(ind0, valores=None): # PEND recomentar docstring
  """
  ¿QUE HACE?
  Busca en la base de datos segun la tabla indicada por "ind0" y segun el año presente en valores (no lo estará en el caso de ind0="clasificaciones").
  
  ¿COMO LO HACE?
  0-) Se tienen 4 bloques condicionales "if", "elif" para cada una de las tablas posibles indicadas por el parametro (ind0): "partidos", "fixtures", "grupos" y "clasificaciones"
    
    0.0-) Si (ind0) es "partidos": 
      
      0.0.0-) Se crea una lista vacia (df0)
      
      0.0.1-) Se itera sobre todos los objetos de la tabla "partidos" de la base de datos, accedida mediante el ORM de Django

      0.0.2-) Si el atributo "agno" del objeto iterado (des) es igual al parametro (valores) (se asume que es un año) y (valores) es diferente a None:

        0.0.2.0-) Se agrega a la lista (df0) una sublista con los atributos "home", "score", "away" y "agno" recuperados del objeto iterado (des)

      0.0.3-) Si el parametro (valores) es igual a None:

        0.0.3.0-) Se agrega a la lista (df0) una sublista con los atributos "home", "score", "away" y "agno" recuperados del objeto iterado (des). Esto ultimo deberia agrgar todas las filas de la tabla "partidos" a (df0)

      0.0.4-) Se crea un df (df) a partir un array con lista de sublistas basada en partidos del año indicado en la variable (valores) con columnas "home", "score", "away" y "agno"

      0.0.5-) Se convierten los tipos de las columnas del df (df) a los tipos adecuados: "O" para cadenas y "int64" para enteros
      0.0.6-) Se retorna el df (df)
    
    0.1-) Si (ind0) es "fixtures":
      0.1.0-) Se crea una lista vacia (df0)
      
      0.1.1-) Se itera sobre todos los objetos de la tabla "fixtures" de la base de datos, accedida mediante el ORM de Django

      0.1.2-) Si el atributo "agno" del objeto iterado (des) es igual al parametro (valores) (se asume que es un año):

        0.1.2.0-) Se agrega a la lista (df0) una sublista con los atributos "home", "score", "away" y "agno" recuperados del objeto iterado (des)

      0.1.3-) Se crea un df (df) a partir de la lista de sublistas (vacia si no hay filas) basada en fixtures del año indicado en la variable (valores) con columnas "home", "score", "away" y "year"

      0.1.4-) Se convierten los tipos de las columnas del df (df) a los tipos adecuados: "O" para cadenas y "int64" para enteros

      0.1.5-) Se retorna el df (df)
    
    0.2-) Si (ind0) es "grupos":
      0.2.0-) Se crea una lista vacia (df0)
      
      0.2.1-) Se itera sobre todos los objetos de la tabla "grupos" de la base de datos, accedida mediante el ORM de Django

      0.2.2-) Si el atributo "agno" del objeto iterado (des) es igual al parametro (valores) (se asume que es un año):
        0.2.2.0-) Se agrega a la lista (df0) una sublista con los atributos "pais", "Pts", "PJ", "PG", "PP", "PE", "GF", "GC", "Dif", "Grupo" y "agno" recuperados del objeto iterado (des)
      
      0.2.3-) Se crea un df (df) a partir de la lista de sublistas (vacia si no hay filas) basada en grupos del año indicado en la variable (valores) con columnas "pais", "Pts", "PJ", "PG", "PP", "PE", "GF", "GC", "Dif", "Grupo" y "agno"

      0.2.4-) Se convierten los tipos de las columnas del df (df) a los tipos adecuados: "O" para cadenas y "int64" para enteros

      0.2.5-) Se retorna el df (df)

    0.3-) Si (ind0) es "clasificaciones":

      0.3.0-) Se crea una lista vacia (df0)
      
      0.3.1-) Se itera sobre todos los objetos de la tabla "clasificaciones" de la base de datos, accedida mediante el ORM de Django

      0.3.2-) Se agrega a la lista (df0) una sublista con los atributos "home", "score_0", "score_1", "away", "tournament" y "agno" recuperados del objeto iterado (des)

      0.3.3-) Se crea un df (df) a partir de la lista de sublistas (vacia si no hay filas) basada en clasificaciones con columnas "home", "score_0", "score_1", "away", "tournament" y "agno"  

      0.3.4-) Se convierten los tipos de las columnas del df (df) a los tipos adecuados: "O" para cadenas y "int64" para enteros

      0.3.5-) Se retorna el df (df)
      
  Args:
  
  - ind0: cadena que indica a que tabla se hará la búsqueda

  - valores: año en que se hará la búsqueda (a no ser que sea un ind0="clasificaciones")

  Returns:

  - df: dataframe con el contenido de la tabla que se ha consultado
  
  """
  if ind0 == "partidos":
    df0 = list()
    for des in partidos.objects.all():
      if des.agno == valores and valores is not None:
        df0.append([des.home, des.score, des.away, des.agno, des.penales, des.jugado])
      elif valores is None:
        df0.append([des.home, des.score, des.away, des.agno, des.penales, des.jugado])
    df = pd.DataFrame(df0, columns=["home", "score", "away", "agno", "penales", "jugado"])
    df.home, df.score, df.away, df.agno = df.home.astype('O'), df.score.astype('O'), df.away.astype('O'), df.agno.astype('int64')
    df.jugado = df.jugado.astype('int64')
    return df

  elif ind0 == "fixtures":
    df0 = list()
    for des in fixtures.objects.all():
      if des.agno == valores:
        df0.append([des.home, des.score, des.away, des.agno])
    df = pd.DataFrame(df0, columns=["home", "score", "away", "year"])
    df.home, df.score, df.away, df.year = df.home.astype('O'), df.score.astype('O'), df.away.astype('O'), df.year.astype('int64')
    return df

  elif ind0 == "grupos":
    df0 = list()
    for des in grupos.objects.all():
      if des.agno == valores:
        df0.append([des.pais, des.Pts, des.PJ, des.PG, des.PP, des.PE, des.GF, des.GC, des.Dif, des.Grupo, des.agno])
    df = pd.DataFrame(df0, columns=["pais", "Pts", "PJ", "PG", "PP", "PE", "GF", "GC", "Dif", "Grupo", "agno"])
    df.pais, df.Pts, df.PJ, df.PG, df.PP, df.PE, df.GF, df.GC, df.Dif, df.Grupo, df.agno = df.pais.astype('O'), df.Pts.astype('int64'), df.PJ.astype('int64'), df.PG.astype('int64'), df.PP.astype('int64'), df.PE.astype('int64'), df.GF.astype('int64'), df.GC.astype('int64'), df.Dif.astype('O'), df.Grupo.astype('O'), df.agno.astype('int64')
    return df

  elif ind0 == "clasificaciones":
    df0 = list()
    for des in clasificaciones.objects.all():
      df0.append([des.home, des.score_0, des.score_1, des.away, des.tournament, des.agno])
    df = pd.DataFrame(df0, columns=["home", "score_0", "score_1", "away", "tournament", "agno"])
    df.home, df.score_0, df.score_1, df.away, df.tournament, df.agno = df.home.astype('O'), df.score_0.astype('int64'), df.score_1.astype('int64'), df.away.astype('O'), df.tournament.astype('O'), df.agno.astype('int64')
    return df
  
  elif ind0 == "internacionales":
    # todos los partidos de selecciones en orden de fecha (etl/busca_internacionales.py)
    columnas = ["fecha", "agno", "home", "away", "score_0", "score_1", "tournament", "categoria", "neutral"]
    df = pd.DataFrame(list(internacionales.objects.order_by("fecha", "id").values(*columnas)), columns=columnas)
    df[["agno", "score_0", "score_1", "neutral"]] = df[["agno", "score_0", "score_1", "neutral"]].astype("int64")
    return df

  elif ind0 == "proxy_desempegno":
    df = pd.DataFrame(
      list(proxy_desempegno.objects.values("pais", "agno", "PJ", *METRICAS_ABSTRACTAS, "pk_real", "sot_real")),
      columns=["pais", "agno", "PJ", *METRICAS_ABSTRACTAS, "pk_real", "sot_real"],
    )
    # los campos son CharField: se pasan a número (float, porque SOT y PKATT pueden traer decimales); agno vacio -> NaN
    numericas = ["agno", "PJ", *METRICAS_ABSTRACTAS, "pk_real", "sot_real"]
    df[numericas] = df[numericas].apply(pd.to_numeric, errors="coerce")
    return df

def historico_abstractas(df_proxy, pais_id, agno=None):
  """
  ¿QUE HACE?
  Suma por pais las METRICAS_ABSTRACTAS de los mundiales anteriores a (agno), con el pais traducido a id. Es el punto de partida del progreso de las abstractas en una consulta o en el dataset de entrenamiento

  ¿COMO LO HACE?
  0-) Si (agno) no es None, conserva solo las filas con agno < (agno): un mundial nunca ve datos de si mismo ni de los siguientes (sin fuga de datos)

  1-) Suma por pais cada metrica; un pais sin ningun dato real de una metrica queda en NaN (no 0), y "valor_previo" le aplicara su valor por defecto

  2-) Traduce "pais" a id con (pais_id) y descarta los paises que no estan en el mapa

  Args:

    - df_proxy: df con columnas pais, agno y METRICAS_ABSTRACTAS (formar_dataset_real("proxy_desempegno"))

    - pais_id: diccionario nombre del país -> id

    - agno: año del mundial que se va a jugar; None suma todos los años

  Returns:

    - df con columnas pais (id) y METRICAS_ABSTRACTAS, una fila por pais
  """
  df = df_proxy if agno is None else df_proxy[df_proxy["agno"] < agno]
  # PJ: partidos de esos mundiales, denominador de las tasas por partido del historico (proxy anteriores a la version 2 no lo traen)
  columnas = METRICAS_ABSTRACTAS + (["PJ"] if "PJ" in df.columns else [])
  df = df.groupby("pais", as_index=False)[columnas].sum(min_count=1)
  df["pais"] = df["pais"].map(pais_id)
  df = df.dropna(subset=["pais"])
  df["pais"] = df["pais"].astype("int64")
  return df.reset_index(drop=True)

def reiniciar_opcionales(pais_id, agno=None):
  """
  ¿QUE HACE?
  Carga en la variable global (opcionales) el histórico de METRICAS_ABSTRACTAS por país previo al mundial (agno), con el país traducido a id, que es como viajan los equipos en los df de fase. Se llama al inicio de cada consulta para que una predicción no arranque desde el progreso que dejó la anterior.

  ¿COMO LO HACE?
  0-) Recupera las filas por pais y mundial con "formar_dataset_real" ("proxy_desempegno")

  1-) Las resume con "historico_abstractas" (solo mundiales anteriores a (agno))

  Args:

    - pais_id: diccionario nombre del país -> id (atributo "pais_id" de la clase "normali")

    - agno: año del mundial consultado; None usa todos los años

  Globales usadas:

    - opcionales: se reemplaza por el histórico recién cargado
  """
  global opcionales
  opcionales = historico_abstractas(formar_dataset_real("proxy_desempegno"), pais_id, agno)

def funcion_tabla_desempegno(df0, pais, indx=None, agno=None, ind0=0):
    """
    ¿QUE HACE?
    Calcula y acumula las métricas de desempeño base (BASE_METRICS: PTS, PJ,
    PG, PP, PE, GF, GC, D) para el equipo (pais), iterando los partidos de
    df0, y con ind0=1 imputa además el progreso de las métricas abstractas
    (METRICAS_ABSTRACTAS: VI, SOT, PKATT, PKATTALLOW) que luego usa
    calculo_metricas_0 para derivar gkps, mds, mos y mms. Puede usarse para
    generar el rendimiento acumulado en la fase eliminatoria previa al
    mundial (ind0=0), o para ajustar dicho rendimiento en df0 tras las
    predicciones de goles del modelo (ind0=1).

    Las métricas base se calculan con los goles "score_0" y "score_1". Las
    abstractas se imputan a partir del progreso de cada equipo: el histórico
    previo guardado en la variable global opcionales, avanzado partido a
    partido en una copia local (cop_opcionales) que no altera la global. El
    acumulado corresponde a pais, tanto si juega como home como si juega como
    away. Con ind0=1 retorna además el progreso de pais, para que
    calculo_metricas_1 actualice opcionales tras procesar todos los países.

    Esta función no recibe las métricas descartadas ni las opcionales
    elegidas por el usuario: la simulación fase a fase siempre es completa y
    solo lo que entra al modelo se neutraliza (calculo_metricas_0).

    ¿COMO LO HACE?
    0-) Declara opcionales como variable global.

    1-) Si ind0 es 1 y opcionales es None, lanza RuntimeError: el histórico
        debe cargarse antes con reiniciar_opcionales(pais_id), para que las
        claves sean ids como en df0. Con ind0=0 las abstractas no se usan.

    2-) Si ind0 es 1, crea una copia local cop_opcionales de opcionales y la
        indexa por pais. Si hay varios registros para un país, conserva el
        primero. Define dos funciones internas:
      2.0-) valor_previo: devuelve el valor de la métrica para el equipo en
            cop_opcionales, o un valor por defecto (0.1) si el equipo o la
            métrica no existen o el valor es NaN.
      2.1-) avanzar: suma un incremento a la métrica del equipo en
            cop_opcionales, agregando al equipo si no tenía histórico.

    3-) Inicializa el acumulador dic_dsp en None; se llenará en el primer
        partido de pais.

    4-) Recorre las filas de una copia de df0 como tuplas, conservando el índice
        original en partido.Index.

    5-) Determina en qué lado juega pais:
      5.0-) Si pais está en home, establece sufijo "_0", el rival y los goles
            correspondientes.
      5.1-) Si pais está en away, establece sufijo "_1", el rival y los goles
            correspondientes.
      5.2-) Si pais no participa en el partido, pasa al siguiente.
      5.3-) Si es el primer partido de pais, inicializa dic_dsp con los
            valores de la fila para cada métrica con el sufijo de pais (lo
            traído de la fase previa o de eliminatorias), o 0 si df0 no tiene
            esa columna o el valor es NaN.

    6-) Obtiene el PJ previo (del progreso local) de pais y del rival con
        valor_previo.

    7-) Si ind0 es 1, imputa las métricas abstractas con el progreso previo
        al partido:
      7.0-) VI: 1 si el rival no marcó goles, 0 en otro caso.
      7.1-) SOT: promedio de tiros a puerta de pais, ajustado según los goles
            que marcó en el partido.
      7.2-) PKATT: promedio de penales intentados por pais, por 0.9.
      7.3-) PKATTALLOW: promedio de penales intentados por el rival, por 0.9
            (los penales que intenta el rival son los que permite pais).
      7.4-) Avanza el progreso local de pais: suma 1 a PJ y lo imputado a VI,
            SOT, PKATT y PKATTALLOW.
      7.5-) Avanza el progreso local del rival: suma 1 a PJ y a PKATT lo
            imputado como PKATTALLOW de pais, para equilibrar los próximos
            cruces de pais contra ese rival.

    8-) Actualiza las métricas base según el resultado del partido:
      8.0-) Si pais gana, suma 1 a PG y 3 a PTS.
      8.1-) Si pais pierde, suma 1 a PP.
      8.2-) Si empata, suma 1 a PE y 1 a PTS.
      8.3-) Actualiza GF, GC, D y PJ.

    9-) Si ind0 es 1, actualiza df0 con las métricas acumuladas de pais para
        cada partido:
      9.0-) Escribe las métricas en las columnas correspondientes al lado en
            que jugó pais ("_0" para home o "_1" para away).

    10-) Si ind0 es 0:
      10.0-) Si pais no jugó ningún partido, deja dic_dsp en 0 para cada
             métrica.
      10.1-) Retorna un DataFrame de una fila con year, pais y las métricas
             acumuladas, usando indx como índice.

    11-) Si ind0 es 1:
      11.0-) Retorna solo la fila de progreso de pais en cop_opcionales (el
             progreso local del rival se descarta, pues su propia pasada
             calcula el suyo), o None si pais no tiene progreso (ni
             histórico ni partidos en df0).

    Args:
      df0: DataFrame con partidos y las columnas home, away, score_0 y score_1.
           Debe estar en orden cronológico. Si ind0=1, se actualiza in-place
           con las columnas de rendimiento.
      pais: Equipo cuyo rendimiento se calcula o actualiza (id si ind0=1).
      indx: Índice de la fila retornada cuando ind0=0.
      agno: Año del mundial (columna "year" del retorno con ind0=0).
      ind0: 0 para retornar el acumulado; 1 para actualizar df0 in-place y
            retornar el progreso de pais.

    Globales usadas:
      BASE_METRICS: Métricas base (columnas de los df de fase).
      opcionales: DataFrame con columnas pais (id), PJ, VI, SOT, PKATT y
                  PKATTALLOW con el progreso previo por país. Solo se lee;
                  la carga reiniciar_opcionales y la actualiza
                  calculo_metricas_1.

    Return:
      Si ind0 es 0, retorna un DataFrame de una fila con year, pais y las
      métricas acumuladas, usando indx como índice.
      Si ind0 es 1, actualiza df0 in-place y retorna un DataFrame de una fila
      con el progreso de pais (o None si pais no tiene progreso).
    """

    # 0-) Declara opcionales como variable global.
    global opcionales

    # 1-) Con ind0=1 el histórico ya debe estar cargado con ids.
    if ind0 == 1 and opcionales is None:
        raise RuntimeError("Llamar a reiniciar_opcionales(pais_id) antes de actualizar el desempeño")

    # 2-) Copia local indexada por país: aquí avanza el progreso sin tocar la global.
    # Conserva el primer registro si hay países duplicados.
    cop_opcionales = None
    if ind0 == 1:
        cop_opcionales = opcionales.drop_duplicates(subset="pais", keep="first").copy()
        cop_opcionales.index = cop_opcionales["pais"]

    # 2.0-) Valor del progreso local o el valor por defecto.
    def valor_previo(metrica, equipo, defecto=0.1):
        if cop_opcionales is None or metrica not in cop_opcionales.columns:
            return defecto
        valor = cop_opcionales[metrica].get(equipo, defecto)
        return defecto if pd.isna(valor) else valor

    # 2.1-) Suma lo imputado al progreso local; agrega al equipo si no tenía histórico.
    def avanzar(metrica, equipo, incremento, defecto=0.1):
        cop_opcionales.loc[equipo, "pais"] = equipo
        cop_opcionales.loc[equipo, metrica] = valor_previo(metrica, equipo, defecto) + incremento

    # 3-) El acumulador se inicializa en el primer partido de pais.
    dic_dsp = None

    # 4-) Recorre una copia de los partidos, conservando sus índices originales.
    for partido in df0.copy().itertuples():
        # 5-) Determina el lado en que juega pais y asigna el rival y los goles.
        if partido.home == pais:
            # 5.0-) pais juega como home.
            sufijo, rival = "_0", partido.away
            gf_pais, gf_rival = partido.score_0, partido.score_1
        elif partido.away == pais:
            # 5.1-) pais juega como away.
            sufijo, rival = "_1", partido.home
            gf_pais, gf_rival = partido.score_1, partido.score_0
        else:
            # 5.2-) pais no participa en este partido.
            continue

        # 5.3-) Primer partido de pais: arranca desde lo que trae la fila, o 0.
        if dic_dsp is None:
            dic_dsp = {}
            for m in BASE_METRICS:
                v = getattr(partido, f"{m}{sufijo}", 0)
                dic_dsp[m] = 0 if pd.isna(v) else v

        # 6-7-) Imputa las métricas abstractas con el progreso previo al partido.
        if cop_opcionales is not None:
            # 6-) Obtiene los partidos previos (progreso local) de pais y del rival.
            pj_previo = valor_previo("PJ", pais)
            pj_previo_rival = valor_previo("PJ", rival)

            # 7.0-) Valla invicta si el rival no marcó.
            vi_imp = 1 if gf_rival == 0 else 0
            # 7.1-) Tiros a puerta de pais ajustados por sus goles.
            avg_sot = valor_previo("SOT", pais) / max(1, pj_previo)
            ratio = gf_pais / max(0.5, avg_sot * 0.3)
            sot_imp = round(avg_sot * (0.7 + 0.3 * min(ratio, 2)), 1)
            # 7.2-) Penales intentados por pais.
            pkatt_imp = round(valor_previo("PKATT", pais) / max(1, pj_previo) * 0.9, 2)
            # 7.3-) Penales intentados por el rival = permitidos por pais.
            pkatt_rival_imp = round(valor_previo("PKATT", rival) / max(1, pj_previo_rival) * 0.9, 2)

            # 7.4-) Avanza el progreso local de pais.
            avanzar("PJ", pais, 1, defecto=0)
            avanzar("VI", pais, vi_imp, defecto=0)
            avanzar("SOT", pais, sot_imp)
            avanzar("PKATT", pais, pkatt_imp)
            avanzar("PKATTALLOW", pais, pkatt_rival_imp)

            # 7.5-) Avanza el progreso local del rival para los próximos cruces de pais contra él.
            avanzar("PJ", rival, 1, defecto=0)
            avanzar("PKATT", rival, pkatt_rival_imp)

        # 8-) Actualiza las métricas base según el resultado.
        if gf_pais > gf_rival:
            # 8.0-) Victoria.
            dic_dsp["PG"] += 1
            dic_dsp["PTS"] += 3
        elif gf_pais < gf_rival:
            # 8.1-) Derrota.
            dic_dsp["PP"] += 1
        else:
            # 8.2-) Empate.
            dic_dsp["PE"] += 1
            dic_dsp["PTS"] += 1

        # 8.3-) Actualiza goles, diferencia y partidos jugados.
        dic_dsp["GF"] += gf_pais
        dic_dsp["GC"] += gf_rival
        dic_dsp["D"] = dic_dsp["GF"] - dic_dsp["GC"]
        dic_dsp["PJ"] += 1

        # 9-) Si corresponde, actualiza df0 con el acumulado hasta este partido.
        if ind0 == 1:
            # 9.0-) Escribe las métricas en las columnas del lado de pais.
            cols_to_update = [f"{m}{sufijo}" for m in BASE_METRICS]
            # pandas 3 no sube int64 -> float64 al asignar: los goles predichos son decimales (GF, GC, D)
            if any(df0[c].dtype.kind != "f" for c in cols_to_update):
                df0[cols_to_update] = df0[cols_to_update].astype("float64")
            df0.loc[partido.Index, cols_to_update] = [dic_dsp[m] for m in BASE_METRICS]

    # 10-) Si se solicitó el acumulado para la fase eliminatoria, lo retorna.
    if ind0 == 0:
        # 10.0-) pais sin partidos: acumulado en 0.
        if dic_dsp is None:
            dic_dsp = {m: 0 for m in BASE_METRICS}
        # 10.1-) Construye y retorna el DataFrame de una fila.
        return pd.DataFrame({"year": agno, "pais": pais, **dic_dsp}, index=[indx])

    # 11-) Tras predicción: retorna el progreso de pais para actualizar la global.
    if cop_opcionales is not None and pais in cop_opcionales.index:
        # 11.0-) Solo la fila de pais; el progreso local del rival se descarta.
        return cop_opcionales.loc[[pais]].reset_index(drop=True)
    return None

def _por_partido(valor, PJ):
    """Divide fila a fila por PJ; NaN donde PJ es 0 (cada calc_* lo pasa luego a 0)."""
    return valor / PJ.where(PJ > 0)

# tasas por partido para un equipo sin partidos en su historico abstracto: ~30 % de vallas invictas, ~4 tiros a puerta y
# ~0,12 penales a favor y en contra por partido (las de etl/construye_proxy.py)
TASAS_REFERENCIA = {"VI": 0.30, "SOT": 4.0, "PKATT": 0.12, "PKATTALLOW": 0.12}
# parte de los penales que terminan en gol, para descontar los goles de penal
CONVERSION_PENAL = 0.75

def _tasa_historica(valor, PJ_hist, metrica):
    """(valor) del historico abstracto por partido de ese mismo historico (PJ_hist); la tasa de referencia si el equipo no tiene partidos en su historico"""
    return (valor / PJ_hist.where(PJ_hist > 0)).fillna(TASAS_REFERENCIA[metrica])

def _tope(serie, referencia):
    """serie / referencia, sin pasar de 1: cuanto de un valor de referencia alcanza el equipo"""
    return (serie / referencia).clip(upper=1)

def calc_gkps(PJ, GC, VI, PKATTALLOW, PJ_hist):
    """
    ¿QUE HACE?
    Calcula, partido a partido, el puntaje de portería (goalkeeping score, 0 a 100): premia las vallas invictas y castiga los goles recibidos que no vinieron de penal

    Args:

      - PJ, GC: Series con el acumulado de partidos jugados y goles en contra del ciclo hasta cada partido

      - VI, PKATTALLOW, PJ_hist: Series con el progreso de vallas invictas, penales permitidos y partidos del historico abstracto

    Return:

      - Serie con 60 * VI/PJ_hist + 40 - 10 * (GC/PJ - 0,75 * PKATTALLOW/PJ_hist), entre 0 y 100, o 0 donde PJ es 0
    """
    vi, pka = _tasa_historica(VI, PJ_hist, "VI"), _tasa_historica(PKATTALLOW, PJ_hist, "PKATTALLOW")
    gc_sin_penal = (_por_partido(GC, PJ) - CONVERSION_PENAL * pka).clip(lower=0)
    return (vi * 60 + 40 - gc_sin_penal * 10).clip(0, 100).fillna(0)

def calc_mds(PJ, GC, VI, PKATTALLOW, PJ_hist):
    """
    ¿QUE HACE?
    Calcula, partido a partido, el puntaje defensivo (mean defensive score, 0 a 100): premia las vallas invictas y castiga goles recibidos y penales permitidos

    Args:

      - PJ, GC: Series con el acumulado de partidos jugados y goles en contra del ciclo hasta cada partido

      - VI, PKATTALLOW, PJ_hist: Series con el progreso de vallas invictas, penales permitidos y partidos del historico abstracto

    Return:

      - Serie con 50 * VI/PJ_hist + 50 - 10 * (GC/PJ + PKATTALLOW/PJ_hist), entre 0 y 100, o 0 donde PJ es 0
    """
    vi, pka = _tasa_historica(VI, PJ_hist, "VI"), _tasa_historica(PKATTALLOW, PJ_hist, "PKATTALLOW")
    return (vi * 50 + 50 - (_por_partido(GC, PJ) + pka) * 10).clip(0, 100).fillna(0)

def calc_mos(PJ, GF, SOT, PKATT, PJ_hist):
    """
    ¿QUE HACE?
    Calcula, partido a partido, el puntaje ofensivo (mean offensive score, 0 a 100): goles por partido, tiros a puerta por partido y efectividad de los tiros sin contar penales

    Args:

      - PJ, GF: Series con el acumulado de partidos jugados y goles a favor del ciclo hasta cada partido

      - SOT, PKATT, PJ_hist: Series con el progreso de tiros a puerta, penales intentados y partidos del historico abstracto

    Return:

      - Serie con 50 * min(1, (GF/PJ)/3) + 30 * min(1, (SOT/PJ_hist)/6) + 20 * min(1, efectividad/0,5), donde efectividad = (GF/PJ - 0,75 * PKATT/PJ_hist) / (SOT/PJ_hist); 0 donde PJ es 0
    """
    gf, sot, pk = _por_partido(GF, PJ), _tasa_historica(SOT, PJ_hist, "SOT"), _tasa_historica(PKATT, PJ_hist, "PKATT")
    efectividad = ((gf - CONVERSION_PENAL * pk).clip(lower=0) / sot.where(sot > 0)).fillna(0)
    return (_tope(gf, 3) * 50 + _tope(sot, 6) * 30 + _tope(efectividad, 0.5) * 20).fillna(0)

def calc_mms(PJ, PTS, PG, SOT, PJ_hist):
    """
    ¿QUE HACE?
    Calcula, partido a partido, el puntaje de momento (mean momentum score, 0 a 100): proporción de puntos obtenidos sobre los posibles en el ciclo y tiros a puerta por partido del historico. (PG) no se usa: la proporción de victorias ya está dentro de los puntos

    Args:

      - PJ, PTS, PG: Series con el acumulado de partidos jugados, puntos y partidos ganados del ciclo hasta cada partido

      - SOT, PJ_hist: Series con el progreso de tiros a puerta y partidos del historico abstracto

    Return:

      - Serie con 60 * PTS/(3*PJ) + 40 * min(1, (SOT/PJ_hist)/5), o 0 donde PJ es 0
    """
    return (_por_partido(PTS, PJ * 3) * 60 + _tope(_tasa_historica(SOT, PJ_hist, "SOT"), 5) * 40).fillna(0)

def calc_rate(valor, PJ):
    """
    ¿QUE HACE?
    Calcula, partido a partido, la tasa de una métrica acumulada (GF o GC) por cada 10 partidos

    Args:

      - valor: Serie con el acumulado de la métrica (GF o GC) hasta cada partido

      - PJ: Serie con el acumulado de partidos jugados; un PJ de 0 se reemplaza por 0.085 para no dividir por 0

    Return:

      - Serie con (valor/PJ)*10
    """
    return (valor / PJ.replace(0, 0.085)) * 10

def metricas_apagadas(drop_metricas, metricas_opcionales):
    """Métricas que el modelo recibe neutralizadas: las base descartadas más las opcionales no aceptadas."""
    return set(drop_metricas) | (set(OPTIONAL_METRICS) - set(metricas_opcionales))

def validar_valores_neutros(drop_metricas, metricas_opcionales):
    """
    ¿QUE HACE?
    Verifica, antes de cargar el modelo, que toda métrica que se vaya a neutralizar tenga su valor en VALOR_NEUTRO

    Args:

      - drop_metricas: métricas base descartadas por el usuario

      - metricas_opcionales: métricas opcionales aceptadas por el usuario

    Return:

      - No retorna; lanza ValueError con las métricas sin valor neutro configurado
    """
    faltantes = sorted(m for m in metricas_apagadas(drop_metricas, metricas_opcionales) if VALOR_NEUTRO.get(m) is None)
    if faltantes:
        raise ValueError(f"Aún no hay valor neutro configurado para {faltantes}; no se pueden ignorar todavía")

def neutralizar(features, drop_metricas, metricas_opcionales):
    """
    ¿QUE HACE?
    Reemplaza por su VALOR_NEUTRO cada métrica descartada (base) o no aceptada (opcional). Nunca quita columnas, así la entrada al modelo conserva sus dimensiones

    Args:

      - features: diccionario métrica -> Serie con los valores de un lado (home o away)

      - drop_metricas: métricas base descartadas por el usuario

      - metricas_opcionales: métricas opcionales aceptadas por el usuario

    Return:

      - features con las métricas apagadas reemplazadas
    """
    for m in metricas_apagadas(drop_metricas, metricas_opcionales):
        features[m] = pd.Series(VALOR_NEUTRO[m], index=features[m].index, dtype=float)
    return features

def calculo_metricas_0(df_desempegno, drop_metricas=DROP_METRICS, metricas_opcionales=OPTIONAL_METRICS):
    """
    ¿QUE HACE?
    Arma la entrada al modelo a partir de un df de fase: para cada partido y cada lado (home, away) toma las métricas base acumuladas, deriva las opcionales (gkps, mds, mos, mms, rate_GC, rate_GF) con el progreso de METRICAS_ABSTRACTAS en (opcionales), y neutraliza lo que el usuario descartó o no aceptó. La salida siempre tiene las mismas columnas en el mismo orden, sin importar la elección del usuario

    ¿COMO LO HACE?
    0-) Si (opcionales) es None lanza RuntimeError: debe cargarse antes con "reiniciar_opcionales"

    1-) Indexa (opcionales) por id de país, con un registro por país

    2-) Para cada lado (idx 0 = home, idx 1 = away):

      2.0) Toma las métricas base de ese lado por su nombre con sufijo (PTS_0, PJ_0, ...)

      2.1) Busca el progreso abstracto del equipo de cada fila (VI, SOT, PKATT, PKATTALLOW y sus partidos PJ); si el país no tiene registros queda en 0 y las calc_* usan las tasas de referencia (TASAS_REFERENCIA)

      2.2) Calcula fila a fila las opcionales con "calc_gkps", "calc_mds", "calc_mos", "calc_mms" y "calc_rate"; cada partido solo ve su propio acumulado

      2.3) Neutraliza con "neutralizar" las métricas apagadas

      2.4) Arma el bloque [equipo, ALL_METRICS con sufijo, score]

    3-) Une los bloques home y away por columnas y retorna el resultado

    Args:

      - df_desempegno: df de fase con home, away, score_0, score_1 y BASE_METRICS con sufijos _0 y _1

      - drop_metricas: métricas base descartadas por el usuario

      - metricas_opcionales: métricas opcionales aceptadas por el usuario

    Globales usadas:

      - opcionales: progreso de METRICAS_ABSTRACTAS por id de país (solo se lee)

    Return:

      - df con columnas [home, ALL_METRICS_0, score_0, away, ALL_METRICS_1, score_1] (COLS_MODELO_POR_EQUIPO * 2), una fila por partido y en el mismo orden que (df_desempegno)
    """
    # 0-) El histórico debe estar cargado con ids.
    if opcionales is None:
        raise RuntimeError("Llamar a reiniciar_opcionales(pais_id) antes de calcular métricas")

    # 1-) Un registro por país.
    historico = opcionales.drop_duplicates(subset="pais", keep="first").set_index("pais")
    historico = historico.reindex(columns=METRICAS_ABSTRACTAS + ["PJ"])
    bloques = []

    # 2-) Cada lado por separado.
    for idx, lado in ((0, "home"), (1, "away")):
        # 2.0) Métricas base de este lado.
        base = {m: df_desempegno[f"{m}_{idx}"].astype(float) for m in BASE_METRICS}

        # 2.1) Progreso abstracto del equipo de cada fila.
        # sin historico: 0 partidos, y las calc_* usan las tasas de referencia
        abst = historico.reindex(df_desempegno[lado].to_numpy()).fillna(0.0)
        abst.index = df_desempegno.index

        # 2.2) Opcionales calculadas fila a fila.
        features = {
            **base,
            "gkps": calc_gkps(base["PJ"], base["GC"], abst["VI"], abst["PKATTALLOW"], abst["PJ"]),
            "mds": calc_mds(base["PJ"], base["GC"], abst["VI"], abst["PKATTALLOW"], abst["PJ"]),
            "mos": calc_mos(base["PJ"], base["GF"], abst["SOT"], abst["PKATT"], abst["PJ"]),
            "mms": calc_mms(base["PJ"], base["PTS"], base["PG"], abst["SOT"], abst["PJ"]),
            "rate_GC": calc_rate(base["GC"], base["PJ"]),
            "rate_GF": calc_rate(base["GF"], base["PJ"]),
        }

        # 2.3) Lo apagado recibe su valor neutro.
        features = neutralizar(features, drop_metricas, metricas_opcionales)

        # 2.4) Bloque con orden fijo.
        bloques.append(pd.DataFrame({
            lado: df_desempegno[lado],
            **{f"{m}_{idx}": features[m] for m in ALL_METRICS},
            f"score_{idx}": df_desempegno[f"score_{idx}"],
        }))

    # 3-) Siempre COLS_MODELO_POR_EQUIPO * 2 columnas.
    return pd.concat(bloques, axis=1)

def calculo_metricas_1(df_desempegno):
  """
  ¿QUE HACE?
  Ajusta, tras alguna predicción de goles del modelo, las métricas de
  rendimiento de df_desempegno para cada país único que juega como home o
  away, llamando a funcion_tabla_desempegno con ind0=1. Luego vuelca en la
  variable global opcionales el progreso de cada país, para que la siguiente
  fase impute a partir de él.

  ¿COMO LO HACE?
  0-) Declara opcionales como variable global.

  1-) Obtiene la lista paises con los equipos únicos de las columnas home y
      away de df_desempegno.

  2-) Recorre paises:
    2.0-) Llama a funcion_tabla_desempegno(df_desempegno, pais, ind0=1), que
          actualiza df_desempegno in-place. Cada país avanza en local desde la
          misma global, sin ver el progreso de los demás.
    2.1-) Si retorna progreso (no None), lo agrega a la lista progresos.

  3-) Si hay progresos, ya procesados todos los países:
    3.0-) Une los progresos en el DataFrame nuevos.
    3.1-) Reemplaza en opcionales las filas de esos países por las de nuevos,
          conservando las de los países que no jugaron.

  4-) Retorna df_desempegno.

  Args:
    df_desempegno: DataFrame con los partidos de alguna fase del mundial y las
                   predicciones de goles del modelo en score_0 y score_1.

  Globales usadas:
    opcionales: DataFrame con el progreso previo de métricas por país; se
                actualiza aquí.

  Return:
    df_desempegno con las métricas de rendimiento actualizadas.
  """
  # 0-) Declara opcionales como variable global.
  global opcionales

  # 1-) Países únicos de la fase.
  paises = pd.concat([df_desempegno.home, df_desempegno.away], axis=0).unique()
  progresos = []

  # 2-) Cada país avanza en local desde la misma global.
  for indip, pais in enumerate(paises):
    # 2.0-) Ajusta df_desempegno in-place y obtiene el progreso de pais.
    progreso = funcion_tabla_desempegno(df_desempegno, pais, ind0=1)
    # 2.1-) Guarda el progreso si existe.
    if progreso is not None:
      progresos.append(progreso)

  # 3-) Ya procesados todos, vuelca el progreso en la global.
  if progresos:
    # 3.0-) Une los progresos.
    nuevos = pd.concat(progresos, ignore_index=True)
    # 3.1-) Reemplaza las filas de los países que jugaron.
    opcionales = pd.concat(
      [nuevos, opcionales[~opcionales["pais"].isin(nuevos["pais"])]],
      ignore_index=True,
    )
    # los países agregados en local llegan como float: se mantiene el id entero
    opcionales["pais"] = opcionales["pais"].astype("int64")

  # 4-) Retorna el df actualizado.
  return df_desempegno

def agregar_features(df_mundial, dic_ag, ind=None):
  """
  ¿QUE HACE?
  Arroja un df mas completo a partir de uno de entrada que es el parámetro (df_mundial), conteniendo, además de los nombres de equipos home y away, los puntajes para cada uno por partido y el año en que se disputó el mundial, las diferentes métricas de rendimiento que mostrarán los equipos para la inferencia con el modelo, que serán (PTS, PJ, PG, PP, PE, GF, GC, D)
  
  ¿COMO LO HACE?
  0-) Se define una variable (df5) que sera un df vacio. Se crea una variable (llaves) con la lista de llaves del parametro (dic_ag). Se crean unas variables (c_locales, c_visitantes) contenedoras de los nombres de las columnas correspondientes a las metricas de rendimiento para los equipos home y away

  1-) Se itera a la inversa sobre las filas en forma de tupla y con los indices reseteados del df (df_mundial). Se tienen 2 condicionales

    1.0) Si el tipo del año iterado de las filas del df (df_mundial) no es str. Se guarda en las variables (agno0, agno1) el año del mundial en estudio para iniciar la busqueda de informacion de rendimiento en años de clasificacion, casteando el contenido de la columna "agno" a str

    1.1) Si es tipo str el tipo del año iterado de las filas del df (df_mundial). Se guarda en las variables (agno0, agno1) el año del mundial en estudio para iniciar la busqueda de informacion de rendimiento en años de clasificacion en el tipo en que viene
  
  2-) Se guarda en la variable (mundial_actual) la informacion de rendimiento en fase de clasificacion. Se guarda en unas variables (prt_nv_h, prt_nv_a) la informacion de rendimiento registrada para los equipos que juegan como home y away. Se actualizan las variables (agno0, agno1) al indice en la lista (llaves) del año del mundial actual -1.
  
  3-) Se inicia un bucle while en caso de que (prt_nv_h) este vacio no habiendo informacion de rendimiento para el año actual en fase de clasificacion

    3.0) Se actualiza la variable (mundial_actual) con la informacion de rendimiento en clasificaciones del año anterior al del mundial actual. Se actualiza la variable (prt_nv_h) segun el nuevo contenido de (mundial_actual). Se tienen 3 condicionales

      3.0.0) Si (prt_nv_h) no esta vacio. Se rompe el bucle while con break

      3.0.1) Si el indice en (agno0) es 0 y (prt_nv_h) está vacio. Se actualiza (prt_nv_h) con todas las metricas de rendimiento seteadas a 0, con las columnas correspondientes al equipo home

      3.0.2) Si el indice en (agno0) no es 0 y (prt_nv_h) está vacio. Se resta 1 al (agno0)
  
  4-) Se inicia un bucle while en caso de que (prt_nv_a) este vacio no habiendo informacion de rendimiento para el año actual en fase de clasificacion

    4.0) Se actualiza la variable (mundial_actual) con la informacion de rendimiento en clasificaciones del año anterior al del mundial actual. Se actualiza la variable (prt_nv_a) segun el nuevo contenido de (mundial_actual). Se tienen 3 condicionales

      4.0.0) Si (prt_nv_a) no esta vacio. Se rompe el bucle while con break

      4.0.1) Si el indice en (agno0) es 0 y (prt_nv_a) está vacio. Se actualiza (prt_nv_a) con todas las metricas de rendimiento seteadas a 0, con las columnas correspondientes al equipo away

      4.0.2) Si el indice en (agno0) no es 0 y (prt_nv_a) está vacio. Se resta 1 al (agno0)
      
  5-) Se crean 2 condicionales

    5.0) Si el parámetro (ind) es None. Se coloca en un diccionario (data_metricas) toda la informacion de rendimiento obtenida previamente, con claves-columna:
    
    ("home, PTS_0, PJ_0, PG_0, PP_0, PE_0, GF_0, GC_0, D_0, score_0, away, PTS_1, PJ_1, PG_1, PP_1, PE_1, GF_1, GC_1, D_1, score_1")
    
    y valor, la informacion iterada y la registrada en (prt_nv_h) y (prt_nv_a)
    
    5.1) Si el parámetro (ind) es 0. Se coloca en un diccionario (data_metricas) toda la informacion de rendimiento obtenida previamente, con claves-columna:
    
    ("home, PTS_0, PJ_0, PG_0, PP_0, PE_0, GF_0, GC_0, D_0, score_0, away, PTS_1, PJ_1, PG_1, PP_1, PE_1, GF_1, GC_1, D_1, score_1")
    
    Este condicional se encarga de los df fixture, por lo que para los años "1934, 1938" se empieza de una con "score" porque, al no haber fase de grupos (en el fixture) se empieza directamente con la columna score en formato de referencia a partidos (Match X) teniendose unas clave-columna:

    ("home, PTS_0, PJ_0, PG_0, PP_0, PE_0, GF_0, GC_0, D_0, score, away, PTS_1, PJ_1, PG_1, PP_1, PE_1, GF_1, GC_1, D_1, score")

  6-) Se guarda en el df (df5) el contenido de (agf_df)

  7-) Se retorna el contenido de (df5) con filas invertidas e indices reseteados

  Args:

    - df_mundial: df contenedor de los partidos del mundial actual, para obtener las metricas de rendimiento en la fase de clasificaciones

    - dic_ag: diccionario contenedor de df con los parametros de rendimiento (PTS, PJ, PG, PP, PE, GF, GC, D) and the nombre del pais (pais), obtenidos por los equipos durante la fase de clasificacion previa a cada mundial, con clave igual al año del mundial de interés (en formato string) y valor igual al df de rendimiento por equipo
    
    - ind: indicador de si el df (df_mundial) es de partidos historicos o de fixture del mundial en estudio
  
  Returns:

    - df5: df con la informacion del rendimiento de los equipos home y away que jugarán la fase de grupos del mundial actual, durante la fase de clasificacion previa al mundial. Tiene los indices reseteados
  """
  df5 = pd.DataFrame()
  llaves = list(dic_ag.keys())
  c_locales = [f"{m}_0" for m in BASE_METRICS]
  c_visitantes = [f"{m}_1" for m in BASE_METRICS]

  for mnd in (df_mundial[::-1].reset_index(drop=True).itertuples(index=True)):
    if not isinstance(mnd.agno, str):
      agno0 = str(mnd.agno)
      agno1 = str(mnd.agno)
    else:
      agno0 = mnd.agno
      agno1 = mnd.agno

    mundial_actual = dic_ag[agno0]

    prt_nv_h = mundial_actual[mundial_actual.pais==mnd.home][mundial_actual.columns[1:-1]].copy()
    prt_nv_a = mundial_actual[mundial_actual.pais==mnd.away][mundial_actual.columns[1:-1]].copy()
    
    agno0 = llaves.index(agno0) - 1
    agno1 = llaves.index(agno1) - 1

    while prt_nv_h.empty:
      mundial_actual = dic_ag[llaves[agno0]]
      prt_nv_h = mundial_actual[mundial_actual.pais==mnd.home][mundial_actual.columns[1:-1]].copy()

      if not prt_nv_h.empty:
        break

      if agno0 == 0 and prt_nv_h.empty:
        prt_nv_h = pd.DataFrame(np.zeros((1, len(c_locales))), columns=c_locales)
      elif agno0 > 0 and prt_nv_h.empty:
        agno0 -= 1

    while prt_nv_a.empty:
      mundial_actual = dic_ag[llaves[agno1]]
      prt_nv_a = mundial_actual[mundial_actual.pais==mnd.away][mundial_actual.columns[1:-1]].copy()

      if not prt_nv_a.empty:
        break
      
      if agno1 == 0 and prt_nv_a.empty:
        prt_nv_a = pd.DataFrame(np.zeros((1, len(c_visitantes))), columns=c_visitantes)
      elif agno1 > 0:
        agno1 -= 1

    local_mapping = {}
    for col_0, col_1 in zip(prt_nv_h.columns, c_locales):
      local_mapping[col_1] = [prt_nv_h[col_0].values[0]]
      
    visitor_mapping = {}
    for col_0, col_1 in zip(prt_nv_a.columns, c_visitantes):
      visitor_mapping[col_1] = [prt_nv_a[col_0].values[0]]

    if ind is None:
      data_metricas = {
          "home": [mnd.home],
          **local_mapping,
          "score_0": [mnd.score_0],
          "away": [mnd.away],
          **visitor_mapping,
          "score_1": [mnd.score_1],
      }
    elif ind == 0:
      data_metricas = {
          "home": [mnd.home],
          **local_mapping,
          "score_0": [mnd.score_0] if not (mnd.agno in (1934, 1938)) else [mnd.score],
          "away": [mnd.away],
          **visitor_mapping,
          "score_1": [mnd.score_1] if not (mnd.agno in (1934, 1938)) else [mnd.score],
      }
    agf_df = pd.DataFrame(data=data_metricas)
    df5 = pd.concat([df5, agf_df], axis=0)

  return df5.iloc[::-1].reset_index(drop=True)

class normali():
  def __init__(self, ind_partidos, ind_fixtures, df_prediccion_usuario, clsif, obj0="-", evitar="-", embedding_dim=3):
    """
    ¿QUE HACE?
    Genera una serie de df y capas de embedding necesarios para dar funcionamiento a la busqueda de nombres de grupos para equipos y el orden de juego en cada mundial (los emparejamientos por mejores terceros de 1986-1994 viven fuera de la clase, en "alinear_llave_terceros")

    ¿COMO LO HACE?
    0-) Se coloca en una variable (df0) el resultado de la funcion "formar_dataset_real" ingresandole el argumento (ind_partidos), que será un df contenedor de todos los partidos de cierto mundial 

    1-) En el atributo de instancia (self.df2) se almacena una copia del df (df0) con el índice reseteado, conteniendo todos los partidos recuperados de la tabla indicada por (ind_partidos) 

    2-) Luego deja en un atributo de instancia (self.df2) una copia del o los mundiales recuperados, por "formar_dataset_real" 

    3-) En otro atributo de instancia (self.df3) se almacena el fixture del mundial de interés, usando la funcion "formar_dataset_real" con argumentos "fixtures" y el parametro (ind_fixtures) que contiene el año del mundial de interés en formato string, para la búsqueda del fixture correspondiente en la DB

    4-) Se crea un df contenedor (self.paises) de los nombres de todos los paises que jugaron en la fase de clasificacion previa al mundial de interés y los paises que juegan el mundial en si mismo 

    5-) PEND (buscar si se puede optimizar con tokenize)

    6-) Luego se deja en un atributo de instancia (self.obj0) el contenido del parametro (obj0) para indicar cuando se trabaja con o sin prediccion tras consulta de un usuario 

    7-) Se almacena en un atributo de instancia (self.df5) el df contendor de la consulta del usuario (df_prediccion_usuario) 

    8-) Se crea una capa embedding de keras indicando el tamaño de la entrada y salida de la misma para futura conversion de datos categoricos referentes a los nombres de los paises que juegan el mundial de interes

    - Args:

      - ind_partidos: valor de tipo str que indica a la funcion "formar_dataset_real" que tabla de la DB debe consultar
      
      - ind_fixtures: valor de tipo str que indica a la funcion "formar_dataset_real" que año debe buscar en la tabla fixture de la DB, proveniente de un slicing a los ultimos 4 elementos del argumento (pregunta_0) de la funcion "consulta_general"
      
      - df_prediccion_usuario: df consulta del usuario, básicamente un df de la fase con datos editados por usuario para hacer inferencia
      
      - clsif: df contenedor de partidos de clasificacion previa al mundial de interés a predecir

      - obj0: puede ser un str de la forma "-", o un slicing a los ultimos 4 elementos del argumento (pregunta_0) de la funcion "consulta_general", indicando en el 1er caso que el 1er valor de retorno de la funcion que usa este argumento desde el atributo de instancia (self.obj0), será igual a todos los partidos del mundial actual, e indicando para el 2do caso, que este 1er valor de retorno se divide entre los partidos de la fase final y los partidos del df de consulta del usuario
      
      - embedding_dim: valor indicador del tamaño de los vectores embedding para cada palabra
    """
    df0 = formar_dataset_real(ind_partidos)
    self.df2 = df0.copy().reset_index(drop=True)
    self.df3 = formar_dataset_real("fixtures", ind_fixtures)
    
    self.paises = pd.concat([self.df2.home, self.df2.away, self.df3.home, self.df3.away, clsif.home, clsif.away], axis=0)

    self.pais_id = {pais: id for id, pais in enumerate(self.paises.unique())}
    self.id_pais = {id: pais for pais, id in self.pais_id.items()}

    self.obj0 = obj0
    self.df5 = df_prediccion_usuario
    self.embedding_layer = tf.keras.layers.Embedding(input_dim=len(self.pais_id), output_dim=embedding_dim)

  def ajuste_score_partidos(self, x, dictionary, ind=0):
    """
    ¿QUE HACE?
    Sustituye caracteres basura resultantes del scraping, por los que convienen al preprocesamiento, entrenamiento e inferencia con los datos

    ¿COMO LO HACE?
    0-) Se crea un parametro (translate_table) que usa el metodo "str.maketrans" para dar soporte al metodo ".translate" con una tabla de referncia para uno a uno caracteres. Se crea un parametro (translated_text) para traducir caracteres de una cadena, segun (translate_table) mediante ".translate". Desde el paramtro (new_string) se obtiene una particion de la cadena (translated_text) tomando "," como separador

    1-) Si el parametro (ind) es igual a 0 o igual a 1

      1.0) Se define una variable (goles) que representa el numero de goles para el equipo home o away, segun el valor que tenga (ind), para buscar en (new_string)

      1.1) Retorna la variable (goles)
      
    Args:

        - x: string a limpiar

        - dictionary: diccionario con clave (caracter a reemplazar), valor (caracter de reemplazo)

        - ind: indica si retornaremos el resultado formateado segun el diccionario para los goles
              del equipo 0 o equipo 1 (score_0, score_1)

    Returns:

        - goles: el numero de goles para el equipo de interes como un integer
    """
    translate_table = str.maketrans(dictionary)
    translated_text = x.translate(translate_table)
    new_string = translated_text.split(",")
    
    if (ind == 0) or (ind == 1):
      goles = int(new_string[ind])
      return goles
    
  def hacer_embedding_a_equipos(self, entrada):
    """
    ¿QUE HACE?
    convierte a tensor los valores categoricos referentes a los nombres de los equipos que juegan cieta fase del mundial. Se concatena la version embedding como array tipo float de los nombres de los equipo de la fase trabajada, junto con los valores de rendimiento recuperados de (entrada) y se entrega ese nuevo array concatenado como resultado 

    ¿COMO LO HACE?
    0-) Se convierte a tensor los valores de (entrada) que corresponden a los nombres de los equipos que jugarian cierto round de cierta fase ya sea como local o visitante, pasandolos todos a int64 y guardandolo en (indices_tensor)

    1-) Referenciando a (self.embedding_layer) se accede al objeto de la funcion Embedding de tensorflow, se le pasa el (indices_tensor) como parametro y se guarda este formato embedding como array numpy en (embeddings_array)

    2-) En la variable (result) se concatena como columna el array de embeddings (embeddings_array) con el array de todos los valores de rendimiento (entrada[:,1:]), correspondientes a todas las columnas de (entrada) a partir de la posicion 1, omitiendo unicamente la columna 0 que contiene el id del equipo. Se retorna (result)

    - Args:

      - entrada: array numpy contendor del orden de juego por equipo local o visitante segun se pase y los valores de rendimiento de cada uno

    - Returns:

      - result: array concatenado del codigo embedding para cada equipo y sus valores de rendimiento, entresacados de (entrada)
    """
    indices_tensor = tf.convert_to_tensor(entrada[:, 0], dtype=tf.int64)

    embeddings = self.embedding_layer(indices_tensor)
    embeddings_array = embeddings.numpy()
    result = np.concatenate((embeddings_array, entrada[:, 1:].astype(float)), axis=1)
    return result

  def c_s(self, data):
    '''
    ¿QUE HACE?
    Recive un array en el parametro (data) contenedor de los partidos del round de la fase trabajada con los equipos codificados y ejecuta la eliminacion de la columna de "score_n", luego va guardando en unas listas vacias (X, y) los lotes de informacion segun el tamaño de paso establecido en ---n_steps0--- y su respuesta esperrada para el "score_n". Al final se devuelve una tupla con features y valores esperados referentes a los goles que estos equipos, con estos valores de rendimiento, habrian de anotar
    
    ¿COMO LO HACE?
    0-) Se crean unas listas vacias (X, y). Se elimina de (entrada) la columna correspondiente al score_n segun si se trabaja con equipo local o visitante
    
    1-) Se almacena en una variable (n_steps0) el tamaño de paso con que se definirán los lotes de informacion con que se alimentará el modelo para la inferencia

    2-) Se itera un rango desde la variable de iteracion (i) correspondiente al numero de filas que tendria el array pasado en (data), menos el tamaño de paso. Se guarda en la lista (X) el lote de filas con todas las columnas de informacion de equipo y rendimiento, desde que marca (i) hasta (i + n_steps0) y dejando para (y) justo el valor que corresponderia a las columnas de la fila (i + n_steps0). Finalmente se retorna una tupla con ambos arrays (X, y)
    
    - Args:

      - data: array contendor de nombres de los equipos codificados como embedding y sus valores de rendimiento 

    - returns:

      - np.array(X), np.array(y): tupla contenedora de identidad de cada equipo (nombre codificado, valores de rendimiento) en (X) y su correspondiente respuesta esperada en goles almacenada en "score_n" dentro de la lista (y)
    '''
    X, y = [], []
    data0 = np.delete(data, C_S_DELETE_INDEX, axis=1)

    n_steps0 = N_STEPS

    for i in range(len(data) - n_steps0 + 1):
      X.append(data0[i:i + n_steps0, :])
      y.append(data[i + n_steps0 - 1, C_S_Y_INDEX]) # la ventana [i, i + n_steps0) termina en el partido cuyo score se predice

    return np.array(X), np.array(y)

  def grupos_anio_interes(self, year):
    """
    ¿QUE HACE?
    Toma los valores de una consulta a DB con ORM usando de parámetro la tabla y el valor de la columna, para pasar el nombre categorico a indice, en los partidos que habria judado un equipo, formar un diccionario con claves (nombre del grupo) valor (valores de rendimiento actualizados tras las eliminatorias para cada equipo del grupo). Finalmente, ese diccionario se retorna a la salida de la funcion

    ¿COMO LO HACE?
    0-) Se emplea la funcion "formar_dataset_real" para recuperar con el ORM de django, las filas de la base de datos postgresql, de la tabla grupos, donde la columna agno sea igual a (year). Se guarda toda esta informacion en una variable llamada (df0)

    1-) Toma del df devuelto en (df0) y transforma el contenido de la columan "pais" a indices para facilitar su compatibilidad con las entradas del modelo de deep learning

    2-) Se usa un ".loc" para afectar la direccion en memoria de todas las filas y columnas, menos la primera y la ultima, de (df0) igualando a 0 el conteniddo dentro de este rango
    
    3-) genera el diccionario (dic_0) con la iteracion de los valores unicos de la columna del (df0) correspondiente a los grupos para cada pais y deja como clave (nombre del grupo) y valor (valores de rendimiento actualizados tras las eliminatorias para cada equipo del grupo)
    
    - Args:

      - year: año del mundial del que interesa recuperar los grupos

    - Returns:

      - dic_0: diccionario contenedor del nombre de cada grupo como valor los valores de rendimiento para cada equipo del grupo, acumulados durante la fase eliminatoria previa al mundial
    """
    df0 = formar_dataset_real("grupos", year)
    df0["pais"] = df0["pais"].map(self.pais_id)
    metrics_to_zero = [col for col in df0.columns if col not in ("pais", "Grupo", "agno")] # PEND (quizas requiera un condicional si (if self.lista_fases.index(self.ind0[0]) < self.lista_fases.index(self.ind0[1]):))
    df0.loc[:, metrics_to_zero] = 0
    dic_0 = {des0: df0[df0["Grupo"] == des0] for des0 in set(df0["Grupo"].tolist())}
    return dic_0

  def create_features(self, dic_ag):
    """
    ¿QUE HACE?
    Arroja las secuencias necesarias tanto de la data historia del mundial de interes, como del fixture del mismo, apuntando a adaptarse al formato del torneo que tuvo ese mundial en particular
    
    Arrojará especificamente el historico de partidos de la fase final del mundial de interés (extraídos del df historico completo o acotados solo a la fase final segun el valor del atributo obj0), la secuencia definida por el usuario para hacer inferencia, la fase de grupos para el mundial de interés, la estructura de fixture de fase final, todo en ese orden. La idea de arrojar estos datos, es permitir la inferencia en el orden marcado por la estructura que tendria el mundial de interés para el usuario, presente desde el propio fixture de ese mundial

    Puede tambien arrojar solo el historico de partidos completo del mundial, el historico de partidos de la fase de grupos, el fixture de la fase final

    ¿COMO LO HACE?
    0-) Se iteran todos los grupos del mundial presentes en (dic_ag) y se actualiza en la columna "pais" al indice de este segun el atributo (pais_id)

    1-) Se crea una variable (df01) y se le asigna una copia del contenido del mundial de interés en el atributo (df2). Se crea una variable (year0) que almacena como una serie, la columna del año o "year" de (df2). Con (pais_id) se actualizan los nombres de los equipos que juegan como home y away en el df (df01)

    2-) (data_limpia_0, data_limpia_1) se almacenan series correspondientes a la division entre goles del equipo home y goles del equipo away para todos los partidos registrados en (df2), mediante el uso del metodo (ajuste_score_partidos)

    3-) Se crea un atributo df (df1) para guardar el nombre id de los equipos home y away, los puntajes guardados en (data_limpia_0, data_limpia_1) aqui etiquetados como "score_0, score_1", el año de juego del mundial de itneres y la especificacion de indices respecto al tamaño del mundial presente en (df01)

    4-) Se actualiza en (df1) todas las metricas de rendimiento desarrolladas por cada equipo home y away, during the fase de clasificaciones, usando la funcion "agregar_features". Se crea una variable (df02) con una copia del fixture del mundial actual

    5-) Mediante una referencia index con negacion booleana (~) se obtiene toda la seccion del (df02) que contiene la fase de grupos de el mundial de interes, luego se obtiene el tamaño de esta fase y se almacena en (longitud_fase_grupos). Mediante este valor numerico se obtiene un slicing con todos los partidos de fase de grupos y se actualiza con estos datos, la variable (df02)

    6-) Se crean 2 condicionales

        6.0) Si el año del fixture del mundial actual es 1934 o 1938. Se actualizan los nombres de los equipos home y away a su indice segun el atributo (pais_id) en la variable (df02)

        6.1) Si el año del fixture del mundial actual no es 1934 o 1938. Si el año del fixture del mundial actual es 1934 o 1938. Se actualizan los nombres de los equipos home y away a su indice segun el atributo (pais_id) en la variable (df02). En unas variables (data_limpia_2, data_limpia_3) se almacenan series correspondientes a la division entre goles del equipo home y goles del equipo away para todos los partidos registrados en la fase de grupos de (df3), mediante el uso del metodo (ajuste_score_partidos)

          6.1.0) Se actualiza (df02) a un df con los nombres de los equipos home y away que juegan la fase de grupos del mundial de interes, los puntajes de cada partido "score_0" y "score_1", el año de ese mundial y los indices del df usando el tamaño del df (df02)

    7-) Se actualizan las metricas de rendimiento del df (df02) dentro de esa misma variable, usando la funcion "agregar_features". Se crean unas variables (df2, df3) que seran una copia de la fase de grupos del fixture (df02) y un slice de la fase final del fixture (df02) 
    
    8-) Se crean 2 condicionales 

      8.0) Si el atributo (obj0) indica que se tomarán datos del usuario. Se actualiza el df (df1) a los partidos de fase final. En el atributo (df5) con el df enviado por el usuario se actualizan los nombres de los equipos home y away a su forma id. Se actualiza (df1) a una tupla con la fase final no fixture en (df1) y el df del usuario en el atributo (df5). Se retorna: (df1) con fase final y df del usuario, (df2) con la fase de grupos del mundial y (df3) con la fase final del fixture

      8.1) Se retornan: (df1) se deja como contenedor de los partidos del mundial, df2 con el contenido de la fase de grupos en el fixture y df3 con el contenido de la fase final en el fixture

    Args:

        - dic_ag: diccionario con las metricas de rendimiento de cada equipo segun lo jugado en la fase de clasificaciones

    Returns: 

        - CASO 1: (df1, df2, df3). (fase final del mundial jugado, df consulta del usuario), (fase de grupos del fixture), (fase final del fixture) 

        - CASO 2: (df1, df2, df3). (todos los partidos con formato correcto del mundial jugado), (fase de grupos del fixture), (fase final del fixture)
    """
    for des0 in dic_ag:
      dic_ag[des0].pais = dic_ag[des0].pais.map(self.pais_id)

    df01 = self.df2.copy()
    year0 = self.df2.agno
    df01.home = df01.home.map(self.pais_id)
    df01.away = df01.away.map(self.pais_id)
    data_limpia_0 = pd.Series(df01["score"]).map(lambda x: self.ajuste_score_partidos(x, {"(": r"", ")": r"", ":": r",", " ": r","})).to_list()
    data_limpia_1 = pd.Series(df01["score"]).map(lambda x: self.ajuste_score_partidos(x, {"(": r"", ")": r"", ":": r",", " ": r","}, ind=1)).to_list()

    df1 = pd.DataFrame({"home": df01.home.to_list(), "score_0": data_limpia_0, "score_1": data_limpia_1, "away": df01.away.to_list(), "agno": year0.to_list()}, index=[list(range(len(df01)))])
    
    df1 = agregar_features(df1, dic_ag)

    df02 = self.df3.copy()
    longitud_fase_grupos = df02[~df02["away"].map(lambda x: any(term in x for term in ["Match", "Grupo"]))].shape[0]
    df02 = self.df3.iloc[:longitud_fase_grupos, :]
    
    if df02.year.unique().tolist()[0] in (1934, 1938):
      df02.home = df02.home.map(self.pais_id)
      df02.away = df02.away.map(self.pais_id)
      df02 = df02.rename(columns={"year": "agno"}) # "agregar_features" lee el año como "agno"
      
    elif df02.year.unique().tolist()[0] not in (1934, 1938):
      df02.home = df02.home.map(self.pais_id)
      df02.away = df02.away.map(self.pais_id)
      data_limpia_2 = pd.Series(df02["score"]).map(lambda x: self.ajuste_score_partidos(x, {"(": r"", ")": r"", ":": r",", " ": r","}))
      data_limpia_3 = pd.Series(df02["score"]).map(lambda x: self.ajuste_score_partidos(x, {"(": r"", ")": r"", ":": r",", " ": r","}, ind=1))

      df02 = pd.DataFrame({"home": df02.home.to_list(), "score_0": data_limpia_2, "score_1": data_limpia_3, "away": df02.away.to_list(), "agno": df02.year.to_list()}, index=list(range(len(df02))))

    df02 = agregar_features(df02, dic_ag, ind=0)
    df2, df3 = df02.copy(), self.df3.iloc[longitud_fase_grupos:, :]
    
    if self.obj0 != "-":
      df1 = df1.iloc[longitud_fase_grupos:, :]

      self.df5.home = self.df5.home.map(self.pais_id)
      self.df5.away = self.df5.away.map(self.pais_id)

      df1 = (df1, self.df5)
      return df1, df2, df3

    else:
      return df1, df2, df3

# =====================================================================
#  EMPAREJAMIENTO POR MEJORES TERCEROS (MUNDIALES 1986, 1990 Y 1994)
# =====================================================================

_POSICIONES = {"1": "Winner", "2": "Runners-up", "3": "Third-p"}

@dataclass(frozen=True)
class FormatoTorneo:
  """
  ¿QUE HACE?
  Datos del formato de un mundial con mejores terceros (lo que cambia de un año a otro son datos, no codigo): grupos, que ganadores de grupo enfrentan a un mejor tercero, cuantos terceros pasan, el orden de juego del 1er round de la fase final (llave) y los cruces oficiales FIFA

  Atributos:

    - grupos: letras de los grupos ("ABCDEF" en 1986-1994, "ABCDEFGHIJKL" en 2026)

    - ganadores_con_tercero: letras de los ganadores que enfrentan a un mejor tercero ("ABCD"; en 2026 "ABDEGIKL")

    - n_terceros: cuantos terceros pasan (4; en 2026, 8)

    - llave: partidos del 1er round en notacion "<posicion><grupo>-<posicion><grupo>": "2E" = Runners-up Grupo E, "1A-3" = el ganador del grupo A enfrenta al mejor tercero que le asignen los cruces

    - cruces: mapeo frozenset(grupos con mejor tercero) -> {ganador: grupo del 3ro rival}
  """
  grupos: str
  ganadores_con_tercero: str
  n_terceros: int
  llave: tuple
  cruces: MappingProxyType

  @property
  def partidos_llave(self):
    """1ros y 2dos de cada grupo mas los mejores terceros, de a dos por partido"""
    return (2 * len(self.grupos) + self.n_terceros) // 2

  @property
  def combinaciones(self):
    return frozenset(frozenset(c) for c in combinations(self.grupos, self.n_terceros))

  @property
  def clasificados_fijos(self):
    return frozenset(p + g for p in "12" for g in self.grupos)

  @property
  def patron_partido(self):
    letra = f"[{self.grupos[0]}-{self.grupos[-1]}]"
    return re.compile(rf"[12]{letra}-(?:[12]{letra}|3)")

def _cruces_desde_texto(ganadores, tabla):
  """{"ABCD": "CDAB", ...} -> mapeo frozenset(grupos) -> {ganador: grupo del 3ro rival}"""
  return MappingProxyType({frozenset(grupos): MappingProxyType(dict(zip(ganadores, rivales))) for grupos, rivales in tabla.items()})

def _cruces_2026():
  """Las 495 combinaciones del reglamento FIFA 2026 (data/partidos/terceros_2026.csv, de etl/construye_mundial_2026.py)"""
  ruta = settings.BASE_DIR.parent / "data" / "partidos" / "terceros_2026.csv"
  return dict(pd.read_csv(ruta, dtype=str).itertuples(index=False)) if ruta.exists() else {}

# Reglamento FIFA (el mismo en 1986, 1990 y 1994): segun que 4 grupos aporten mejor tercero,
# de que grupo es el tercero que enfrenta a cada ganador de los grupos A, B, C y D
_CRUCES_1986_1994 = {
  "ABCD": "CDAB", "ABCE": "CABE", "ABCF": "CABF", "ABDE": "DABE", "ABDF": "DABF",
  "ABEF": "EABF", "ACDE": "CDAE", "ACDF": "CDAF", "ACEF": "CAFE", "ADEF": "DAFE",
  "BCDE": "CDBE", "BCDF": "CDBF", "BCEF": "ECBF", "BDEF": "EDBF", "CDEF": "CDFE",
}
_LLAVES = {
  1986: ("1A-3", "1F-2E", "2A-2C", "1B-3", "1C-3", "2F-2B", "1E-2D", "1D-3"),
  1990: ("1A-3", "1E-2D", "2F-2B", "1B-3", "2A-2C", "1C-3", "1D-3", "1F-2E"),
  1994: ("2A-2C", "1A-3", "1B-3", "1F-2E", "1C-3", "1E-2D", "1D-3", "2F-2B"),
  # dieciseisavos 2026, Match 73 a 88
  2026: ("2A-2B", "1E-3", "1F-2C", "1C-2F", "1I-3", "2E-2I", "1A-3", "1L-3",
         "1D-3", "1G-3", "2K-2L", "1H-2J", "1B-3", "1J-2H", "1K-3", "2D-2G"),
}
FORMATOS = MappingProxyType({
  **{agno: FormatoTorneo("ABCDEF", "ABCD", 4, _LLAVES[agno], _cruces_desde_texto("ABCD", _CRUCES_1986_1994)) for agno in (1986, 1990, 1994)},
  2026: FormatoTorneo("ABCDEFGHIJKL", "ABDEGIKL", 8, _LLAVES[2026], _cruces_desde_texto("ABDEGIKL", _cruces_2026())),
})
# mundiales cuyo reglamento se puede editar desde el frontend (el editor dibuja 6 grupos)
EDITABLES_EN_FRONTEND = (1986, 1990, 1994)

def _etiqueta(codigo):
  """Traduce un codigo de llave ("2E") a la etiqueta que entiende "camb_grp_elm_smpl" ("Runners-up Grupo E")"""
  return f"{_POSICIONES[codigo[0]]} Grupo {codigo[1]}"

class ReglamentoInvalido(ValueError):
  """Error con la lista completa de problemas del reglamento, para mostrarlos todos de una vez en el frontend"""
  def __init__(self, errores):
    self.errores = list(errores)
    super().__init__("Reglamento de mejores terceros invalido:\n- " + "\n- ".join(self.errores))

@dataclass(frozen=True)
class ReglamentoTerceros:
  """
  ¿QUE HACE?
  Representa el reglamento de mejores terceros de un mundial: el orden de los partidos del 1er round de la fase final (llave) y, para cada combinacion de grupos con mejor tercero, el rival de cada ganador que enfrenta a un tercero (cruces). Es inmutable y se valida al crearse contra el (FormatoTorneo) de su año, venga del reglamento FIFA o del frontend

  ¿COMO LO HACE?
  0-) "__post_init__" junta todos los errores de la llave y de los cruces, y si hay alguno lanza (ReglamentoInvalido)

  1-) La llave es valida si tiene los partidos del formato (8 en 1986-1994, 16 en 2026) con formato "1A-2C" / "1A-3", cada 1ro y 2do de grupo aparece exactamente una vez, solo los ganadores del formato enfrentan a un mejor tercero y no hay cruces entre equipos del mismo grupo

  2-) Los cruces son validos si estan todas las combinaciones (15 en 1986-1994, 495 en 2026), cada combinacion reparte sus grupos entre los ganadores con tercero y ningun ganador enfrenta al tercero de su propio grupo

  Atributos:

    - agno: año del mundial (una clave de FORMATOS: 1986, 1990, 1994 o 2026)

    - llave: tupla de partidos en notacion "<posicion><grupo>-<posicion><grupo>"

    - cruces: mapeo frozenset(grupos) -> {ganador: grupo del 3ro rival}
  """
  agno: int
  llave: tuple
  cruces: MappingProxyType

  @property
  def formato(self):
    return FORMATOS[self.agno]

  def __post_init__(self):
    if self.agno not in FORMATOS:
      raise ReglamentoInvalido([f"el mundial {self.agno} no usa el formato de mejores terceros (solo {', '.join(map(str, FORMATOS))})"])
    errores = [*self._errores_llave(), *self._errores_cruces()]
    if errores:
      raise ReglamentoInvalido(errores)

  def _errores_llave(self):
    f = self.formato
    if len(self.llave) != f.partidos_llave:
      yield f"la llave debe tener {f.partidos_llave} partidos y tiene {len(self.llave)}"
    malos = [p for p in self.llave if not (isinstance(p, str) and f.patron_partido.fullmatch(p))]
    if malos:
      yield f"partidos con formato invalido: {malos}"
      return
    cruces = [p.split("-") for p in self.llave]
    fijos = [c for cruce in cruces for c in cruce if c != "3"]
    if repetidos := sorted({c for c in fijos if fijos.count(c) > 1}):
      yield f"equipos repetidos en la llave: {repetidos}"
    if faltan := sorted(f.clasificados_fijos - set(fijos)):
      yield f"equipos sin partido en la llave: {faltan}"
    contra_tercero = {local for local, visita in cruces if visita == "3"}
    if contra_tercero != {"1" + g for g in f.ganadores_con_tercero}:
      esperados = ", ".join("1" + g for g in f.ganadores_con_tercero)
      yield f"solo {esperados} enfrentan a un mejor tercero (recibido: {sorted(contra_tercero)})"
    if mismos := [f"{l}-{v}" for l, v in cruces if v != "3" and l[1] == v[1]]:
      yield f"cruces entre equipos del mismo grupo: {mismos}"

  def _errores_cruces(self):
    f = self.formato
    if set(self.cruces) != f.combinaciones:
      yield f"deben definirse los cruces para las {len(f.combinaciones)} combinaciones de {f.n_terceros} grupos"
      return
    for grupos, rival_de in self.cruces.items():
      nombre = "".join(sorted(grupos))
      if sorted(rival_de) != list(f.ganadores_con_tercero) or sorted(rival_de.values()) != sorted(grupos):
        yield f"{nombre}: cada grupo debe aportar exactamente un rival a {', '.join('1' + g for g in f.ganadores_con_tercero)}"
      elif mismos := [g for g, rival in rival_de.items() if g == rival]:
        yield f"{nombre}: el ganador de {mismos} no puede enfrentar al tercero de su propio grupo"

  @classmethod
  def oficial(cls, agno):
    """Reglamento FIFA del año (agno)"""
    formato = FORMATOS.get(agno)
    return cls(agno, formato.llave if formato else (), formato.cruces if formato else MappingProxyType({}))

  @classmethod
  def desde_json(cls, agno, datos):
    """Construye (y valida) el reglamento enviado por el frontend: {"llave": [...], "cruces": {"ABCD": "CDAB", ...}}"""
    ganadores = FORMATOS[agno].ganadores_con_tercero if agno in FORMATOS else ""
    try:
      llave = tuple(datos["llave"])
      cruces = _cruces_desde_texto(ganadores, datos["cruces"])
    except (KeyError, TypeError, AttributeError):
      raise ReglamentoInvalido(['se esperaba {"llave": [partidos], "cruces": {"ABCD": "CDAB", ...}}']) from None
    return cls(agno, llave, cruces)

  def a_json(self):
    """Inverso de "desde_json", para precargar el editor del frontend"""
    return {"llave": list(self.llave),
            "cruces": {"".join(sorted(grupos)): "".join(rival_de[g] for g in self.formato.ganadores_con_tercero)
                       for grupos, rival_de in sorted(self.cruces.items(), key=lambda kv: sorted(kv[0]))}}

def reglamentos_oficiales():
  """Diccionario {agno: reglamento FIFA en formato JSON} para el editor del template (los mundiales de EDITABLES_EN_FRONTEND)"""
  return {agno: ReglamentoTerceros.oficial(agno).a_json() for agno in EDITABLES_EN_FRONTEND}

def tabla_terceros(dict_table, df_grupos=None):
  """
  ¿QUE HACE?
  Genera un df con el 3er lugar de cada grupo ordenado segun el criterio FIFA: puntos, diferencia de gol y goles a favor. La letra del grupo se usa como ultimo desempate para que el resultado sea determinista

  ¿COMO LO HACE?
  0-) Se toma la fila de indice 2 (3er lugar) de cada tabla de (dict_table), ya ordenada por "fase_de_grupos"

  1-) Si se entrega (df_grupos), se acumulan los goles a favor y en contra de cada equipo como home y away, y se agregan las columnas "DG" y "GF" a los terceros; si no, valen 0

  2-) Se ordena de mejor a peor tercero

  Args:

    - dict_table: diccionario de salida de "fase_de_grupos" con clave "Grupo X" y valor el df (pais, Pts) ordenado

    - df_grupos: df de la fase de grupos ya jugada con columnas "home", "away", "score_0" y "score_1" (opcional, para desempatar)

  Returns:

    - df con columnas "Grupo", "pais", "Pts", "DG", "GF" ordenado de mejor a peor tercero
  """
  terceros = pd.DataFrame(
    [(grupo[-1], tabla["pais"].iloc[2], tabla["Pts"].iloc[2]) for grupo, tabla in dict_table.items()],
    columns=["Grupo", "pais", "Pts"]).astype({"Pts": float})

  if df_grupos is not None:
    columnas = ["pais", "GF", "GC"]
    goles = pd.concat([df_grupos[["home", "score_0", "score_1"]].set_axis(columnas, axis=1),
                       df_grupos[["away", "score_1", "score_0"]].set_axis(columnas, axis=1)])
    goles = goles.astype({"GF": float, "GC": float}).groupby("pais").sum()
    terceros = terceros.join(goles.assign(DG=goles.GF - goles.GC)[["DG", "GF"]], on="pais")
  terceros = terceros.reindex(columns=["Grupo", "pais", "Pts", "DG", "GF"]).fillna({"DG": 0, "GF": 0})

  return terceros.sort_values(["Pts", "DG", "GF", "Grupo"], ascending=[False, False, False, True], ignore_index=True)

def con_reglamento_terceros(func):
  """
  ¿QUE HACE?
  Decorador que entrega a la funcion decorada el reglamento de mejores terceros a usar: el personalizado si quien llama lo pasa en (personalizado), o el oficial FIFA de (agno) si no. La funcion decorada solo alinea equipos con las tablas de grupos ya jugadas, sin saber de donde salio el reglamento
  """
  @wraps(func)
  def envoltura(dict_table, agno, *args, personalizado=None, **kwargs):
    reglamento = personalizado if personalizado is not None else ReglamentoTerceros.oficial(agno)
    if reglamento.agno != agno:
      raise ValueError(f"el reglamento es del mundial {reglamento.agno} y se pidio el {agno}")
    return func(dict_table, reglamento, *args, **kwargs)
  return envoltura

@con_reglamento_terceros
def alinear_llave_terceros(dict_table, reglamento, df_grupos=None):
  """
  ¿QUE HACE?
  Genera el df (home, away) del 1er round de la fase final segun los mejores terceros (4 en 1986-1994, 8 en 2026) de las tablas de grupos ya jugadas y el (reglamento) que inyecta "con_reglamento_terceros". Quien llama pasa (dict_table, agno, df_grupos=..., personalizado=...)

  ¿COMO LO HACE?
  0-) Se obtienen los mejores terceros del formato con "tabla_terceros"

  1-) Se busca en los cruces del (reglamento) el rival de cada ganador con tercero para esa combinacion de grupos

  2-) Se recorre la llave del (reglamento) traduciendo cada codigo a etiqueta; los "3" se reemplazan por el tercero asignado al ganador local

  Args:

    - dict_table: salida de "fase_de_grupos"

    - reglamento: (ReglamentoTerceros) inyectado por el decorador; quien llama pasa el año (int) y opcionalmente (personalizado)

    - df_grupos: df de la fase de grupos jugada, para desempatar por diferencia de gol y goles a favor

  Returns:

    - df con columnas "home" y "away" en formato "Winner Grupo A" / "Third-p Grupo C", con los mejores terceros en attrs["mejores_terceros"]
  """
  terceros = tabla_terceros(dict_table, df_grupos).iloc[:reglamento.formato.n_terceros]
  rival_de = reglamento.cruces[frozenset(terceros.Grupo)]
  partidos = []
  for local, visita in (partido.split("-") for partido in reglamento.llave):
    if visita == "3":
      visita += rival_de[local[1]]
    partidos.append((_etiqueta(local), _etiqueta(visita)))
  llave = pd.DataFrame(partidos, columns=["home", "away"])
  llave.attrs["mejores_terceros"] = terceros
  return llave

class func_prediccion_orden(normali):
  def __init__(self, ind_partidos, ind_fixtures, df_prediccion_usuario, clsif, obj0="-", evitar="-", embedding_dim=3, ind0=[None, None], n_partidos=0, drop_metricas=DROP_METRICS, metricas_opcionales=OPTIONAL_METRICS):
    """
    ¿QUE HACE?
    Inicializa la clase (func_prediccion_orden), que hereda de (normali), llamando al constructor del padre y definiendo los atributos adicionales necesarios para gestionar el flujo de predicciones secuenciales entre fases del mundial: la lista ordenada de fases, el diccionario de resultados por partido, el diccionario de fuerzas de equipos, el indicador de fase actual y objetivo, y el contador de partidos procesados

    ¿COMO LO HACE?
    0-) Se llama al constructor de la clase padre (normali) pasandole todos los parametros recibidos para inicializar los atributos heredados de esa clase

    1-) Se define el atributo lista (self.lista_fases) con los nombres de las fases del mundial en el orden en que se juegan, para su uso como referencia de comparacion entre la fase actual y la fase objetivo de la inferencia

    2-) Se inicializa el atributo diccionario (self.dic_matches) en None, para su posterior llenado con los resultados (ganador y perdedor) de cada partido procesado

    3-) Se inicializa el atributo diccionario (self.dic_fuerza) vacio, para su posterior llenado con los promedios de goles de cada equipo como local y visitante

    4-) Se almacena en el atributo (self.ind0) el parametro (ind0), que es una lista de 2 elementos indicando respectivamente la fase actual de inferencia (ind0[0]) y la fase objetivo marcada por el usuario como inicio de prediccion (ind0[1])

    5-) Se almacena en el atributo (self.n_partidos) el parametro (n_partidos), que lleva la cuenta acumulada del numero total de partidos procesados

    Args:

      - ind_partidos: valor de tipo str que indica a la funcion "formar_dataset_real" que tabla de la DB debe consultar

      - ind_fixtures: valor de tipo str con el año del mundial de interés para indicar la búsqueda correcta del fixture en la DB

      - df_prediccion_usuario: df de consulta del usuario con los partidos y puntajes editados para hacer inferencias personalizadas

      - clsif: df contenedor de partidos de clasificacion previa al mundial de interés

      - obj0: cadena que indica si el primer valor de retorno de (create_features) incluirá todos los partidos del mundial o solo la fase final separada del df de consulta del usuario

      - evitar: parametro reservado para uso futuro

      - embedding_dim: valor indicador del tamaño de los vectores embedding para cada equipo

      - ind0: lista de 2 elementos indicando la fase actual de inferencia (ind0[0]) y la fase objetivo marcada por el usuario (ind0[1])

      - n_partidos: numero inicial de partidos ya procesados, generalmente 0 al inicio

      - drop_metricas: metricas base que el usuario descarta; llegan neutralizadas al modelo (se guardan en self.drop_metricas)

      - metricas_opcionales: metricas opcionales que el usuario acepta; las no aceptadas llegan neutralizadas al modelo (se guardan en self.metricas_opcionales)

    Return:

      - No retorna valor, solo inicializa los atributos de la instancia
    """
    normali.__init__(self, ind_partidos, ind_fixtures, df_prediccion_usuario, clsif, obj0=obj0, evitar=evitar, embedding_dim=embedding_dim)

    self.lista_fases = ["grupo", "doce", "dieciseisavos", "knockout", "quarter", "semi", "third", "final", "Fase Final"]
    self.dic_matches = {}
    self.dic_fuerza = {}
    self.ind0 = ind0
    self.n_partidos = n_partidos
    self.drop_metricas = tuple(drop_metricas)
    self.metricas_opcionales = tuple(metricas_opcionales)

  def contexto_simulacion(self, tabla):
    """
    ¿QUE HACE?
    Agrega a la (tabla) de entrada al predictor (ids home_id y away_id de la app) lo que el LSTM recibe como contexto, con las mismas columnas que "construir_dataset": nombres de los equipos, categoria ("mundial"), neutral (0 si juega un anfitrion), local_0 / local_1 (el anfitrion juega en su cancha) y si el historico de metricas abstractas de cada equipo tiene penales o tiros reales (pk_real, sot_real). Agrega tambien el contexto de EXTRAS que la app sabe calcular (dataset.EXTRAS_EN_APP): elo_0/1 al inicio del mundial, local_continental_0/1 y fase_final. Usa los atributos (anfitriones), (con_datos_reales) y (primera_fase) que deja "consulta_general"; sin ellos, todo partido es neutral y sin datos reales
    """
    anfitriones = getattr(self, "anfitriones", set())
    reales = getattr(self, "con_datos_reales", {})
    tabla = tabla.assign(
      home=tabla["home_id"].map(self.id_pais), away=tabla["away_id"].map(self.id_pais), categoria="mundial",
      neutral=[0 if h in anfitriones or a in anfitriones else 1 for h, a in zip(tabla["home_id"], tabla["away_id"])],
      local_0=tabla["home_id"].isin(anfitriones).astype(int), local_1=tabla["away_id"].isin(anfitriones).astype(int),
    )
    for marca in ("pk_real", "sot_real"):
      for i, lado in ((0, "home_id"), (1, "away_id")):
        tabla[f"{marca}_{i}"] = tabla[lado].isin(reales.get(marca, set())).astype(int)
    # contexto adicional (EXTRAS de dataset.py) igual que "agregar_contexto": Elo al inicio del mundial, si ya paso la
    # primera fase y si el equipo juega en la confederacion del anfitrion
    from .contexto import ELO_INICIAL, contexto_desde_bd
    mundial = contexto_desde_bd().get(self.obj0, {}) if isinstance(self.obj0, int) else {}
    elo, confed = mundial.get("elo", {}), mundial.get("confederacion", {})
    for i, lado in ((0, "home"), (1, "away")):
      tabla[f"elo_{i}"] = tabla[lado].map(lambda p: elo.get(p, ELO_INICIAL))
      tabla[f"local_continental_{i}"] = tabla[lado].map(lambda p: int(confed.get(p) in mundial.get("confederacion_anfitrion", set())))
    tabla["fase_final"] = int(self.ind0[0] != getattr(self, "primera_fase", "grupo"))
    return tabla

  def intsc_prob_goles(self, X, model, alterno=0):
    """
    ¿QUE HACE?
    Juega los partidos de la entrada (X): con el predictor (alterno=0) o con sus resultados reales (alterno=1, fases anteriores a la objetivo), y entrega para cada uno los goles y los puntos de home y away

    ¿COMO LO HACE?
    0-) alterno = 0: (X) se convierte con "lote_desde_entrada" y "contexto_simulacion" en el (Lote) que recibe cualquier predictor (ver predictores.py), que devuelve los goles esperados (n, 2)

      0.0) "resultado_y_marcador" (evaluacion.py) decide el resultado (empate si las probabilidades de ganar de los dos estan a menos del umbral de empate del modelo) y el marcador mas probable dentro de ese resultado: los mismos que mide la evaluacion

      0.1) Puntos segun el resultado (3 por victoria, 2 hasta 1990, 1 por empate). Si en una eliminatoria el marcador es empate, pasa el que tenia mas probabilidad de ganar: se guarda en el atributo (ganador_empate) para "jugar"

    1-) alterno = 1: (X) es el df de la fase (o la Serie de un partido): para cada pareja (home, away) se toma con "marcador_real" el resultado que tuvo en el mundial real (atributo (resultados_reales), lo arma "consulta_general") y se entrega con los puntos que da ese marcador

    Args:

      - X: si (alterno) es 0, array de "calculo_metricas_0" con las entradas (home, away) de cada partido de la fase; si es 1, df (o Serie) de la fase con las columnas "home" y "away" (ids)

      - model: predictor (interfaz "Predictor" de predictores.py) listo para hacer inferencia

      - alterno: pametro predefinid a 0 para decidir si el metodo realiza inferencia con el modelo o arroja los valores default (esto se aplica cuando no se ha llegado a la fase objetivo marcada por el usuario para iniciar la inferencia)

    Return:
      - yield((y_home, points_home), (y_away, points_away)): un iterable con unas tuplas de 2 valores; el numero de goles default o predicho por el modelo para home y away respectivamente y la probabilidad de anotacion del home y away respectivamente
    """
    if alterno == 0:
      # X viene de "calculo_metricas_0" (esquema fijo); el predictor devuelve (n, 2) goles esperados de home y away
      from .predictores import Lote, lote_desde_entrada
      from .evaluacion import resultado_y_marcador
      entrada = Lote(self.contexto_simulacion(lote_desde_entrada(X).tabla))
      Y0 = model.predecir(entrada)
      resultado, marcador, probs = resultado_y_marcador(Y0, getattr(model, "config", {}).get("umbral_empate", 0.0),
                                                        model.probabilidades(entrada) if hasattr(model, "probabilidades") else None)
      victoria = 2 if isinstance(self.obj0, int) and self.obj0 < 1994 else 3
      self.__dict__.setdefault("ganador_empate", {})
      for r, (g0, g1), p, fila in zip(resultado, marcador, probs, np.atleast_2d(X)):
        self.ganador_empate[(fila[0], fila[COLS_MODELO_POR_EQUIPO])] = "home" if p[0] >= p[2] else "away"
        puntos = (victoria, 0) if r == 1 else (0, victoria) if r == -1 else (1, 1)
        yield ((float(g0), puntos[0]), (float(g1), puntos[1]))
      return

    elif alterno == 1:
      # X es el df de la fase (o la Serie de un partido): el marcador se busca por pareja en el mundial real
      filas = X.to_frame().T if isinstance(X, pd.Series) else X
      Y0 = np.array([marcador_real(self.resultados_reales, h, a) for h, a in zip(filas["home"], filas["away"])], dtype=float)
      # resultado real: los puntos salen del marcador, no de la interseccion de Poisson (2 por victoria hasta 1990)
      victoria = 2 if isinstance(self.obj0, int) and self.obj0 < 1994 else 3
      for y_home, y_away in Y0:
        p_hm, p_aw = (victoria, 0) if y_home > y_away else (0, victoria) if y_away > y_home else (1, 1)
        yield ((y_home, p_hm), (y_away, p_aw))
      return


  def registrar_partido(self, partido, ganador, perdedor, empate):
    """
    ¿QUE HACE?
    Guarda en (dic_matches) el ganador y el perdedor de (partido). Si es la repeticion de un partido anterior que termino empatado (1934 y 1938 desempataban jugando de nuevo, sin penales), el ganador de la repeticion pasa a ser tambien el del partido original, que es al que se refieren las rondas siguientes ("Winner Match 12")
    """
    self.dic_matches[partido]["ganador"], self.dic_matches[partido]["perdedor"] = ganador, perdedor
    pendientes = self.__dict__.setdefault("empates_por_repetir", {})
    pareja = frozenset((ganador, perdedor))
    original = pendientes.pop(pareja, None)
    if original is not None:
      self.dic_matches[original]["ganador"], self.dic_matches[original]["perdedor"] = ganador, perdedor
    elif empate:
      pendientes[pareja] = partido

  def ganador_perdedor(self, paso):
    """
    ¿QUE HACE?
    Deja en el atributo diccionario "self.dic_matches" otros diccionarios contenedores del ganador y perdedor por partido

    ¿COMO LO HACE?
    0-) Se define una variable (value_in) igual al numero de partidos sobre los que se ha hecho ya la inferencia, referenciando al atributo (n_partidos). Se define una variable (value_fin) igual al numero de partidos con los que se ha trabajado mas el numero de partidos se va a inferir con el modelo ahora "la nueva fase".

    1-) Se actualiza el atributo (dic_matches) con estructura (clave: partido (numero iterado del partido de la fase actual)), valor: (clave: ganador-perdedor, None-None)

    2-) Se actualiza el atributo (n_partidos) a (value_fin)

    - Args:
    
      - paso: numero que indica el tamaño de paso para dar nombre a los partidos del siguiente 
              encuentro

    - Return:

      - no retorna mas que un cambio de diccionario en "self.dic_matches" y actualiza el
        atributo "self.n_partidos" con el numero de indicador del primer partido del siguiente encuentro
    """
    value_in = self.n_partidos
    value_fin = value_in + paso
    # se acumula: el 3er lugar y la final buscan "Loser/Winner Match N" de rondas anteriores
    self.dic_matches.update({f"Match {col0}": dict(ganador=None, perdedor=None) for col0 in range(value_in, value_fin)})
    self.n_partidos = value_fin

  def fase_de_grupos(self, dic_t, df_fixture_, model):
    """
    ¿QUE HACE?
    Ejecuta la inferencia con el modelo para los partidos de la fase de grupos, usandose una funcion específica para esta fase porque el modo de eliminacion y seleccion dado aqui es diferente al de la fase final que cumple con los parametros de un torneo de eliminacion simple, mientras que la fase de grupos es un torneo tipo round robin
    
    Para hacerlo, se vale de la seccion del fixture que incluye los partidos de esta fase y del diccionario que incluye a los df con la informacion del rendimiento para cada equipo de cada grupo

    ¿COMO LO HACE?
    1-) Se crea una variable array de recepcion de puntajes por partido (x)

    2-) Se iteran las clave-valor del diccionario (dic_t). Se obtienen los paises de la columna "pais" del df del grupo iterado y se dejan en una variable (paises). Se define un df (df_fix_group_n) que será seccion del (df_fixture_) con los partidos en que jugaran, como home o away, los equipos almacenados en la variable (paises). se definen y agregan las tasas de goles anotados y recibidos mas la diferencia entre ambos, para los equipos home y away. Se define una variable array copia de (df_fix_group_n) con el nombre (X)

    3-) Se tienen 2 condicionales

      3.0) Si la fase actual "la de grupos" está antes de la fase objetivo para una prediccion. Se almacena en una variable (puntos) el resultado de aplicar intsc_prob_goles a (X) con el parámetro "alterno" seteado como 1 "lo que indica que no se predice y solo se toma el historico de resultados para los partidos previos a la fase desde la que se quiere predecir" lo que arrojará tanto los goles por partido como el puntaje obtenido para cada equipo home y away

      3.1) Si la fase de grupos es la fase objetivo entonces. Se define una variable (puntos) como el resultado de intsc_prob_goles, con el parametro alterno inicializado como 0, "indicando que se ejecutará la inferencia con el modelo" lo que arrojará la prediccion de goles y el puntaje obtenido para cada equipo home y away

    4-) Se iteran en conjunto los puntajes para cada partido y los partidos sobre los que se infirió. Se toman los nombres de los equipos home y away. Para el home y el away del partido, se guarda como una fila en el array de recepcion (x) los goles predichos o arrojados por intsc_prob_goles. Se actualiza el puntaje con lo arrojado por intsc_prob_goles, dentro del df que referencia el (dic_t) para el grupo y los equipos home y away iterados

    5-) En el df que referencia el grupo iterado dentro de (dic_t) se reordenan las filas del mayor al menor puntaje. Se deja en ese df solo las columnas de pais y Pts. Se redondean los resultados numericos en el df a 0 puntos decimales

    6-) Los puntajes almacenados en (x) se almacenan en las columnas "score_0" y "score_1" del df_fixture_ de fase de grupos. Se actualizan las metricas de rendimiento mediante la funcion "calculo_metricas_1". Se retorna el (dic_t) actualizado con los nuevos puntajes

    Args:

      - dic_t: diccionario con clave "grupo del mundial" (Grupo X) valor "df con nombres y metricas de rendimiento de cada equipo
      
      - df_fixture_: df con el orden de juego y nombres de cada equipo para la fase de grupos
      
      - model: model para hacer la inferencia

    Returns:

      - dic_t: es el mismo diccionario (dic_t) que entra como parámetro pero con el puntaje de cada equipo actualizado

      - df_fixture_: es el mismo df (df_fixture_) que entra como parámetro pero con los parametros de rendimiento actualizados

    """
    # marcadores por indice del fixture: los grupos se recorren en el orden de (dic_t), no en el del fixture
    marcadores = pd.DataFrame(index=df_fixture_.index, columns=["score_0", "score_1"], dtype=float)

    for group, dfs in dic_t.items():
      paises = dfs['pais'].values
      df_fix_group_n = df_fixture_[(df_fixture_['home'].isin(paises)) & (df_fixture_['away'].isin(paises))]
      if self.lista_fases.index(self.ind0[0]) < self.lista_fases.index(self.ind0[1]):
        # fase anterior a la objetivo: se juega con los resultados reales del df de la fase
        puntos = self.intsc_prob_goles(df_fix_group_n, model, alterno=1)
      else:
        # X se arma con la salida de "calculo_metricas_0" (esquema fijo del modelo y métricas neutralizadas)
        df_modelo = calculo_metricas_0(df_fix_group_n, self.drop_metricas, self.metricas_opcionales)
        X = df_modelo.to_numpy()
        puntos = self.intsc_prob_goles(X, model, alterno=0)

      goles = {pais: [0.0, 0.0] for pais in paises} # pais -> [goles a favor, goles en contra]
      for points_, row in zip(puntos, df_fix_group_n.itertuples()):
        home, away = row.home, row.away
        g0, g1 = points_[0][0], points_[1][0]
        marcadores.loc[row.Index] = [g0, g1]
        dic_t[group].loc[dic_t[group]['pais'] == home, 'Pts'] += points_[0][1]
        dic_t[group].loc[dic_t[group]['pais'] == away, 'Pts'] += points_[1][1]
        for equipo, a_favor, en_contra in ((home, g0, g1), (away, g1, g0)):
          goles[equipo][0] += a_favor
          goles[equipo][1] += en_contra

      # orden FIFA: puntos, diferencia de gol y goles a favor
      dic_t[group]["DG"] = dic_t[group]["pais"].map(lambda p: goles[p][0] - goles[p][1])
      dic_t[group]["GF"] = dic_t[group]["pais"].map(lambda p: goles[p][0])
      dic_t[group] = dic_t[group].sort_values(["Pts", "DG", "GF"], ascending=False).reset_index(drop=True)
      dic_t[group] = dic_t[group][['pais', 'Pts']]
      dic_t[group] = dic_t[group].round(0)

    df_fixture_["score_0"] = marcadores["score_0"]
    df_fixture_["score_1"] = marcadores["score_1"]

    calculo_metricas_1(df_fixture_)

    return dic_t

  def jugar(self, df_fixture_updated, *args):
    """
    ¿QUE HACE?
    Ejecuta la inferencia para los partidos de la fase final, empleandose una funcion completa para abarcar la inferencia solo en esta fase porque el formato de la misma, para eliminacion y seleccion entre cada round, es diferente al que se tiene para la fase de grupos
    
    Para hacerlo recurre a la actualizacion del fixture formateado entregado por parametro y al modelo que ejecuta la inferencia, ademas del registro del ganador y perdedor para cada partido, en el atributo "dic_matches"

    ¿COMO LO HACE?
    0-) Se define un array (Y0) para recepcion de informacion y un string vacio (partido). Se guarda el numero de filas del df de parametro (df_fixture_updated). Se genera el diccionario correspondiente a las victorias y derrotas para los partidos a disputar segun el (df_fixture_updated). Se actualiza el (df_fixture_updated) para que contenga las tasas de goles anotados y recibidos para los home y away, todo dentro de una nueva variable (df_fixture_partidos). Se guarda en una variable (X) un array salido de la copia del (df_fixture_partidos)

    1-) Se tienen 2 condicionales
      
      1.0) cuando el (X) es de dimension 2
        
        1.0.0) Se tienen a su vez, 2 condicionales

          1.0.0.0) Si la fase actual esta antes de la fase objetivo para iniciar la inferencia. Se almacena en la variable (puntos) la impresion de resultados default del fixture para antes de la inferencia

          1.0.0.1) Si la fase actual es la objetivo o despues de esta. Se almacena en la variable (puntos) la inferencia del modelo para el array (X)
      
        1.0.1) Se itera el contenido de la variable (puntos) y las filas del df (df_fixture_updated) al mismo tiempo. Se almacena en las variables (home, away) el nombre de los equipos que jugaron como home y away segun el df (df_fixture_updated). Se almacena en las variables (point_h, point_a) los goles arrojados por el metodo de inferencia "intsc_prob_goles"

        1.0.2) Se tienen 2 condicionales

          1.0.2.0) Si el puntaje del home es mayor que el del away. Se deja la variable (ganador) como "home" y la variable (perdedor) como "away"

          1.0.2.1) Si el puntaje del home es menor que el del away. Se deja la variable (ganador) como "away" y la variable (perdedor) como "home"

        1.0.3) Se va almacenando en la variable (indice) el numero de partido iterado. Se almacena en la variable (partido) el nombre del partido con el formato (partido X). Se almacena en las claves "ganador" y "perdedor" de ese partido en (dic_matches) a las variables (ganador, perdedor). En el array (Y0) se almacena como fila, a los puntajes home y away

        1.0.4) para el df (df_fixture_updated) se actualizan las columans "score_0" y "score_1". Al df (df_fixture_updated) se le actualizan las metricas de rendimiento dado el resultado arrojado por el metodo de inferencia "intsc_prob_goles"

      1.1) cuando el (X) es de dimension 1. Se deja en la variable (índice) la resta entre el nuero de partidos total y el numero de partidos en la fase actual. Se deja en la variable (partido) el nombre del partido actual

      1.2) Se tienen 2 condicionales

        1.2.0) Si la fase actual esta antes de la fase objetivo para iniciar la inferencia. Se almacena en la variable (puntos) la impresion de resultados default del fixture para antes de la inferencia

        1.2.1) Si la fase actual es la objetivo o despues de esta. Se almacena en la variable (puntos) la inferencia del modelo para el array (X)

      1.3) Se itera el contenido de la variable (puntos). Se almacena en unas variables (home, away) el nombre de los equipos que jugaron como home y away en el partido estudiado. Se almacena en unas variables (point_h, point_a) el numero de goles que arroja el metodo de inferencia para los equipos home y away

      1.4) Se tienen 2 condicionales

        1.4.0) Si (point_h) es mayor que (point_a) se deja la variable (ganador) como home y la variable (perdedor) como away
        
        1.4.1) Si (point_h) es menor que (point_a) sw deja la variable (ganador) como away y la variable (perdedor) como home
        
      1.5) Se actualizan las claves "ganador", "perdedor" para el partido actual en (dic_matches). Se actualiza el array (Y0) con el numero de goles arrojado por "intsc_prob_goles" 

      1.6) para el df (df_fixture_updated) se actualizan las columans "score_0" y "score_1". Al df (df_fixture_updated) se le actualizan las metricas de rendimiento dado el resultado arrojado por el metodo de inferencia "intsc_prob_goles"
      
    2-) Se crean unas variables (homes, aways, Winners, Losers) alojadoras de unas listas vacias. Se iteran los partidos de la fase actual (df_fixture_updated). Se va almacenando en la variable (indice) el numero de partido iterado. Se almacena en la variable (partido) el nombre del partido con el formato (partido X). Se actualizan estas listas vacias con los nombres de los equipos home, away, ganador y perdedor del partido actual

    3-) Se crea un df (df_nombres) con los nombres de los equipos que desputan, ganan y pierden los partidos de la fase actuql. Se imprime dicho df (df_nombres)

    4-) Se retorna el df (df_fixture_updated)
    
    Args:

      - df_fixture_updated: df con los partidos de la fase actual
   
      - model: modelo ara hacer inferencia sobre los partidos de la fase actual en (df_fixture_updated)

    Return:

      - df_fixture_updated: df actualizado de la fase actual, con el resultado arrojado por el metodo de inferencia "intsc_prob_goles" y las metricas de rendimiento actualizadas con "calculo_metricas_1"
    """
    if len(args) == 1:
      model = args[0]
    elif len(args) == 2:
      model = args[1]
    else:
      raise TypeError("get_winner() expects the model parameter.")

    Y0 = np.array([])
    partido = ""
    tamagno = df_fixture_updated.shape[0]
    self.ganador_perdedor(tamagno)
    # fase anterior a la objetivo: se juega con los resultados reales (alterno=1)
    rejugada = self.lista_fases.index(self.ind0[0]) < self.lista_fases.index(self.ind0[1])
    if not hasattr(self, "desempates"):
      self.desempates = {}

    if isinstance(df_fixture_updated, pd.DataFrame): # varios partidos (una fila por partido)
      if self.lista_fases.index(self.ind0[0]) < self.lista_fases.index(self.ind0[1]):
        puntos = self.intsc_prob_goles(df_fixture_updated, model, alterno=1) # resultados reales de la fase
      else:
        # PEND (agegar parametro 'valor=agno')
        # PEND recomentar docstring
        df_fixture_partidos = calculo_metricas_0(df_fixture_updated, self.drop_metricas, self.metricas_opcionales)
        X = df_fixture_partidos.copy().to_numpy()
        puntos = self.intsc_prob_goles(X, model, alterno=0)

      for points_, row in zip(puntos, df_fixture_updated.itertuples()):
        home, away = row.home, row.away
        point_h, point_a = points_[0][0], points_[1][0]
        # empate en eliminatoria: en una fase anterior a la objetivo lo decide la tanda real; si se predijo, el mas probable
        desempate_local = point_h == point_a and (gana_local_desempate(self.desempates, home, away) if rejugada
                                                  else self.__dict__.get("ganador_empate", {}).get((home, away)) == "home")

        if point_h > point_a or desempate_local:
          ganador = home
          perdedor = away
        else:
          ganador = away
          perdedor = home
        
        indice = (self.n_partidos - tamagno) + row.Index
        partido = f"Match {indice}"
        self.registrar_partido(partido, ganador, perdedor, point_h == point_a)

        Y0 = np.append(Y0, [point_h, point_a])

      df_fixture_updated["score_0"], df_fixture_updated["score_1"] = Y0[0::2], Y0[1::2]
      df_fixture_updated = calculo_metricas_1(df_fixture_updated)

    else: # un solo partido como Series
      indice = self.n_partidos - tamagno
      partido = f"Match {indice}"
      
      if self.lista_fases.index(self.ind0[0]) < self.lista_fases.index(self.ind0[1]):
        puntos = self.intsc_prob_goles(df_fixture_updated, model, alterno=1) # resultados reales de la fase
      else:
        # PEND (agegar parametro 'valor=agno')
        # PEND recomentar docstring
        df_fixture_partidos = calculo_metricas_0(df_fixture_updated, self.drop_metricas, self.metricas_opcionales)
        X = df_fixture_partidos.copy().to_numpy()
        puntos = self.intsc_prob_goles(X, model, alterno=0)

      for points_ in puntos:
        home, away = df_fixture_updated["home"], df_fixture_updated["away"]
        point_h, point_a = points_[0][0], points_[1][0]
        desempate_local = point_h == point_a and (gana_local_desempate(self.desempates, home, away) if rejugada
                                                  else self.__dict__.get("ganador_empate", {}).get((home, away)) == "home")

        if point_h > point_a or desempate_local:
          ganador = home
          perdedor = away
        else:
          ganador = away
          perdedor = home

        self.registrar_partido(partido, ganador, perdedor, point_h == point_a)

        Y0 = np.append(Y0, [point_h, point_a])

      df_fixture_updated["score_0"], df_fixture_updated["score_1"] = Y0[0::2], Y0[1::2]
      df_fixture_updated = calculo_metricas_1(df_fixture_updated)

    homes, aways, Winners, Losers = [], [], [], []
    for linea in df_fixture_updated.itertuples():
      indice = (self.n_partidos - tamagno) + linea.Index
      partido = f"Match {indice}"

      homes.append(self.id_pais[linea.home])
      aways.append(self.id_pais[linea.away])
      Winners.append(self.id_pais[self.dic_matches[partido]["ganador"]])
      Losers.append(self.id_pais[self.dic_matches[partido]["perdedor"]])
    
    df_nombres = pd.DataFrame(data={"home": homes, "away": aways, "Winner": Winners, "Loser": Losers}, index=range(len(homes)))
    print(df_nombres)

    return df_fixture_updated

  def seleccionar(self, df_fixture_, equipo, ind0):
    """
    ¿QUE HACE?
    Entrega las ultiimas metricas de rendimiento registradas, ya sea por fase de grupos o por alguno de los rounds de la fase final en el parámetro (df_fixture_), para el equipo de interés entregado en el parámetro (equipo), ordenando la respuesta segun se indique el el parámetro (ind0)

    ¿COMO LO HACE?
    0-) se definen unas variables (indice_0, indice_1) que poseen los indices de las filas del df (df_fixture_) donde esté el (equipo), ya sea como home o away

    1-) se define una variable (mas_g) con el index máximo entre (indice_0, indice_1)

    2-) se tienen 2 condicionles para actuar segun si el index máximo está entre los partidos jugados como home o away

      2.0) si el máximo está en la lista de partidos home: se toman dentro de una variable (seleccion) las primeras 10 columnas (0-9) con la informacion del rendimiento acumulada ahi (home, PTS_0, PJ_0, PG_0, PP_0, PE_0, GF_0, GC_0, D_0, score_0), se sustituye el nombre home por away si (ind0) dice away y se entrega (seleccion)

      2.1) si el máximo está en la lista de partidos away: se toman dentro de una variable (seleccion) las ultimas 10 columnas (10-19) con la informacion del rendimiento acumulada ahi (away, PTS_1, PJ_1, PG_1, PP_1, PE_1, GF_1, GC_1, D_1, score_1), se sustituye el nombre away por home si (ind0) dice home y se entrega (seleccion)

    Args:

      - df_fixture_: será el df (df_encuentro_jugado) respecto al 1er round de la fase final o 
      a la 2da fase de grupos

      - equipo: será el nombre del equipo a buscar dentro de los partidos jugados en 
      (df_fixture_)

      - ind0: indica si la respuesta debe ser dirigida a un partido como home o away
    
    Returns:

      - seleccion: Serie contenedora de la informacion de rendimiento registrada para el ultimo partido jugado por (equipo) en (df_fixture_), formateada segun el lugar que tomará el (equipo) en la próxima fase "home o away"
    """
    indice_0 = df_fixture_[df_fixture_.home == equipo].index
    indice_1 = df_fixture_[df_fixture_.away == equipo].index
    
    mas_g = max(indice_0.tolist() + indice_1.tolist())
    
    if mas_g in indice_0:
      seleccion = df_fixture_.iloc[mas_g, :COLS_FASE_POR_EQUIPO]
      if ind0 == "away":
        seleccion.rename(index={"home": "away"}, inplace=True)
      return seleccion

    elif mas_g in indice_1:
      seleccion = df_fixture_.iloc[mas_g, COLS_FASE_POR_EQUIPO:]
      if ind0 == "home":
        seleccion.rename(index={"away": "home"}, inplace=True)
      return seleccion

  def camb_grp_elm_smpl(self, df_encuentro_jugado, df_encuentro_siguiente, dt, year=None):
    """
    ¿QUE HACE?
    Organiza a los equipos  que jugaran el primer round de la fase final (generalmente un knockout u octavos de final en un torneo de eliminacion simple) tras acabar la fase de grupos y haberse definido los 1ros a 4tos lugares por grupo dentro de la misma, por su acumulacion de puntos
    
    ¿COMO LO HACE?
    0-) Se almacenann las columnas de la fase de grupos (df_encuentro_jugado), se crea un diccionario (dic) con claves lugar obtenido en fase de grupos y valor numero correspondiente a dicho lugar (ej. "Third-p":2, en referencia a el index en df para ese equipo). Un diccionario (dic_f20) vacio par el caso de tener una 2da fase de grupos, la dimension del df de el proximo round (tamagno_0, el 1ro de la fase final), el numero de columnas (tamagno_1) para el df de el proximo round, finalmente, el df vacio para generar el correspondiente al 1er round de la fase final a disputarse (dffix)

    1-) Se definen 2 condicionales para los casos en que se jugará una fase de grupos (para mundiales pasados esto solo ocurre entre 1974 y 1982, de darse de nuevo este formato en el futuro, debe tenerse al menos el nombre y cantidad de los grupos para esta 2da fase). Lo que se hace es agrupar en variables, el contenido del df (df_encuentro_siguiente) correspondiente a los partidos de esa 2da fase, teniendo cada variable el numero de partidos por grupo para esta fase. Se almacenan todos en un diccionario que debera retornarse al final de la funcion con claves nombre de los grupos y valor variables con los partidos por grupos

    2-) Se define un condicional para trabajar con rounds de mas de 1 partido, o de 1 solo partido. para el 1er caso
      
      2.0-) se iteran en forma de tupla los partidos del df_encuentro_siguiente, se almacena de cada partido iterado el equipo local (home) y visitante (away), se definen unas variables vacias para su uso futuro (equipo0, equipo1)

        2.0.0) para el caso de tener un df_encuentro_siguiente de 3 columnas, se busca: el índice correspondiente a la posición por puntaje alcanzada por el equipo home y away, según el (dic) en fase de grupos (ubicacion0, ubicacion1), el grupo correspondiente al equipo home y away que juegan ese partido iterado (g0, g1), la búsqueda del nombre del equipo, según la posición alcanzada dentro del grupo definido (equipo0, equipo1), mediante la funcion "loc"

        2.0.1) Para el caso de tener 2 o 4 columnas (entrada del usuario) (equipo0, equipo1) serán iguales a los (home, away) que se iteran del df (df_encuentro_siguiente)
      
      Luego se usa el metodo (seleccionar) para obtener las metricas de rendimiento para los (equipo0, equipo1) segun el df (df_encuentro_jugado) y se almacenan estos valores en un df (rndTeam0 y rndTeam1) que luego se concatena como columnas a un df (partido) y este a su vez se concatena como fila a un (dffix)

      2.1-) se almacena en unas variables (home0, away0) los nombres de los equipos home y away que juegan la siguiete round (el 1ro tras la fase de grupos)
      
        2.1.0-) para el caso de tener un df_encuentro_siguiente de 3 columnas, se almacena en unas variables (home0, away0) los nombres de los equipos home y away que juegan la siguiete round (el 1ro tras la fase de grupos). Se obtiene el lugar e indice de fila en el df del grupo para los equipos home y away mediante la funcion "split" para el home y el away, mediante el (dic) y se les almacena en las variables (ubicacion0, ubicacion1). Luego se crea el nombre del grupo a buscar en (g0, g1), luego se busca el df de ese grupo en el (dt) y se obtiene el nombre de país correspondiente al lugar del home o away mediante la función "loc" (equipo0, equipo1)

        2.1.1-) para el caso de tener un df_encuentro_siguiente de 2 o 4 columnas (entrada del usuario) (equipo0, equipo1) serán iguales a los (home, away) que se obtienen del df (df_encuentro_siguiente)
      
      Luego se usa el metodo (seleccionar) para obtener las metricas de rendimiento para los (equipo0, equipo1) segun el df (df_encuentro_jugado) y se almacenan estos valores en unos df (rndTeam0 y rndTeam1) que luego se concatena como columnas a un df (partido) y este a su vez se concatena como fila a un df (dffix)
    
    3-) se definen las columans para (dffix) y se depuran los indices

    4-) se divide la respuesta final entre 2 condicionales

      4.0) el año del mundial en que se predice es tal que se presenta una 2da fase de grupos (1974, 1978, 1982)

        4.0.0-) Si la dimension del df (df_encuentro_siguiente) es propia de entradas historicas (3) o si no se esta entre los estado actual y objetivo (grupo, doce); se define un diccionario vacio para recepción (dic_f21), se itera el contenido de los puntajes por grupo de la primera fase de grupos en (dic_f20), se define un df para recepción (new_df), se iteran las filas del df de desempeño por grupo en la primera fase de grupos, se deja en una variable (ef2) el desempeño del grupo definido para los índices correspondientes al grupo dejado en la columna (pais) de la fila iterada del (dic_f20) y la posición del equipo de interés presente hasta los últimos 8 índices de la entrada (pais) de la fila iterada según el índice marcado por (dic), se concatena a (new_df) el contenido de (ef2), se almacena (new_df) en el grupo iterado desde (dic_f20) dentro del diccionario de recepcion (dic_f21), finalmente se retornan el df (dffix) con el orden e informacion para los partidos del siguiente round y el diccionario (dic_f21)
    
        4.0.1-) Si la dimensión del df (df_encuentro_siguiente) es propia de entradas del usuario (2 o 4), se iteran los grupos del diccionario (dic_f20), se genera un df y una variable para almacen de puntajes (punto), se iteran los equipos de la columna (pais) del grupo iterado, se iteran los grupos del diccionario de la fase de grupos previa y se actualiza la variable de (punto) con el puntaje del equipo iterado en el grupo de la fase previa donde este se ubicó, se actualiza la variable (punto) a un df con 2 filas (pais, punto), se concatena como fila en un df (df0) el ahora array (punto), se almacena en ese (dic_f20) el grupo iterado de la nueva fase y el (df0) con los (pais, punto) actualizados, finalmente, se entregan el df (dffix) junto con el diccionario (dic_f20)
      
      4.1) se esta en un mundial que no incluye una 2da fase de grupos. La respuesta es solo el df (dffix) con el orden e informacion para los partidos del siguiente round

    Args:
    
      - df_encuentro_jugado: df con el orden e informacion del rendimiento de cada equipo de la fase de grupos o fase previa al 1er round de la fase final

      - df_encuentro_siguiente: df con el orden y nombres de equipos o ids de busqueda para los equipos que disputaran el 1er round de la fase final, tras la fase de grupos

      - dt: diccionario con informacion del rendimiento acumulado durante la disputa de la fase de grupos, organizada en clave: nombre del grupo (Grupo X) y valor: df de 4 filas y columnas "pais, puntaje"

      - year: define si el año del mundial en estudio es de los que han tenido una 2da fase de grupos

    Returns:

      - (dffix, dic_f21): el primero es el df con el orden e informacion de los equipos que disputarán el siguiente round. el segundo es el diccionario actualizado desde las claves de busqueda de partidos de la 1ra fase de grupos para los equipos de cada grupo con la informacion de desempeño acumulada en esa fase
    
      - (dffix, dic_f20): el primero es el df con el orden e informacion de los equipos que disputarán el siguiente round. el segundo es el diccionario actualizado segun la informacion del usuario, para la disputa de la 2da fase de grupos

      - (dffix): para los mundiales con solo una fase de grupos
    """
    columnas = df_encuentro_jugado.columns.to_list()
    dic = {"Winner": 0, "Runners-up": 1, "Third-p": 2, "Fourth-p": 3}
    dic_f20 = {}
    tamagno_0 = len(df_encuentro_siguiente.shape)
    tamagno_1 = df_encuentro_siguiente.shape[-1]
    dffix = pd.DataFrame()

    if year in (1974, 1978):
      uno = pd.concat([df_encuentro_siguiente.home.iloc[:6], df_encuentro_siguiente.away.iloc[:6]], axis=0).unique().tolist()
      dos = pd.concat([df_encuentro_siguiente.home.iloc[6:12], df_encuentro_siguiente.away.iloc[6:12]], axis=0).unique().tolist()
      uno = pd.DataFrame(data=np.array(uno), columns=["pais"]).reset_index(drop=True)
      dos = pd.DataFrame(data=np.array(dos), columns=["pais"]).reset_index(drop=True)
      dic_f20 = {"Grupo A": uno, "Grupo B": dos}
    
    elif year in (1982,):
      uno = pd.concat([df_encuentro_siguiente.home.iloc[:3], df_encuentro_siguiente.away.iloc[:3]], axis=0).unique().tolist()
      dos = pd.concat([df_encuentro_siguiente.home.iloc[3:6], df_encuentro_siguiente.away.iloc[3:6]], axis=0).unique().tolist()
      tres = pd.concat([df_encuentro_siguiente.home.iloc[6:9], df_encuentro_siguiente.away.iloc[6:9]], axis=0).unique().tolist()
      cuatro = pd.concat([df_encuentro_siguiente.home.iloc[9:12], df_encuentro_siguiente.away.iloc[9:12]], axis=0).unique().tolist()

      uno = pd.DataFrame(data=np.array(uno), columns=["pais"]).reset_index(drop=True)
      dos = pd.DataFrame(data=np.array(dos), columns=["pais"]).reset_index(drop=True)
      tres = pd.DataFrame(data=np.array(tres), columns=["pais"]).reset_index(drop=True)
      cuatro = pd.DataFrame(data=np.array(cuatro), columns=["pais"]).reset_index(drop=True)

      dic_f20 = {"Grupo A": uno,
                 "Grupo B": dos,
                 "Grupo C": tres,
                 "Grupo D": cuatro}

    if tamagno_0 > 1:
      for row_fix in df_encuentro_siguiente.itertuples():
        home, away = row_fix.home, row_fix.away
        equipo0, equipo1 = None, None

        if tamagno_1 == 3:
          ubicacion0, ubicacion1 = dic[home[:home.find(" ")]], dic[away[:away.find(" ")]]
          g0, g1 = home[home.find(" ")+1:], away[away.find(" ")+1:]
          equipo0, equipo1 = dt[g0].loc[ubicacion0, "pais"], dt[g1].loc[ubicacion1, "pais"]

        elif (tamagno_1 == 2 or tamagno_1 == 4):
          equipo0, equipo1 = home, away

        rndTeam0 = self.seleccionar(df_encuentro_jugado, equipo0, "home")
        rndTeam0 = pd.DataFrame(data=[rndTeam0.values], columns=rndTeam0.index).reset_index(drop=True)

        rndTeam1 = self.seleccionar(df_encuentro_jugado, equipo1, "away")
        rndTeam1 = pd.DataFrame(data=[rndTeam1.values], columns=rndTeam1.index).reset_index(drop=True)

        partido = pd.concat([rndTeam0, rndTeam1], axis=1, ignore_index=True)
        dffix = pd.concat([dffix, partido], axis=0)

    elif tamagno_0 == 1:
      home0, away0 = df_encuentro_siguiente.home, df_encuentro_siguiente.away
      if tamagno_1 == 3:
        home1, away1 = home0.split(" "), away0.split(" ")
        ubicacion0, ubicacion1 = dic[home1[0]], dic[away1[0]]
        g0, g1 = home1[1]+" "+home1[2], away1[1]+" "+away1[2]
        equipo0, equipo1 = dt[g0].loc[ubicacion0, "pais"], dt[g1].loc[ubicacion1, "pais"]

      elif (tamagno_1 == 2 or tamagno_1 == 4):
        equipo0, equipo1 = home0, away0

      rndTeam0 = self.seleccionar(df_encuentro_jugado, equipo0, "home")
      rndTeam0 = pd.DataFrame(data=[rndTeam0.values], columns=rndTeam0.index).reset_index(drop=True)
      
      rndTeam1 = self.seleccionar(df_encuentro_jugado, equipo1, "away")
      rndTeam1 = pd.DataFrame(data=[rndTeam1.values], columns=rndTeam1.index).reset_index(drop=True)
      
      partido = pd.concat([rndTeam0, rndTeam1], axis=1, ignore_index=True)
      dffix = pd.concat([dffix, partido], axis=0)

    dffix.columns = columnas
    dffix = dffix.reset_index(drop=True)

    if year in (1974, 1978, 1982): # PEND (IA, dime si aqui tambien combiene poner la logica condicional (if self.lista_fases.index(self.ind0[0]) < self.lista_fases.index(self.ind0[1]):))
      dic_f21 = {}
      if (not self.ind0[0] == "grupo" and not self.ind0[1] == "doce") or (tamagno_1 == 3):
        for des0 in dic_f20.items():
          new_df = pd.DataFrame(columns=("pais", "Pts"))
          for des1 in des0[1].itertuples():
            ef2 = dt[des1.pais[-7:]].iloc[dic[des1.pais[:-8]]]
            new_df = pd.concat([new_df, pd.DataFrame([ef2], columns=new_df.columns)], axis=0)
          dic_f21[des0[0]] = new_df
        return dffix, dic_f21

      else: # PEND (IA, dime si aqui tambien combiene poner la logica condicional (if self.lista_fases.index(self.ind0[0]) < self.lista_fases.index(self.ind0[1]):))
        for des0 in dic_f20.items():
          df0 = pd.DataFrame()
          punto = 0
          for des00 in des0[1].pais.values:
            for des01 in dt.keys():
              if des00 in dt[des01].pais.values:
                punto = dt[des01][dt[des01].pais == des00]["Pts"].values[0]

            punto = pd.DataFrame(data=np.array([[des00, punto]]), columns=["pais", "Pts"])
            df0 = pd.concat([df0, punto], axis=0)
          dic_f20[des0[0]] = df0
        return dffix, dic_f20

    else:
      return dffix

  def ordenar_elm_smpl(self, df_encuentro_jugado, df_encuentro_siguiente, tipo="Winner"):
    """
    ¿QUE HACE?
    Organiza los equipos que jugarán los partidos de cada round de la fase final desde el primero hasta el ultimo, valiendose de los nombres presentes en la seccion del fixture entregada como(df_encuentro_siguiente) para obtener a los equipos desde el df (df_encuentro_jugado), según las victorias y derrotas por partido, marcadas en (dic_matches), dejando en un df (df_encuentro_siguiente) nuevo con el orden y la informacion de rendimiento necesaria para la inferencia en la funcion (get_winner)

    ¿COMO LO HACE?
    0-) se define un df (dffix) vacio para la recepcion del orden de juego y rendimiento de cada equipo, una variable (columnas) con el nombre de las columnas del df (df_encuentro_jugado), una variable (tamagno_1) con el número de columnas en el df (df_encuentro_siguiente) para determinar si se organiza hacia una prediccion del usuario o tras esta
    
    1-) Se tienen 2 condicionales para dirigir el ordenamiento de partidos segun el numero de partidos del siguiente round
      
      1.0) Para cuando (tipo) sea "Winner" y no sea la final (1 solo partido por round). Se iteran los nombres de los equipos que jugarán como home y away en el próximo round df (df_encuentro_siguiente)

        1.0.0) se tienen 2 condicionales

          1.0.0.0) para cuando el numero de columnas del df del siguiente round sea 3. En este caso home y away corresponderan a claves de la forma (Winner-Looser Partido X) que se almacenan por separado en unas variables (equipo0, equipo1), con estas se busca al ganador registrado para dicha clave buscando en el diccionario (dic_matches) con la clave de nombre del partido que redirige a otro diccionario cuya clave corresponde al parámetro (tipo)

          1.0.0.1) para cuando el numero de columnas del df del siguiente round sea 2 o 4. En este caso lo entregado corresponderá a una consulta del usuario, por lo que home y away corresponderán a los nombres directamente de los equipos que disputan el proximo round y no la clave mostrada para el condicional anterior, dichos nombres se almacenan en las variables (equipo0, equipo1)

        1.0.1) Se usa el metodo (seleccionar) para obtener las metricas de rendimiento para los (equipo0, equipo1) segun el df (df_encuentro_jugado) y se almacenan estos valores en unos df (rndTeam0 y rndTeam1) que luego se concatena como columnas a un df (partido) y este a su vez se concatena como fila a un df (dffix)

      1.1) para cuando (tipo) sea "Looser" o sea la final (1 solo partido por round). Se toman las columnas home y away del df (df_encuentro_siguiente) y se almacenan en unas variables (home, away)

        1.1.0) se tienen 2 condicionales

          1.1.0.0) cuando el numero de columnas del df (df_encuentro_siguiente) es 3, se tiene una estructura de home y away (Winner-Looser Partido X) y se busca en el diccionario (dic_matches) con clave (Partido X) redirigiéndose a un diccionario con clave (tipo), almacenándose el resultado del ganador o perdedor del partido referenciado en unas variables (equipo0, equipo1)

          1.1.0.1) cuando el numero de columnas del df (df_encuentro_siguiente) es 2 o 4,  home y away son solo los nombres de los equipos que, segun la inferencia buscada por el usuario, disputarian el siguiente round, almacenándose el resultado del ganador o perdedor del partido referenciado en unas variables (equipo0, equipo1)
        
        1.1.1) Se usa el metodo (seleccionar) para obtener las metricas de rendimiento para los (equipo0, equipo1) segun el df (df_encuentro_jugado) y se almacenan estos valores en unos df (rndTeam0 y rndTeam1) que luego se concatena como columnas a un df (partido) y este a su vez se concatena como fila a un df (dffix)

    2-) se resetean los indices del df (dffix) y se le adaptan las columnas de la variable (columnas) correspondientes al df (df_encuentro_jugado)

    Args:

      - df_encuentro_jugado: df con la informacion actualizada para cada equipo que disputara el round previo y que jugará en el proximo
      
      - df_encuentro_siguiente: df contenedor de los partidos que habrán de jugarse en el siguiente round, con el nombre de los equipos definidos por una consulta del usuario (2 o 4 columnas) o las claves para buscar a los equipos que disputarán el siguiente round de la forma (Winner-Looser Partido X) (3 columnas)
      
      - tipo: el resultado del partido previo que define el lugar disputado en el siguiente round

    Return:

      - dffix: df contenedor del orden de juego del siguiente round, con los equipos que lo disputaran y su informacion de rendimiento

    """
    dffix = pd.DataFrame()
    # solo las columnas de los dos equipos (equipo + base + score por lado); descarta extras al final si las hubiera
    df_jugado = df_encuentro_jugado.iloc[:, :2 * COLS_FASE_POR_EQUIPO]
    columnas = df_jugado.columns.to_list()
    tamagno_1 = df_encuentro_siguiente.shape[-1]
    resultado = {"Winner": "ganador", "Loser": "perdedor"}[tipo] # tipo viene en ingles (fixture); dic_matches guarda "ganador"/"perdedor"

    if tipo == "Winner" and not len(df_encuentro_siguiente.shape) == 1:
      for home, away in zip(df_encuentro_siguiente["home"], df_encuentro_siguiente["away"]):
        if tamagno_1 == 3:
          equipo0 = self.dic_matches[home[len(tipo)+1:]][resultado]
          equipo1 = self.dic_matches[away[len(tipo)+1:]][resultado]
        elif (tamagno_1 == 2 or tamagno_1 == 4):
          equipo0, equipo1 = home, away

        rndTeam0 = self.seleccionar(df_jugado, equipo0, "home") 
        rndTeam0 = pd.DataFrame(data=[rndTeam0.values], columns=rndTeam0.index).reset_index(drop=True)
        
        rndTeam1 = self.seleccionar(df_jugado, equipo1, "away")
        rndTeam1 = pd.DataFrame(data=[rndTeam1.values], columns=rndTeam1.index).reset_index(drop=True)
        
        partido = pd.concat([rndTeam0, rndTeam1], axis=1, ignore_index=True)
        dffix = pd.concat([dffix, partido], axis=0)

    elif tipo == "Loser" or len(df_encuentro_siguiente.shape) == 1:
      home, away = df_encuentro_siguiente["home"], df_encuentro_siguiente["away"]

      if tamagno_1 == 3:
        equipo0 = self.dic_matches[home[len(tipo) + 1:]][resultado]
        equipo1 = self.dic_matches[away[len(tipo) + 1:]][resultado]
      elif (tamagno_1 == 2 or tamagno_1 == 4):
        equipo0, equipo1 = home.item(), away.item()

      rndTeam0 = self.seleccionar(df_jugado, equipo0, "home")
      rndTeam0 = pd.DataFrame(data=[rndTeam0.values], columns=rndTeam0.index).reset_index(drop=True)

      rndTeam1 = self.seleccionar(df_jugado, equipo1, "away")
      rndTeam1 = pd.DataFrame(data=[rndTeam1.values], columns=rndTeam1.index).reset_index(drop=True)

      partido = pd.concat([rndTeam0, rndTeam1], axis=1, ignore_index=True)
      dffix = pd.concat([dffix, partido], axis=0)

    dffix = dffix.reset_index(drop=True)
    dffix.columns = columnas

    return dffix

# Global setups: se construyen la primera vez que se piden, no al importar (asi migrate y load_data funcionan con la BD vacia)
agnos = [1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022, 2026]
_cache_global = {}

def obtener_clasificaciones():
  """Tabla "clasificaciones" como df (formar_dataset_real); se consulta a la BD una sola vez por proceso"""
  if "clasificaciones" not in _cache_global:
    _cache_global["clasificaciones"] = formar_dataset_real("clasificaciones")
  return _cache_global["clasificaciones"]

def obtener_dic_agnos():
  """
  ¿QUE HACE?
  Devuelve el diccionario año del mundial -> df de rendimiento de cada pais en las eliminatorias previas a ese mundial. Se calcula la primera vez que se pide y luego se sirve desde (_cache_global)

  ¿COMO LO HACE?
  0-) Si no esta en cache: para cada año de (agnos) junta los partidos de clasificacion jugados despues del mundial anterior y hasta ese año, y arma con "funcion_tabla_desempegno" la fila de rendimiento de cada pais

  1-) Devuelve una copia de cada df, porque "create_features" reescribe la columna "pais" con ids y no debe alterar la cache que usara la siguiente consulta

  Returns:

    - diccionario {"1934": df, ..., "2022": df}
  """
  if "dic_agnos" not in _cache_global:
    clsf = obtener_clasificaciones()
    dic_agnos = {}

    for des0 in agnos:
      df_partidos_agnos = pd.DataFrame(columns=["home", "score_0", "score_1", "away", "agno"])

      for des1 in set(clsf["agno"].tolist()):
        if des0 > agnos[0]:
          if (int(des1) <= des0 and int(des1) > agnos[agnos.index(des0)-1]):
            eliminatoria = clsf.loc[clsf["agno"] == des1][["home", "score_0", "score_1", "away", "agno"]]
            df_partidos_agnos = pd.concat([df_partidos_agnos, eliminatoria], axis=0)

        elif not (des0 > agnos[0]):
          if int(des1) <= des0:
            eliminatoria = clsf.loc[clsf["agno"] == des1][["home", "score_0", "score_1", "away", "agno"]]
            df_partidos_agnos = pd.concat([df_partidos_agnos, eliminatoria], axis=0)

      eliminatorias = pd.DataFrame()
      for idx, pais in enumerate(set(df_partidos_agnos.home.tolist() + df_partidos_agnos.away.tolist())):
        eliminatorias = pd.concat([eliminatorias, funcion_tabla_desempegno(df_partidos_agnos, pais, idx, des0)], axis=0)

      dic_agnos[str(des0)] = eliminatorias

    _cache_global["dic_agnos"] = dic_agnos

  return {agno: df.copy() for agno, df in _cache_global["dic_agnos"].items()}

def consulta_general(pregunta_0, df5, reglamento_terceros=None, drop_metricas=DROP_METRICS, metricas_opcionales=OPTIONAL_METRICS):
  """
  ¿QUE HACE?
  Realiza la predicción de resultados para un mundial específico, creando el objeto de la clase (func_prediccion_orden) necesario segun el año al que quiere aplicarse inferencia, generando los datasets necesarios, generando parametros para calcular la fuerza aproximada de cada equipo, cargando el modelo entrenado y aplicando las funciones de predicción para obtener los resultados esperados

  ¿COMO LO HACE?

  0-) Se crea el objeto (objs) de la clase (func_prediccion_orden) con los parametros:

        - ind_partidos: cadena con la forma ("partidos") para indicar la busqueda correcta en DB con el ORM de django

        - ind_fixtures: cadena contenedora del año del mundial de interes para indicar la busqueda correcta de su fixture en DB con el ORM de django
      
        - df_prediccion_usuario: df de consulta del usuario para hacer inferencias personalizadas. Idealmente se incluye o no en la salida del metodo (create_features) la ---modalidad de inferencia---
      
        - clsif: df con las clasificaciones historicas de mundiales para obtener el total absoluto de equipos participantes en el mundial a predecir
    
        - obj0: un entero con el año del mundial a predecir, para indicar si el componente de indice 0 de la salida del metodo (create_features) incluye (fase final fixture del mundial jugado, df consulta del usuario) o (contenido del mundial jugado)

        - drop_metricas, metricas_opcionales: la eleccion de metricas del usuario, que (calculo_metricas_0) usa para neutralizar la entrada al modelo

    0.0-) Se recarga la variable global (opcionales) con "reiniciar_opcionales", traduciendo los paises a los ids de (objs)

  1-) Se llama al metodo (create_features) del objeto (objs) usando como argumento el (dic_agnos) para generar los datasets necesarios para la predicción, almacenándolos en las variables (df1, df2, df3):

  2- Se crean atributos:

    - df1: (df conenedor de la fase final fixture del mundial jugado, df consulta del usuario) o (contenido del mundial jugado)
  
    - df2: (df conenedor de la fase de grupos del fixture)
    
    - df3: (df conenedor de la fase final del fixture)

  3-) Se crea la variable (agg0) como un df que concatena en filas y reiniciando el índice con el parametro (ignore_index=True), las columnas (home) de (df2) y (df1[0]), renombrando la columna resultante como (Team) mediante el metodo (rename)

  4-) Se crea la variable (pts0) como un df que concatena en filas y reiniciando el índice con el parametro (ignore_index=True), las columnas (score_0) de (df2) y (df1[0]), renombrando la columna resultante como (goles) mediante el metodo (rename)

  5-) Se concatena (agg0) y (pts0) en columnas para generar un df (agg0) con las columnas (Team, goles), luego se agrupa por (Team) y se calcula el promedio de goles por equipo mediante el metodo (groupby) y (mean)

  6-) Se crea la variable (agg1) como un df que concatena en filas y reiniciando el índice con el parametro (ignore_index=True), las columnas (away) de (df2) y (df1[0]), renombrando la columna resultante como (Team) mediante el metodo (rename)

  7-) Se crea la variable (pts1) como un df que concatena en filas y reiniciando el índice con el parametro (ignore_index=True), las columnas (score_1) de (df2) y (df1[0]), renombrando la columna resultante como (goles) mediante el metodo (rename)
  
  8-) Se crea la variable (pts1) como un df que concatena en filas y reiniciando el índice con el parametro (ignore_index=True), las columnas (score_1) de (df2) y (df1[0]), renombrando la columna resultante como (goles) mediante el metodo (rename)

  9-) Se concatena (agg1) y (pts1) en columnas para generar un df (agg1) con las columnas (Team, goles), luego se agrupa por (Team) y se calcula el promedio de goles por equipo mediante el metodo (groupby) y (mean)

  10-) Se actualiza el diccionario (dic_fuerza) del objeto (objs) con  llaves (home) y (away) respectivamente y como claves los df (agg0) y (agg1)

  11-) Se define la variable (dir_modelo) con la carpeta del predictor en uso: (settings.MODEL_PATH) / (settings.MODEL_DIR), configurable desde .env

  12-) Se obtiene el predictor con "predictor_en_uso" ("cargar_predictor", que rechaza un modelo entrenado con otras metricas, la primera vez; luego desde memoria) y se almacena en la variable (modelo)

  13-) Se crea una variable (dict_g) llamando al medtodo (grupos_anio_interes) del objeto (objs) con parametro igual al año del mundial de interes y se obtiene un diccionario con claves de cada grupo que jugó ese mundial ("Grupo A", "Grupo B", pj) y valor un df contenedor de los equipos del grupo, parametros de rendimiento seteados a 0, el grupo mismo y el año de juego

  14-) Se crean unos condicionales para definir si el mundial a predecir esta entre alguno de los periodos en que el formato del torneo tuvo algun cambio significativo:

    - df_ent: df con los resultados predichos para cada fase del mundial de interes y nombres de equipos ya en formato texto, con las alteraciones que el usuario haya hecho en su consulta, concatenando los dfs de cada fase del mundial de interes (fase de grupos, 2da fase de grupos, semifinales, partido por el tercer lugar y final) cada uno con sus resultados predichos y nombres de equipos ya en formato texto

    - Para 1986, 1990 y 1994 el 1er round de la fase final se arma con "alinear_llave_terceros" DESPUES de jugar la fase de grupos, y (df_ent.attrs) lleva las tablas "mejores_terceros" y "llave_octavos" para mostrarlas en el template

  Args:

    - pregunta_0: cadena con la fase objetivo y el año del mundial (ej. "grupo1986")

    - df5: df de consulta del usuario

    - reglamento_terceros: (ReglamentoTerceros) personalizado desde el frontend para 1986-1994; None usa el reglamento FIFA oficial

    - drop_metricas: metricas base que el usuario descarta (llegan al modelo con su VALOR_NEUTRO)

    - metricas_opcionales: metricas opcionales que el usuario acepta (las demas llegan al modelo con su VALOR_NEUTRO)
  """

  objs = func_prediccion_orden("partidos", pregunta_0[-4:], df5, obtener_clasificaciones(), obj0=int(pregunta_0[-4:]), drop_metricas=drop_metricas, metricas_opcionales=metricas_opcionales)
  # el progreso de metricas abstractas arranca desde el historico en cada consulta (no desde la anterior)
  reiniciar_opcionales(objs.pais_id, agno=int(pregunta_0[-4:]))

  # copia propia de esta consulta: create_features le pasa "pais" a ids y agregar_features la usa asi mas abajo
  dic_agnos = obtener_dic_agnos()
  df_features = objs.create_features(dic_agnos)
  df1, df2, df3 = df_features[0], df_features[1], df_features[2]

  print("primero: (df1)")
  print(df1)
  print("")
  print("")
  print("segundo: (df2)")
  print(df2)
  print("")
  print("")
  print("tercero: (df3)")
  print(df3)

  # media de goles por equipo como local (agg0) y como visitante (agg1): df con indice "Team" y columna "goles"
  agg0 = pd.DataFrame({
    "Team": pd.concat([df2.home, df1[0].home], axis=0, ignore_index=True),
    "goles": pd.to_numeric(pd.concat([df2.score_0, df1[0].score_0], axis=0, ignore_index=True), errors="coerce"),
  }).groupby("Team").mean()

  agg1 = pd.DataFrame({
    "Team": pd.concat([df2.away, df1[0].away], axis=0, ignore_index=True),
    "goles": pd.to_numeric(pd.concat([df2.score_1, df1[0].score_1], axis=0, ignore_index=True), errors="coerce"),
  }).groupby("Team").mean()

  objs.dic_fuerza["home"] = agg0
  objs.dic_fuerza["away"] = agg1

  # create_features deja en df1[0] el historico de todos los mundiales (menos sus primeras filas); desde aqui las fases
  # anteriores a la objetivo se arman con la fase final real del mundial consultado, y se juegan con sus marcadores reales
  reales = objs.df2[objs.df2["agno"].astype(int) == int(pregunta_0[-4:])]
  limpieza = {"(": r"", ")": r"", ":": r",", " ": r","}
  partidos_agno = pd.DataFrame({
    "home": reales["home"].map(objs.pais_id).to_list(),
    "score_0": reales["score"].map(lambda x: objs.ajuste_score_partidos(x, limpieza)).to_list(),
    "score_1": reales["score"].map(lambda x: objs.ajuste_score_partidos(x, limpieza, ind=1)).to_list(),
    "away": reales["away"].map(objs.pais_id).to_list(),
    "score": reales["score"].to_list(), # la rama 1986-1994 lee el marcador original ("marcadores_knockout")
    "penales": reales["penales"].to_list(),
    "jugado": reales["jugado"].to_list(),
  })
  objs.resultados_reales = resultados_reales(partidos_agno)
  # quien paso en los empates reales (tanda de penales o partido no jugado), para rejugar las fases previas
  objs.desempates = desempates_reales(partidos_agno)
  # 1934 y 1938 no tienen fase de grupos: su primera fase son los octavos (para el contexto "fase_final")
  objs.primera_fase = "knockout" if int(pregunta_0[-4:]) in (1934, 1938) else "grupo"
  # contexto del LSTM: anfitriones del mundial (no juegan en campo neutral) y equipos con penales / tiros reales en su historico
  agno_consulta = int(pregunta_0[-4:])
  objs.anfitriones = {objs.pais_id[n] for n in internacionales.objects.filter(categoria="mundial", agno=str(agno_consulta), neutral="0")
                      .values_list("home", flat=True).distinct() if n in objs.pais_id}
  proxy = formar_dataset_real("proxy_desempegno")
  objs.con_datos_reales = {marca: {objs.pais_id[n] for n in proxy.loc[(proxy["agno"] < agno_consulta) & (proxy[marca] == 1), "pais"] if n in objs.pais_id}
                           for marca in ("pk_real", "sot_real")}
  df1 = (partidos_agno.iloc[len(df2):].reset_index(drop=True), df1[1])

  dir_modelo = settings.MODEL_PATH / settings.MODEL_DIR
  modelo = predictor_en_uso(dir_modelo)
  
  dict_g = objs.grupos_anio_interes(pregunta_0[-4:])

  if "1998" in pregunta_0 or "2002" in pregunta_0 or "2006" in pregunta_0 or "2010" in pregunta_0 or "2014" in pregunta_0 or "2018" in pregunta_0 or "2022" in pregunta_0:
    df_fixture_group_, df_fixture_knockout, df_fixture_quarter, df_fixture_semi, df_fixture_third, df_fixture_final = None, None, None, None, None, None

    if "grupo" in pregunta_0:
      objs.ind0[1] = "grupo"
      df_fixture_group_ = agregar_features(df1[1].copy(), dic_agnos, ind=0)
      df_fixture_knockout = df3[["home", "score", "away"]].iloc[0:8].copy()
      df_fixture_quarter = df3[["home", "score", "away"]].iloc[8:12].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[12:14].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "knockout" in pregunta_0:
      objs.ind0[1] = "knockout"
      df_fixture_group_ = df2.copy()
      df_fixture_knockout = df1[1].copy()
      df_fixture_quarter = df3[["home", "score", "away"]].iloc[8:12].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[12:14].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "quarter" in pregunta_0:
      objs.ind0[1] = "quarter"
      df_fixture_group_ = df2.copy()
      df_fixture_knockout = df1[0][["home", "away"]].iloc[0:8].copy()
      df_fixture_quarter = df1[1].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[12:14].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "semi" in pregunta_0:
      objs.ind0[1] = "semi"
      df_fixture_group_ = df2.copy()
      df_fixture_knockout = df1[0][["home", "away"]].iloc[0:8].copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[8:12].copy()
      df_fixture_semi = df1[1].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "third" in pregunta_0:
      objs.ind0[1] = "third"
      df_fixture_group_ = df2.copy()
      df_fixture_knockout = df1[0][["home", "away"]].iloc[0:8].copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[8:12].copy()
      df_fixture_semi = df1[0][["home", "away"]].iloc[12:14].copy()
      df_fixture_third = df1[1].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "final" in pregunta_0:
      objs.ind0[1] = "final"
      df_fixture_group_ = df2.copy()
      df_fixture_knockout = df1[0][["home", "away"]].iloc[0:8].copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[8:12].copy()
      df_fixture_semi = df1[0][["home", "away"]].iloc[12:14].copy()
      df_fixture_third = df1[0][["home", "away"]].iloc[14].copy()
      df_fixture_final = df1[1].copy()

    objs.ind0[0] = "grupo"
    dict_table = objs.fase_de_grupos(dict_g, df_fixture_group_, modelo) 

    objs.n_partidos = df_fixture_group_.shape[0] + 1 # el fixture numera los octavos desde (partidos de grupos + 1)
    df_fixture_knockout = objs.camb_grp_elm_smpl(df_fixture_group_, df_fixture_knockout, dict_table) 

    objs.ind0[0] = "knockout"
    objs.jugar(df_fixture_knockout, modelo) 

    print("indice base", objs.n_partidos)
    df_fixture_quarter = objs.ordenar_elm_smpl(df_fixture_knockout, df_fixture_quarter) 

    objs.ind0[0] = "quarter"
    objs.jugar(df_fixture_quarter, modelo)

    print("indice base", objs.n_partidos)
    df_fixture_semi = objs.ordenar_elm_smpl(df_fixture_quarter, df_fixture_semi)

    objs.ind0[0] = "semi"
    objs.jugar(df_fixture_semi, modelo)

    df_fixture_third = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_third, tipo="Loser")

    objs.ind0[0] = "third"
    objs.jugar(df_fixture_third, modelo)

    df_fixture_final = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_final)

    print(f"\n\n----------\n----------\n\nFINAL - FINAL - FINAL: --- {pregunta_0} ---\n----------\n----------\n")
    objs.ind0[0] = "final"
    objs.jugar(df_fixture_final, modelo)

    df_ent = pd.concat([df_fixture_group_[["home", "score_0", "score_1", "away"]], df_fixture_knockout[["home", "score_0", "score_1", "away"]], df_fixture_quarter[["home", "score_0", "score_1", "away"]], df_fixture_semi[["home", "score_0", "score_1", "away"]], df_fixture_third[["home", "score_0", "score_1", "away"]], df_fixture_final[["home", "score_0", "score_1", "away"]]], axis=0)

    df_ent.home = df_ent.home.map(objs.id_pais)
    df_ent.away = df_ent.away.map(objs.id_pais)

    return df_ent

  elif "1934" in pregunta_0 or "1938" in pregunta_0:
    # tras octavos el fixture trae cuartos (4 en 1934, 5 en 1938 por un desempate), 2 semis, 3er lugar y final
    nq = len(df3) - 4

    if "knockout" in pregunta_0:
      objs.ind0[1] = "knockout"
      # 1ra fase del torneo: los partidos del usuario toman las metricas de las eliminatorias previas (como la fase de grupos en otros años)
      df_fixture_knockout = agregar_features(df1[1].copy().assign(agno=int(pregunta_0[-4:])), dic_agnos)
      df_fixture_quarter = df3[["home", "score", "away"]].iloc[:nq].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[nq:nq + 2].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[nq + 2].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[nq + 3].copy()

    if "quarter" in pregunta_0:
      objs.ind0[1] = "quarter"
      df_fixture_knockout = df2.copy()
      df_fixture_quarter = df1[1].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[nq:nq + 2].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[nq + 2].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[nq + 3].copy()

    if "semi" in pregunta_0:
      objs.ind0[1] = "semi"
      df_fixture_knockout = df2.copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[:nq].copy()
      df_fixture_semi = df1[1].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[nq + 2].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[nq + 3].copy()

    if "third" in pregunta_0:
      objs.ind0[1] = "third"
      df_fixture_knockout = df2.copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[:nq].copy()
      df_fixture_semi = df1[0][["home", "away"]].iloc[nq:nq + 2].copy()
      df_fixture_third = df1[1].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[nq + 3].copy()

    if "final" in pregunta_0:
      objs.ind0[1] = "final"
      df_fixture_knockout = df2.copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[:nq].copy()
      df_fixture_semi = df1[0][["home", "away"]].iloc[nq:nq + 2].copy()
      df_fixture_third = df1[0][["home", "away"]].iloc[nq + 2].copy()
      df_fixture_final = df1[1].copy()

    objs.n_partidos = 1 # sin fase de grupos: el fixture arranca en "Match 1"
    objs.ind0[0] = "knockout"
    objs.jugar(df_fixture_knockout, modelo)

    df_fixture_quarter = objs.ordenar_elm_smpl(df_fixture_knockout, df_fixture_quarter)

    objs.ind0[0] = "quarter"
    objs.jugar(df_fixture_quarter, modelo)

    df_fixture_semi = objs.ordenar_elm_smpl(df_fixture_quarter, df_fixture_semi)

    objs.ind0[0] = "semi"
    objs.jugar(df_fixture_semi, modelo)

    df_fixture_third = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_third, tipo="Loser")

    objs.ind0[0] = "third"
    objs.jugar(df_fixture_third, modelo)

    df_fixture_final = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_final)

    print(f"\n\n----------\n----------\n\nFINAL - FINAL - FINAL: --- {pregunta_0} ---\n----------\n----------\n")
    objs.ind0[0] = "final"
    objs.jugar(df_fixture_final, modelo)

    df_ent = pd.concat([df_fixture_knockout[["home", "score_0", "score_1", "away"]], df_fixture_quarter[["home", "score_0", "score_1", "away"]], df_fixture_semi[["home", "score_0", "score_1", "away"]], df_fixture_third[["home", "score_0", "score_1", "away"]], df_fixture_final[["home", "score_0", "score_1", "away"]]], axis=0)

    df_ent.home = df_ent.home.map(objs.id_pais)
    df_ent.away = df_ent.away.map(objs.id_pais)

    return df_ent

  elif "1950" in pregunta_0:

    if "grupo" in pregunta_0:
      objs.ind0[1] = "grupo"
      df_fixture_group_ = agregar_features(df1[1].copy(), dic_agnos, ind=0)
      df_fixture_quarter = df3[["home", "score", "away"]].iloc[:6].copy()

    elif "Fase Final" in pregunta_0:
      objs.ind0[1] = "Fase Final"
      df_fixture_group_ = df2.copy()
      df_fixture_quarter = df1[1].copy()

    objs.ind0[0] = "grupo"
    dict_table = objs.fase_de_grupos(dict_g, df_fixture_group_, modelo)

    df_fixture_quarter = objs.camb_grp_elm_smpl(df_fixture_group_, df_fixture_quarter, dict_table)

    dt = {}
    equipos = pd.concat([df_fixture_quarter.home, df_fixture_quarter.away], axis=0).unique().tolist()
    dt["Fase Final"] = pd.DataFrame(data={"pais": equipos, "Pts": [0, 0, 0, 0]})

    objs.ind0[0] = "Fase Final"
    dict_table = objs.fase_de_grupos(dt, df_fixture_quarter, modelo)

    print(f"\n\n----------\n----------\n\nFINAL - FINAL - FINAL: --- {pregunta_0} ---\n----------\n----------\n")

    paises, puntajes = [], []
    dtable = dict_table["Fase Final"].copy()
    dtable.rename(columns={"Pts.": "Pts"}, inplace=True)
    for linea in dtable.itertuples():
      paises.append(objs.id_pais[linea.pais])
      puntajes.append(linea.Pts)
    df_nombres = pd.DataFrame(data={"pais": paises, "Pts": puntajes}, index=range(len(paises)))
    print(df_nombres)

    df_ent = pd.concat([df_fixture_group_[["home", "score_0", "score_1", "away"]], df_fixture_quarter[["home", "score_0", "score_1", "away"]]], axis=0)

    df_ent.home = df_ent.home.map(objs.id_pais)
    df_ent.away = df_ent.away.map(objs.id_pais)

    return df_ent

  elif "1954" in pregunta_0 or "1958" in pregunta_0 or "1962" in pregunta_0 or "1966" in pregunta_0 or "1970" in pregunta_0:

    if "grupo" in pregunta_0:
      objs.ind0[1] = "grupo"
      df_fixture_group_ = agregar_features(df1[1].copy(), dic_agnos, ind=0)
      df_fixture_quarter = df3[["home", "score", "away"]].iloc[:4].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[4:6].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[6].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[7].copy()

    elif "quarter" in pregunta_0:
      objs.ind0[1] = "quarter"
      df_fixture_group_ = df2.copy()
      df_fixture_quarter = df1[1].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[4:6].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[6].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[7].copy()

    elif "semi" in pregunta_0:
      objs.ind0[1] = "semi"
      df_fixture_group_ = df2.copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[:4].copy()
      df_fixture_semi = df1[1].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[6].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[7].copy()

    elif "third" in pregunta_0:
      objs.ind0[1] = "third"
      df_fixture_group_ = df2.copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[:4].copy()
      df_fixture_semi = df1[0][["home", "away"]].iloc[4:6].copy()
      df_fixture_third = df1[1].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[7].copy()

    elif "final" in pregunta_0:
      objs.ind0[1] = "final"
      df_fixture_group_ = df2.copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[0:4].copy()
      df_fixture_semi = df1[0][["home", "away"]].iloc[4:6].copy()
      df_fixture_third = df1[0][["home", "away"]].iloc[6].copy()
      df_fixture_final = df1[1].copy()

    objs.ind0[0] = "grupo"
    dict_table = objs.fase_de_grupos(dict_g, df_fixture_group_, modelo)

    df_fixture_quarter = objs.camb_grp_elm_smpl(df_fixture_group_, df_fixture_quarter, dict_table)

    objs.n_partidos = df_fixture_group_.shape[0] + 1 # cuartos: partidos de grupos + 1
    objs.ind0[0] = "quarter"
    objs.jugar(df_fixture_quarter, modelo)

    df_fixture_semi = objs.ordenar_elm_smpl(df_fixture_quarter, df_fixture_semi)

    objs.ind0[0] = "semi"
    objs.jugar(df_fixture_semi, modelo)

    df_fixture_third = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_third, tipo="Loser")

    objs.ind0[0] = "third"
    objs.jugar(df_fixture_third, modelo)

    df_fixture_final = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_final)

    print(f"\n\n----------\n----------\n\nFINAL - FINAL - FINAL: --- {pregunta_0} ---\n----------\n----------\n")
    objs.ind0[0] = "final"
    objs.jugar(df_fixture_final, modelo)

    df_ent = pd.concat([df_fixture_group_[["home", "score_0", "score_1", "away"]], df_fixture_quarter[["home", "score_0", "score_1", "away"]], df_fixture_semi[["home", "score_0", "score_1", "away"]], df_fixture_third[["home", "score_0", "score_1", "away"]], df_fixture_final[["home", "score_0", "score_1", "away"]]], axis=0)

    df_ent.home = df_ent.home.map(objs.id_pais)
    df_ent.away = df_ent.away.map(objs.id_pais)

    return df_ent
  
  elif "2026" in pregunta_0:
    # 48 equipos en 12 grupos; pasan los 2 primeros y los 8 mejores terceros a dieciseisavos (Match 73 a 88), que se arman
    # con el reglamento de mejores terceros (FORMATOS[2026], o el personalizado). Despues del fixture de 2026 (df3):
    # filas 0-15 dieciseisavos, 16-23 octavos, 24-27 cuartos, 28-29 semifinales, 30 tercer puesto y 31 final
    agno = 2026
    filas = lambda a, b: df3[["home", "score", "away"]].iloc[a:b].copy()
    marcadores_32 = df3
    df_fixture_group_ = df2.copy()
    df_fixture_knockout, df_fixture_quarter, df_fixture_semi = filas(16, 24), filas(24, 28), filas(28, 30)
    df_fixture_third, df_fixture_final = df3[["home", "score", "away"]].iloc[30].copy(), df3[["home", "score", "away"]].iloc[31].copy()

    if "dieciseisavos" in pregunta_0:
      objs.ind0[1] = "dieciseisavos"
      # la consulta del usuario trae score_0 y score_1; la llave lee "score" ("3:1")
      marcadores_32 = df1[1].assign(score=df1[1]["score_0"].astype(str) + ":" + df1[1]["score_1"].astype(str))
    elif "grupo" in pregunta_0:
      objs.ind0[1] = "grupo"
      df_fixture_group_ = agregar_features(df1[1].copy(), dic_agnos, ind=0)
    elif "knockout" in pregunta_0:
      objs.ind0[1] = "knockout"
      df_fixture_knockout = df1[1].copy()
    elif "quarter" in pregunta_0:
      objs.ind0[1] = "quarter"
      df_fixture_quarter = df1[1].copy()
    elif "semi" in pregunta_0:
      objs.ind0[1] = "semi"
      df_fixture_semi = df1[1].copy()
    elif "third" in pregunta_0:
      objs.ind0[1] = "third"
      df_fixture_third = df1[1].copy()
    elif "final" in pregunta_0:
      objs.ind0[1] = "final"
      df_fixture_final = df1[1].copy()

    objs.ind0[0] = "grupo"
    dict_table = objs.fase_de_grupos(dict_g, df_fixture_group_, modelo)

    # dieciseisavos con las tablas de grupos YA jugadas y el reglamento de mejores terceros
    df_fixture_32 = alinear_llave_terceros(dict_table, agno, df_grupos=df_fixture_group_, personalizado=reglamento_terceros)
    mejores_terceros = df_fixture_32.attrs["mejores_terceros"]
    cruces_32 = df_fixture_32[["home", "away"]].rename(columns={"home": "cruce_local", "away": "cruce_visita"})
    df_fixture_32.insert(1, "score", marcadores_32["score"].iloc[:16].to_numpy())
    df_fixture_32 = objs.camb_grp_elm_smpl(df_fixture_group_, df_fixture_32, dict_table)

    objs.n_partidos = df_fixture_group_.shape[0] + 1 # dieciseisavos: partidos de grupos + 1 (Match 73)
    objs.ind0[0] = "dieciseisavos"
    objs.jugar(df_fixture_32, modelo)

    df_fixture_knockout = objs.ordenar_elm_smpl(df_fixture_32, df_fixture_knockout)
    objs.ind0[0] = "knockout"
    objs.jugar(df_fixture_knockout, modelo)

    df_fixture_quarter = objs.ordenar_elm_smpl(df_fixture_knockout, df_fixture_quarter)
    objs.ind0[0] = "quarter"
    objs.jugar(df_fixture_quarter, modelo)

    df_fixture_semi = objs.ordenar_elm_smpl(df_fixture_quarter, df_fixture_semi)
    objs.ind0[0] = "semi"
    objs.jugar(df_fixture_semi, modelo)

    df_fixture_third = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_third, tipo="Loser")
    objs.ind0[0] = "third"
    objs.jugar(df_fixture_third, modelo)

    df_fixture_final = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_final)
    objs.ind0[0] = "final"
    objs.jugar(df_fixture_final, modelo)

    columnas = ["home", "score_0", "score_1", "away"]
    df_ent = pd.concat([df[columnas] for df in (df_fixture_group_, df_fixture_32, df_fixture_knockout, df_fixture_quarter,
                                                  df_fixture_semi, df_fixture_third, df_fixture_final)], axis=0)
    df_ent.home = df_ent.home.map(objs.id_pais)
    df_ent.away = df_ent.away.map(objs.id_pais)
    df_ent.attrs["mejores_terceros"] = mejores_terceros.assign(pais=mejores_terceros.pais.map(objs.id_pais))
    df_ent.attrs["llave_octavos"] = cruces_32.assign(local=df_fixture_32.home.map(objs.id_pais).to_numpy(),
                                                     visita=df_fixture_32.away.map(objs.id_pais).to_numpy())
    return df_ent

  elif "1986" in pregunta_0 or "1990" in pregunta_0 or "1994" in pregunta_0:
    agno = int(pregunta_0[-4:])

    if "grupo" in pregunta_0:
      objs.ind0[1] = "grupo"
      df_fixture_group_ = agregar_features(df1[1].copy(), dic_agnos, ind=0)
      marcadores_knockout = df3
      df_fixture_quarter = df3[["home", "score", "away"]].iloc[8:12].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[12:14].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "knockout" in pregunta_0:
      objs.ind0[1] = "knockout"
      df_fixture_group_ = df2.copy()
      # la consulta del usuario trae score_0 y score_1; la llave lee "score" ("3:1")
      marcadores_knockout = df1[1].assign(score=df1[1]["score_0"].astype(str) + ":" + df1[1]["score_1"].astype(str))
      df_fixture_quarter = df3[["home", "score", "away"]].iloc[8:12].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[12:14].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "quarter" in pregunta_0:
      objs.ind0[1] = "quarter"
      df_fixture_group_ = df2.copy()
      marcadores_knockout = df1[0]
      df_fixture_quarter = df1[1].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[12:14].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "semi" in pregunta_0:
      objs.ind0[1] = "semi"
      df_fixture_group_ = df2.copy()
      marcadores_knockout = df1[0]
      df_fixture_quarter = df1[0][["home", "away"]].iloc[8:12].copy()
      df_fixture_semi = df1[1].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "third" in pregunta_0:
      objs.ind0[1] = "third"
      df_fixture_group_ = df2.copy()
      marcadores_knockout = df1[0]
      df_fixture_quarter = df1[0][["home", "away"]].iloc[8:12].copy()
      df_fixture_semi = df1[0][["home", "away"]].iloc[12:14].copy()
      df_fixture_third = df1[1].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "final" in pregunta_0:
      objs.ind0[1] = "final"
      df_fixture_group_ = df2.copy()
      marcadores_knockout = df1[0]
      df_fixture_quarter = df1[0][["home", "away"]].iloc[8:12].copy()
      df_fixture_semi = df1[0][["home", "away"]].iloc[12:14].copy()
      df_fixture_third = df1[0][["home", "away"]].iloc[14].copy()
      df_fixture_final = df1[1].copy()

    objs.ind0[0] = "grupo"
    dict_table = objs.fase_de_grupos(dict_g, df_fixture_group_, modelo)

    # la llave se arma con las tablas de grupos YA jugadas, con el reglamento FIFA o el personalizado desde el frontend
    df_fixture_knockout = alinear_llave_terceros(dict_table, agno, df_grupos=df_fixture_group_, personalizado=reglamento_terceros)
    mejores_terceros = df_fixture_knockout.attrs["mejores_terceros"]
    cruces_octavos = df_fixture_knockout[["home", "away"]].rename(columns={"home": "cruce_local", "away": "cruce_visita"})
    df_fixture_knockout.insert(1, "score", marcadores_knockout["score"].iloc[:8].to_numpy())

    df_fixture_knockout = objs.camb_grp_elm_smpl(df_fixture_group_, df_fixture_knockout, dict_table)

    objs.n_partidos = df_fixture_group_.shape[0] + 1 # octavos: partidos de grupos + 1
    objs.ind0[0] = "knockout"
    objs.jugar(df_fixture_knockout, modelo)

    df_fixture_quarter = objs.ordenar_elm_smpl(df_fixture_knockout, df_fixture_quarter)

    objs.ind0[0] = "quarter"
    objs.jugar(df_fixture_quarter, modelo)

    df_fixture_semi = objs.ordenar_elm_smpl(df_fixture_quarter, df_fixture_semi)

    objs.ind0[0] = "semi"
    objs.jugar(df_fixture_semi, modelo)

    df_fixture_third = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_third, tipo="Loser")

    objs.ind0[0] = "third"
    objs.jugar(df_fixture_third, modelo)

    df_fixture_final = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_final)

    print(f"\n\n----------\n----------\n\nFINAL - FINAL - FINAL: --- {pregunta_0} ---\n----------\n----------\n")
    objs.ind0[0] = "final"
    objs.jugar(df_fixture_final, modelo)

    df_ent = pd.concat([df_fixture_group_[["home", "score_0", "score_1", "away"]], df_fixture_knockout[["home", "score_0", "score_1", "away"]], df_fixture_quarter[["home", "score_0", "score_1", "away"]], df_fixture_semi[["home", "score_0", "score_1", "away"]], df_fixture_third[["home", "score_0", "score_1", "away"]], df_fixture_final[["home", "score_0", "score_1", "away"]]], axis=0)

    df_ent.home = df_ent.home.map(objs.id_pais)
    df_ent.away = df_ent.away.map(objs.id_pais)
    df_ent.attrs["mejores_terceros"] = mejores_terceros.assign(pais=mejores_terceros.pais.map(objs.id_pais))
    df_ent.attrs["llave_octavos"] = cruces_octavos.assign(local=df_fixture_knockout.home.map(objs.id_pais).to_numpy(),
                                                          visita=df_fixture_knockout.away.map(objs.id_pais).to_numpy())

    return df_ent

  elif "1974" in pregunta_0 or "1978" in pregunta_0:

    if "grupo" in pregunta_0:
      objs.ind0[1] = "grupo"
      df_fixture_group_ = agregar_features(df1[1].copy(), dic_agnos, ind=0)
      df_fixture_doce = df3[["home", "score", "away"]].iloc[:12].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[12].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[13].copy()

    elif "doce" in pregunta_0:
      objs.ind0[1] = "doce"
      df_fixture_group_ = df2.copy()
      df_fixture_doce = df1[1].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[12].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[13].copy()

    elif "third" in pregunta_0:
      objs.ind0[1] = "third"
      df_fixture_group_ = df2.copy()
      df_fixture_doce = df1[0][["home", "away"]].iloc[:12].copy()
      df_fixture_third = df1[1].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[13].copy()

    elif "final" in pregunta_0:
      objs.ind0[1] = "final"
      df_fixture_group_ = df2.copy()
      df_fixture_doce = df1[0][["home", "away"]].iloc[:12].copy()
      df_fixture_third = df1[0][["home", "away"]].iloc[12].copy()
      df_fixture_final = df1[1].copy()

    dict_g_1974G0 = {"Grupo 1": dict_g.get("Grupo 1"),
                     "Grupo 2": dict_g.get("Grupo 2"),
                     "Grupo 3": dict_g.get("Grupo 3"),
                     "Grupo 4": dict_g.get("Grupo 4")}

    objs.ind0[0] = "grupo"
    dict_table = objs.fase_de_grupos(dict_g_1974G0, df_fixture_group_, modelo)

    objs.n_partidos = df_fixture_group_.shape[0] + df_fixture_doce.shape[0] + 1
    df_fixture_doce = objs.camb_grp_elm_smpl(df_fixture_group_, df_fixture_doce, dict_table, year=int(pregunta_0[-4:]))

    objs.ind0[0] = "doce"
    dict_table = objs.fase_de_grupos(df_fixture_doce[1], df_fixture_doce[0].iloc[:, :2 * COLS_FASE_POR_EQUIPO], modelo)

    df_fixture_third = objs.camb_grp_elm_smpl(df_fixture_doce[0].iloc[:, :2 * COLS_FASE_POR_EQUIPO], df_fixture_third, dict_table)

    objs.ind0[0] = "third"
    if df_fixture_third.shape[0] > 1:
      objs.jugar(df_fixture_third.iloc[0:-1], modelo)
    elif df_fixture_third.shape[0] == 1:
      objs.jugar(df_fixture_third, modelo)

    df_fixture_final = objs.camb_grp_elm_smpl(df_fixture_doce[0].iloc[:, :2 * COLS_FASE_POR_EQUIPO], df_fixture_final, dict_table)

    print(f"\n\n----------\n----------\n\nFINAL - FINAL - FINAL: --- {pregunta_0} ---\n----------\n----------\n")
    objs.ind0[0] = "final"
    objs.jugar(df_fixture_final, modelo)

    df_ent = pd.concat([df_fixture_group_[["home", "score_0", "score_1", "away"]], df_fixture_doce[0][["home", "score_0", "score_1", "away"]], df_fixture_third[["home", "score_0", "score_1", "away"]], df_fixture_final[["home", "score_0", "score_1", "away"]]], axis=0)

    df_ent.home = df_ent.home.map(objs.id_pais)
    df_ent.away = df_ent.away.map(objs.id_pais)

    return df_ent

  elif "1982" in pregunta_0:

    if "grupo" in pregunta_0:
      objs.ind0[1] = "grupo"
      df_fixture_group_ = agregar_features(df1[1].copy(), dic_agnos, ind=0)
      df_fixture_doce = df3[["home", "score", "away"]].iloc[:12].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[12:14].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "doce" in pregunta_0:
      objs.ind0[1] = "doce"
      df_fixture_group_ = df2.copy()
      df_fixture_doce = df1[1].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[12:14].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "semi" in pregunta_0:
      objs.ind0[1] = "semi"
      df_fixture_group_ = df2.copy()
      df_fixture_doce = df1[0][["home", "away"]].iloc[:12].copy().copy()
      df_fixture_semi = df1[1].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "third" in pregunta_0:
      objs.ind0[1] = "third"
      df_fixture_group_ = df2.copy()
      df_fixture_doce = df1[0][["home", "away"]].iloc[:12].copy()
      df_fixture_semi = df1[0][["home", "away"]].iloc[12:14].copy()
      df_fixture_third = df1[1].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "final" in pregunta_0:
      objs.ind0[1] = "final"
      df_fixture_group_ = df2.copy()
      df_fixture_doce = df1[0][["home", "away"]].iloc[:12].copy()
      df_fixture_semi = df1[0][["home", "away"]].iloc[12:14].copy()
      df_fixture_third = df1[0][["home", "away"]].iloc[14].copy()
      df_fixture_final = df1[1].copy()

    dict_g_1982G0 = {"Grupo 1": dict_g.get("Grupo 1"),
                     "Grupo 2": dict_g.get("Grupo 2"),
                     "Grupo 3": dict_g.get("Grupo 3"),
                     "Grupo 4": dict_g.get("Grupo 4"),
                     "Grupo 5": dict_g.get("Grupo 5"),
                     "Grupo 6": dict_g.get("Grupo 6")}

    objs.ind0[0] = "grupo"
    dict_table = objs.fase_de_grupos(dict_g_1982G0, df_fixture_group_, modelo)

    df_fixture_doce = objs.camb_grp_elm_smpl(df_fixture_group_, df_fixture_doce, dict_table, year=int(pregunta_0[-4:]))

    objs.ind0[0] = "doce"
    dict_table = objs.fase_de_grupos(df_fixture_doce[1], df_fixture_doce[0].iloc[:, :2 * COLS_FASE_POR_EQUIPO], modelo)

    objs.n_partidos = df_fixture_group_.shape[0] + df_fixture_doce[0].shape[0] + 1
    
    df_fixture_semi = objs.camb_grp_elm_smpl(df_fixture_doce[0].iloc[:, :2 * COLS_FASE_POR_EQUIPO], df_fixture_semi, dict_table)

    objs.ind0[0] = "semi"
    objs.jugar(df_fixture_semi, modelo)

    df_fixture_third = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_third, tipo="Loser")

    objs.ind0[0] = "third"
    objs.jugar(df_fixture_third, modelo)

    df_fixture_final = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_final)

    print(f"\n\n----------\n----------\n\nFINAL - FINAL - FINAL: --- {pregunta_0} ---\n----------\n----------\n")
    objs.ind0[0] = "final"
    objs.jugar(df_fixture_final, modelo)

    df_ent = pd.concat([df_fixture_group_[["home", "score_0", "score_1", "away"]], df_fixture_doce[0][["home", "score_0", "score_1", "away"]], df_fixture_semi[["home", "score_0", "score_1", "away"]], df_fixture_third[["home", "score_0", "score_1", "away"]], df_fixture_final[["home", "score_0", "score_1", "away"]]], axis=0)

    df_ent.home = df_ent.home.map(objs.id_pais)
    df_ent.away = df_ent.away.map(objs.id_pais)

    return df_ent
