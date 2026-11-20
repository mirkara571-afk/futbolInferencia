import numpy as np
import pandas as pd
import tensorflow as tf

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


# re-exportados desde "modulo" (hub de H.2) para montecarlo.py y tests.py
from .datos import marcador_real, desempates_reales, resultados_reales  # noqa: F401
from .modelo import predictor_en_uso  # noqa: F401

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
      if valores is not None and des.agno == int(valores):
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
      if des.agno == int(valores):
        df0.append([des.home, des.score, des.away, des.agno])
    df = pd.DataFrame(df0, columns=["home", "score", "away", "year"])
    df.home, df.score, df.away, df.year = df.home.astype('O'), df.score.astype('O'), df.away.astype('O'), df.year.astype('int64')
    return df

  elif ind0 == "grupos":
    df0 = list()
    for des in grupos.objects.all():
      if des.agno == int(valores):
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

# métricas derivadas: extraídas a metricas.py en el refactor H.2
from .metricas import calc_gkps, calc_mds, calc_mms, calc_mos, calc_rate


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

# emparejamiento por mejores terceros: extraido a formatos.py en el refactor H.2; re-exporta
# ReglamentoTerceros, ReglamentoInvalido, FORMATOS, alinear_llave_terceros y reglamentos_oficiales
from .formatos import *  # noqa: F403


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

