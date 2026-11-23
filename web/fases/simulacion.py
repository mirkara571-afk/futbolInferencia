"""Simulacion de torneos: normali, func_prediccion_orden y consulta_general.

Extraido de modulo.py en el refactor H.2. Importa los helpers de datos/metricas/modelo desde
.modulo; modulo.py NO importa este modulo para evitar el import circular.
"""
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from django.conf import settings

from .datos import desempates_reales, gana_local_desempate, marcador_real, resultados_reales
from .formatos import alinear_llave_terceros
from .modelo import predictor_en_uso
from .models import internacionales
from .modulo import (
    C_S_DELETE_INDEX, C_S_Y_INDEX, COLS_FASE_POR_EQUIPO, COLS_MODELO_POR_EQUIPO,
    DROP_METRICS, N_STEPS, OPTIONAL_METRICS,
    agregar_features, calculo_metricas_0, calculo_metricas_1, formar_dataset_real,
    obtener_clasificaciones, obtener_dic_agnos, reiniciar_opcionales,
)

logger = logging.getLogger(__name__)


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
    # D.4: historial de partidos previos por equipo (df cronologico) para armar secuencias k>1; se carga perezoso
    self.historia = None

  def _historia_inicial(self):
    """D.4: historial previo de cada equipo (sus partidos internacionales recientes) para armar secuencias k>1.

    Carga el dataset del ciclo internacionales (partidos_features_internacionales.csv) con las mismas
    columnas que "construir_secuencias"; si no existe, devuelve un df vacio (las secuencias arrancan en cero)."""
    from pathlib import Path

    from django.conf import settings
    ruta = Path(settings.BASE_DIR).parent / "data" / "datasets" / "partidos_features_internacionales.csv"
    if not ruta.exists():
      return pd.DataFrame()
    return pd.read_csv(ruta)

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
    from .dataset import ranking_por_agno
    ranking = ranking_por_agno() if isinstance(self.obj0, int) else {}
    for i, lado in ((0, "home"), (1, "away")):
      tabla[f"ranking_{i}"] = tabla[lado].map(lambda p: ranking.get((self.obj0, p), 200))
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
      tabla = self.contexto_simulacion(lote_desde_entrada(X).tabla)
      if getattr(model, "config", {}).get("k", 1) > 1:
        # D.4: secuencias k>1 armadas con el historial partido a partido; se acumula tras cada fase
        if self.historia is None:
          self.historia = self._historia_inicial()
        tabla_mapeada = model.mapear_tabla(tabla)
        entrada = Lote(tabla, secuencias=None, historia=self.historia)
      else:
        tabla_mapeada = None
        entrada = Lote(tabla)
      Y0 = model.predecir(entrada)
      if tabla_mapeada is not None:
        self.historia = pd.concat([self.historia, tabla_mapeada], ignore_index=True)
      resultado, marcador, probs = resultado_y_marcador(Y0, getattr(model, "config", {}).get("umbral_empate", 0.0),
                                                        model.probabilidades(entrada) if hasattr(model, "probabilidades") else None)
      victoria = 2 if isinstance(self.obj0, int) and self.obj0 < 1994 else 3
      self.__dict__.setdefault("ganador_empate", {})
      self.__dict__.setdefault("probs_partido", {})
      for r, (g0, g1), p, fila in zip(resultado, marcador, probs, np.atleast_2d(X)):
        self.ganador_empate[(fila[0], fila[COLS_MODELO_POR_EQUIPO])] = "home" if p[0] >= p[2] else "away"
        self.probs_partido[(int(fila[0]), int(fila[COLS_MODELO_POR_EQUIPO]))] = [float(x) for x in p]
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
      if df_fix_group_n.empty:
        raise ValueError(f"Recepción incompleta: el {group} no tiene partidos; la fase de grupos debe venir completa")
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
    logger.debug("df_nombres:\n%s", df_nombres)

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

  logger.debug("primero: (df1)\n%s\n\n\nsegundo: (df2)\n%s\n\n\ntercero: (df3)\n%s", df1, df2, df3)

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
  objs.anfitriones = {objs.pais_id[n] for n in internacionales.objects.filter(categoria="mundial", agno=agno_consulta, neutral="0")
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

    logger.debug("indice base %s", objs.n_partidos)
    df_fixture_quarter = objs.ordenar_elm_smpl(df_fixture_knockout, df_fixture_quarter) 

    objs.ind0[0] = "quarter"
    objs.jugar(df_fixture_quarter, modelo)

    logger.debug("indice base %s", objs.n_partidos)
    df_fixture_semi = objs.ordenar_elm_smpl(df_fixture_quarter, df_fixture_semi)

    objs.ind0[0] = "semi"
    objs.jugar(df_fixture_semi, modelo)

    df_fixture_third = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_third, tipo="Loser")

    objs.ind0[0] = "third"
    objs.jugar(df_fixture_third, modelo)

    df_fixture_final = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_final)

    logger.debug("FINAL --- %s", pregunta_0)
    objs.ind0[0] = "final"
    objs.jugar(df_fixture_final, modelo)

    df_ent = pd.concat([df_fixture_group_[["home", "score_0", "score_1", "away"]], df_fixture_knockout[["home", "score_0", "score_1", "away"]], df_fixture_quarter[["home", "score_0", "score_1", "away"]], df_fixture_semi[["home", "score_0", "score_1", "away"]], df_fixture_third[["home", "score_0", "score_1", "away"]], df_fixture_final[["home", "score_0", "score_1", "away"]]], axis=0)

    if hasattr(objs, "probs_partido"):
      df_ent.attrs["probs"] = [objs.probs_partido.get((int(h), int(a))) for h, a in zip(df_ent["home"], df_ent["away"])]
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

    logger.debug("FINAL --- %s", pregunta_0)
    objs.ind0[0] = "final"
    objs.jugar(df_fixture_final, modelo)

    df_ent = pd.concat([df_fixture_knockout[["home", "score_0", "score_1", "away"]], df_fixture_quarter[["home", "score_0", "score_1", "away"]], df_fixture_semi[["home", "score_0", "score_1", "away"]], df_fixture_third[["home", "score_0", "score_1", "away"]], df_fixture_final[["home", "score_0", "score_1", "away"]]], axis=0)

    if hasattr(objs, "probs_partido"):
      df_ent.attrs["probs"] = [objs.probs_partido.get((int(h), int(a))) for h, a in zip(df_ent["home"], df_ent["away"])]
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

    logger.debug("FINAL --- %s", pregunta_0)

    paises, puntajes = [], []
    dtable = dict_table["Fase Final"].copy()
    dtable.rename(columns={"Pts.": "Pts"}, inplace=True)
    for linea in dtable.itertuples():
      paises.append(objs.id_pais[linea.pais])
      puntajes.append(linea.Pts)
    df_nombres = pd.DataFrame(data={"pais": paises, "Pts": puntajes}, index=range(len(paises)))
    logger.debug("df_nombres:\n%s", df_nombres)

    df_ent = pd.concat([df_fixture_group_[["home", "score_0", "score_1", "away"]], df_fixture_quarter[["home", "score_0", "score_1", "away"]]], axis=0)

    if hasattr(objs, "probs_partido"):
      df_ent.attrs["probs"] = [objs.probs_partido.get((int(h), int(a))) for h, a in zip(df_ent["home"], df_ent["away"])]
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

    logger.debug("FINAL --- %s", pregunta_0)
    objs.ind0[0] = "final"
    objs.jugar(df_fixture_final, modelo)

    df_ent = pd.concat([df_fixture_group_[["home", "score_0", "score_1", "away"]], df_fixture_quarter[["home", "score_0", "score_1", "away"]], df_fixture_semi[["home", "score_0", "score_1", "away"]], df_fixture_third[["home", "score_0", "score_1", "away"]], df_fixture_final[["home", "score_0", "score_1", "away"]]], axis=0)

    if hasattr(objs, "probs_partido"):
      df_ent.attrs["probs"] = [objs.probs_partido.get((int(h), int(a))) for h, a in zip(df_ent["home"], df_ent["away"])]
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
    if hasattr(objs, "probs_partido"):
      df_ent.attrs["probs"] = [objs.probs_partido.get((int(h), int(a))) for h, a in zip(df_ent["home"], df_ent["away"])]
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

    logger.debug("FINAL --- %s", pregunta_0)
    objs.ind0[0] = "final"
    objs.jugar(df_fixture_final, modelo)

    df_ent = pd.concat([df_fixture_group_[["home", "score_0", "score_1", "away"]], df_fixture_knockout[["home", "score_0", "score_1", "away"]], df_fixture_quarter[["home", "score_0", "score_1", "away"]], df_fixture_semi[["home", "score_0", "score_1", "away"]], df_fixture_third[["home", "score_0", "score_1", "away"]], df_fixture_final[["home", "score_0", "score_1", "away"]]], axis=0)

    if hasattr(objs, "probs_partido"):
      df_ent.attrs["probs"] = [objs.probs_partido.get((int(h), int(a))) for h, a in zip(df_ent["home"], df_ent["away"])]
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

    logger.debug("FINAL --- %s", pregunta_0)
    objs.ind0[0] = "final"
    objs.jugar(df_fixture_final, modelo)

    df_ent = pd.concat([df_fixture_group_[["home", "score_0", "score_1", "away"]], df_fixture_doce[0][["home", "score_0", "score_1", "away"]], df_fixture_third[["home", "score_0", "score_1", "away"]], df_fixture_final[["home", "score_0", "score_1", "away"]]], axis=0)

    if hasattr(objs, "probs_partido"):
      df_ent.attrs["probs"] = [objs.probs_partido.get((int(h), int(a))) for h, a in zip(df_ent["home"], df_ent["away"])]
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

    logger.debug("FINAL --- %s", pregunta_0)
    objs.ind0[0] = "final"
    objs.jugar(df_fixture_final, modelo)

    df_ent = pd.concat([df_fixture_group_[["home", "score_0", "score_1", "away"]], df_fixture_doce[0][["home", "score_0", "score_1", "away"]], df_fixture_semi[["home", "score_0", "score_1", "away"]], df_fixture_third[["home", "score_0", "score_1", "away"]], df_fixture_final[["home", "score_0", "score_1", "away"]]], axis=0)

    if hasattr(objs, "probs_partido"):
      df_ent.attrs["probs"] = [objs.probs_partido.get((int(h), int(a))) for h, a in zip(df_ent["home"], df_ent["away"])]
    df_ent.home = df_ent.home.map(objs.id_pais)
    df_ent.away = df_ent.away.map(objs.id_pais)

    return df_ent
