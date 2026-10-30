from django.shortcuts import render
from django.http import JsonResponse
from .models import partidos, fixtures, grupos, clasificaciones
from django.core.serializers.json import DjangoJSONEncoder
import json
import pandas as pd
import numpy as np

from .modulo import *

agnos = ["1934", "1938", "1950", "1954", "1958", "1962", "1966", "1970", "1974", "1978", "1982", "1986", "1990", "1994", "1998", "2002", "2006", "2010", "2014", "2018", "2026", "2022"]

def formar_dataset_consulta(valores):
    """
    Recive del template y arroja un df con formato adecuado para predecir

    - Args:
        - valores: lista de listas con datos para hacer un dataframe

    - Returns: 
        - dfd: dataframe con indices reiniciados y con columnas "home", "score_0", "score_1", "away"
    """
    dfd = pd.DataFrame()
    for val0, val1 in zip(valores[2:], range(len(valores[2:]))):
        print(f"val0\n{val0}")
        dfd = pd.concat([dfd, pd.DataFrame(data=np.array([[val0[0], int(val0[2]), int(val0[3]), val0[1]]]), columns=["home", "score_0", "score_1", "away"])], axis=0)
    dfd = dfd.reset_index(drop=True)#.iloc[:,1:]

    #print(f"dfd\n{dfd[["home", "score_0", "score_1", "away"]]}, {dfd.shape}")
    return(dfd)

dic_torneos = {"Fase de Grupos": "grupo", "Fase Inicial":"doce", "Fase Final": "Fase Final", "16vos": "dieciseisavos", "octavos": "knockout", "cuartos":"quarter", "semifinal":"semi", "3er_lugar":"third", "final": "final"}

def leer_reglamento(agno, datos):
    """
    Construye el reglamento de mejores terceros enviado por el template

    - Args:
        - agno: año del mundial (str o int)
        - datos: None (usar reglamento FIFA oficial) o {"llave": [...8], "cruces": {"ABCD": "CDAB", ...}}

    - Returns:
        - None o un ReglamentoTerceros ya validado (lanza ReglamentoInvalido antes de correr el modelo)
    """
    if not datos:
        return None
    return ReglamentoTerceros.desde_json(int(agno), datos)

def leer_metricas(datos, catalogo, nombre, defecto):
    """
    Valida una lista de metricas enviada por el template contra su catalogo de modulo.py

    - Args:
        - datos: None (usar el valor por defecto) o lista de nombres (ej. ["GF", "GC"])
        - catalogo: BASE_METRICS u OPTIONAL_METRICS
        - nombre: clave del JSON, para el mensaje de error
        - defecto: valor si el template no envio la clave (DROP_METRICS u OPTIONAL_METRICS)

    - Returns:
        - tupla con las metricas (lanza ValueError si llega algo fuera del catalogo)
    """
    if datos is None:
        return tuple(defecto)
    if not isinstance(datos, list) or not all(isinstance(m, str) for m in datos):
        raise ValueError(f"{nombre} debe ser una lista de nombres de metricas")
    desconocidas = set(datos) - set(catalogo)
    if desconocidas:
        raise ValueError(f"{nombre}: metricas no validas {sorted(desconocidas)}")
    return tuple(datos)

def catalogo_metricas():
    """Metricas que el template muestra como casillas (fuente de verdad: modulo.py)"""
    return {"base": BASE_METRICS, "opcionales": OPTIONAL_METRICS}

def procesar_recepcion_0(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        opcion_0 = data.get('fecha', None)
        if(opcion_0 in agnos):
            df = formar_dataset_real("partidos", opcion_0)
            print(f"df:\n{df}")
            json_records_0 = df.to_json(orient='records')
            context = {'partidos_reales': json_records_0}
            print(f"Estos son mis partidos\n{df}")
            return(JsonResponse(context))

    return(render(request, "prediccion.html", {"reglamentos_terceros": reglamentos_oficiales(), "catalogo_metricas": catalogo_metricas()}))

def predecir_recepcion(data):
    """
    Ejecuta una prediccion completa a partir del JSON que envia la web o la API.

    - data: {"recepcion": [...], "reglamento": ..., "drop_metricas": [...], "metricas_opcionales": [...]}

    - Returns: DataFrame de respuesta de consulta_general (con attrs)
    - Lanza ValueError o ReglamentoInvalido si la peticion no es valida
    """
    recepcion = data.get('recepcion', None)
    if not recepcion:
        raise ValueError("Falta 'recepcion'")
    reglamento = leer_reglamento(recepcion[0], data.get('reglamento'))
    drop_metricas = leer_metricas(data.get('drop_metricas'), BASE_METRICS, "drop_metricas", DROP_METRICS)
    metricas_opcionales = leer_metricas(data.get('metricas_opcionales'), OPTIONAL_METRICS, "metricas_opcionales", OPTIONAL_METRICS)
    validar_valores_neutros(drop_metricas, metricas_opcionales)
    recepc = formar_dataset_consulta(recepcion)
    if recepcion[1] == "Fase de Grupos":
        recepc["agno"] = int(recepcion[0])
    return consulta_general(dic_torneos[recepcion[1]] + recepcion[0], recepc, reglamento_terceros=reglamento,
                            drop_metricas=drop_metricas, metricas_opcionales=metricas_opcionales)


def _respuesta_json(respuesta, recepcion):
    """DataFrame de consulta_general -> JSON de la web/API (partidos, terceros y llave)"""
    extras = dict(respuesta.attrs)
    respuesta = respuesta.reset_index(drop=True)
    partidos = [{"home": row.home, "score": f"{int(row.score_0)}:{int(row.score_1)}", "away": row.away}
                for row in respuesta.itertuples()]
    payload = {"year": int(recepcion[0]), "partidos": partidos}
    for clave in ("mejores_terceros", "llave_octavos"):
        if clave in extras:
            payload[clave] = json.loads(extras[clave].to_json(orient="records"))
    return payload


def procesar_recepcion_1(request):
    """Vista usada al precionar el boton "predecir" """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            recepcion = data.get('recepcion', None)
            if recepcion:
                try:
                    respuesta = predecir_recepcion(data)
                except (ReglamentoInvalido, ValueError) as error:
                    return JsonResponse({'error': str(error)}, status=400)
                # la web espera los campos como strings JSON (los parsea en el template); la API los devuelve como listas
                payload = _respuesta_json(respuesta, recepcion)
                for clave in ("partidos", "mejores_terceros", "llave_octavos"):
                    if clave in payload:
                        payload[clave] = json.dumps(payload[clave], ensure_ascii=False)
                return JsonResponse(payload)
        except json.JSONDecodeError as e:
            print(f"Error al decodificar JSON: {e}")
            return JsonResponse({'error': 'JSON inválido'}, status=400)

    return JsonResponse({'error': 'Método no permitido'}, status=405)
