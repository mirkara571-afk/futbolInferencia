"""Carga del predictor en uso (extraido de modulo.py en el refactor H.2)."""
# predictor cargado por carpeta: cargar un LSTM de Keras tarda segundos y no conviene repetirlo en cada consulta
_PREDICTORES = {}

def predictor_en_uso(carpeta):
  """Predictor de (carpeta) con "cargar_predictor", guardado en memoria; se vuelve a cargar si cambia su esquema.json"""
  from .predictores import ESQUEMA, cargar_predictor
  version = (carpeta / ESQUEMA).stat().st_mtime
  if _PREDICTORES.get(carpeta, (None,))[0] != version:
    _PREDICTORES[carpeta] = (version, cargar_predictor(carpeta)[0])
  return _PREDICTORES[carpeta][1]
