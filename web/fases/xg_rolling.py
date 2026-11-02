"""
Features rolling de xG sin fuga (plan 07, tarea XG-01).

El xG real solo existe en los mundiales de 2018 y 2022 (xG_disponible == 1); el resto de partidos no aporta xG. Este
modulo construye, para cada equipo y lado, la media de xG anotado y recibido en sus partidos previos (en orden
(agno, orden)), el numero de partidos que aportan y un indicador de cobertura que permite al modelo ignorar el relleno.

Se usa tal cual en el brazo B del backtest (rolling real) y, alimentado con columnas xg_* ya imputadas, en el brazo C.
"""
import numpy as np

from .modelos_estadisticos import _columna


def construir_rolling(dataset, ventana=None, media_global=None):
  """
  ¿QUE HACE?
  Devuelve el mismo (dataset) con 8 columnas nuevas: por lado i en (0,1),
    - xg_roll_for_i     = media de xG anotado por el equipo en sus partidos previos con xG
    - xg_roll_against_i = media de xG recibido en esos mismos partidos previos
    - xg_roll_n_i       = numero de partidos previos que aportan xG
    - xg_roll_has_i     = 1 si xg_roll_n_i > 0, si no 0
  Cuando no hay previos, for/against se rellenan con (media_global) o 0.0 y n = has = 0.

  ¿COMO LO HACE?
  Ordena los partidos por (agno, orden) y recorre a cada equipo manteniendo su historial de xG anotado/recibido. Las
  medias de un partido p se calculan ANTES de incorporar el xG del propio p, por lo que el rolling nunca usa el partido
  objetivo (no-fuga). Solo aportan los partidos con xG disponible (columna xg_* no nula); la ventana W usa los ultimos
  W previos y None usa todos (expanding).

  Args:
    - dataset: DataFrame con al menos home_id, away_id, agno, orden, xg_0 y xg_1 (el xG falta como NaN)
    - ventana: None (expanding) o entero W >= 1 (ultimos W previos)
    - media_global: valor de relleno para for/against cuando no hay previos (None => 0.0)

  Returns:
    - el mismo DataFrame + columnas xg_roll_*
  """
  if ventana is not None and (not isinstance(ventana, (int, np.integer)) or int(ventana) < 1):
    raise ValueError(f"ventana debe ser None o un entero >= 1, se recibio {ventana!r}")
  ventana = int(ventana) if ventana is not None else None
  relleno = float(media_global) if media_global is not None else 0.0

  ds = dataset.copy()
  agno = _columna(ds, "agno")
  orden = _columna(ds, "orden")
  home = _columna(ds, "home_id").astype("int64")
  away = _columna(ds, "away_id").astype("int64")
  # disponibilidad explicita: si existe xG_disponible, una fila con 0 no aporta aunque sus xg_* no sean NaN
  disponible = _columna(ds, "xG_disponible", 1.0) != 0
  xg_0 = np.where(disponible, _columna(ds, "xg_0", np.nan), np.nan)
  xg_1 = np.where(disponible, _columna(ds, "xg_1", np.nan), np.nan)

  n = len(ds)
  for_cols = np.full((n, 2), relleno)
  against_cols = np.full((n, 2), relleno)
  n_cols = np.zeros((n, 2), dtype="int64")
  has_cols = np.zeros((n, 2), dtype="int64")

  # orden global cronologico: primero agno, luego orden dentro del torneo (lexsort: la ultima clave es la primaria)
  pos = np.lexsort((orden, agno))
  historia_for = {}
  historia_against = {}

  for p in pos:
    eq_h, eq_a = int(home[p]), int(away[p])
    for j, eq in enumerate((eq_h, eq_a)):
      historial = historia_for.get(eq)
      if historial:
        previos_for = historial[-ventana:] if ventana else historial
        m = len(previos_for)
        if m:
          previos_against = historia_against[eq][-m:]
          for_cols[p, j] = float(np.mean(previos_for))
          against_cols[p, j] = float(np.mean(previos_against))
          n_cols[p, j] = m
          has_cols[p, j] = 1
      # sin historial: se conserva el relleno y n = has = 0

    # incorporar el xG del propio partido solo despues de calcular sus medias (no-fuga estricta)
    if not np.isnan(xg_0[p]) and not np.isnan(xg_1[p]):
      historia_for.setdefault(eq_h, []).append(float(xg_0[p]))
      historia_against.setdefault(eq_h, []).append(float(xg_1[p]))
      historia_for.setdefault(eq_a, []).append(float(xg_1[p]))
      historia_against.setdefault(eq_a, []).append(float(xg_0[p]))

  for j, sufijo in enumerate(("0", "1")):
    ds[f"xg_roll_for_{sufijo}"] = for_cols[:, j]
    ds[f"xg_roll_against_{sufijo}"] = against_cols[:, j]
    ds[f"xg_roll_n_{sufijo}"] = n_cols[:, j]
    ds[f"xg_roll_has_{sufijo}"] = has_cols[:, j]
  return ds
