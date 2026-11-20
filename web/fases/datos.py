"""Helpers puros de resultados y desempates reales (extraido de modulo.py en el refactor H.2)."""
import pandas as pd


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
