"""
Descarga y resume datos de StatsBomb Open Data (xG, tiros y tiros a puerta) para las competiciones de selecciones.

StatsBomb Open Data es publico y gratuito para investigacion no comercial; revisa la licencia antes de usarlo en
produccion comercial. Los nombres de equipo quedan como los trae StatsBomb (ingles): la traduccion a los nombres de la
BD se hace aparte, igual que en construye_proxy.py.

Por partido se calcula, desde los eventos de tipo "Shot":
  - xG: suma de statsbomb_xg por equipo
  - tiros: numero de eventos Shot
  - tiros a puerta: tiros con outcome Goal o Saved

Uso:
    python etl/busca_statsbomb.py --competicion 43 --temporada 3
    python etl/busca_statsbomb.py --competicion 43 --temporada 3 106
"""
import argparse
import json
import time
import urllib.request

from pathlib import Path

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
SALIDA = Path("data") / "partidos" / "statsbomb"


def bajar(ruta):
  with urllib.request.urlopen(BASE + ruta, timeout=60) as respuesta:
    return json.loads(respuesta.read().decode("utf-8"))


def resumir_partido(eventos):
  datos = {"xg": 0.0, "tiros": 0, "sot": 0}
  resumen = {}
  for evento in eventos:
    if evento.get("type", {}).get("name") != "Shot":
      continue
    equipo = evento.get("team", {}).get("name", "?")
    fila = resumen.setdefault(equipo, {"xg": 0.0, "tiros": 0, "sot": 0})
    fila["xg"] += float(evento.get("shot", {}).get("statsbomb_xg") or 0.0)
    fila["tiros"] += 1
    if evento.get("shot", {}).get("outcome", {}).get("name") in ("Goal", "Saved"):
      fila["sot"] += 1
  return resumen


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--competicion", type=int, default=43, help="43 = FIFA World Cup")
  parser.add_argument("--temporada", type=int, nargs="+", default=[3, 106], help="3 = 2018, 106 = 2022")
  args = parser.parse_args()

  SALIDA.mkdir(parents=True, exist_ok=True)
  for temporada in args.temporada:
    ruta = f"/matches/{args.competicion}/{temporada}.json"
    partidos = bajar(ruta)
    filas = []
    for partido in partidos:
      try:
        eventos = bajar(f"/events/{partido['match_id']}.json")
      except Exception as error:
        print(f"  ! sin eventos {partido['match_id']}: {error}")
        continue
      resumen = resumir_partido(eventos)
      equipos = [partido["home_team"]["home_team_name"], partido["away_team"]["away_team_name"]]
      marcador = partido["home_score"], partido["away_score"]
      fila = {"match_id": partido["match_id"], "fecha": partido["match_date"],
              "home": equipos[0], "away": equipos[1], "score_0": marcador[0], "score_1": marcador[1]}
      for lado, equipo in ((0, equipos[0]), (1, equipos[1])):
        r = resumen.get(equipo, {"xg": 0.0, "tiros": 0, "sot": 0})
        fila[f"xg_{lado}"] = round(r["xg"], 4)
        fila[f"tiros_{lado}"] = r["tiros"]
        fila[f"sot_{lado}"] = r["sot"]
      filas.append(fila)
      time.sleep(0.3)
    import pandas as pd
    destino = SALIDA / f"{args.competicion}_{temporada}.csv"
    pd.DataFrame(filas).to_csv(destino, index=False, encoding="utf-8")
    print(f"{args.competicion}/{temporada}: {len(filas)} partidos -> {destino}")


if __name__ == "__main__":
  main()
