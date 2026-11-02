"""
Une los resumenes de StatsBomb (xG, tiros, tiros a puerta) a partidos_features.csv y escribe una copia moderna
partidos_features_xg.csv con las columnas xg_0/1, tiros_0/1, sot_0/1 y xG_disponible.

El xG del partido es una observacion post-partido, no una feature pre-partido: sirve para el modelo jerarquico y para
diagnostico, no para predecir el propio partido. xG_disponible marca donde el dato existe (2018 y 2022).

Uso:
    python etl/unir_statsbomb.py
"""
import pandas as pd

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DATASET = RAIZ / "data" / "datasets" / "partidos_features.csv"
SALIDA = RAIZ / "data" / "datasets" / "partidos_features_xg.csv"
STATSBOMB = RAIZ / "data" / "partidos" / "statsbomb"
NOMBRES = RAIZ / "data" / "partidos" / "nombres_en_es.csv"


def main():
  nombres = dict(pd.read_csv(NOMBRES, encoding="utf-8").itertuples(index=False))
  tablas = []
  for ruta in sorted(STATSBOMB.glob("43_*.csv")):
    df = pd.read_csv(ruta)
    df["agno"] = pd.to_datetime(df["fecha"]).dt.year
    df["home"] = df["home"].map(nombres)
    df["away"] = df["away"].map(nombres)
    tablas.append(df)
  statsbomb = pd.concat(tablas, ignore_index=True)

  ds = pd.read_csv(DATASET)
  # StatsBomb guarda el marcador con prorroga/penales ya resueltos; el dataset guarda goles a los 90/120 en goles_0/1
  moderno = ds.merge(
    statsbomb[["agno", "home", "away", "score_0", "score_1", "xg_0", "xg_1", "tiros_0", "tiros_1", "sot_0", "sot_1"]],
    left_on=["agno", "home", "away", "goles_0", "goles_1"],
    right_on=["agno", "home", "away", "score_0", "score_1"],
    how="left",
  )
  moderno["xG_disponible"] = moderno["xg_0"].notna().astype(int)
  moderno = moderno.drop(columns=["score_0", "score_1"], errors="ignore")
  SALIDA.parent.mkdir(parents=True, exist_ok=True)
  moderno.to_csv(SALIDA, index=False, encoding="utf-8")

  cubiertos = moderno.loc[moderno["agno"].isin([2018, 2022]), "xG_disponible"].sum()
  total = moderno.loc[moderno["agno"].isin([2018, 2022])].shape[0]
  print(f"{SALIDA} -> {moderno.shape}; xG en {cubiertos}/{total} partidos de 2018/2022")
  sin_union = moderno[(moderno["agno"].isin([2018, 2022])) & (moderno["xG_disponible"] == 0)]
  if len(sin_union):
    print("Sin unir (revisar nombres/marcador):")
    print(sin_union[["agno", "home", "away", "goles_0", "goles_1"]].to_string(index=False))


if __name__ == "__main__":
  main()
