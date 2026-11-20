"""
E.3 · Carga y normalización del ranking FIFA.

Fuente recomendada (del informe docs/especificacion/10-fuentes-e3.md):
  - Kaggle `cashncarry/fifaworldranking` (CSV): columnas rank_date, country_full, country_abrv,
    total_points, previous_points, rank, rank_change, confederation. Cubre 1992-08 -> hoy.
  - Sin cuenta de Kaggle se puede usar un mirror de GitHub (ajustar FUENTE_URL).

Salida: data/ranking_fifa.csv con (fecha, pais, posicion, puntos), listo para:
  1) cargarse a una tabla nueva (p. ej. fases.RankingFifa) con un management command, o
  2) usarse como contexto externo del LSTM (un "extra" tipo `elo`), previa regen de
     `construir_dataset` y reentrenamiento.

La traducción FIFA (inglés) -> nombre del proyecto (español) reutiliza `construye_proxy.traducir`
(que ya conoce los nombres oficiales de FIFA: "Korea Republic", "IR Iran", "USA", "Czechia", ...)
y el mapa English->Spanish de martj42 (`data/partidos/nombres_en_es.csv`).
"""
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "etl"))
from construye_proxy import traducir  # noqa: E402

SALIDA = RAIZ / "data" / "ranking_fifa.csv"
PARTIDOS = RAIZ / "data" / "partidos"
FUENTE_URL = "https://www.kaggle.com/datasets/cashncarry/fifaworldranking"  # descargar manualmente si no hay token


def _equipos_y_nombres():
    """Equipos del proyecto (nombres en español) y mapa English->Spanish de martj42."""
    partidos = pd.read_csv(PARTIDOS / "partidos.csv", encoding="utf-8")
    equipos = set(partidos["home"].dropna()) | set(partidos["away"].dropna())
    nombres = dict(pd.read_csv(PARTIDOS / "nombres_en_es.csv", encoding="utf-8").itertuples(index=False))
    return equipos, nombres


def cargar_csv(csv: Path, pais_id: dict | None = None, desde: str | None = "1992-01-01") -> pd.DataFrame:
    """Lee el CSV de Kaggle y devuelve (fecha, pais, posicion, puntos) normalizado.

    - (csv): ruta al archivo descargado de Kaggle (columnas rank_date, country_full, total_points, rank).
    - (pais_id): opcional; diccionario nombre(español) -> id del proyecto. Si se omite, no se agrega la columna id.
    - (desde): fecha mínima a conservar (el ranking FIFA oficial arranca en 1992-08).
    """
    equipos, nombres = _equipos_y_nombres()
    df = pd.read_csv(csv)
    df = df.rename(columns={"rank_date": "fecha", "country_full": "pais_fifa",
                            "total_points": "puntos", "rank": "posicion"})
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha", "pais_fifa", "puntos"])
    if desde:
        df = df[df["fecha"] >= pd.to_datetime(desde)]
    df["pais"] = df["pais_fifa"].map(lambda e: traducir(e, equipos, nombres))
    df = df.dropna(subset=["pais"]).drop(columns=["pais_fifa"])
    columnas = ["fecha", "pais", "posicion", "puntos"]
    if pais_id:
        df["pais_id"] = df["pais"].map(pais_id)
        df = df.dropna(subset=["pais_id"])
        df["pais_id"] = df["pais_id"].astype(int)
        columnas = ["fecha", "pais", "pais_id", "posicion", "puntos"]
    return df[columnas].sort_values(["pais", "fecha"])


def main(csv: Path, pais_id: dict | None = None) -> None:
    """Guarda data/ranking_fifa.csv con el CSV descargado de Kaggle y (opcional) el mapa de ids del proyecto."""
    df = cargar_csv(csv, pais_id)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SALIDA, index=False, encoding="utf-8")
    print(f"{len(df)} filas -> {SALIDA} ({df['pais'].nunique()} selecciones)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python etl/busca_fifa_ranking.py <ruta/al/csv_kaggle>")
        print("Descarga manual (Kaggle):", FUENTE_URL)
        raise SystemExit(1)
    main(Path(sys.argv[1]))
