from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from fases.models import rankingfifa

RANKING = Path(settings.BASE_DIR).parent / "data" / "ranking_fifa.csv"


def leer_ranking(ruta):
  """
  ¿QUE HACE?
  Lee data/ranking_fifa.csv (fecha,pais,posicion,puntos) y lo normaliza a los tipos del modelo: fecha a date,
  posicion a int y puntos a float

  ¿COMO LO HACE?
  0-) Lee el csv con pandas y convierte cada columna a su tipo (las filas invalidas se descartan)

  1-) Devuelve una lista de dicts con las claves del modelo (fecha, pais, posicion, puntos), lista para bulk_create
  """
  if not Path(ruta).exists():
    raise CommandError(f"No existe {ruta}")
  df = pd.read_csv(ruta, encoding="utf-8")
  df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
  df["posicion"] = pd.to_numeric(df["posicion"], errors="coerce")
  df["puntos"] = pd.to_numeric(df["puntos"], errors="coerce")
  df = df.dropna(subset=["fecha", "pais", "posicion", "puntos"])
  df["fecha"] = df["fecha"].dt.date
  df["posicion"] = df["posicion"].astype(int)
  filas = []
  for registro in df[["fecha", "pais", "posicion", "puntos"]].to_dict("records"):
    filas.append({
      "fecha": registro["fecha"],
      "pais": registro["pais"],
      "posicion": int(registro["posicion"]),
      "puntos": float(registro["puntos"]),
    })
  return filas


class Command(BaseCommand):
  """
  ¿QUE HACE?
  "python manage.py cargar_ranking_fifa" llena la tabla (rankingfifa) con data/ranking_fifa.csv, para reconstruir
  el ranking desde cero en cualquier maquina

  ¿COMO LO HACE?
  0-) No ejecuta los system checks (requires_system_checks = []): cargar urls -> views -> modulo arrastra tensorflow,
  y no hace falta para cargar datos

  1-) Lee y normaliza el csv con (leer_ranking)

  2-) Dentro de una sola transaccion, vacia la tabla y la recarga con bulk_create (idempotente)
  """
  help = "Carga data/ranking_fifa.csv en la tabla rankingfifa"
  requires_system_checks = []

  def handle(self, *args, **opciones):
    filas = leer_ranking(RANKING)
    with transaction.atomic():
      rankingfifa.objects.all().delete()
      rankingfifa.objects.bulk_create([rankingfifa(**fila) for fila in filas], batch_size=1000)
    self.stdout.write(self.style.SUCCESS(f"rankingfifa: {len(filas)} filas cargadas"))
