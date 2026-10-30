import io
import json
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from fases.modulo import consulta_general, formar_dataset_real

RAIZ = Path(settings.BASE_DIR).parent
SNAPSHOT = RAIZ / "experimentos" / "snapshot_consulta.json"
# el fixture de grupos de estos mundiales tiene 48 y 36 partidos; el resto son eliminatorias
GRUPOS = {2018: 48, 1986: 36}


class Command(BaseCommand):
  """
  ¿QUE HACE?
  Guarda o verifica una "foto" de consulta_general para 2018 y 1986: la simulacion completa (grupos + eliminatorias)
  a partir de los resultados reales de la fase de grupos. Sirve de red de seguridad antes y despues del refactor H.2:
  si el refactor cambia un marcador, la verificacion falla.

  ¿COMO LO HACE?
  0-) Toma los partidos reales del mundial (formar_dataset_real), se queda con la fase de grupos (GRUPOS) y arma el
      df que espera consulta_general: home, score_0, score_1, away y agno

  1-) Llama a consulta_general("grupo<agno>", df) y guarda solo las columnas home, score_0, score_1 y away de la salida

  2-) Con --guardar escribe experimentos/snapshot_consulta.json; con --verificar compara la salida actual contra ese
      archivo y falla si cambia algun partido

  Uso:
    python manage.py snapshot_consulta --guardar
    python manage.py snapshot_consulta --verificar
  """
  help = "Foto de consulta_general para 2018/1986: --guardar crea el JSON y --verificar compara la salida actual"
  requires_system_checks = []

  def add_arguments(self, parser):
    parser.add_argument("--guardar", action="store_true", help="Escribe experimentos/snapshot_consulta.json")
    parser.add_argument("--verificar", action="store_true", help="Compara la salida actual contra el JSON guardado")
    parser.add_argument("--anios", nargs="+", type=int, default=sorted(GRUPOS), help="Años a fotografiar (por defecto 2018 y 1986)")

  def consulta(self, agno):
    """Salida de consulta_general para (agno) con los resultados reales de la fase de grupos"""
    if agno not in GRUPOS:
      raise CommandError(f"No hay conteo de partidos de grupos para {agno}; añádelo a GRUPOS")
    base = formar_dataset_real("partidos", str(agno)).iloc[:GRUPOS[agno]]
    goles = base["score"].str.split(":", expand=True).astype(int)
    df = base[["home", "away"]].copy()
    df["score_0"], df["score_1"] = goles[0], goles[1]
    df["agno"] = agno
    df = df[["home", "score_0", "score_1", "away", "agno"]]
    with redirect_stdout(io.StringIO()):
      salida = consulta_general(f"grupo{agno}", df)
    salida = salida[["home", "score_0", "score_1", "away"]].copy()
    salida["score_0"], salida["score_1"] = salida["score_0"].astype(int), salida["score_1"].astype(int)
    return salida.to_dict("records")

  def handle(self, *args, **opciones):
    if not opciones["guardar"] and not opciones["verificar"]:
      raise CommandError("Indica --guardar o --verificar")
    anios = opciones["anios"]
    actual = {str(a): self.consulta(a) for a in anios}
    if opciones["guardar"]:
      SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
      SNAPSHOT.write_text(json.dumps({"fecha": date.today().isoformat(), "anios": anios, "partidos": actual},
                                     indent=2, ensure_ascii=False), encoding="utf-8")
      self.stdout.write(self.style.SUCCESS(f"Snapshot -> {SNAPSHOT} ({sum(len(v) for v in actual.values())} partidos)"))
    if opciones["verificar"]:
      if not SNAPSHOT.exists():
        raise CommandError(f"No existe {SNAPSHOT}; ejecuta antes --guardar")
      guardado = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["partidos"]
      fallos = 0
      for agno, filas in actual.items():
        esperadas = guardado.get(agno)
        if esperadas is None or len(esperadas) != len(filas):
          self.stderr.write(f"{agno}: falta el snapshot o cambia el numero de partidos")
          fallos += 1
          continue
        for i, (a, b) in enumerate(zip(esperadas, filas)):
          if a != b:
            self.stderr.write(f"{agno} partido {i}: {a} != {b}")
            fallos += 1
            if fallos >= 10:
              break
      if fallos:
        raise CommandError(f"La foto cambio en {fallos} partidos; revisa el refactor antes de continuar")
      self.stdout.write(self.style.SUCCESS(f"Snapshot OK: {SNAPSHOT}"))
