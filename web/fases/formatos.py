"""
Reglas de formato de torneo con mejores terceros (1986, 1990, 1994 y 2026).

Extraido de modulo.py en el refactor H.2: lo que cambia entre mundiales son datos
(FormatoTorneo / ReglamentoTerceros), no codigo. No toca la base de datos.
"""
import re
from dataclasses import dataclass
from functools import wraps
from itertools import combinations
from types import MappingProxyType

import pandas as pd
from django.conf import settings


# =====================================================================
#  EMPAREJAMIENTO POR MEJORES TERCEROS (MUNDIALES 1986, 1990 Y 1994)
# =====================================================================

_POSICIONES = {"1": "Winner", "2": "Runners-up", "3": "Third-p"}

@dataclass(frozen=True)
class FormatoTorneo:
  """
  ¿QUE HACE?
  Datos del formato de un mundial con mejores terceros (lo que cambia de un año a otro son datos, no codigo): grupos, que ganadores de grupo enfrentan a un mejor tercero, cuantos terceros pasan, el orden de juego del 1er round de la fase final (llave) y los cruces oficiales FIFA

  Atributos:

    - grupos: letras de los grupos ("ABCDEF" en 1986-1994, "ABCDEFGHIJKL" en 2026)

    - ganadores_con_tercero: letras de los ganadores que enfrentan a un mejor tercero ("ABCD"; en 2026 "ABDEGIKL")

    - n_terceros: cuantos terceros pasan (4; en 2026, 8)

    - llave: partidos del 1er round en notacion "<posicion><grupo>-<posicion><grupo>": "2E" = Runners-up Grupo E, "1A-3" = el ganador del grupo A enfrenta al mejor tercero que le asignen los cruces

    - cruces: mapeo frozenset(grupos con mejor tercero) -> {ganador: grupo del 3ro rival}
  """
  grupos: str
  ganadores_con_tercero: str
  n_terceros: int
  llave: tuple
  cruces: MappingProxyType

  @property
  def partidos_llave(self):
    """1ros y 2dos de cada grupo mas los mejores terceros, de a dos por partido"""
    return (2 * len(self.grupos) + self.n_terceros) // 2

  @property
  def combinaciones(self):
    return frozenset(frozenset(c) for c in combinations(self.grupos, self.n_terceros))

  @property
  def clasificados_fijos(self):
    return frozenset(p + g for p in "12" for g in self.grupos)

  @property
  def patron_partido(self):
    letra = f"[{self.grupos[0]}-{self.grupos[-1]}]"
    return re.compile(rf"[12]{letra}-(?:[12]{letra}|3)")

def _cruces_desde_texto(ganadores, tabla):
  """{"ABCD": "CDAB", ...} -> mapeo frozenset(grupos) -> {ganador: grupo del 3ro rival}"""
  return MappingProxyType({frozenset(grupos): MappingProxyType(dict(zip(ganadores, rivales))) for grupos, rivales in tabla.items()})

def _cruces_2026():
  """Las 495 combinaciones del reglamento FIFA 2026 (data/partidos/terceros_2026.csv, de etl/construye_mundial_2026.py)"""
  ruta = settings.BASE_DIR.parent / "data" / "partidos" / "terceros_2026.csv"
  return dict(pd.read_csv(ruta, dtype=str).itertuples(index=False)) if ruta.exists() else {}

# Reglamento FIFA (el mismo en 1986, 1990 y 1994): segun que 4 grupos aporten mejor tercero,
# de que grupo es el tercero que enfrenta a cada ganador de los grupos A, B, C y D
_CRUCES_1986_1994 = {
  "ABCD": "CDAB", "ABCE": "CABE", "ABCF": "CABF", "ABDE": "DABE", "ABDF": "DABF",
  "ABEF": "EABF", "ACDE": "CDAE", "ACDF": "CDAF", "ACEF": "CAFE", "ADEF": "DAFE",
  "BCDE": "CDBE", "BCDF": "CDBF", "BCEF": "ECBF", "BDEF": "EDBF", "CDEF": "CDFE",
}
_LLAVES = {
  1986: ("1A-3", "1F-2E", "2A-2C", "1B-3", "1C-3", "2F-2B", "1E-2D", "1D-3"),
  1990: ("1A-3", "1E-2D", "2F-2B", "1B-3", "2A-2C", "1C-3", "1D-3", "1F-2E"),
  1994: ("2A-2C", "1A-3", "1B-3", "1F-2E", "1C-3", "1E-2D", "1D-3", "2F-2B"),
  # dieciseisavos 2026, Match 73 a 88
  2026: ("2A-2B", "1E-3", "1F-2C", "1C-2F", "1I-3", "2E-2I", "1A-3", "1L-3",
         "1D-3", "1G-3", "2K-2L", "1H-2J", "1B-3", "1J-2H", "1K-3", "2D-2G"),
}
FORMATOS = MappingProxyType({
  **{agno: FormatoTorneo("ABCDEF", "ABCD", 4, _LLAVES[agno], _cruces_desde_texto("ABCD", _CRUCES_1986_1994)) for agno in (1986, 1990, 1994)},
  2026: FormatoTorneo("ABCDEFGHIJKL", "ABDEGIKL", 8, _LLAVES[2026], _cruces_desde_texto("ABDEGIKL", _cruces_2026())),
})
# mundiales cuyo reglamento se puede editar desde el frontend (el editor dibuja 6 grupos)
EDITABLES_EN_FRONTEND = (1986, 1990, 1994)

def _etiqueta(codigo):
  """Traduce un codigo de llave ("2E") a la etiqueta que entiende "camb_grp_elm_smpl" ("Runners-up Grupo E")"""
  return f"{_POSICIONES[codigo[0]]} Grupo {codigo[1]}"

class ReglamentoInvalido(ValueError):
  """Error con la lista completa de problemas del reglamento, para mostrarlos todos de una vez en el frontend"""
  def __init__(self, errores):
    self.errores = list(errores)
    super().__init__("Reglamento de mejores terceros invalido:\n- " + "\n- ".join(self.errores))

@dataclass(frozen=True)
class ReglamentoTerceros:
  """
  ¿QUE HACE?
  Representa el reglamento de mejores terceros de un mundial: el orden de los partidos del 1er round de la fase final (llave) y, para cada combinacion de grupos con mejor tercero, el rival de cada ganador que enfrenta a un tercero (cruces). Es inmutable y se valida al crearse contra el (FormatoTorneo) de su año, venga del reglamento FIFA o del frontend

  ¿COMO LO HACE?
  0-) "__post_init__" junta todos los errores de la llave y de los cruces, y si hay alguno lanza (ReglamentoInvalido)

  1-) La llave es valida si tiene los partidos del formato (8 en 1986-1994, 16 en 2026) con formato "1A-2C" / "1A-3", cada 1ro y 2do de grupo aparece exactamente una vez, solo los ganadores del formato enfrentan a un mejor tercero y no hay cruces entre equipos del mismo grupo

  2-) Los cruces son validos si estan todas las combinaciones (15 en 1986-1994, 495 en 2026), cada combinacion reparte sus grupos entre los ganadores con tercero y ningun ganador enfrenta al tercero de su propio grupo

  Atributos:

    - agno: año del mundial (una clave de FORMATOS: 1986, 1990, 1994 o 2026)

    - llave: tupla de partidos en notacion "<posicion><grupo>-<posicion><grupo>"

    - cruces: mapeo frozenset(grupos) -> {ganador: grupo del 3ro rival}
  """
  agno: int
  llave: tuple
  cruces: MappingProxyType

  @property
  def formato(self):
    return FORMATOS[self.agno]

  def __post_init__(self):
    if self.agno not in FORMATOS:
      raise ReglamentoInvalido([f"el mundial {self.agno} no usa el formato de mejores terceros (solo {', '.join(map(str, FORMATOS))})"])
    errores = [*self._errores_llave(), *self._errores_cruces()]
    if errores:
      raise ReglamentoInvalido(errores)

  def _errores_llave(self):
    f = self.formato
    if len(self.llave) != f.partidos_llave:
      yield f"la llave debe tener {f.partidos_llave} partidos y tiene {len(self.llave)}"
    malos = [p for p in self.llave if not (isinstance(p, str) and f.patron_partido.fullmatch(p))]
    if malos:
      yield f"partidos con formato invalido: {malos}"
      return
    cruces = [p.split("-") for p in self.llave]
    fijos = [c for cruce in cruces for c in cruce if c != "3"]
    if repetidos := sorted({c for c in fijos if fijos.count(c) > 1}):
      yield f"equipos repetidos en la llave: {repetidos}"
    if faltan := sorted(f.clasificados_fijos - set(fijos)):
      yield f"equipos sin partido en la llave: {faltan}"
    contra_tercero = {local for local, visita in cruces if visita == "3"}
    if contra_tercero != {"1" + g for g in f.ganadores_con_tercero}:
      esperados = ", ".join("1" + g for g in f.ganadores_con_tercero)
      yield f"solo {esperados} enfrentan a un mejor tercero (recibido: {sorted(contra_tercero)})"
    if mismos := [f"{l}-{v}" for l, v in cruces if v != "3" and l[1] == v[1]]:
      yield f"cruces entre equipos del mismo grupo: {mismos}"

  def _errores_cruces(self):
    f = self.formato
    if set(self.cruces) != f.combinaciones:
      yield f"deben definirse los cruces para las {len(f.combinaciones)} combinaciones de {f.n_terceros} grupos"
      return
    for grupos, rival_de in self.cruces.items():
      nombre = "".join(sorted(grupos))
      if sorted(rival_de) != list(f.ganadores_con_tercero) or sorted(rival_de.values()) != sorted(grupos):
        yield f"{nombre}: cada grupo debe aportar exactamente un rival a {', '.join('1' + g for g in f.ganadores_con_tercero)}"
      elif mismos := [g for g, rival in rival_de.items() if g == rival]:
        yield f"{nombre}: el ganador de {mismos} no puede enfrentar al tercero de su propio grupo"

  @classmethod
  def oficial(cls, agno):
    """Reglamento FIFA del año (agno)"""
    formato = FORMATOS.get(agno)
    return cls(agno, formato.llave if formato else (), formato.cruces if formato else MappingProxyType({}))

  @classmethod
  def desde_json(cls, agno, datos):
    """Construye (y valida) el reglamento enviado por el frontend: {"llave": [...], "cruces": {"ABCD": "CDAB", ...}}"""
    ganadores = FORMATOS[agno].ganadores_con_tercero if agno in FORMATOS else ""
    try:
      llave = tuple(datos["llave"])
      cruces = _cruces_desde_texto(ganadores, datos["cruces"])
    except (KeyError, TypeError, AttributeError):
      raise ReglamentoInvalido(['se esperaba {"llave": [partidos], "cruces": {"ABCD": "CDAB", ...}}']) from None
    return cls(agno, llave, cruces)

  def a_json(self):
    """Inverso de "desde_json", para precargar el editor del frontend"""
    return {"llave": list(self.llave),
            "cruces": {"".join(sorted(grupos)): "".join(rival_de[g] for g in self.formato.ganadores_con_tercero)
                       for grupos, rival_de in sorted(self.cruces.items(), key=lambda kv: sorted(kv[0]))}}

def reglamentos_oficiales():
  """Diccionario {agno: reglamento FIFA en formato JSON} para el editor del template (los mundiales de EDITABLES_EN_FRONTEND)"""
  return {agno: ReglamentoTerceros.oficial(agno).a_json() for agno in EDITABLES_EN_FRONTEND}

def tabla_terceros(dict_table, df_grupos=None):
  """
  ¿QUE HACE?
  Genera un df con el 3er lugar de cada grupo ordenado segun el criterio FIFA: puntos, diferencia de gol y goles a favor. La letra del grupo se usa como ultimo desempate para que el resultado sea determinista

  ¿COMO LO HACE?
  0-) Se toma la fila de indice 2 (3er lugar) de cada tabla de (dict_table), ya ordenada por "fase_de_grupos"

  1-) Si se entrega (df_grupos), se acumulan los goles a favor y en contra de cada equipo como home y away, y se agregan las columnas "DG" y "GF" a los terceros; si no, valen 0

  2-) Se ordena de mejor a peor tercero

  Args:

    - dict_table: diccionario de salida de "fase_de_grupos" con clave "Grupo X" y valor el df (pais, Pts) ordenado

    - df_grupos: df de la fase de grupos ya jugada con columnas "home", "away", "score_0" y "score_1" (opcional, para desempatar)

  Returns:

    - df con columnas "Grupo", "pais", "Pts", "DG", "GF" ordenado de mejor a peor tercero
  """
  terceros = pd.DataFrame(
    [(grupo[-1], tabla["pais"].iloc[2], tabla["Pts"].iloc[2]) for grupo, tabla in dict_table.items()],
    columns=["Grupo", "pais", "Pts"]).astype({"Pts": float})

  if df_grupos is not None:
    columnas = ["pais", "GF", "GC"]
    goles = pd.concat([df_grupos[["home", "score_0", "score_1"]].set_axis(columnas, axis=1),
                       df_grupos[["away", "score_1", "score_0"]].set_axis(columnas, axis=1)])
    goles = goles.astype({"GF": float, "GC": float}).groupby("pais").sum()
    terceros = terceros.join(goles.assign(DG=goles.GF - goles.GC)[["DG", "GF"]], on="pais")
  terceros = terceros.reindex(columns=["Grupo", "pais", "Pts", "DG", "GF"]).fillna({"DG": 0, "GF": 0})

  return terceros.sort_values(["Pts", "DG", "GF", "Grupo"], ascending=[False, False, False, True], ignore_index=True)

def con_reglamento_terceros(func):
  """
  ¿QUE HACE?
  Decorador que entrega a la funcion decorada el reglamento de mejores terceros a usar: el personalizado si quien llama lo pasa en (personalizado), o el oficial FIFA de (agno) si no. La funcion decorada solo alinea equipos con las tablas de grupos ya jugadas, sin saber de donde salio el reglamento
  """
  @wraps(func)
  def envoltura(dict_table, agno, *args, personalizado=None, **kwargs):
    reglamento = personalizado if personalizado is not None else ReglamentoTerceros.oficial(agno)
    if reglamento.agno != agno:
      raise ValueError(f"el reglamento es del mundial {reglamento.agno} y se pidio el {agno}")
    return func(dict_table, reglamento, *args, **kwargs)
  return envoltura

@con_reglamento_terceros
def alinear_llave_terceros(dict_table, reglamento, df_grupos=None):
  """
  ¿QUE HACE?
  Genera el df (home, away) del 1er round de la fase final segun los mejores terceros (4 en 1986-1994, 8 en 2026) de las tablas de grupos ya jugadas y el (reglamento) que inyecta "con_reglamento_terceros". Quien llama pasa (dict_table, agno, df_grupos=..., personalizado=...)

  ¿COMO LO HACE?
  0-) Se obtienen los mejores terceros del formato con "tabla_terceros"

  1-) Se busca en los cruces del (reglamento) el rival de cada ganador con tercero para esa combinacion de grupos

  2-) Se recorre la llave del (reglamento) traduciendo cada codigo a etiqueta; los "3" se reemplazan por el tercero asignado al ganador local

  Args:

    - dict_table: salida de "fase_de_grupos"

    - reglamento: (ReglamentoTerceros) inyectado por el decorador; quien llama pasa el año (int) y opcionalmente (personalizado)

    - df_grupos: df de la fase de grupos jugada, para desempatar por diferencia de gol y goles a favor

  Returns:

    - df con columnas "home" y "away" en formato "Winner Grupo A" / "Third-p Grupo C", con los mejores terceros en attrs["mejores_terceros"]
  """
  terceros = tabla_terceros(dict_table, df_grupos).iloc[:reglamento.formato.n_terceros]
  rival_de = reglamento.cruces[frozenset(terceros.Grupo)]
  partidos = []
  for local, visita in (partido.split("-") for partido in reglamento.llave):
    if visita == "3":
      visita += rival_de[local[1]]
    partidos.append((_etiqueta(local), _etiqueta(visita)))
  llave = pd.DataFrame(partidos, columns=["home", "away"])
  llave.attrs["mejores_terceros"] = terceros
  return llave

