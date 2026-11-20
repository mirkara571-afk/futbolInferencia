"""
Parser de las páginas de porteros de FBref guardadas a mano (E.1).

¿QUÉ HACE?
Convierte cada `data/proxy_desempegno/fbref/fbref_<año>.html` (la página
/en/comps/1/<año>/keepers/ que se guarda con Ctrl+S cuando Cloudflare no deja
automatizar la descarga) en el CSV que espera `construye_proxy.py`:
`fbref_<año>.csv` con columnas `equipo, VI, SOT, PKATT, PKATTALLOW`.

¿CÓMO LO HACE?
1-) Abre el HTML con BeautifulSoup y busca las dos tablas por su id:
    `stats_squads_keeper_for` (lo que recibió cada selección) y
    `stats_squads_keeper_against` (lo que hicieron sus rivales, filas "vs X").
2-) De la primera toma CS -> VI y PKatt -> PKATTALLOW; de la segunda toma
    PKatt -> PKATT y SoTA -> SOT. Las celdas vacías quedan NaN (FBref no tiene
    penales antes de 1998 ni tiros a puerta antes de 2018).
3-) Une las dos tablas por equipo, quita el prefijo "vs " y guarda el CSV con
    los nombres de FBref en inglés; la traducción al español la hace
    `construye_proxy.py`.

¿PARA QUÉ SIRVE?
Para poder ejecutar E.1 sin Selenium: tú guardas las páginas, este script las
parsea y `construye_proxy.py` rellena `proxy_desempegno` con PKATT/SOT reales.

Uso:
    python etl/parsea_fbref.py
    python etl/parsea_fbref.py --anio 1998 --validar
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

RAIZ = Path(__file__).resolve().parent.parent
FBREF = RAIZ / "data" / "proxy_desempegno" / "fbref"
PARTIDOS = RAIZ / "data" / "partidos"
sys.path.insert(0, str(RAIZ / "etl"))
from construye_proxy import traducir  # noqa: E402

TABLA_FOR = "stats_squads_keeper_for"
TABLA_AGAINST = "stats_squads_keeper_against"

# FBref abrevia algunos nombres en la tabla; se normalizan antes de traducir.
# "S & M" = Serbia and Montenegro; "Trin & Tobago" = Trinidad and Tobago;
# "Bosnia & Herz'na" aparece cortado/dañado en el HTML.
NORMALIZAR = {"S & M": "Serbia and Montenegro", "Trin & Tobago": "Trinidad & Tobago"}


def normalizar(equipo):
    """Nombre de FBref -> nombre completo que entiende TRADUCCION_FBREF."""
    equipo = equipo.strip()
    if equipo in NORMALIZAR:
        return NORMALIZAR[equipo]
    if equipo.startswith("Bosnia") and "Herz" in equipo:
        return "Bosnia & Herz'na"
    return equipo


def numero(texto):
    """Celda de FBref -> float; vacía o no numérica -> NaN."""
    texto = (texto or "").strip().replace(",", "")
    if not texto:
        return float("nan")
    try:
        return float(texto)
    except ValueError:
        return float("nan")


def nombre_equipo(fila):
    """Nombre de la selección desde la celda data-stat='team' (quita el código de bandera)."""
    celda = fila.find(["th", "td"], {"data-stat": "team"})
    if celda is None:
        return None
    enlace = celda.find("a")
    nombre = enlace.get_text(" ", strip=True) if enlace else celda.get_text(" ", strip=True)
    return re.sub(r"^[a-z]{2}\s+", "", nombre).strip()


def leer_tabla(sopa, id_tabla, columnas):
    """Devuelve {equipo: {metrica: valor}} de la tabla (id_tabla)."""
    tabla = sopa.find("table", id=id_tabla)
    if tabla is None:
        return {}
    cuerpo = tabla.find("tbody")
    if cuerpo is None:
        return {}
    salida = {}
    for fila in cuerpo.find_all("tr"):
        clases = fila.get("class") or []
        if "thead" in clases:          # fila de cabecera repetida dentro del tbody
            continue
        equipo = nombre_equipo(fila)
        if not equipo or equipo == "Squad":
            continue
        equipo = re.sub(r"^vs\s+", "", equipo, flags=re.IGNORECASE)
        equipo = normalizar(equipo)
        valores = {}
        for data_stat, metrica in columnas.items():
            celda = fila.find(["th", "td"], {"data-stat": data_stat})
            valores[metrica] = numero(celda.get_text(" ", strip=True) if celda else "")
        salida[equipo] = valores
    return salida


def parsear(html):
    """HTML de FBref -> DataFrame (equipo, VI, SOT, PKATT, PKATTALLOW)."""
    sopa = BeautifulSoup(html, "html.parser")
    recibido = leer_tabla(sopa, TABLA_FOR, {"gk_clean_sheets": "VI", "gk_pens_att": "PKATTALLOW"})
    hecho = leer_tabla(sopa, TABLA_AGAINST, {"gk_pens_att": "PKATT", "gk_shots_on_target_against": "SOT"})
    equipos = sorted(set(recibido) | set(hecho))
    filas = []
    for equipo in equipos:
        fila = {"equipo": equipo}
        fila.update(recibido.get(equipo, {}))
        fila.update(hecho.get(equipo, {}))
        filas.append(fila)
    columnas = ["equipo", "VI", "SOT", "PKATT", "PKATTALLOW"]
    df = pd.DataFrame(filas, columns=columnas)
    return df[columnas]


def equipos_del_agno(anio, partidos):
    """Selecciones que jugaron el Mundial (anio) según data/partidos/partidos.csv."""
    del_agno = partidos[partidos["year"] == anio]
    return set(del_agno["home"].dropna()) | set(del_agno["away"].dropna())


def validar(anio, df, partidos, nombres):
    """Compara los equipos del CSV con los del Mundial y traduce los nombres de FBref."""
    esperados = equipos_del_agno(anio, partidos)
    traducidos = {equipo: traducir(equipo, esperados, nombres) for equipo in df["equipo"]}
    sin_traducir = sorted(e for e, t in traducidos.items() if t is None)
    faltan = sorted(esperados - set(t for t in traducidos.values() if t))
    print(f"\nValidación {anio}: {len(df)} equipos en FBref, {len(esperados)} en partidos.csv")
    if sin_traducir:
        print(f"  Sin traducir ({len(sin_traducir)}): {sin_traducir}")
    else:
        print("  Todos los nombres de FBref se traducen a un equipo del Mundial.")
    if faltan:
        print(f"  Faltan en FBref ({len(faltan)}): {faltan}")
    return sin_traducir, faltan


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--anio", type=int, help="procesar solo ese año (por defecto, todos los .html)")
    parser.add_argument("--validar", action="store_true", help="compara los equipos con data/partidos/partidos.csv")
    args = parser.parse_args()

    partidos = pd.read_csv(PARTIDOS / "partidos.csv", encoding="utf-8")
    nombres = dict(pd.read_csv(PARTIDOS / "nombres_en_es.csv", encoding="utf-8").itertuples(index=False))
    archivos = sorted(FBREF.glob(f"fbref_{args.anio}.html" if args.anio else "fbref_*.html"))
    if not archivos:
        raise SystemExit(f"No hay HTML en {FBREF}")

    for ruta in archivos:
        anio = int(re.search(r"fbref_(\d{4})", ruta.name).group(1))
        df = parsear(ruta.read_text(encoding="utf-8", errors="replace"))
        salida = ruta.with_suffix(".csv")
        df.to_csv(salida, index=False, encoding="utf-8")
        print(f"{anio}: {len(df)} equipos -> {salida.name} "
              f"(VI {df['VI'].notna().sum()}, SOT {df['SOT'].notna().sum()}, "
              f"PKATT {df['PKATT'].notna().sum()}, PKATTALLOW {df['PKATTALLOW'].notna().sum()})")
        if args.validar:
            validar(anio, df, partidos, nombres)


if __name__ == "__main__":
    main()
