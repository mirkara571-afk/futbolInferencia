"""
Descarga de fbref, por cada mundial, las metricas de porteros por seleccion que alimentan proxy_desempegno.

Todo sale de la pagina de porteros (/keepers/), que existe para todos los mundiales; la de tiros (/shooting/) solo
desde 2018. Esa pagina tiene dos tablas:

  - Squad Stats (stats_squads_keeper_for): lo que recibio cada seleccion -> CS = VI, PKatt = PKATTALLOW
  - Opponent Stats (stats_squads_keeper_against, filas "vs X"): lo que hicieron los rivales de los porteros de X, es
    decir lo que hizo X -> PKatt = PKATT, SoTA = SOT

Lo que fbref tiene (revisado a mano): CS todos los años; penales desde 1998; tiros a puerta desde 2018. Las celdas
vacias quedan NaN y "construye_proxy.py" las imputa.

Guarda data/proxy_desempegno/fbref/fbref_<año>.csv con los nombres de fbref (en ingles): la traduccion a los nombres
de la BD la hace "construye_proxy.py", que la valida contra los equipos de cada mundial.

Uso (abre Firefox; fbref bloquea las descargas sin navegador):
    python etl/busca_metricas.py --anio 1998
    python etl/busca_metricas.py --todos
"""
import argparse

from fake_useragent import UserAgent

from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

import re
import time
from pathlib import Path
import numpy as np
import pandas as pd

from rutas import PROXY_DESEMPEGNO

# comp 1 = FIFA World Cup en fbref; la url sigue el patron /en/comps/1/{agno}/keepers/{agno}-World-Cup-Stats
MUNDIALES = [1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978, 1982, 1986, 1990, 1994, 1998, 2002,
             2006, 2010, 2014, 2018, 2022]
SALIDA = PROXY_DESEMPEGNO / "fbref"

# fbref bloquea por encima de ~10 peticiones/minuto; se espera entre paginas
ESPERA_ENTRE_PAGINAS = 7

# id de la tabla en la pagina de porteros -> {data-stat: metrica}
TABLAS = {
    "stats_squads_keeper_for": {"gk_clean_sheets": "VI", "gk_pens_att": "PKATTALLOW"},
    "stats_squads_keeper_against": {"gk_pens_att": "PKATT", "gk_shots_on_target_against": "SOT"},
}


def crear_driver():
    useragent = UserAgent()
    options = Options()
    # 'general.useragent.override' es la preferencia que realmente cambia el user agent en Firefox
    options.set_preference("general.useragent.override", useragent.random)
    # Firefox debe estar instalado; se da la ruta explícita por si el registro no la expone
    firefox = r"C:\Program Files\Mozilla Firefox\firefox.exe"
    if Path(firefox).exists():
        options.binary_location = firefox
    # geckodriver por defecto crea el perfil en %TEMP% y en algunos equipos falla con
    # "Unable to write to temporary directory" (salida 64); se usa un perfil escribible
    # dentro del propio proyecto (data/proxy_desempegno/.firefox_profile).
    perfil = PROXY_DESEMPEGNO / ".firefox_profile"
    perfil.mkdir(parents=True, exist_ok=True)
    service = Service(service_args=["--profile-root", str(perfil)])
    driver = Firefox(options=options, service=service)
    driver.implicitly_wait(7)
    driver.maximize_window()
    return driver


def descomentar_tablas(driver):
    """
    fbref envia la mayoria de sus tablas dentro de comentarios html (<!-- ... -->) y las
    muestra por javascript. Si la tabla no aparece en el DOM, se sacan los comentarios
    del div contenedor y se inyectan como html real.
    """
    driver.execute_script("""
        document.querySelectorAll('div.placeholder').forEach(function(p){
            var padre = p.parentNode;
            padre.childNodes.forEach(function(n){
                if (n.nodeType === 8) { padre.insertAdjacentHTML('beforeend', n.nodeValue); }
            });
        });
    """)


def leer_tabla_equipos(driver, id_tabla, columnas):
    """
    ¿QUE HACE?
    Lee la tabla de equipos (id_tabla) de la pagina abierta en (driver) y retorna un df
    con la columna "pais" y una columna por cada metrica de (columnas).

    ¿COMO LO HACE?
    0-) Espera la tabla; si no aparece, descomenta las tablas ocultas y la busca de nuevo.
    1-) Recorre las filas del tbody, ignorando las de cabecera repetida (class 'thead').
    2-) Toma el nombre del equipo del <a> de la celda 'team' (o 'squad' en versiones viejas de fbref).
    3-) Por cada data-stat de (columnas) toma el texto de la celda; si no existe o esta vacia, pone NaN.
    """
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, id_tabla)))
    except TimeoutException:
        descomentar_tablas(driver)

    try:
        tabla = driver.find_element(By.XPATH, f"//table[@id='{id_tabla}']")
    except NoSuchElementException:
        print(f"No se encontró la tabla {id_tabla}")
        return pd.DataFrame(columns=["pais"] + list(columnas.values()))

    filas = tabla.find_elements(By.XPATH, ".//tbody//tr[not(contains(@class, 'thead'))]")
    registros = []

    for des0 in filas:
        equipo = des0.find_elements(By.XPATH, ".//*[@data-stat='team' or @data-stat='squad']//a")
        if not equipo:
            continue
        registro = {"pais": equipo[0].text.strip()}

        for des1, des2 in columnas.items():
            celda = des0.find_elements(By.XPATH, f".//td[@data-stat='{des1}']")
            texto = celda[0].text.strip().replace(",", "") if celda else ""
            registro[des2] = float(texto) if re.fullmatch(r"-?\d+(\.\d+)?", texto) else np.nan

        registros.append(registro)

    return pd.DataFrame(registros)


def get_metricas_abstractas(year, driver):
    """
    ¿QUE HACE?
    Obtiene VI, PKATTALLOW, PKATT y SOT por seleccion para el mundial (year) desde la pagina de porteros de fbref y
    los guarda en SALIDA / fbref_<year>.csv, con los nombres de fbref

    ¿COMO LO HACE?
    0-) Abre la pagina de porteros y lee sus dos tablas de equipos (TABLAS)

    1-) En la tabla de rivales los equipos vienen como "vs Argentina": se quita el "vs " para unir por equipo

    2-) Une las dos tablas por "equipo" (outer, para no perder selecciones) y guarda el csv
    """
    driver.get(f"https://fbref.com/en/comps/1/{year}/keepers/{year}-World-Cup-Stats")
    df_metricas = None
    for id_tabla, columnas in TABLAS.items():
        df_tabla = leer_tabla_equipos(driver, id_tabla, columnas).rename(columns={"pais": "equipo"})
        df_tabla["equipo"] = df_tabla["equipo"].str.replace(r"^vs\s+", "", regex=True)
        print(f"{year} {id_tabla}: {len(df_tabla)} selecciones")
        df_metricas = df_tabla if df_metricas is None else df_metricas.merge(df_tabla, on="equipo", how="outer")

    df_metricas = df_metricas.reindex(columns=["equipo", "VI", "SOT", "PKATT", "PKATTALLOW"])
    SALIDA.mkdir(parents=True, exist_ok=True)
    df_metricas.to_csv(SALIDA / f"fbref_{year}.csv", index=False, encoding="utf-8")
    return df_metricas


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--anio", type=int, choices=MUNDIALES)
    grupo.add_argument("--todos", action="store_true")
    args = parser.parse_args()

    driver = crear_driver()
    try:
        for year in (MUNDIALES if args.todos else [args.anio]):
            print(get_metricas_abstractas(year, driver))
            time.sleep(ESPERA_ENTRE_PAGINAS)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
