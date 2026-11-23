"""
Etapa G · Cuadro de los 5 campeones más probables.

Motor Montecarlo en numpy/Python puro: no pasa por ``consulta_general``. Recibe los goles
esperados (lambda de Poisson) de todas las parejas de equipos que siguen vivos, congela las
métricas al inicio de la fase simulada y sortea cada marcador de su Poisson.

Alcance v2:
- Desde la fase de grupos: 1934/1938 (eliminatoria), 1950 (liguilla final), 1954–1970
  (4 grupos → cuartos; 1954 es aproximado: los playoffs de desempate se cuentan como partidos
  de grupo y se usan 3 puntos), 1974/1978/1982 (segunda fase), 1986/1990/1994 (mejores terceros),
  1998–2022 (llave de 16) y 2026 (12 grupos, 8 mejores terceros, llave de 32).
- Desde fases knockout: 1998–2022 y 2026 (dieciseisavos, octavos, cuartos, semifinal y final),
  rejugando las fases previas con sus resultados reales.
- Reglamento de mejores terceros personalizado (1986/1990/1994/2026).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from django.conf import settings

from .modulo import (
    BASE_METRICS,
    ReglamentoTerceros,
    calculo_metricas_0,
    desempates_reales,
    formar_dataset_real,
    internacionales,
    obtener_clasificaciones,
    obtener_dic_agnos,
    predictor_en_uso,
    reiniciar_opcionales,
    resultados_reales,
)
from .simulacion import func_prediccion_orden

_HISTORICOS_8 = (1954, 1958, 1962, 1966, 1970)                # 4 grupos numerados -> cuartos (8)
_MODERNOS_16 = (1998, 2002, 2006, 2010, 2014, 2018, 2022)     # 8 grupos -> octavos (16)
_TERCEROS = (1986, 1990, 1994, 2026)                          # mejores terceros
_ELIMINACION = (1934, 1938)                                    # eliminatoria pura (sin grupos)
_FINAL_ROUND = (1950,)                                         # grupos + liguilla final de 4
_SEGUNDA_FASE_4 = (1974, 1978)                                 # grupos + 2da fase (2 grupos de 4) + final
_SEGUNDA_FASE_3 = (1982,)                                      # grupos + 2da fase (4 grupos de 3) + semis + final

# etiqueta de fase del frontend -> fase interna de func_prediccion_orden.lista_fases
_FASES = {
    "Fase de Grupos": "grupo", "grupo": "grupo",
    "octavos": "knockout", "knockout": "knockout",
    "cuartos": "quarter", "quarter": "quarter",
    "semifinal": "semi", "semi": "semi",
    "dieciseisavos": "dieciseisavos",
    "final": "final",
}
_KNOCKOUT_MODERNO = ("knockout", "quarter", "semi", "final")
_KNOCKOUT_2026 = ("dieciseisavos", "knockout", "quarter", "semi", "final")


def _ganador(rng, lambdas, i, j):
    """Sortea el marcador de (i local, j visitante) y devuelve (ganador, perdedor)."""
    lh, la = float(lambdas[i, j][0]), float(lambdas[i, j][1])
    gh, ga = rng.poisson([lh, la])
    if gh > ga:
        return i, j
    if ga > gh:
        return j, i
    return (i, j) if lh >= la else (j, i)


def simular_knockout(rng, lambdas, equipos):
    """Una llave de eliminación directa con ``equipos`` en el orden del cuadro (potencia de 2).

    Devuelve ``(campeon, finalista, semifinalistas)``."""
    vivos = list(equipos)
    semifinalistas = ()
    finalistas = ()
    while len(vivos) > 1:
        if len(vivos) == 4:
            semifinalistas = tuple(vivos)
        if len(vivos) == 2:
            finalistas = tuple(vivos)
        ganadores = []
        for k in range(0, len(vivos), 2):
            g, _ = _ganador(rng, lambdas, vivos[k], vivos[k + 1])
            ganadores.append(g)
        vivos = ganadores
    campeon = vivos[0]
    finalista = next(t for t in finalistas if t != campeon)
    return campeon, finalista, semifinalistas


def _jugar_grupo(rng, lambdas, partidos):
    """Round-robin de un grupo en Python puro. Devuelve la lista ordenada por FIFA."""
    equipos = sorted({e for h, a in partidos for e in (h, a)})
    pts = {e: 0 for e in equipos}
    gf = {e: 0 for e in equipos}
    gc = {e: 0 for e in equipos}
    for h, a in partidos:
        gh, ga = rng.poisson([float(lambdas[h, a][0]), float(lambdas[h, a][1])])
        gf[h] += gh
        gc[h] += ga
        gf[a] += ga
        gc[a] += gh
        if gh > ga:
            pts[h] += 3
        elif ga > gh:
            pts[a] += 3
        else:
            pts[h] += 1
            pts[a] += 1
    orden = sorted(equipos, key=lambda e: (-pts[e], -(gf[e] - gc[e]), -gf[e], e))
    return [{"pais": e, "Pts": pts[e], "DG": gf[e] - gc[e], "GF": gf[e]} for e in orden]


def simular_grupo(rng, lambdas, partidos):
    """Round-robin de un grupo (versión pública para tests): DataFrame ordenado por FIFA."""
    return pd.DataFrame(_jugar_grupo(rng, lambdas, partidos), columns=["pais", "Pts", "DG", "GF"])


def _orden_cuartos_8(tablas):
    """Orden de los 8 clasificados de 4 grupos (1958–1970, grupos "Grupo 1".."Grupo 4")."""
    cruces = [("1", "1"), ("2", "2"), ("1", "3"), ("2", "4"),
              ("1", "2"), ("2", "1"), ("1", "4"), ("2", "3")]
    pos = {"1": 0, "2": 1}
    return [tablas["Grupo " + g][pos[p]]["pais"] for p, g in cruces]


def _orden_octavos_16(tablas):
    """Orden de los 16 clasificados de 8 grupos de 4 (formato estándar 1998–2022)."""
    cruces = ["1A", "2B", "1C", "2D", "1E", "2F", "1G", "2H",
              "1B", "2A", "1D", "2C", "1F", "2E", "1H", "2G"]
    pos = {"1": 0, "2": 1}
    return [tablas["Grupo " + codigo[1]][pos[codigo[0]]]["pais"] for codigo in cruces]


def _orden_llave_terceros(tablas, reglamento):
    """Orden de los clasificados según la llave oficial y los cruces de mejores terceros."""
    terceros = sorted(
        ({**tablas[g][2], "Grupo": g[-1]} for g in tablas),
        key=lambda t: (-t["Pts"], -t["DG"], -t["GF"], t["Grupo"]),
    )
    rival_de = reglamento.cruces[frozenset(t["Grupo"] for t in terceros[:reglamento.formato.n_terceros])]
    pos = {"1": 0, "2": 1}
    orden = []
    for partido in reglamento.llave:
        local, visita = partido.split("-")
        for lado in (local, visita):
            if lado == "3":
                grupo, posicion = rival_de[local[1]], 2
            else:
                grupo, posicion = lado[1], pos[lado[0]]
            orden.append(tablas["Grupo " + grupo][posicion]["pais"])
    return orden


def _simular_un_torneo(rng, lambdas, grupos_partidos, orden_fn):
    """Una simulación completa desde la fase de grupos."""
    tablas = {grupo: _jugar_grupo(rng, lambdas, partidos) for grupo, partidos in grupos_partidos.items()}
    return simular_knockout(rng, lambdas, orden_fn(tablas))


# --- preparación (estado, métricas congeladas y lambdas) ------------------------------------

def _construir_objs(agno, drop_metricas, metricas_opcionales):
    """Replica la preparación de ``consulta_general`` y devuelve (objs, df2, df3, dict_g, modelo)."""
    objs = func_prediccion_orden(
        "partidos", str(agno), pd.DataFrame(), obtener_clasificaciones(), obj0="-",
        drop_metricas=drop_metricas, metricas_opcionales=metricas_opcionales,
    )
    reiniciar_opcionales(objs.pais_id, agno=agno)
    dic_agnos = obtener_dic_agnos()
    _, df2, df3 = objs.create_features(dic_agnos)
    # contexto_simulacion usa obj0 (int) para el Elo
    objs.obj0 = agno

    # resultados reales y desempates (para rejugar las fases previas con marcador real)
    reales = objs.df2[objs.df2["agno"].astype(int) == agno]
    limpieza = {"(": "", ")": "", ":": ",", " ": ","}
    partidos_agno = pd.DataFrame({
        "home": reales["home"].map(objs.pais_id).to_list(),
        "score_0": reales["score"].map(lambda x: objs.ajuste_score_partidos(x, limpieza)).to_list(),
        "score_1": reales["score"].map(lambda x: objs.ajuste_score_partidos(x, limpieza, ind=1)).to_list(),
        "away": reales["away"].map(objs.pais_id).to_list(),
        "score": reales["score"].to_list(),
        "penales": reales["penales"].to_list(),
        "jugado": reales["jugado"].to_list(),
    })
    objs.resultados_reales = resultados_reales(partidos_agno)
    objs.desempates = desempates_reales(partidos_agno)
    objs.primera_fase = "grupo"
    objs.ind0 = ["grupo", "grupo"]
    objs.anfitriones = {
        objs.pais_id[n] for n in internacionales.objects.filter(categoria="mundial", agno=agno, neutral="0")
        .values_list("home", flat=True).distinct() if n in objs.pais_id
    }
    proxy = formar_dataset_real("proxy_desempegno")
    objs.con_datos_reales = {
        marca: {objs.pais_id[n] for n in proxy.loc[(proxy["agno"] < agno) & (proxy[marca] == 1), "pais"] if n in objs.pais_id}
        for marca in ("pk_real", "sot_real")
    }
    modelo = predictor_en_uso(settings.MODEL_PATH / settings.MODEL_DIR)
    # D.4: con un modelo k>1, sembrar el historial internacional previo para armar las secuencias
    if getattr(modelo, "config", {}).get("k", 1) > 1:
      objs.historia = objs._historia_inicial()
    dict_g = objs.grupos_anio_interes(str(agno))
    return objs, df2, df3, dict_g, modelo


def _base_por_equipo(df_fase):
    """Métricas base congeladas por equipo (última fila donde aparece) y la lista de equipos."""
    equipos = sorted({int(h) for h in df_fase["home"]} | {int(a) for a in df_fase["away"]})
    base = {}
    for e in equipos:
        h = df_fase[df_fase["home"] == e]
        if len(h):
            r = h.iloc[-1]
            base[e] = {m: float(r[f"{m}_0"]) for m in BASE_METRICS}
        else:
            r = df_fase[df_fase["away"] == e].iloc[-1]
            base[e] = {m: float(r[f"{m}_1"]) for m in BASE_METRICS}
    return equipos, base


def _lambdas_desde_base(objs, equipos, base, modelo, drop_metricas, metricas_opcionales):
    """Goles esperados de todas las parejas de ``equipos`` con métricas congeladas.

    Devuelve ``(max_id+1, max_id+1, 2)`` con ``lambdas[i, j] = [lambda_i_local, lambda_j_visitante]``."""
    from .predictores import Lote, lote_desde_entrada

    pares = []
    for i, j in itertools.permutations(equipos, 2):
        pares.append({"home": i, "away": j, "score_0": 0, "score_1": 0,
                      **{f"{m}_0": base[i][m] for m in BASE_METRICS},
                      **{f"{m}_1": base[j][m] for m in BASE_METRICS}})
    df_pares = pd.DataFrame(pares)
    X = calculo_metricas_0(df_pares, drop_metricas, metricas_opcionales).to_numpy()
    tabla = objs.contexto_simulacion(lote_desde_entrada(X).tabla)
    Y = modelo.predecir(Lote(tabla, secuencias=None, historia=getattr(objs, "historia", None)))

    n = max(equipos) + 1
    lambdas = np.zeros((n, n, 2))
    for (i, j), (lh, la) in zip(((int(r["home"]), int(r["away"])) for r in pares), Y):
        lambdas[i, j] = [lh, la]
    return lambdas


def _orden_desde_df(df_fase):
    """Ids de los equipos de una fase knockout en orden de cuadro (home/away intercalados)."""
    return [int(x) for pareja in zip(df_fase["home"], df_fase["away"]) for x in pareja]


def _es_primera_fase(nombre):
    """True si el grupo es de primera fase ("Grupo 1".."Grupo 9"); False si es de segunda ("Grupo A"..)."""
    return nombre.rsplit(" ", 1)[-1].isdigit()


def _pares_3(equipos):
    return [(equipos[0], equipos[1]), (equipos[1], equipos[2]), (equipos[0], equipos[2])]


def _orden_1934_38(df2):
    """Orden de cuadro de 1934/1938: los equipos únicos en el orden del fixture (ignora replays)."""
    orden = []
    for pareja in zip(df2["home"], df2["away"]):
        for x in pareja:
            x = int(x)
            if x not in orden:
                orden.append(x)
    return orden


def _simular_1950(rng, lambdas, grupos_partidos):
    """1950: 4 grupos -> liguilla final de 4 (todos contra todos). El 1ro es el campeón."""
    campeones = [_jugar_grupo(rng, lambdas, partidos)[0]["pais"] for partidos in grupos_partidos.values()]
    partidos = [(a, b) for i, a in enumerate(campeones) for b in campeones[i + 1:]]
    tabla = _jugar_grupo(rng, lambdas, partidos)
    return tabla[0]["pais"], tabla[1]["pais"], tuple(campeones)


def _simular_1974_78(rng, lambdas, grupos_partidos):
    """1974/1978: 4 grupos -> 2 grupos de 4 (segunda fase, composición fija) -> final."""
    tablas = {g: _jugar_grupo(rng, lambdas, partidos) for g, partidos in grupos_partidos.items()}
    a = [tablas["Grupo 3"][0]["pais"], tablas["Grupo 4"][1]["pais"], tablas["Grupo 2"][1]["pais"], tablas["Grupo 1"][0]["pais"]]
    b = [tablas["Grupo 2"][0]["pais"], tablas["Grupo 1"][1]["pais"], tablas["Grupo 3"][1]["pais"], tablas["Grupo 4"][0]["pais"]]
    ta = _jugar_grupo(rng, lambdas, [(a[i], a[j]) for i in range(4) for j in range(i + 1, 4)])
    tb = _jugar_grupo(rng, lambdas, [(b[i], b[j]) for i in range(4) for j in range(i + 1, 4)])
    campeon, perdedor = _ganador(rng, lambdas, ta[0]["pais"], tb[0]["pais"])
    semifinalistas = tuple(sorted([ta[0]["pais"], tb[0]["pais"], ta[1]["pais"], tb[1]["pais"]]))
    return campeon, perdedor, semifinalistas


def _simular_1982(rng, lambdas, grupos_partidos):
    """1982: 6 grupos -> 4 grupos de 3 (segunda fase) -> semis (A-C, B-D) -> final."""
    tablas = {g: _jugar_grupo(rng, lambdas, partidos) for g, partidos in grupos_partidos.items()}
    a = [tablas["Grupo 1"][0]["pais"], tablas["Grupo 3"][0]["pais"], tablas["Grupo 6"][1]["pais"]]
    b = [tablas["Grupo 2"][0]["pais"], tablas["Grupo 4"][0]["pais"], tablas["Grupo 5"][1]["pais"]]
    c = [tablas["Grupo 1"][1]["pais"], tablas["Grupo 3"][1]["pais"], tablas["Grupo 6"][0]["pais"]]
    d = [tablas["Grupo 2"][1]["pais"], tablas["Grupo 4"][1]["pais"], tablas["Grupo 5"][0]["pais"]]
    ta = _jugar_grupo(rng, lambdas, _pares_3(a))
    tb = _jugar_grupo(rng, lambdas, _pares_3(b))
    tc = _jugar_grupo(rng, lambdas, _pares_3(c))
    td = _jugar_grupo(rng, lambdas, _pares_3(d))
    return simular_knockout(rng, lambdas, [ta[0]["pais"], tc[0]["pais"], tb[0]["pais"], td[0]["pais"]])


def _replay_moderno(objs, df2, df3, dict_g, modelo, fase_int):
    """Rejuega el formato moderno (1998–2022) hasta ``fase_int`` con resultados reales.

    Devuelve ``(equipos, base, orden)`` de los vivos con métricas congeladas y orden de cuadro."""
    objs.ind0 = ["grupo", fase_int]
    df_grupos = df2.copy()
    dict_table = objs.fase_de_grupos(dict_g, df_grupos, modelo)

    if fase_int == "knockout":
        df_fase = objs.camb_grp_elm_smpl(df_grupos, df3[["home", "score", "away"]].iloc[0:8].copy(), dict_table)
        equipos, base = _base_por_equipo(df_fase)
        return equipos, base, _orden_desde_df(df_fase)

    objs.n_partidos = df_grupos.shape[0] + 1
    objs.ind0[0] = "knockout"
    df_knockout = objs.camb_grp_elm_smpl(df_grupos, df3[["home", "score", "away"]].iloc[0:8].copy(), dict_table)
    df_knockout = objs.jugar(df_knockout, modelo)
    if fase_int == "quarter":
        df_fase = objs.ordenar_elm_smpl(df_knockout, df3[["home", "score", "away"]].iloc[8:12].copy())
        equipos, base = _base_por_equipo(df_fase)
        return equipos, base, _orden_desde_df(df_fase)

    objs.ind0[0] = "quarter"
    df_quarter = objs.ordenar_elm_smpl(df_knockout, df3[["home", "score", "away"]].iloc[8:12].copy())
    df_quarter = objs.jugar(df_quarter, modelo)
    if fase_int == "semi":
        df_fase = objs.ordenar_elm_smpl(df_quarter, df3[["home", "score", "away"]].iloc[12:14].copy())
        equipos, base = _base_por_equipo(df_fase)
        return equipos, base, _orden_desde_df(df_fase)

    objs.ind0[0] = "semi"
    df_semi = objs.ordenar_elm_smpl(df_quarter, df3[["home", "score", "away"]].iloc[12:14].copy())
    df_semi = objs.jugar(df_semi, modelo)
    df_final = objs.ordenar_elm_smpl(df_semi, df3[["home", "score", "away"]].iloc[15].copy())
    equipos, base = _base_por_equipo(df_final)
    return equipos, base, _orden_desde_df(df_final)


def _replay_2026(objs, df2, df3, dict_g, modelo, fase_int, reglamento_obj):
    """Rejuega el formato 2026 (12 grupos → llave de 32) hasta ``fase_int`` con resultados reales.

    Devuelve ``(equipos, base, orden)`` de los vivos con métricas congeladas y orden de cuadro."""
    from .formatos import alinear_llave_terceros

    objs.ind0 = ["grupo", fase_int]
    df_grupos = df2.copy()
    dict_table = objs.fase_de_grupos(dict_g, df_grupos, modelo)

    # dieciseisavos (Match 73–88) con las tablas de grupos ya jugadas y el reglamento de terceros
    df_32 = alinear_llave_terceros(dict_table, reglamento_obj.agno, df_grupos=df_grupos, personalizado=reglamento_obj)
    df_32.insert(1, "score", df3[["home", "score", "away"]].iloc[:16]["score"].to_numpy())
    df_32 = objs.camb_grp_elm_smpl(df_grupos, df_32, dict_table)

    if fase_int == "dieciseisavos":
        equipos, base = _base_por_equipo(df_32)
        return equipos, base, _orden_desde_df(df_32)

    objs.n_partidos = df_grupos.shape[0] + 1
    objs.ind0[0] = "dieciseisavos"
    df_32 = objs.jugar(df_32, modelo)

    df_knockout = objs.ordenar_elm_smpl(df_32, df3[["home", "score", "away"]].iloc[16:24].copy())
    if fase_int == "knockout":
        equipos, base = _base_por_equipo(df_knockout)
        return equipos, base, _orden_desde_df(df_knockout)

    objs.ind0[0] = "knockout"
    df_knockout = objs.jugar(df_knockout, modelo)
    df_quarter = objs.ordenar_elm_smpl(df_knockout, df3[["home", "score", "away"]].iloc[24:28].copy())
    if fase_int == "quarter":
        equipos, base = _base_por_equipo(df_quarter)
        return equipos, base, _orden_desde_df(df_quarter)

    objs.ind0[0] = "quarter"
    df_quarter = objs.jugar(df_quarter, modelo)
    df_semi = objs.ordenar_elm_smpl(df_quarter, df3[["home", "score", "away"]].iloc[28:30].copy())
    if fase_int == "semi":
        equipos, base = _base_por_equipo(df_semi)
        return equipos, base, _orden_desde_df(df_semi)

    objs.ind0[0] = "semi"
    df_semi = objs.jugar(df_semi, modelo)
    df_final = objs.ordenar_elm_smpl(df_semi, df3[["home", "score", "away"]].iloc[31].copy())
    equipos, base = _base_por_equipo(df_final)
    return equipos, base, _orden_desde_df(df_final)


def probabilidades_campeon(recepcion, drop_metricas=(), metricas_opcionales=(), reglamento=None,
                           n=5000, semilla=0):
    """Simula ``n`` torneos desde la fase indicada y devuelve el cuadro de campeones.

    ``reglamento`` (dict) es un reglamento de terceros personalizado (1986/1990/1994/2026)."""
    agno = int(recepcion[0])
    fase = _FASES.get(recepcion[1])
    if fase is None:
        raise ValueError(f"Fase no soportada: {recepcion[1]!r}")
    objs, df2, df3, dict_g, modelo = _construir_objs(agno, drop_metricas, metricas_opcionales)

    reglamento_obj = None
    if agno in _TERCEROS:
        reglamento_obj = (ReglamentoTerceros.desde_json(agno, reglamento) if reglamento is not None
                          else ReglamentoTerceros.oficial(agno))

    rng = np.random.default_rng(semilla)

    if fase == "grupo":
        equipos, base = _base_por_equipo(df2)
        lambdas = _lambdas_desde_base(objs, equipos, base, modelo, drop_metricas, metricas_opcionales)
        cuenta = {}

        if agno in _ELIMINACION:
            orden = _orden_1934_38(df2)
            for _ in range(n):
                campeon, finalista, semifinalistas = simular_knockout(rng, lambdas, orden)
                cuenta.setdefault(campeon, [0, 0, 0])[0] += 1
                cuenta.setdefault(finalista, [0, 0, 0])[1] += 1
                for equipo in semifinalistas:
                    cuenta.setdefault(equipo, [0, 0, 0])[2] += 1
        else:
            grupos_partidos = {}
            for grupo, df in dict_g.items():
                if agno in _SEGUNDA_FASE_4 + _SEGUNDA_FASE_3 and not _es_primera_fase(grupo):
                    continue
                ids = {int(p) for p in df["pais"]}
                grupos_partidos[grupo] = [(int(r.home), int(r.away)) for r in df2.itertuples()
                                          if int(r.home) in ids and int(r.away) in ids]
            for _ in range(n):
                if agno in _FINAL_ROUND:
                    campeon, finalista, semifinalistas = _simular_1950(rng, lambdas, grupos_partidos)
                elif agno in _SEGUNDA_FASE_4:
                    campeon, finalista, semifinalistas = _simular_1974_78(rng, lambdas, grupos_partidos)
                elif agno in _SEGUNDA_FASE_3:
                    campeon, finalista, semifinalistas = _simular_1982(rng, lambdas, grupos_partidos)
                elif agno in _HISTORICOS_8:
                    campeon, finalista, semifinalistas = _simular_un_torneo(rng, lambdas, grupos_partidos, _orden_cuartos_8)
                elif agno in _MODERNOS_16:
                    campeon, finalista, semifinalistas = _simular_un_torneo(rng, lambdas, grupos_partidos, _orden_octavos_16)
                elif agno in _TERCEROS:
                    campeon, finalista, semifinalistas = _simular_un_torneo(rng, lambdas, grupos_partidos, lambda tablas: _orden_llave_terceros(tablas, reglamento_obj))
                else:
                    raise ValueError(f"Montecarlo no soporta el formato de {agno} todavía (1954)")
                cuenta.setdefault(campeon, [0, 0, 0])[0] += 1
                cuenta.setdefault(finalista, [0, 0, 0])[1] += 1
                for equipo in semifinalistas:
                    cuenta.setdefault(equipo, [0, 0, 0])[2] += 1

    elif fase in _KNOCKOUT_MODERNO and agno in _MODERNOS_16:
        equipos, base, orden = _replay_moderno(objs, df2, df3, dict_g, modelo, fase)
        lambdas = _lambdas_desde_base(objs, equipos, base, modelo, drop_metricas, metricas_opcionales)
        cuenta = {}
        for _ in range(n):
            campeon, finalista, semifinalistas = simular_knockout(rng, lambdas, orden)
            cuenta.setdefault(campeon, [0, 0, 0])[0] += 1
            cuenta.setdefault(finalista, [0, 0, 0])[1] += 1
            for equipo in semifinalistas:
                cuenta.setdefault(equipo, [0, 0, 0])[2] += 1

    elif fase in _KNOCKOUT_2026 and agno == 2026:
        equipos, base, orden = _replay_2026(objs, df2, df3, dict_g, modelo, fase, reglamento_obj)
        lambdas = _lambdas_desde_base(objs, equipos, base, modelo, drop_metricas, metricas_opcionales)
        cuenta = {}
        for _ in range(n):
            campeon, finalista, semifinalistas = simular_knockout(rng, lambdas, orden)
            cuenta.setdefault(campeon, [0, 0, 0])[0] += 1
            cuenta.setdefault(finalista, [0, 0, 0])[1] += 1
            for equipo in semifinalistas:
                cuenta.setdefault(equipo, [0, 0, 0])[2] += 1

    else:
        raise ValueError(f"Montecarlo no soporta la fase {recepcion[1]!r} de {agno} todavía")

    filas = []
    for equipo, (c, f, s) in cuenta.items():
        filas.append({"pais": objs.id_pais[equipo], "campeon": c / n, "final": f / n, "semifinal": s / n})
    filas.sort(key=lambda r: (-r["campeon"], -r["final"], r["pais"]))
    return {"year": agno, "simulaciones": n, "campeones": filas[:5]}
