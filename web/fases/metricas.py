"""
Métricas derivadas por partido (gkps, mds, mos, mms, rate_GF/rate_GC).

Extraído de modulo.py en el refactor H.2: funciones puras sobre Series de pandas, sin acceso
a la base de datos. Las usa calculo_metricas_0 (modulo.py).
"""


def _por_partido(valor, PJ):
    """Divide fila a fila por PJ; NaN donde PJ es 0 (cada calc_* lo pasa luego a 0)."""
    return valor / PJ.where(PJ > 0)

# tasas por partido para un equipo sin partidos en su historico abstracto: ~30 % de vallas invictas, ~4 tiros a puerta y
# ~0,12 penales a favor y en contra por partido (las de etl/construye_proxy.py)
TASAS_REFERENCIA = {"VI": 0.30, "SOT": 4.0, "PKATT": 0.12, "PKATTALLOW": 0.12}
# parte de los penales que terminan en gol, para descontar los goles de penal
CONVERSION_PENAL = 0.75

def _tasa_historica(valor, PJ_hist, metrica):
    """(valor) del historico abstracto por partido de ese mismo historico (PJ_hist); la tasa de referencia si el equipo no tiene partidos en su historico"""
    return (valor / PJ_hist.where(PJ_hist > 0)).fillna(TASAS_REFERENCIA[metrica])

def _tope(serie, referencia):
    """serie / referencia, sin pasar de 1: cuanto de un valor de referencia alcanza el equipo"""
    return (serie / referencia).clip(upper=1)

def calc_gkps(PJ, GC, VI, PKATTALLOW, PJ_hist):
    """
    ¿QUE HACE?
    Calcula, partido a partido, el puntaje de portería (goalkeeping score, 0 a 100): premia las vallas invictas y castiga los goles recibidos que no vinieron de penal

    Args:

      - PJ, GC: Series con el acumulado de partidos jugados y goles en contra del ciclo hasta cada partido

      - VI, PKATTALLOW, PJ_hist: Series con el progreso de vallas invictas, penales permitidos y partidos del historico abstracto

    Return:

      - Serie con 60 * VI/PJ_hist + 40 - 10 * (GC/PJ - 0,75 * PKATTALLOW/PJ_hist), entre 0 y 100, o 0 donde PJ es 0
    """
    vi, pka = _tasa_historica(VI, PJ_hist, "VI"), _tasa_historica(PKATTALLOW, PJ_hist, "PKATTALLOW")
    gc_sin_penal = (_por_partido(GC, PJ) - CONVERSION_PENAL * pka).clip(lower=0)
    return (vi * 60 + 40 - gc_sin_penal * 10).clip(0, 100).fillna(0)

def calc_mds(PJ, GC, VI, PKATTALLOW, PJ_hist):
    """
    ¿QUE HACE?
    Calcula, partido a partido, el puntaje defensivo (mean defensive score, 0 a 100): premia las vallas invictas y castiga goles recibidos y penales permitidos

    Args:

      - PJ, GC: Series con el acumulado de partidos jugados y goles en contra del ciclo hasta cada partido

      - VI, PKATTALLOW, PJ_hist: Series con el progreso de vallas invictas, penales permitidos y partidos del historico abstracto

    Return:

      - Serie con 50 * VI/PJ_hist + 50 - 10 * (GC/PJ + PKATTALLOW/PJ_hist), entre 0 y 100, o 0 donde PJ es 0
    """
    vi, pka = _tasa_historica(VI, PJ_hist, "VI"), _tasa_historica(PKATTALLOW, PJ_hist, "PKATTALLOW")
    return (vi * 50 + 50 - (_por_partido(GC, PJ) + pka) * 10).clip(0, 100).fillna(0)

def calc_mos(PJ, GF, SOT, PKATT, PJ_hist):
    """
    ¿QUE HACE?
    Calcula, partido a partido, el puntaje ofensivo (mean offensive score, 0 a 100): goles por partido, tiros a puerta por partido y efectividad de los tiros sin contar penales

    Args:

      - PJ, GF: Series con el acumulado de partidos jugados y goles a favor del ciclo hasta cada partido

      - SOT, PKATT, PJ_hist: Series con el progreso de tiros a puerta, penales intentados y partidos del historico abstracto

    Return:

      - Serie con 50 * min(1, (GF/PJ)/3) + 30 * min(1, (SOT/PJ_hist)/6) + 20 * min(1, efectividad/0,5), donde efectividad = (GF/PJ - 0,75 * PKATT/PJ_hist) / (SOT/PJ_hist); 0 donde PJ es 0
    """
    gf, sot, pk = _por_partido(GF, PJ), _tasa_historica(SOT, PJ_hist, "SOT"), _tasa_historica(PKATT, PJ_hist, "PKATT")
    efectividad = ((gf - CONVERSION_PENAL * pk).clip(lower=0) / sot.where(sot > 0)).fillna(0)
    return (_tope(gf, 3) * 50 + _tope(sot, 6) * 30 + _tope(efectividad, 0.5) * 20).fillna(0)

def calc_mms(PJ, PTS, PG, SOT, PJ_hist):
    """
    ¿QUE HACE?
    Calcula, partido a partido, el puntaje de momento (mean momentum score, 0 a 100): proporción de puntos obtenidos sobre los posibles en el ciclo y tiros a puerta por partido del historico. (PG) no se usa: la proporción de victorias ya está dentro de los puntos

    Args:

      - PJ, PTS, PG: Series con el acumulado de partidos jugados, puntos y partidos ganados del ciclo hasta cada partido

      - SOT, PJ_hist: Series con el progreso de tiros a puerta y partidos del historico abstracto

    Return:

      - Serie con 60 * PTS/(3*PJ) + 40 * min(1, (SOT/PJ_hist)/5), o 0 donde PJ es 0
    """
    return (_por_partido(PTS, PJ * 3) * 60 + _tope(_tasa_historica(SOT, PJ_hist, "SOT"), 5) * 40).fillna(0)

def calc_rate(valor, PJ):
    """
    ¿QUE HACE?
    Calcula, partido a partido, la tasa de una métrica acumulada (GF o GC) por cada 10 partidos

    Args:

      - valor: Serie con el acumulado de la métrica (GF o GC) hasta cada partido

      - PJ: Serie con el acumulado de partidos jugados; un PJ de 0 se reemplaza por 0.085 para no dividir por 0

    Return:

      - Serie con (valor/PJ)*10
    """
    return (valor / PJ.replace(0, 0.085)) * 10

