from django.db import models

# home,score,away,agno,penales,jugado (score a los 90 o 120 minutos; la tanda va aparte en penales, "4:2" o "")
class partidos(models.Model):
    home = models.CharField(max_length=50)
    score = models.CharField(max_length=6)
    away = models.CharField(max_length=50)
    agno = models.IntegerField()
    penales = models.CharField(max_length=6, blank=True, default="")
    jugado = models.CharField(max_length=1, default="1")

    class Meta:
        indexes = [models.Index(fields=['agno'])]

# home,score,away,year
class fixtures(models.Model):
    home = models.CharField(max_length=50)
    score = models.CharField(max_length=10)
    away = models.CharField(max_length=50)
    agno = models.IntegerField()

    class Meta:
        indexes = [models.Index(fields=['agno'])]

# pais,Pts,PJ,PG,PE,PP,GF,GC,Dif,Grupo,agno
class grupos(models.Model):
    pais = models.CharField(max_length=50)
    Pts = models.CharField(max_length=6)
    PJ = models.CharField(max_length=6)
    PG = models.CharField(max_length=6)
    PE = models.CharField(max_length=6)
    PP = models.CharField(max_length=6)
    GF = models.CharField(max_length=6)
    GC = models.CharField(max_length=6)
    Dif = models.CharField(max_length=6)
    Grupo = models.CharField(max_length=50)
    agno = models.IntegerField()

    class Meta:
        indexes = [models.Index(fields=['agno'])]

# agno, home, away, score_0, score_1, tournament
class clasificaciones(models.Model):
    agno = models.IntegerField()
    home = models.CharField(max_length=50)
    away = models.CharField(max_length=50)
    score_0 = models.CharField(max_length=10)
    score_1 = models.CharField(max_length=10)
    tournament = models.CharField(max_length=50)

    class Meta:
        indexes = [models.Index(fields=['agno'])]

# pais, agno, gkps, mds, mos, mms, VI, SOT, PKATT, PKATTALLOW (una fila por pais y mundial)
class proxy_desempegno(models.Model):
    pais = models.CharField(max_length=50)
    agno = models.IntegerField(null=True)
    gkps = models.CharField(max_length=50)
    mds = models.CharField(max_length=50)
    mos = models.CharField(max_length=50)
    mms = models.CharField(max_length=50)
    VI = models.CharField(max_length=50)
    SOT = models.CharField(max_length=50)
    PKATT = models.CharField(max_length=50)
    PKATTALLOW = models.CharField(max_length=50)
    # partidos de esa seleccion en ese mundial: denominador de las tasas por partido del historico
    PJ = models.CharField(max_length=6, blank=True, default="")
    # 1 si los penales (PKATT, PKATTALLOW) o los tiros a puerta (SOT) son de fbref; 0 si los imputo etl/construye_proxy.py
    pk_real = models.CharField(max_length=1, default="0")
    sot_real = models.CharField(max_length=1, default="0")

    class Meta:
        indexes = [models.Index(fields=['agno'])]

# fecha, agno, home, away, score_0, score_1, tournament, categoria, neutral (etl/busca_internacionales.py)
class internacionales(models.Model):
    fecha = models.CharField(max_length=10)
    agno = models.IntegerField()
    home = models.CharField(max_length=50)
    away = models.CharField(max_length=50)
    score_0 = models.CharField(max_length=3)
    score_1 = models.CharField(max_length=3)
    tournament = models.CharField(max_length=80)
    categoria = models.CharField(max_length=30)
    neutral = models.CharField(max_length=1)

    class Meta:
        indexes = [models.Index(fields=['agno'])]

# fecha,pais,posicion,puntos (ranking FIFA, etl/busca_fifa_ranking.py)
class rankingfifa(models.Model):
    fecha = models.DateField()
    pais = models.CharField(max_length=50)
    posicion = models.SmallIntegerField()
    puntos = models.FloatField()

    class Meta:
        indexes = [models.Index(fields=['pais', 'fecha'])]
