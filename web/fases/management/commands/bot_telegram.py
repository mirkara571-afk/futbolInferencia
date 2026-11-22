"""H.5 · bot de Telegram (opcional).

Responde a dos comandos contra el motor de prediccion de la app:

  /start, /help               -> ayuda
  /favoritos <agno>           -> los 5 campeones mas probables (Etapa G, Montecarlo)
  /partido <home> <away>      -> probabilidades de un partido en cancha neutral (Elo head-to-head)

Uso:

  $env:TELEGRAM_TOKEN = "<token que da @BotFather con /newbot>"
  python manage.py bot_telegram
"""
import os
import time
from pathlib import Path

import pandas as pd
import requests
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from fases.contexto import ELO_INICIAL, contexto_desde_bd
from fases.montecarlo import probabilidades_campeon
from fases.predictores import Lote, PredictorElo

API = "https://api.telegram.org/bot{token}/{metodo}"
RAIZ = Path(settings.BASE_DIR).parent


class Command(BaseCommand):
  help = "Bot de Telegram (H.5): responde /favoritos y /partido con el motor de prediccion"

  def add_arguments(self, parser):
    parser.add_argument("--token", default="", help="Token de BotFather (o variable de entorno TELEGRAM_TOKEN)")

  def handle(self, *args, **opciones):
    token = opciones["token"] or os.environ.get("TELEGRAM_TOKEN", "")
    if not token:
      raise CommandError("Falta TELEGRAM_TOKEN: crealo con @BotFather (/newbot) y pasalo con --token o la variable TELEGRAM_TOKEN")
    self.elo = self._elo_entrenado()
    self.avisar("Bot activo. Envia /help en Telegram.")
    offset = 0
    while True:
      try:
        actualizaciones = self._get_updates(token, offset)
        for actualizacion in actualizaciones:
          offset = actualizacion["update_id"] + 1
          self._responder(token, actualizacion)
      except requests.RequestException as error:
        self.avisar(f"Error de red (reintento en 5s): {error}")
        time.sleep(5)
      else:
        time.sleep(1)

  def _elo_entrenado(self):
    """Entrena el Elo una vez al arrancar (barato, scipy sobre ~10k filas)."""
    ds = pd.read_csv(RAIZ / "data" / "datasets" / "partidos_features.csv")
    return PredictorElo.entrenar(ds)

  def _get_updates(self, token, offset):
    respuesta = requests.get(API.format(token=token, metodo="getUpdates"),
                             params={"timeout": 30, "offset": offset}, timeout=60)
    respuesta.raise_for_status()
    return respuesta.json().get("result", [])

  def _responder(self, token, actualizacion):
    mensaje = actualizacion.get("message") or actualizacion.get("edited_message")
    if not mensaje or "chat" not in mensaje:
      return
    chat_id = mensaje["chat"]["id"]
    texto = (mensaje.get("text") or "").strip()
    self._enviar(token, chat_id, self._respuesta(texto))

  def _enviar(self, token, chat_id, texto):
    requests.post(API.format(token=token, metodo="sendMessage"),
                  json={"chat_id": chat_id, "text": texto, "parse_mode": "HTML"}, timeout=30)

  def _respuesta(self, texto):
    partes = texto.split()
    comando = partes[0].lower() if partes else ""
    if comando in ("/start", "/help", "/ayuda"):
      return ("<b>Mundiales · bot</b>\n\n"
              "/favoritos 2026  → los 5 campeones más probables\n"
              "/partido Brasil Argentina  → probabilidades de un partido neutral")
    if comando == "/favoritos":
      return self._favoritos(partes[1] if len(partes) > 1 else "2026")
    if comando == "/partido":
      if len(partes) < 3:
        return "Uso: /partido &lt;home&gt; &lt;away&gt;  (ej. /partido Brasil Argentina)"
      return self._partido(" ".join(partes[1:-1]), partes[-1])
    return "No entiendo ese comando. Prueba /help."

  def _favoritos(self, agno):
    try:
      resultado = probabilidades_campeon([str(int(agno)), "Fase de Grupos"])
    except (ValueError, KeyError) as error:
      return f"No pude simular {agno}: {error}"
    lineas = [f"<b>Mundial {resultado['year']}</b> · {resultado['simulaciones']} simulaciones"]
    for c in resultado["campeones"]:
      lineas.append(f"• {c['pais']}: campeón {c['campeon']:.1%} · final {c['final']:.1%} · semis {c['semifinal']:.1%}")
    return "\n".join(lineas)

  def _partido(self, home, away):
    ctx = contexto_desde_bd()
    agno = max(ctx) if ctx else None
    elos = ctx[agno]["elo"] if agno is not None else {}
    tabla = pd.DataFrame([{"elo_0": elos.get(home, ELO_INICIAL), "elo_1": elos.get(away, ELO_INICIAL),
                           "local_0": 0, "local_1": 0}])
    lote = Lote(tabla)
    probs = self.elo.probabilidades(lote)[0]
    goles = self.elo.predecir(lote)[0]
    linea = "  ".join(f"{e}: {p:.1%}" for e, p in zip((home, "empate", away), probs))
    return (f"<b>{home} vs {away}</b> (neutral)\n{linea}\n"
            f"Goles esperados: {home} {goles[0]:.2f} · {away} {goles[1]:.2f}")
