# API, Docker y CI (H.1, H.3, H.6)

**Fecha:** 2026-10-30 · **Rama:** `exp/etapa-a-medicion`

## 1. API REST (H.1)

Implementada con Django REST Framework + drf-spectacular.

| Ruta | Método | Estado |
|---|---|---|
| `/api/prediccion/` | POST | Funciona; misma `recepcion` que la web |
| `/api/probabilidades/` | POST | **501**: pendiente de la Etapa G (Montecarlo de campeones) |
| `/api/schema/` | GET | OpenAPI 3 |
| `/api/docs/` | GET | Swagger UI |

La lógica no se duplica: `fases/views.py` expone `predecir_recepcion(data)` y tanto la web como la API la usan. La web sigue devolviendo `partidos` como string JSON (el template lo parsea); la API devuelve listas JSON.

**Ejemplo:**

```powershell
curl -X POST http://127.0.0.1:8000/api/prediccion/ `
  -H "Content-Type: application/json" `
  -d '{"recepcion":["2018","Fase de Grupos",["Rusia",0,"Arabia Saudita",0,0]]}'
```

Respuesta:

```json
{
  "year": 2018,
  "partidos": [{"home": "Rusia", "score": "1:0", "away": "Arabia Saudita"}],
  "mejores_terceros": null,
  "llave_octavos": null
}
```

**Tests:** payload inválido → 400; predicción con `consulta_general` mockeada → 200; `/api/schema/` → 200; `/api/probabilidades/` → 501; la web sigue devolviendo strings JSON. Total de la suite: **60 tests**.

## 2. Docker (H.3)

- `Dockerfile`: Python 3.12-slim, `tensorflow-cpu`, gunicorn y WhiteNoise; `collectstatic` en build.
- `docker-compose.yml`: PostgreSQL 16 + web; `depends_on` con healthcheck; volumen `pgdata`.
- `.dockerignore`: excluye `.venv`, `.git`, `.spec-tmp`, `agents`, notebooks, experimentos y docs.

```powershell
docker compose config      # validado
docker compose up --build  # no ejecutado aqui
```

**Limitación:** en este entorno el daemon de Docker no está disponible (`dockerDesktopLinuxEngine` no existe / `.docker/buildx/.lock` denegado), así que la imagen **no se construyó**. El `docker-compose.yml` sí pasó `docker compose config`.

**Nota:** el `CMD` ejecuta `migrate` y `load_data --reemplazar` en cada arranque; para una demo es cómodo, pero reinicia los datos. Para producción habría que separar la carga en un job/entrypoint condicional.

## 3. CI (H.6)

`.github/workflows/ci.yml`:

- Python 3.12, PostgreSQL 16 como servicio.
- `pip install -r requirements.txt` y `ruff`.
- `ruff check web --select E9,F63,F7,F82` (solo sintaxis/nombres indefinidos).
- `python manage.py check`.
- `python manage.py test fases -v 1`.

**No ejecutado todavía:** el workflow se validó como YAML y los comandos se ejecutaron en local, pero GitHub Actions no ha corrido. El badge no se añadió para no mostrar un estado falso.

## 4. Qué falta

- **G:** Montecarlo de los 5 campeones y `/api/probabilidades/` real.
- **H.2:** refactor de `modulo.py` con la foto de A-06.
- **H.4:** rediseño de la UI.
- **H.5:** bot de Telegram (opcional).
- **H.7:** LinkedIn/Fiverr y demo pública.
- **H.3:** construir y desplegar la imagen en un entorno con Docker.
