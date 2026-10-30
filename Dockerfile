FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WHITENOISE=True \
    DEBUG=False

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# tensorflow-cpu pesa menos y no necesita CUDA; el resto de requirements se instala igual
RUN sed 's/^tensorflow==/tensorflow-cpu==/' requirements.txt > /tmp/requirements-docker.txt \
    && pip install -r /tmp/requirements-docker.txt \
    && pip install "gunicorn>=23,<24" "whitenoise>=6.9,<7"

COPY . .

WORKDIR /app/web
RUN python manage.py collectstatic --noinput

EXPOSE 8000

CMD ["sh", "-c", "python manage.py migrate --noinput && python manage.py load_data --reemplazar && gunicorn mundial.wsgi:application --bind 0.0.0.0:8000 --workers 2 --timeout 180"]
