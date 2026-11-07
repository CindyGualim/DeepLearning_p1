# Imagen del servicio Centinela.
#
# Pensada para Hugging Face Spaces (SDK: docker), que expone el puerto 7860, pero
# corre igual en Render, Fly o local. No entrena nada: los pesos y el banco de casos
# viajan en el repositorio, asi que el arranque es de segundos.
#
# El indice de torch es el de CPU a proposito: la rueda con CUDA pesa ~2.5 GB y no
# aporta nada aqui. La GRU tiene 128 unidades y puntuar una tarjeta toma milisegundos.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

COPY requisitos-servicio.txt .
RUN pip install --no-cache-dir -r requisitos-servicio.txt

COPY src/        ./src/
COPY api/        ./api/
COPY artefactos/ ./artefactos/
COPY datos_demo/banco_casos.json datos_demo/metricas.json ./datos_demo/

# Usuario sin privilegios: en Spaces el proceso corre como uid 1000 y necesita poder
# escribir el registro de uso en datos_demo/.
RUN useradd -m -u 1000 centinela && chown -R centinela:centinela /app
USER centinela

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/salud')"

CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]
