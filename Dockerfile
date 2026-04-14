FROM python:3.12-slim

WORKDIR /app

# Instalar SOLO las dependencias necesarias (minimalista)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements-deploy.txt .

# Instalar dependencias Python (NO compilar desde cero)
# Usar wheels pre-compilados de PyPI
RUN pip install --no-cache-dir --prefer-binary -r requirements-deploy.txt

# Copiar código fuente
COPY src/ ./src/
COPY app.py .

# Puerto
EXPOSE 8050

# Healthcheck (start-period alto para carga de datos grandes + Mahalanobis)
HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=300s \
    CMD curl -f http://localhost:8050/ || exit 1

# Comando de inicio
CMD ["python", "app.py"]