# Dockerfile para DISICO Dashboard
# EasyPanel - KVM 2 (8GB RAM, 100GB NVMe)

FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para compilación
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias Python
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copiar código fuente
COPY src/ ./src/
COPY app.py .

# Puerto expuesto
EXPOSE 8050

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8050/ || exit 1

# Comando de inicio con Gunicorn
# --workers 2: 2 workers para manejar concurrencia
# --timeout 120: timeout de 120 segundos para carga inicial
# --bind: escuchar en todas las interfaces
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8050", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:server"]
