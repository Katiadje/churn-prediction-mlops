# Multi-stage build pour optimiser la taille
FROM python:3.9-slim as builder

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Installer dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements
WORKDIR /build
COPY requirements.txt .

# Installer dépendances Python
RUN pip install --user -r requirements.txt

# Stage final
FROM python:3.9-slim

# Installer curl pour le healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Créer user non-root pour sécurité
RUN useradd -m -u 1000 mlops && \
    mkdir -p /app /app/artifacts /app/logs && \
    chown -R mlops:mlops /app

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PATH=/home/mlops/.local/bin:$PATH \
    PYTHONPATH=/app:$PYTHONPATH \
    APP_HOME=/app

# Copier dépendances depuis builder
COPY --from=builder --chown=mlops:mlops /root/.local /home/mlops/.local

# Copier le code
WORKDIR /app
COPY --chown=mlops:mlops . .

# Créer dossiers nécessaires
RUN mkdir -p /app/mlruns /app/artifacts

# Switch vers user non-root
USER mlops

# Exposer le port
EXPOSE 8000

# Commande de démarrage - CORRIGÉE
CMD ["uvicorn", "ready.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]