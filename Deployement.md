# 🚀 Guide de Déploiement Production

## Prérequis
- Kubernetes cluster (GKE, EKS, AKS)
- Helm 3.x
- Docker registry

## Déploiement sur Kubernetes

### 1. Build et Push des Images
```bash
docker build -t your-registry/churn-api:latest -f Dockerfile .
docker push your-registry/churn-api:latest