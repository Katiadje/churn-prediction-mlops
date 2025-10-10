# 🏗️ Infrastructure

Configuration Infrastructure as Code (IaC) pour déployer la plateforme MLOps en production.

## 📁 Structure

```
infrastructure/
├── monitoring/
│   ├── prometheus.yml          # Config Prometheus
│   └── grafana/
│       ├── datasources/        # Datasources Grafana
│       └── dashboards/         # Dashboards JSON
├── nginx/
│   └── nginx.conf              # Config reverse proxy
├── terraform/
│   ├── main.tf                 # Infrastructure AWS
│   ├── variables.tf            # Variables Terraform
│   └── outputs.tf              # Outputs
└── kubernetes/
    └── deployment.yaml         # Déploiement K8s
```

## 🚀 Déploiements Disponibles

### Option 1: Docker Compose (Développement/Demo)

Le plus simple pour démarrer:

```bash
# À la racine du projet
docker-compose up -d
```

**Services disponibles:**
- 🤖 API: http://localhost:8000
- 📊 Dashboard: http://localhost:8501
- 📈 MLflow: http://localhost:5000
- 📉 Grafana: http://localhost:3000 (admin/admin)
- 🔍 Prometheus: http://localhost:9090

### Option 2: Kubernetes (Production)

Pour déploiement scalable en production:

```bash
# Créer le namespace
kubectl apply -f kubernetes/deployment.yaml

# Vérifier les pods
kubectl get pods -n mlops-churn

# Obtenir l'URL du service
kubectl get svc -n mlops-churn churn-api-service
```

**Features:**
- ✅ Auto-scaling (3-10 replicas)
- ✅ Load balancing
- ✅ Health checks
- ✅ Persistent volumes
- ✅ Network policies

### Option 3: Terraform + AWS (Cloud Production)

Infrastructure complète sur AWS:

```bash
cd infrastructure/terraform

# Initialiser Terraform
terraform init

# Voir le plan
terraform plan

# Déployer
terraform apply
```

**Ressources créées:**
- VPC avec subnets publics/privés
- ECS Cluster pour containers
- Application Load Balancer
- ECR pour images Docker
- S3 pour artifacts
- CloudWatch pour logs
- Security Groups

## 🔧 Configuration

### 1. Monitoring (Prometheus + Grafana)

**Prometheus** scrape automatiquement:
- API metrics (`/metrics`)
- MLflow metrics
- System metrics

**Grafana** affiche:
- Request rate
- Response time (p95, p99)
- Error rate
- Prediction volume
- Model performance
- Data drift score

**Accès Grafana:**
```
URL: http://localhost:3000
User: admin
Pass: admin
```

### 2. Nginx Reverse Proxy

Routes configurées:
- `/api/*` → API FastAPI
- `/dashboard` → Streamlit
- `/mlflow/` → MLflow UI
- `/monitoring/` → Grafana

**Features:**
- Rate limiting (100 req/s pour API)
- Gzip compression
- WebSocket support
- Health checks
- Security headers

### 3. Kubernetes

**Secrets à créer:**
```bash
# Snowflake credentials
kubectl create secret generic snowflake-credentials \
  --from-literal=SNOWFLAKE_ACCOUNT='your-account' \
  --from-literal=SNOWFLAKE_USER='your-user' \
  --from-literal=SNOWFLAKE_PASSWORD='your-password' \
  -n mlops-churn
```

**Auto-scaling configuré:**
- Min replicas: 3
- Max replicas: 10
- CPU target: 70%
- Memory target: 80%

### 4. Terraform (AWS)

**Variables à configurer:**
```hcl
# terraform.tfvars
aws_region   = "eu-west-1"
environment  = "production"
project_name = "churn-mlops"
```

**Commandes utiles:**
```bash
# Initialiser
terraform init

# Valider config
terraform validate

# Voir les changements
terraform plan

# Appliquer
terraform apply

# Détruire (attention!)
terraform destroy
```

## 📊 Monitoring & Alerting

### Métriques Collectées

**API:**
- `http_requests_total` - Total requests
- `http_request_duration_seconds` - Request latency
- `predictions_total{risk_level}` - Predictions par niveau de risque
- `model_f1_score` - Performance du modèle
- `data_drift_score` - Score de drift

**Système:**
- CPU usage
- Memory usage
- Disk I/O
- Network traffic

### Dashboards Grafana

Le dashboard MLOps inclut:
1. **Request Rate** - Req/sec au fil du temps
2. **Response Time** - p95/p99 latency
3. **Prediction Volume** - Total predictions
4. **High Risk Count** - Clients à haut risque
5. **Model Performance** - F1 score gauge
6. **Data Drift** - Drift score avec seuil
7. **Error Rate** - 5xx errors
8. **Memory Usage** - Consommation RAM

### Alerting (Prometheus)

À configurer dans `prometheus.yml`:

```yaml
# Exemple d'alerte
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
  for: 5m
  annotations:
    summary: "High error rate detected"
```

## 🔐 Sécurité

### Best Practices Implémentées

1. **Containers:**
   - User non-root
   - Read-only root filesystem (où possible)
   - No privileged mode
   - Resource limits

2. **Network:**
   - Network policies K8s
   - Security groups AWS
   - Rate limiting Nginx
   - HTTPS/TLS (à configurer)

3. **Secrets:**
   - Kubernetes secrets
   - AWS Secrets Manager (Terraform)
   - Jamais de credentials en clair

4. **Headers Sécurité:**
   - X-Frame-Options
   - X-Content-Type-Options
   - X-XSS-Protection

## 🚀 Déploiement CI/CD

### Pipeline GitHub Actions

Le workflow `.github/workflows/ci_cd.yml` gère:

1. **Tests** - Unit + Integration
2. **Build** - Docker image
3. **Push** - ECR/Docker Hub
4. **Deploy** - K8s/ECS automatique

### Déploiement Manuel

```bash
# Build image
docker build -t churn-api:latest .

# Tag pour registry
docker tag churn-api:latest your-registry/churn-api:latest

# Push
docker push your-registry/churn-api:latest

# Deploy K8s
kubectl set image deployment/churn-api \
  api=your-registry/churn-api:latest \
  -n mlops-churn

# Rollout status
kubectl rollout status deployment/churn-api -n mlops-churn
```

## 📈 Scaling

### Horizontal Scaling

**Docker Compose:**
```bash
docker-compose up -d --scale api=5
```

**Kubernetes:**
```bash
kubectl scale deployment churn-api --replicas=5 -n mlops-churn
```

**Auto-scaling (HPA):**
Configuré automatiquement basé sur CPU/Memory.

### Vertical Scaling

Modifier resources dans:
- `docker-compose.yml` (deploy > resources)
- `kubernetes/deployment.yaml` (resources > limits)

## 🔍 Troubleshooting

### Logs

**Docker Compose:**
```bash
docker-compose logs -f api
docker-compose logs -f mlflow
```

**Kubernetes:**
```bash
kubectl logs -f deployment/churn-api -n mlops-churn
kubectl logs -f deployment/mlflow -n mlops-churn
```

### Health Checks

```bash
# API
curl http://localhost:8000/health

# MLflow
curl http://localhost:5000/health

# Prometheus targets
curl http://localhost:9090/api/v1/targets
```

### Debug Pods K8s

```bash
# Exec dans un pod
kubectl exec -it deployment/churn-api -n mlops-churn -- /bin/bash

# Describe pod
kubectl describe pod <pod-name> -n mlops-churn

# Events
kubectl get events -n mlops-churn --sort-by='.lastTimestamp'
```

## 📝 Checklist Déploiement Production

- [ ] Secrets configurés (Snowflake, AWS, etc.)
- [ ] HTTPS/TLS activé
- [ ] Backups configurés (DB, artifacts)
- [ ] Monitoring actif (Prometheus + Grafana)
- [ ] Alerting configuré (email, Slack)
- [ ] Rate limiting en place
- [ ] Auto-scaling configuré
- [ ] Logs centralisés
- [ ] Health checks fonctionnels
- [ ] CI/CD pipeline actif
- [ ] Documentation à jour
- [ ] Disaster recovery plan

## 💡 Améliorations Futures

- [ ] Service Mesh (Istio/Linkerd)
- [ ] Chaos Engineering (Chaos Monkey)
- [ ] Blue/Green deployment
- [ ] Canary releases
- [ ] Multi-region deployment
- [ ] Cost optimization (Spot instances)
- [ ] Backup/Restore automatique
- [ ] Advanced alerting (PagerDuty)

## 📚 Ressources

- [Docker Compose docs](https://docs.docker.com/compose/)
- [Kubernetes docs](https://kubernetes.io/docs/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Prometheus docs](https://prometheus.io/docs/)
- [Grafana docs](https://grafana.com/docs/)

---

**Made with ❤️ for Production MLOps**