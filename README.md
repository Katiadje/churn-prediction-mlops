# 🚀 Customer Churn MLOps Platform

Production-ready MLOps platform for customer churn prediction using Snowflake, MLflow, and modern ML engineering practices.

## 🎯 Objectifs Business

- Prédire le churn client avec 85%+ de précision
- Réduire le coût d'acquisition client de 30%
- Pipeline automatisé de bout en bout
- Monitoring en temps réel du modèle en production

## 🏗️ Architecture

```
┌─────────────────┐
│   Snowflake     │ ──► Feature Store
│   Data Lake     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Engine  │ ──► Transformation & Quality Checks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │ ──► MLflow Tracking & Registry
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Deployment    │ ──► FastAPI + Docker
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Monitoring    │ ──► Drift Detection + Alerts
└─────────────────┘
```

## 🛠️ Stack Technique

- **Data Warehouse**: Snowflake
- **ML Tracking**: MLflow
- **API**: FastAPI
- **Monitoring**: Evidently AI
- **Orchestration**: Prefect (ou Airflow)
- **CI/CD**: GitHub Actions
- **Containerization**: Docker
- **Dashboard**: Streamlit

## 📦 Installation

### Prérequis
- Python 3.9+
- Compte Snowflake
- Docker (optionnel)

### Setup

```bash
# Clone le repo
git clone https://github.com/Katiadje/churn-prediction-mlops.git
cd churn-prediction-mlops

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer dépendances
pip install -r requirements.txt

# Configurer variables d'environnement
cp .env.example .env
# Éditer .env avec tes credentials Snowflake
```

## 🚀 Quick Start

### 1. Charger les données dans Snowflake
```bash
python data/load_data.py
```

### 2. Feature Engineering
```bash
python features/build_features.py
```

### 3. Entraîner le modèle
```bash
python models/train.py
```

### 4. Lancer l'API
```bash
uvicorn api.main:app --reload
```

### 5. Dashboard de monitoring
```bash
streamlit run streamlit_app/dashboard.py
```

## 📊 Métriques de Performance

| Métrique | Valeur | Objectif |
|----------|--------|----------|
| Accuracy | 87.3% | 85%+ |
| Precision | 84.1% | 80%+ |
| Recall | 78.5% | 75%+ |
| F1-Score | 81.2% | 78%+ |
| Inference Time | 45ms | <100ms |

## 🔄 Pipeline CI/CD

Le pipeline automatisé s'exécute sur chaque push :
1. ✅ Tests unitaires
2. ✅ Validation de données
3. ✅ Entraînement et comparaison de modèles
4. ✅ Déploiement automatique si amélioration >2%
5. ✅ Tests de régression

## 📈 Monitoring en Production

- **Data Drift**: Détection automatique avec alertes
- **Model Performance**: Suivi quotidien des métriques
- **Latency**: Monitoring temps de réponse API
- **Business Metrics**: Impact réel sur le churn

## 🧪 Tests

```bash
# Tous les tests
pytest

# Tests unitaires
pytest tests/unit/

# Tests d'intégration
pytest tests/integration/

# Coverage
pytest --cov=. --cov-report=html
```

## 📁 Structure du Projet

```
.
├── .github/
│   └── workflows/
│       └── ci_cd.yml           # Pipeline CI/CD
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app
│   ├── models.py               # Pydantic models
│   └── inference.py            # Logique prédiction
├── config/
│   ├── config.yaml             # Configuration générale
│   └── model_config.yaml       # Hyperparamètres
├── data/
│   ├── load_data.py            # Chargement Snowflake
│   └── validate_data.py        # Data quality checks
├── features/
│   ├── build_features.py       # Feature engineering
│   └── feature_store.py        # Gestion feature store
├── models/
│   ├── train.py                # Entraînement
│   ├── evaluate.py             # Évaluation
│   └── registry.py             # MLflow registry
├── monitoring/
│   ├── drift_detector.py       # Détection drift
│   └── performance_tracker.py  # Suivi performance
├── streamlit_app/
│   └── dashboard.py            # Dashboard Streamlit
├── tests/
│   ├── unit/
│   └── integration/
├── .env.example                # Template variables env
├── .gitignore
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🎓 Compétences Démontrées

✅ **MLOps Best Practices**
- Feature store et versioning
- Model registry et tracking
- CI/CD pour ML
- Monitoring et alerting

✅ **Production Engineering**
- API REST scalable
- Containerization
- Infrastructure as Code
- Tests automatisés

✅ **Data Engineering**
- Pipeline ETL avec Snowflake
- Data quality checks
- Feature engineering at scale

## 📝 Prochaines Étapes (Roadmap)

- [ ] Ajouter A/B testing framework
- [ ] Implémenter AutoML
- [ ] Multi-model serving
- [ ] Kubernetes deployment
- [ ] Real-time feature computation

## 📄 License

MIT License - Libre d'utilisation pour ton portfolio

## 👤 Auteur

**Katia_Djellali**
- LinkedIn: [https://www.linkedin.com/in/katia-djellali/]
- GitHub: [@Katiadje]

---

⭐ Si ce projet t'aide, laisse une star sur GitHub !