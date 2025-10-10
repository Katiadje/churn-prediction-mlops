#!/bin/bash

# Script d'installation et setup du projet MLOps Churn Prediction
# Usage: ./setup.sh

set -e  # Exit on error

echo "🚀 ====================================="
echo "🚀 CUSTOMER CHURN MLOPS SETUP"
echo "🚀 ====================================="

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages
log_info() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Vérifier Python
echo ""
echo "📋 Vérification des prérequis..."
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 n'est pas installé"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d " " -f 2 | cut -d "." -f 1,2)
log_info "Python $PYTHON_VERSION détecté"

# Créer environnement virtuel
echo ""
echo "🐍 Création de l'environnement virtuel..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    log_info "Environnement virtuel créé"
else
    log_warn "Environnement virtuel existe déjà"
fi

# Activer environnement
source venv/bin/activate || source venv/Scripts/activate

# Upgrade pip
log_info "Mise à jour de pip..."
pip install --upgrade pip --quiet

# Installer dépendances
echo ""
echo "📦 Installation des dépendances..."
pip install -r requirements.txt --quiet
log_info "Dépendances installées"

# Créer structure de dossiers
echo ""
echo "📁 Création de la structure de dossiers..."
mkdir -p artifacts
mkdir -p logs
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/saved_models
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p monitoring/reports
mkdir -p mlruns
mkdir -p mlartifacts

log_info "Structure créée"

# Créer fichier .env s'il n'existe pas
echo ""
if [ ! -f ".env" ]; then
    echo "⚙️  Configuration de l'environnement..."
    cp .env.example .env
    log_info "Fichier .env créé à partir de .env.example"
    log_warn "⚠️  IMPORTANT: Édite le fichier .env avec tes credentials Snowflake!"
else
    log_warn "Fichier .env existe déjà"
fi

# Créer fichiers __init__.py
echo ""
echo "📝 Création des fichiers __init__.py..."
touch api/__init__.py
touch data/__init__.py
touch features/__init__.py
touch models/__init__.py
touch monitoring/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py

log_info "Fichiers __init__.py créés"

# Vérifier si Docker est installé
echo ""
if command -v docker &> /dev/null; then
    log_info "Docker détecté"
    
    echo ""
    echo "🐳 Veux-tu lancer les services Docker? (MLflow, Prometheus, Grafana)"
    read -p "(y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose up -d
        log_info "Services Docker lancés"
        log_info "MLflow UI: http://localhost:5000"
        log_info "Grafana: http://localhost:3000 (admin/admin)"
        log_info "Prometheus: http://localhost:9090"
    fi
else
    log_warn "Docker non installé - skip services"
fi

# Tests rapides
echo ""
echo "🧪 Lancement des tests..."
if pytest tests/ -v --maxfail=1 --tb=short 2>/dev/null; then
    log_info "Tests passés ✨"
else
    log_warn "Certains tests ont échoué (normal si Snowflake non configuré)"
fi

# Résumé
echo ""
echo "🎉 ====================================="
echo "🎉 SETUP TERMINÉ AVEC SUCCÈS!"
echo "🎉 ====================================="
echo ""
echo "📋 Prochaines étapes:"
echo ""
echo "1️⃣  Configure tes credentials Snowflake dans .env"
echo "   vim .env"
echo ""
echo "2️⃣  Charge les données dans Snowflake:"
echo "   python data/load_data.py"
echo ""
echo "3️⃣  Créer les features:"
echo "   python features/build_features.py"
echo ""
echo "4️⃣  Entraîner le modèle:"
echo "   python models/train.py"
echo ""
echo "5️⃣  Lancer l'API:"
echo "   uvicorn api.main:app --reload"
echo "   # API disponible sur http://localhost:8000"
echo ""
echo "6️⃣  Lancer le dashboard:"
echo "   streamlit run streamlit_app/dashboard.py"
echo "   # Dashboard disponible sur http://localhost:8501"
echo ""
echo "7️⃣  Monitoring du drift:"
echo "   python monitoring/drift_detector.py"
echo ""
echo "📚 Documentation complète: README.md"
echo "🐛 Tests: pytest tests/ -v"
echo "🚀 CI/CD: Push vers GitHub pour déclencher le pipeline"
echo ""
echo "💡 Besoin d'aide? Consulte le README.md"
echo ""
echo "====================================="
echo "🌟 Bon développement! 🌟"
echo "====================================="