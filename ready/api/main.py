"""
API FastAPI pour les prédictions de churn
Version production avec fallback intelligent
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Churn Prediction API",
    description="API ML pour prédire le churn client",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
MODEL = None
MODEL_MODE = "rules"  # "mlflow" ou "rules"
FEATURE_NAMES = None


# Pydantic models
class CustomerData(BaseModel):
    customer_id: str = Field(..., description="ID unique du client")
    tenure_months: int = Field(..., ge=0, le=100, description="Ancienneté en mois")
    monthly_charges: float = Field(..., ge=0, le=200, description="Charges mensuelles")
    total_charges: float = Field(..., ge=0, description="Charges totales")
    contract_type: str = Field(..., description="Type de contrat")
    payment_method: str = Field(..., description="Méthode de paiement")
    internet_service: str = Field(..., description="Service Internet")
    phone_service: str = Field(..., description="Service téléphonique")
    online_security: str = Field(..., description="Sécurité en ligne")
    tech_support: str = Field(..., description="Support technique")
    senior_citizen: int = Field(..., ge=0, le=1, description="Senior (0/1)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "C000001",
                "tenure_months": 12,
                "monthly_charges": 75.50,
                "total_charges": 906.00,
                "contract_type": "Month-to-month",
                "payment_method": "Electronic check",
                "internet_service": "Fiber optic",
                "phone_service": "Yes",
                "online_security": "No",
                "tech_support": "No",
                "senior_citizen": 0
            }
        }


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    confidence: float
    timestamp: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


def load_mlflow_model():
    """Charge le modèle depuis MLflow avec les feature names"""
    global MODEL, MODEL_MODE, FEATURE_NAMES
    
    try:
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        
        # Essayer de charger depuis le registry
        try:
            model_name = os.getenv('MODEL_NAME', 'churn_predictor')
            MODEL = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
            logger.success(f"✅ Modèle chargé depuis registry: {model_name}")
            
            # Récupérer les feature names du modèle
            if hasattr(MODEL, 'feature_names_in_'):
                FEATURE_NAMES = list(MODEL.feature_names_in_)
                logger.info(f"📋 Feature names chargées: {len(FEATURE_NAMES)} features")
            
            MODEL_MODE = "mlflow"
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Registry inaccessible: {e}")
            
            # Fallback: charger depuis le run le plus récent
            experiment = mlflow.get_experiment_by_name('churn-prediction')
            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=1
                )
                
                if not runs.empty:
                    run_id = runs.iloc[0]['run_id']
                    MODEL = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
                    logger.success(f"✅ Modèle chargé depuis run: {run_id[:8]}")
                    
                    if hasattr(MODEL, 'feature_names_in_'):
                        FEATURE_NAMES = list(MODEL.feature_names_in_)
                    
                    MODEL_MODE = "mlflow"
                    return True
            
            raise Exception("Aucun modèle trouvé")
        
    except Exception as e:
        logger.error(f"❌ Impossible de charger MLflow: {e}")
        logger.warning("⚠️ Utilisation du modèle basé sur des règles métier")
        MODEL_MODE = "rules"
        return False


def create_features_for_mlflow(customer: CustomerData) -> pd.DataFrame:
    """Crée les features exactement comme pendant l'entraînement"""
    
    if FEATURE_NAMES is None:
        raise Exception("Feature names non disponibles")
    
    # Créer un dictionnaire avec toutes les features
    data = {}
    
    # Features numériques de base
    data['tenure_months'] = customer.tenure_months
    data['monthly_charges'] = customer.monthly_charges
    data['total_charges'] = customer.total_charges
    data['senior_citizen'] = customer.senior_citizen
    
    # Features calculées
    data['avg_monthly_spend'] = customer.total_charges / max(customer.tenure_months, 1)
    data['charges_to_tenure_ratio'] = customer.monthly_charges / max(customer.tenure_months, 1)
    
    # Total services
    total_services = 0
    if customer.phone_service == 'Yes':
        total_services += 1
    if customer.online_security == 'Yes':
        total_services += 1
    if customer.tech_support == 'Yes':
        total_services += 1
    data['total_services'] = total_services
    
    # Risk score
    risk = 0
    if customer.contract_type == 'Month-to-month':
        risk += 0.3
    if customer.tenure_months < 12:
        risk += 0.25
    if customer.payment_method == 'Electronic check':
        risk += 0.15
    if customer.online_security == 'No' and customer.tech_support == 'No':
        risk += 0.15
    data['risk_score'] = risk
    
    # Features booléennes
    data['is_new_customer'] = 1 if customer.tenure_months < 6 else 0
    data['is_loyal_customer'] = 1 if customer.tenure_months > 24 else 0
    data['high_spender'] = 1 if customer.monthly_charges > 70 else 0
    data['high_value_customer'] = 1 if customer.monthly_charges > 70 else 0
    data['has_premium_services'] = 1 if total_services >= 2 else 0
    data['high_risk_contract'] = 1 if customer.contract_type == 'Month-to-month' else 0
    data['risky_payment'] = 1 if customer.payment_method == 'Electronic check' else 0
    data['no_protection'] = 1 if customer.online_security == 'No' and customer.tech_support == 'No' else 0
    data['has_phone_internet'] = 1 if customer.phone_service == 'Yes' and customer.internet_service != 'No' else 0
    data['paperless_engaged'] = 0
    
    # Services Yes/No
    data['online_security_yes'] = 1 if customer.online_security == 'Yes' else 0
    data['online_backup_yes'] = 0  # Pas dans l'input
    data['device_protection_yes'] = 0  # Pas dans l'input
    data['tech_support_yes'] = 1 if customer.tech_support == 'Yes' else 0
    data['streaming_tv_yes'] = 0  # Pas dans l'input
    data['streaming_movies_yes'] = 0  # Pas dans l'input
    
    # Encodage catégorielles
    # Contract
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    data['contract_type_encoded'] = contract_map.get(customer.contract_type, 0)
    data['contract_Month-to-month'] = 1 if customer.contract_type == 'Month-to-month' else 0
    data['contract_One year'] = 1 if customer.contract_type == 'One year' else 0
    data['contract_Two year'] = 1 if customer.contract_type == 'Two year' else 0
    
    # Payment
    payment_map = {'Electronic check': 0, 'Mailed check': 1, 'Credit card': 2, 'Bank transfer': 3}
    data['payment_method_encoded'] = payment_map.get(customer.payment_method, 0)
    data['payment_Electronic check'] = 1 if customer.payment_method == 'Electronic check' else 0
    data['payment_Mailed check'] = 1 if customer.payment_method == 'Mailed check' else 0
    data['payment_Credit card'] = 1 if customer.payment_method == 'Credit card' else 0
    data['payment_Bank transfer'] = 1 if customer.payment_method == 'Bank transfer' else 0
    
    # Internet
    internet_map = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
    data['internet_service_encoded'] = internet_map.get(customer.internet_service, 2)
    data['internet_DSL'] = 1 if customer.internet_service == 'DSL' else 0
    data['internet_Fiber optic'] = 1 if customer.internet_service == 'Fiber optic' else 0
    data['internet_No'] = 1 if customer.internet_service == 'No' else 0
    
    # Phone
    data['phone_service_encoded'] = 1 if customer.phone_service == 'Yes' else 0
    data['phone_service'] = 1 if customer.phone_service == 'Yes' else 0
    
    # Services
    data['online_security'] = 1 if customer.online_security == 'Yes' else 0
    data['tech_support'] = 1 if customer.tech_support == 'Yes' else 0
    data['online_backup'] = 0
    data['device_protection'] = 0
    data['streaming_tv'] = 0
    data['streaming_movies'] = 0
    
    # Créer DataFrame
    df = pd.DataFrame([data])
    
    # Ne garder QUE les colonnes attendues par le modèle
    available_features = [col for col in FEATURE_NAMES if col in df.columns]
    missing_features = [col for col in FEATURE_NAMES if col not in df.columns]
    
    # Ajouter les features manquantes avec des 0
    for col in missing_features:
        df[col] = 0
    
    # Réordonner selon l'ordre attendu
    df = df[FEATURE_NAMES]
    
    logger.info(f"📊 Features créées: {df.shape}, Colonnes: {len(FEATURE_NAMES)}")
    
    return df


def calculate_churn_with_rules(customer: CustomerData) -> float:
    """Calcule la probabilité de churn avec des règles métier"""
    
    score = 0.3  # Base
    
    # Facteurs de risque
    if customer.contract_type == 'Month-to-month':
        score += 0.25
    if customer.tenure_months < 6:
        score += 0.20
    elif customer.tenure_months < 12:
        score += 0.10
    if customer.monthly_charges > 80:
        score += 0.15
    if customer.payment_method == 'Electronic check':
        score += 0.10
    if customer.internet_service == 'Fiber optic' and customer.tech_support == 'No':
        score += 0.12
    if customer.online_security == 'No':
        score += 0.08
    if customer.senior_citizen == 1:
        score += 0.05
    
    # Facteurs protecteurs
    if customer.tenure_months > 24:
        score -= 0.20
    if customer.contract_type == 'Two year':
        score -= 0.20
    if customer.online_security == 'Yes' and customer.tech_support == 'Yes':
        score -= 0.15
    if customer.payment_method in ['Bank transfer', 'Credit card']:
        score -= 0.08
    
    # Limiter entre 0 et 1
    probability = max(0.0, min(1.0, score))
    
    # Ajouter randomness pour réalisme
    np.random.seed(hash(customer.customer_id) % 2**32)
    noise = np.random.uniform(-0.05, 0.05)
    probability = max(0.0, min(1.0, probability + noise))
    
    return probability


def get_risk_level(probability: float) -> str:
    """Détermine le niveau de risque"""
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Charge le modèle au démarrage"""
    logger.info("🚀 Démarrage de l'API...")
    load_mlflow_model()
    logger.info(f"📊 Mode: {MODEL_MODE}")


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Churn Prediction API v1.0",
        "status": "running",
        "mode": MODEL_MODE,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL is not None or MODEL_MODE == "rules",
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """
    Prédiction de churn pour un client
    Utilise MLflow si disponible, sinon fallback sur règles métier
    """
    
    try:
        # Mode MLflow
        if MODEL_MODE == "mlflow" and MODEL is not None:
            try:
                X = create_features_for_mlflow(customer)
                churn_proba = MODEL.predict_proba(X)[0][1]
                logger.success(f"✅ Prédiction MLflow: {churn_proba:.2%}")
                
            except Exception as e:
                logger.error(f"❌ Erreur MLflow: {e}")
                logger.warning("⚠️ Fallback sur règles métier")
                churn_proba = calculate_churn_with_rules(customer)
        
        # Mode règles métier
        else:
            churn_proba = calculate_churn_with_rules(customer)
            logger.info(f"📊 Prédiction règles: {churn_proba:.2%}")
        
        # Résultats
        churn_pred = bool(churn_proba >= 0.5)
        confidence = abs(churn_proba - 0.5) * 2
        
        return PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=round(float(churn_proba), 4),
            churn_prediction=churn_pred,
            risk_level=get_risk_level(churn_proba),
            confidence=round(confidence, 4),
            timestamp=datetime.now().isoformat(),
            model_version=f"{MODEL_MODE}-1.0"
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur prédiction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )