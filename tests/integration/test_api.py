"""
Tests d'intégration pour l'API FastAPI
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import numpy as np


@pytest.fixture
def client():
    """Client de test FastAPI"""
    from api.main import app
    return TestClient(app)


@pytest.fixture
def sample_customer_payload():
    """Payload de test pour un client"""
    return {
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


class TestHealthEndpoint:
    """Tests du endpoint health"""
    
    def test_health_check_success(self, client):
        """Test health check quand tout va bien"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data
    
    def test_health_check_format(self, client):
        """Test format de la réponse health"""
        response = client.get("/health")
        data = response.json()
        
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["timestamp"], str)


class TestRootEndpoint:
    """Tests du endpoint root"""
    
    def test_root_endpoint(self, client):
        """Test endpoint racine"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "status" in data
        assert "endpoints" in data
    
    def test_root_endpoints_list(self, client):
        """Test que les endpoints sont listés"""
        response = client.get("/")
        data = response.json()
        
        endpoints = data["endpoints"]
        assert "health" in endpoints
        assert "predict" in endpoints
        assert "batch_predict" in endpoints


class TestPredictionEndpoint:
    """Tests du endpoint de prédiction"""
    
    @patch('api.main.model')
    def test_predict_success(self, mock_model, client, sample_customer_payload):
        """Test prédiction réussie"""
        # Mock du modèle
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        response = client.post("/predict", json=sample_customer_payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "customer_id" in data
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "risk_level" in data
        assert "confidence" in data
        assert "timestamp" in data
    
    @patch('api.main.model')
    def test_predict_high_risk(self, mock_model, client, sample_customer_payload):
        """Test prédiction haut risque"""
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        
        response = client.post("/predict", json=sample_customer_payload)
        data = response.json()
        
        assert data["churn_probability"] == 0.8
        assert data["risk_level"] == "HIGH"
        assert data["churn_prediction"] is True
    
    @patch('api.main.model')
    def test_predict_low_risk(self, mock_model, client, sample_customer_payload):
        """Test prédiction bas risque"""
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        
        response = client.post("/predict", json=sample_customer_payload)
        data = response.json()
        
        assert data["churn_probability"] == 0.2
        assert data["risk_level"] == "LOW"
        assert data["churn_prediction"] is False
    
    @patch('api.main.model')
    def test_predict_medium_risk(self, mock_model, client, sample_customer_payload):
        """Test prédiction risque moyen"""
        mock_model.predict_proba.return_value = np.array([[0.5, 0.5]])
        
        response = client.post("/predict", json=sample_customer_payload)
        data = response.json()
        
        assert data["risk_level"] == "MEDIUM"
    
    def test_predict_without_model(self, client, sample_customer_payload):
        """Test prédiction quand le modèle n'est pas chargé"""
        with patch('api.main.model', None):
            response = client.post("/predict", json=sample_customer_payload)
            
            assert response.status_code == 503
            assert "Modèle non chargé" in response.json()["detail"]
    
    def test_predict_invalid_contract_type(self, client, sample_customer_payload):
        """Test avec type de contrat invalide"""
        sample_customer_payload["contract_type"] = "Invalid"
        
        response = client.post("/predict", json=sample_customer_payload)
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_payment_method(self, client, sample_customer_payload):
        """Test avec méthode de paiement invalide"""
        sample_customer_payload["payment_method"] = "Bitcoin"
        
        response = client.post("/predict", json=sample_customer_payload)
        
        assert response.status_code == 422
    
    def test_predict_negative_tenure(self, client, sample_customer_payload):
        """Test avec tenure négatif"""
        sample_customer_payload["tenure_months"] = -5
        
        response = client.post("/predict", json=sample_customer_payload)
        
        assert response.status_code == 422
    
    def test_predict_missing_field(self, client, sample_customer_payload):
        """Test avec champ manquant"""
        del sample_customer_payload["customer_id"]
        
        response = client.post("/predict", json=sample_customer_payload)
        
        assert response.status_code == 422


class TestBatchPredictionEndpoint:
    """Tests du endpoint de prédiction batch"""
    
    @patch('api.main.model')
    def test_batch_predict_success(self, mock_model, client, sample_customer_payload):
        """Test batch prédiction réussie"""
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        payload = {
            "customers": [
                sample_customer_payload,
                {**sample_customer_payload, "customer_id": "C000002"}
            ]
        }
        
        response = client.post("/predict/batch", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert "total_processed" in data
        assert "timestamp" in data
        assert len(data["predictions"]) == 2
    
    @patch('api.main.model')
    def test_batch_predict_empty_list(self, mock_model, client):
        """Test batch avec liste vide"""
        payload = {"customers": []}
        
        response = client.post("/predict/batch", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_processed"] == 0
    
    def test_batch_predict_without_model(self, client, sample_customer_payload):
        """Test batch sans modèle chargé"""
        with patch('api.main.model', None):
            payload = {"customers": [sample_customer_payload]}
            response = client.post("/predict/batch", json=payload)
            
            assert response.status_code == 503


class TestMetricsEndpoint:
    """Tests du endpoint metrics"""
    
    def test_metrics_endpoint(self, client):
        """Test endpoint metrics"""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model_loaded" in data


class TestDataValidation:
    """Tests de validation des données"""
    
    def test_tenure_boundaries(self, client, sample_customer_payload):
        """Test limites de tenure"""
        # Tenure = 0 (valide)
        sample_customer_payload["tenure_months"] = 0
        response = client.post("/predict", json=sample_customer_payload)
        assert response.status_code in [200, 503]  # 503 si modèle pas chargé
        
        # Tenure = 100 (valide)
        sample_customer_payload["tenure_months"] = 100
        response = client.post("/predict", json=sample_customer_payload)
        assert response.status_code in [200, 503]
        
        # Tenure = 101 (invalide)
        sample_customer_payload["tenure_months"] = 101
        response = client.post("/predict", json=sample_customer_payload)
        assert response.status_code == 422
    
    def test_charges_validation(self, client, sample_customer_payload):
        """Test validation des charges"""
        # Charges négatives
        sample_customer_payload["monthly_charges"] = -10
        response = client.post("/predict", json=sample_customer_payload)
        assert response.status_code == 422
        
        # Charges très élevées
        sample_customer_payload["monthly_charges"] = 300
        response = client.post("/predict", json=sample_customer_payload)
        assert response.status_code == 422
    
    def test_senior_citizen_validation(self, client, sample_customer_payload):
        """Test validation senior_citizen"""
        # Valide: 0 ou 1
        for value in [0, 1]:
            sample_customer_payload["senior_citizen"] = value
            response = client.post("/predict", json=sample_customer_payload)
            assert response.status_code in [200, 503]
        
        # Invalide: autre valeur
        sample_customer_payload["senior_citizen"] = 2
        response = client.post("/predict", json=sample_customer_payload)
        assert response.status_code == 422


class TestErrorHandling:
    """Tests de gestion d'erreurs"""
    
    @patch('api.main.model')
    def test_prediction_exception(self, mock_model, client, sample_customer_payload):
        """Test gestion d'exception lors de la prédiction"""
        mock_model.predict_proba.side_effect = Exception("Model error")
        
        response = client.post("/predict", json=sample_customer_payload)
        
        assert response.status_code == 500
        assert "Erreur lors de la prédiction" in response.json()["detail"]
    
    def test_malformed_json(self, client):
        """Test JSON mal formé"""
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422


class TestResponseFormat:
    """Tests du format des réponses"""
    
    @patch('api.main.model')
    def test_prediction_response_schema(self, mock_model, client, sample_customer_payload):
        """Test schéma de réponse prédiction"""
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        response = client.post("/predict", json=sample_customer_payload)
        data = response.json()
        
        # Vérifier les types
        assert isinstance(data["customer_id"], str)
        assert isinstance(data["churn_probability"], float)
        assert isinstance(data["churn_prediction"], bool)
        assert isinstance(data["risk_level"], str)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["timestamp"], str)
        assert isinstance(data["model_version"], str)
        
        # Vérifier les valeurs
        assert 0 <= data["churn_probability"] <= 1
        assert 0 <= data["confidence"] <= 1
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


class TestConcurrency:
    """Tests de concurrence"""
    
    @patch('api.main.model')
    def test_multiple_concurrent_requests(self, mock_model, client, sample_customer_payload):
        """Test requêtes concurrentes"""
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        # Simuler plusieurs requêtes
        responses = []
        for i in range(10):
            payload = {**sample_customer_payload, "customer_id": f"C{i:06d}"}
            response = client.post("/predict", json=payload)
            responses.append(response)
        
        # Toutes devraient réussir
        assert all(r.status_code in [200, 503] for r in responses)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])