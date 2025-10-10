"""
Tests unitaires pour le feature engineering
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestFeatureEngineering:
    """Tests pour la création de features"""
    
    @pytest.fixture
    def sample_data(self):
        """Données de test"""
        return pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003'],
            'tenure_months': [12, 6, 48],
            'monthly_charges': [75.0, 100.0, 50.0],
            'total_charges': [900.0, 600.0, 2400.0],
            'contract_type': ['Month-to-month', 'One year', 'Two year'],
            'payment_method': ['Electronic check', 'Credit card', 'Bank transfer'],
            'internet_service': ['Fiber optic', 'DSL', 'No'],
            'phone_service': ['Yes', 'Yes', 'No'],
            'online_security': ['No', 'Yes', 'No internet'],
            'tech_support': ['No', 'Yes', 'No internet'],
            'senior_citizen': [0, 1, 0],
            'churned': [1, 0, 0]
        })
    
    def test_create_business_features(self, sample_data):
        """Test création des features métier"""
        # Mock FeatureEngineer
        from features.build_features import FeatureEngineer
        
        with patch('features.build_features.connect'):
            engineer = FeatureEngineer()
            result = engineer.create_business_features(sample_data)
        
        # Vérifier que les nouvelles features sont créées
        assert 'avg_monthly_spend' in result.columns
        assert 'charges_to_tenure_ratio' in result.columns
        assert 'is_new_customer' in result.columns
        assert 'is_loyal_customer' in result.columns
        assert 'risk_score' in result.columns
        
        # Vérifier les valeurs
        assert result['is_new_customer'].iloc[1] == 1  # tenure=6
        assert result['is_loyal_customer'].iloc[2] == 1  # tenure=48
        
        # Risk score doit être entre 0 et 1
        assert (result['risk_score'] >= 0).all()
        assert (result['risk_score'] <= 1).all()
    
    def test_avg_monthly_spend_calculation(self, sample_data):
        """Test calcul avg_monthly_spend"""
        from features.build_features import FeatureEngineer
        
        with patch('features.build_features.connect'):
            engineer = FeatureEngineer()
            result = engineer.create_business_features(sample_data)
        
        # Vérifier le calcul
        expected = sample_data['total_charges'] / (sample_data['tenure_months'] + 1)
        pd.testing.assert_series_equal(
            result['avg_monthly_spend'], 
            expected,
            check_names=False
        )
    
    def test_high_risk_contract_flag(self, sample_data):
        """Test flag high_risk_contract"""
        from features.build_features import FeatureEngineer
        
        with patch('features.build_features.connect'):
            engineer = FeatureEngineer()
            result = engineer.create_business_features(sample_data)
        
        # Month-to-month devrait être à risque
        assert result['high_risk_contract'].iloc[0] == 1
        assert result['high_risk_contract'].iloc[1] == 0
        assert result['high_risk_contract'].iloc[2] == 0
    
    def test_total_services_count(self, sample_data):
        """Test comptage des services"""
        from features.build_features import FeatureEngineer
        
        with patch('features.build_features.connect'):
            engineer = FeatureEngineer()
            result = engineer.create_business_features(sample_data)
        
        # Vérifier que total_services est correct
        assert result['total_services'].iloc[0] == 0  # No services
        assert result['total_services'].iloc[1] >= 2  # Has services
    
    def test_encode_categorical(self, sample_data):
        """Test encodage des variables catégorielles"""
        from features.build_features import FeatureEngineer
        
        with patch('features.build_features.connect'):
            engineer = FeatureEngineer()
            result = engineer.encode_categorical(sample_data)
        
        # Vérifier que les colonnes encodées existent
        assert 'contract_type_encoded' in result.columns
        assert 'payment_method_encoded' in result.columns
        
        # Les valeurs encodées doivent être numériques
        assert result['contract_type_encoded'].dtype in [np.int32, np.int64]
        
        # Vérifier que les encoders sont sauvegardés
        assert 'contract_type' in engineer.label_encoders
    
    def test_scale_numerical_fit(self, sample_data):
        """Test normalisation des features numériques"""
        from features.build_features import FeatureEngineer
        
        with patch('features.build_features.connect'):
            engineer = FeatureEngineer()
            
            # Créer d'abord les features
            sample_data = engineer.create_business_features(sample_data)
            result = engineer.scale_numerical(sample_data, fit=True)
        
        # Vérifier que les colonnes sont normalisées (mean~0, std~1)
        assert abs(result['tenure_months'].mean()) < 0.1
        assert abs(result['monthly_charges'].mean()) < 0.1
        
        # Le scaler doit être fitted
        assert engineer.scaler is not None
    
    def test_select_final_features(self, sample_data):
        """Test sélection des features finales"""
        from features.build_features import FeatureEngineer
        
        with patch('features.build_features.connect'):
            engineer = FeatureEngineer()
            
            # Pipeline complet
            df = engineer.create_business_features(sample_data)
            df = engineer.encode_categorical(df)
            df = engineer.scale_numerical(df, fit=True)
            
            X, y, feature_names = engineer.select_final_features(df)
        
        # Vérifier dimensions
        assert X.shape[0] == len(sample_data)
        assert X.shape[1] > 10  # Au moins 10 features
        
        # Vérifier y
        pd.testing.assert_series_equal(y, sample_data['churned'])
        
        # Vérifier feature_names
        assert len(feature_names) == X.shape[1]
        assert 'tenure_months' in feature_names


class TestDataValidation:
    """Tests pour la validation des données"""
    
    def test_missing_values_detection(self):
        """Test détection des valeurs manquantes"""
        df = pd.DataFrame({
            'col1': [1, 2, None, 4],
            'col2': [1, 2, 3, 4]
        })
        
        missing = df.isnull().sum()
        assert missing['col1'] == 1
        assert missing['col2'] == 0
    
    def test_data_types_validation(self):
        """Test validation des types de données"""
        df = pd.DataFrame({
            'numeric': [1, 2, 3],
            'categorical': ['A', 'B', 'C']
        })
        
        assert df['numeric'].dtype in [np.int64, np.int32]
        assert df['categorical'].dtype == object
    
    def test_outlier_detection(self):
        """Test détection des outliers"""
        data = np.array([1, 2, 3, 4, 5, 100])  # 100 est un outlier
        
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        outliers = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
        
        assert outliers[-1] == True  # 100 est détecté comme outlier
    
    def test_churned_distribution(self):
        """Test distribution de la target"""
        df = pd.DataFrame({
            'churned': [0, 0, 0, 1, 1, 0, 0, 1]
        })
        
        churn_rate = df['churned'].mean()
        
        # Vérifier que le churn rate est raisonnable
        assert 0 < churn_rate < 1
        assert churn_rate == 0.375  # 3/8


class TestFeatureStore:
    """Tests pour le feature store"""
    
    @patch('features.build_features.write_pandas')
    @patch('features.build_features.connect')
    def test_save_to_snowflake(self, mock_connect, mock_write):
        """Test sauvegarde dans Snowflake"""
        from features.build_features import FeatureEngineer
        
        # Mock connexion
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        # Mock write_pandas
        mock_write.return_value = (True, 1, 100, None)
        
        engineer = FeatureEngineer()
        
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        engineer.save_to_snowflake(df, 'test_table')
        
        # Vérifier que write_pandas a été appelé
        mock_write.assert_called_once()
    
    def test_save_artifacts(self, tmp_path):
        """Test sauvegarde des artifacts"""
        import pickle
        from features.build_features import FeatureEngineer
        
        with patch('features.build_features.connect'):
            engineer = FeatureEngineer()
            
            # Créer des artifacts
            engineer.label_encoders = {'col1': 'encoder1'}
            
            # Changer le répertoire vers tmp_path
            with patch('os.makedirs'):
                with patch('builtins.open', create=True) as mock_open:
                    engineer.save_artifacts()
                    
                    # Vérifier que open a été appelé pour sauvegarder
                    assert mock_open.call_count >= 2


# Tests d'intégration légers
class TestFeatureEngineeringPipeline:
    """Tests du pipeline complet"""
    
    @pytest.fixture
    def complete_data(self):
        """Dataset complet pour tests"""
        np.random.seed(42)
        n = 100
        
        return pd.DataFrame({
            'customer_id': [f'C{i:06d}' for i in range(n)],
            'tenure_months': np.random.randint(1, 72, n),
            'monthly_charges': np.random.uniform(20, 150, n),
            'total_charges': np.random.uniform(100, 8000, n),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
            'payment_method': np.random.choice(['Electronic check', 'Credit card'], n),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n),
            'phone_service': np.random.choice(['Yes', 'No'], n),
            'online_security': np.random.choice(['Yes', 'No', 'No internet'], n),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet'], n),
            'senior_citizen': np.random.choice([0, 1], n),
            'churned': np.random.choice([0, 1], n)
        })
    
    @patch('features.build_features.connect')
    def test_full_pipeline(self, mock_connect, complete_data):
        """Test du pipeline complet de A à Z"""
        from features.build_features import FeatureEngineer
        
        engineer = FeatureEngineer()
        
        # 1. Créer features métier
        df = engineer.create_business_features(complete_data)
        assert len(df) == len(complete_data)
        
        # 2. Encoder catégorielles
        df = engineer.encode_categorical(df)
        assert 'contract_type_encoded' in df.columns
        
        # 3. Normaliser
        df = engineer.scale_numerical(df, fit=True)
        
        # 4. Sélectionner features
        X, y, feature_names = engineer.select_final_features(df)
        
        # Vérifications finales
        assert X.shape[0] == len(complete_data)
        assert X.shape[1] > 15
        assert len(y) == len(complete_data)
        assert not X.isnull().any().any()  # Pas de NaN


if __name__ == '__main__':
    pytest.main([__file__, '-v'])