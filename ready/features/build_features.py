"""
Feature Engineering Pipeline
Transforme les données brutes en features ML-ready
"""

import os
import pandas as pd
import numpy as np
from snowflake.connector import connect
from sklearn.preprocessing import LabelEncoder, StandardScaler
from loguru import logger
from dotenv import load_dotenv
import json

load_dotenv()


class FeatureEngineer:
    def __init__(self):
        self.conn = connect(
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database=os.getenv('SNOWFLAKE_DATABASE'),
            schema=os.getenv('SNOWFLAKE_SCHEMA'),
            role=os.getenv('SNOWFLAKE_ROLE')
        )
        self.label_encoders = {}
        self.scaler = StandardScaler()
        logger.info("✅ FeatureEngineer initialisé")

    def load_raw_data(self):
        """Charge les données brutes depuis Snowflake"""
        query = "SELECT * FROM customers_raw"
        
        logger.info("📥 Chargement des données brutes...")
        df = pd.read_sql(query, self.conn)
        logger.info(f"✅ {len(df)} lignes chargées")
        
        return df

    def create_business_features(self, df):
        """Crée des features métier à forte valeur ajoutée"""
        logger.info("🔧 Création des features métier...")
        
        df = df.copy()
        
        # 1. Features financières
        df['avg_monthly_spend'] = df['total_charges'] / (df['tenure_months'] + 1)
        df['charges_to_tenure_ratio'] = df['monthly_charges'] / (df['tenure_months'] + 1)
        df['high_spender'] = (df['monthly_charges'] > df['monthly_charges'].quantile(0.75)).astype(int)
        
        # 2. Features de fidélité
        df['is_new_customer'] = (df['tenure_months'] <= 6).astype(int)
        df['is_loyal_customer'] = (df['tenure_months'] >= 36).astype(int)
        df['tenure_group'] = pd.cut(df['tenure_months'], 
                                     bins=[0, 12, 24, 48, 100], 
                                     labels=['0-1yr', '1-2yr', '2-4yr', '4yr+'])
        
        # 3. Features de services
        service_cols = ['online_security', 'online_backup', 'device_protection', 
                       'tech_support', 'streaming_tv', 'streaming_movies']
        
        for col in service_cols:
            df[f'{col}_yes'] = (df[col] == 'Yes').astype(int)
        
        df['total_services'] = sum(df[f'{col}_yes'] for col in service_cols)
        df['has_premium_services'] = (df['total_services'] >= 4).astype(int)
        
        # 4. Features de risque
        df['high_risk_contract'] = (df['contract_type'] == 'Month-to-month').astype(int)
        df['risky_payment'] = (df['payment_method'] == 'Electronic check').astype(int)
        df['no_protection'] = ((df['online_security'] == 'No') & 
                               (df['tech_support'] == 'No')).astype(int)
        
        # 5. Features d'engagement
        df['has_phone_internet'] = ((df['phone_service'] == 'Yes') & 
                                    (df['internet_service'] != 'No')).astype(int)
        df['paperless_engaged'] = (df['paperless_billing'] == 'Yes').astype(int)
        
        # 6. Score de risque composite (weighted)
        df['risk_score'] = (
            df['is_new_customer'] * 0.3 +
            df['high_risk_contract'] * 0.25 +
            df['risky_payment'] * 0.15 +
            df['no_protection'] * 0.15 +
            (df['total_services'] == 0) * 0.15
        )
        
        logger.info(f"✅ {df.shape[1] - df.shape[1]} nouvelles features créées")
        
        return df

    def encode_categorical(self, df):
        """Encode les variables catégorielles"""
        logger.info("🔧 Encodage des variables catégorielles...")
        
        df = df.copy()
        
        categorical_cols = [
            'contract_type', 'payment_method', 'internet_service',
            'phone_service', 'multiple_lines', 'online_security',
            'online_backup', 'device_protection', 'tech_support',
            'streaming_tv', 'streaming_movies', 'paperless_billing',
            'partner', 'dependents', 'tenure_group'
        ]
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        logger.info(f"✅ {len(categorical_cols)} colonnes encodées")
        
        return df

    def scale_numerical(self, df, fit=True):
        """Normalise les features numériques"""
        logger.info("🔧 Normalisation des features numériques...")
        
        df = df.copy()
        
        numerical_cols = [
            'tenure_months', 'monthly_charges', 'total_charges',
            'avg_monthly_spend', 'charges_to_tenure_ratio',
            'total_services', 'risk_score'
        ]
        
        cols_to_scale = [col for col in numerical_cols if col in df.columns]
        
        if fit:
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        else:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        logger.info(f"✅ {len(cols_to_scale)} colonnes normalisées")
        
        return df

    def select_final_features(self, df):
        """Sélectionne les features finales pour le modèle"""
        
        feature_cols = [
            # Numériques normalisées
            'tenure_months', 'monthly_charges', 'total_charges',
            'avg_monthly_spend', 'charges_to_tenure_ratio',
            'total_services', 'risk_score',
            
            # Binaires
            'senior_citizen', 'is_new_customer', 'is_loyal_customer',
            'high_spender', 'has_premium_services', 'high_risk_contract',
            'risky_payment', 'no_protection', 'has_phone_internet',
            'paperless_engaged',
            
            # Services
            'online_security_yes', 'online_backup_yes', 'device_protection_yes',
            'tech_support_yes', 'streaming_tv_yes', 'streaming_movies_yes',
            
            # Catégorielles encodées
            'contract_type_encoded', 'payment_method_encoded',
            'internet_service_encoded', 'phone_service_encoded'
        ]
        
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features]
        y = df['churned'] if 'churned' in df.columns else None
        
        logger.info(f"✅ {len(available_features)} features sélectionnées")
        
        return X, y, available_features

    def save_to_snowflake(self, df, table_name='customer_features'):
        """Sauvegarde les features dans Snowflake"""
        from snowflake.connector.pandas_tools import write_pandas
        
        logger.info(f"💾 Sauvegarde dans {table_name}...")
        
        cursor = self.conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        success, _, nrows, _ = write_pandas(
            self.conn,
            df,
            table_name.upper(),
            auto_create_table=True,
            overwrite=True
        )
        
        if success:
            logger.info(f"✅ {nrows} lignes sauvegardées dans {table_name}")
        
        cursor.close()

    def save_artifacts(self):
        """Sauvegarde les encoders et scalers"""
        import pickle
        
        os.makedirs('artifacts', exist_ok=True)
        
        with open('artifacts/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        with open('artifacts/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info("✅ Artifacts sauvegardés dans ./artifacts/")

    def close(self):
        self.conn.close()


def main():
    """Pipeline principal de feature engineering"""
    try:
        engineer = FeatureEngineer()
        
        # 1. Charger données brutes
        df = engineer.load_raw_data()
        
        # 2. Créer features métier
        df = engineer.create_business_features(df)
        
        # 3. Encoder catégorielles
        df = engineer.encode_categorical(df)
        
        # 4. Normaliser numériques
        df = engineer.scale_numerical(df, fit=True)
        
        # 5. Sélectionner features finales
        X, y, feature_names = engineer.select_final_features(df)
        
        # 6. Sauvegarder dans Snowflake
        features_df = pd.concat([
            df[['customer_id']], 
            X, 
            df[['churned']]
        ], axis=1)
        
        engineer.save_to_snowflake(features_df, 'customer_features')
        
        # 7. Sauvegarder les artifacts
        engineer.save_artifacts()
        
        # Stats finales
        print("\n📊 Feature Engineering Summary:")
        print(f"✅ Total features: {X.shape[1]}")
        print(f"✅ Total samples: {X.shape[0]}")
        print(f"✅ Churn rate: {y.mean():.2%}")
        print(f"\n📝 Features: {', '.join(feature_names[:10])}...")
        
        engineer.close()
        logger.success("🎉 Feature engineering terminé!")
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        raise


if __name__ == "__main__":
    main()