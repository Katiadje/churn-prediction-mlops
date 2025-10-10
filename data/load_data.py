"""
Script de chargement des données dans Snowflake
Génère un dataset synthétique de churn client pour la démo
"""

import os
import pandas as pd
import numpy as np
from snowflake.connector import connect
from snowflake.connector.pandas_tools import write_pandas
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class SnowflakeDataLoader:
    def __init__(self):
        """Initialise la connexion Snowflake"""
        logger.info("🔌 Connexion à Snowflake...")
        
        self.conn = connect(
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database=os.getenv('SNOWFLAKE_DATABASE'),
            schema=os.getenv('SNOWFLAKE_SCHEMA'),
            role=os.getenv('SNOWFLAKE_ROLE')
        )
        
        logger.success("✅ Connexion Snowflake établie")
        
        cursor = self.conn.cursor()
        result = cursor.execute(
            "SELECT CURRENT_USER(), CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA()"
        ).fetchone()
        logger.info(f"📊 User: {result[0]}")
        logger.info(f"📊 Warehouse: {result[1]}")
        logger.info(f"📊 Database: {result[2]}")
        logger.info(f"📊 Schema: {result[3]}")
        cursor.close()

    def create_schema(self):
        """Crée le schéma Snowflake si nécessaire"""
        logger.info("🏗️ Vérification/Création du schéma...")
        cursor = self.conn.cursor()
        
        queries = [
            f"CREATE DATABASE IF NOT EXISTS {os.getenv('SNOWFLAKE_DATABASE')}",
            f"USE DATABASE {os.getenv('SNOWFLAKE_DATABASE')}",
            f"CREATE SCHEMA IF NOT EXISTS {os.getenv('SNOWFLAKE_SCHEMA')}",
            f"USE SCHEMA {os.getenv('SNOWFLAKE_SCHEMA')}"
        ]
        
        for query in queries:
            cursor.execute(query)
            logger.info(f"✅ {query}")
        
        cursor.close()
        logger.success("✅ Schéma configuré")

    def generate_synthetic_data(self, n_samples=10000):
        """Génère des données synthétiques de churn client"""
        logger.info(f"🔧 Génération de {n_samples} clients synthétiques...")
        
        np.random.seed(42)
        
        data = {
            'customer_id': [f'C{str(i).zfill(6)}' for i in range(n_samples)],
            'tenure_months': np.random.randint(1, 72, n_samples),
            'monthly_charges': np.random.uniform(20, 150, n_samples),
            'total_charges': np.random.uniform(100, 8000, n_samples),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Credit card', 'Bank transfer'], n_samples),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
            'phone_service': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'online_security': np.random.choice(['Yes', 'No', 'No internet'], n_samples),
            'online_backup': np.random.choice(['Yes', 'No', 'No internet'], n_samples),
            'device_protection': np.random.choice(['Yes', 'No', 'No internet'], n_samples),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet'], n_samples),
            'streaming_tv': np.random.choice(['Yes', 'No', 'No internet'], n_samples),
            'streaming_movies': np.random.choice(['Yes', 'No', 'No internet'], n_samples),
            'paperless_billing': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
            'senior_citizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
            'partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5]),
            'dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        }
        
        df = pd.DataFrame(data)
        
        churn_prob = (
            (df['tenure_months'] < 12) * 0.3 +
            (df['contract_type'] == 'Month-to-month') * 0.25 +
            (df['monthly_charges'] > 100) * 0.2 +
            (df['tech_support'] == 'No') * 0.15 +
            (df['senior_citizen'] == 1) * 0.1
        )
        
        df['churned'] = (np.random.random(n_samples) < churn_prob).astype(int)
        df['created_at'] = pd.Timestamp.now()
        
        logger.success(f"✅ Données générées: {len(df)} lignes, {df['churned'].mean():.2%} churn rate")
        return df

    def load_to_snowflake(self, df, table_name='customers_raw'):
        """Charge le DataFrame dans Snowflake"""
        logger.info(f"📤 Chargement de {len(df)} lignes dans {table_name}...")
        
        cursor = self.conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        logger.info(f"🗑️ Table {table_name} supprimée si existante")
        
        success, nchunks, nrows, _ = write_pandas(
            self.conn,
            df,
            table_name.upper(),
            auto_create_table=True,
            overwrite=True
        )
        
        if success:
            logger.success(f"✅ {nrows} lignes chargées dans {table_name}")
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            logger.info(f"✅ Vérification: {count} lignes dans la table")
        else:
            logger.error(f"❌ Erreur lors du chargement")
        
        cursor.close()

    def create_sample_queries(self):
        """Crée des views utiles pour l'analyse"""
        logger.info("📊 Création des views analytiques...")
        cursor = self.conn.cursor()
        
        queries = [
            """
            CREATE OR REPLACE VIEW churn_summary AS
            SELECT 
                "contract_type" AS contract_type,
                COUNT(*) AS total_customers,
                SUM("churned") AS churned_customers,
                ROUND(AVG("churned") * 100, 2) AS churn_rate_pct,
                ROUND(AVG("monthly_charges"), 2) AS avg_monthly_charges,
                ROUND(AVG("tenure_months"), 1) AS avg_tenure_months
            FROM customers_raw
            GROUP BY "contract_type"
            ORDER BY churn_rate_pct DESC
            """,
            
            """
            CREATE OR REPLACE VIEW high_risk_customers AS
            SELECT 
                "customer_id",
                "tenure_months",
                "monthly_charges",
                "contract_type",
                "tech_support",
                "churned"
            FROM customers_raw
            WHERE 
                "tenure_months" < 12 
                AND "contract_type" = 'Month-to-month'
                AND "monthly_charges" > 70
            ORDER BY "monthly_charges" DESC
            """
        ]
        
        for i, query in enumerate(queries, 1):
            try:
                cursor.execute(query)
                logger.success(f"✅ View {i}/2 créée")
            except Exception as e:
                logger.warning(f"⚠️ Impossible de créer la vue {i}/2 : {e}")
        
        cursor.close()
        logger.success("✅ Views analytiques créées")

    def display_sample_data(self):
        """Affiche un échantillon des données"""
        logger.info("📊 Affichage des données...")
        cursor = self.conn.cursor()
        
        queries = [
            ("Total clients", 'SELECT COUNT(*) FROM customers_raw'),
            ("Clients churned", 'SELECT COUNT(*) FROM customers_raw WHERE "churned" = 1'),
            ("Churn rate", 'SELECT ROUND(AVG("churned") * 100, 2) FROM customers_raw'),
            ("Revenu moyen", 'SELECT ROUND(AVG("monthly_charges"), 2) FROM customers_raw'),
        ]
        
        print("\n" + "="*60)
        print("📊 STATISTIQUES DES DONNÉES")
        print("="*60)
        
        for label, query in queries:
            cursor.execute(query)
            value = cursor.fetchone()[0]
            print(f"{label:.<40} {value}")
        
        print("="*60)
        
        print("\n📋 APERÇU DES DONNÉES (5 premières lignes):")
        cursor.execute('SELECT * FROM customers_raw LIMIT 5')
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        df_sample = pd.DataFrame(rows, columns=columns)
        print(df_sample)
        
        cursor.close()

    def close(self):
        """Ferme la connexion"""
        if self.conn:
            self.conn.close()
            logger.info("👋 Connexion Snowflake fermée")


def main():
    try:
        loader = SnowflakeDataLoader()
        loader.create_schema()
        df = loader.generate_synthetic_data(n_samples=10000)
        loader.load_to_snowflake(df, table_name='customers_raw')
        loader.create_sample_queries()
        loader.display_sample_data()
        loader.close()
        print("\n" + "="*60)
        logger.success("🎉 CHARGEMENT DES DONNÉES TERMINÉ AVEC SUCCÈS!")
        print("="*60)
        print("\n💡 Prochaine étape:")
        print("   python data/validate_data.py\n")
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
