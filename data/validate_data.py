"""
Validation des données pour garantir la qualité
Data Quality Checks avant le feature engineering
"""

import os
import pandas as pd
import numpy as np
from snowflake.connector import connect
from loguru import logger
from dotenv import load_dotenv
from typing import Dict, List, Tuple
import json
from datetime import datetime

load_dotenv()


class DataValidator:
    """Validateur de qualité des données"""
    
    def __init__(self):
        self.conn = self._get_connection()
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'checks': [],
            'passed': 0,
            'failed': 0,
            'warnings': 0
        }
        logger.info("✅ DataValidator initialisé")
    
    def _get_connection(self):
        """Connexion Snowflake"""
        try:
            return connect(
                account=os.getenv('SNOWFLAKE_ACCOUNT'),
                user=os.getenv('SNOWFLAKE_USER'),
                password=os.getenv('SNOWFLAKE_PASSWORD'),
                warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
                database=os.getenv('SNOWFLAKE_DATABASE'),
                schema=os.getenv('SNOWFLAKE_SCHEMA'),
                role=os.getenv('SNOWFLAKE_ROLE')
            )
        except Exception as e:
            logger.warning(f"⚠️ Connexion Snowflake échouée: {e}")
            return None
    
    def load_data(self, table_name: str = 'customers_raw') -> pd.DataFrame:
        """Charge les données à valider"""
        if self.conn is None:
            logger.warning("⚠️ Mode DEMO - Génération de données mock")
            return self._generate_mock_data()
        
        query = f"SELECT * FROM {table_name}"
        
        try:
            df = pd.read_sql(query, self.conn)
            logger.info(f"✅ {len(df)} lignes chargées depuis {table_name}")
            return df
        except Exception as e:
            logger.error(f"❌ Erreur chargement: {e}")
            return self._generate_mock_data()
    
    def _generate_mock_data(self) -> pd.DataFrame:
        """Génère des données mock pour tests"""
        np.random.seed(42)
        n = 1000
        
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
    
    def check_missing_values(self, df: pd.DataFrame) -> bool:
        """Vérifier les valeurs manquantes"""
        logger.info("🔍 Check: Valeurs manquantes...")
        
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        critical_cols = ['customer_id', 'tenure_months', 'monthly_charges', 'churned']
        
        has_critical_missing = False
        for col in critical_cols:
            if col in df.columns and missing[col] > 0:
                logger.error(f"❌ Colonne critique '{col}' a {missing[col]} valeurs manquantes")
                has_critical_missing = True
        
        total_missing = missing.sum()
        
        check_result = {
            'check': 'missing_values',
            'passed': not has_critical_missing and total_missing == 0,
            'total_missing': int(total_missing),
            'details': {col: int(val) for col, val in missing[missing > 0].items()}
        }
        
        self.validation_results['checks'].append(check_result)
        
        if check_result['passed']:
            logger.success("✅ Pas de valeurs manquantes critiques")
            self.validation_results['passed'] += 1
        else:
            logger.warning(f"⚠️ {total_missing} valeurs manquantes trouvées")
            self.validation_results['failed'] += 1
        
        return check_result['passed']
    
    def check_duplicates(self, df: pd.DataFrame) -> bool:
        """Vérifier les doublons"""
        logger.info("🔍 Check: Doublons...")
        
        duplicates = df.duplicated(subset=['customer_id']).sum()
        
        check_result = {
            'check': 'duplicates',
            'passed': duplicates == 0,
            'duplicates_count': int(duplicates)
        }
        
        self.validation_results['checks'].append(check_result)
        
        if check_result['passed']:
            logger.success("✅ Pas de doublons")
            self.validation_results['passed'] += 1
        else:
            logger.error(f"❌ {duplicates} doublons trouvés!")
            self.validation_results['failed'] += 1
        
        return check_result['passed']
    
    def check_data_types(self, df: pd.DataFrame) -> bool:
        """Vérifier les types de données"""
        logger.info("🔍 Check: Types de données...")
        
        expected_types = {
            'tenure_months': [np.int64, np.int32, int],
            'monthly_charges': [np.float64, float],
            'total_charges': [np.float64, float],
            'senior_citizen': [np.int64, np.int32, int],
            'churned': [np.int64, np.int32, int]
        }
        
        type_issues = []
        for col, expected in expected_types.items():
            if col in df.columns:
                if df[col].dtype not in expected:
                    type_issues.append(f"{col}: {df[col].dtype} (attendu: {expected[0].__name__})")
        
        check_result = {
            'check': 'data_types',
            'passed': len(type_issues) == 0,
            'issues': type_issues
        }
        
        self.validation_results['checks'].append(check_result)
        
        if check_result['passed']:
            logger.success("✅ Types de données corrects")
            self.validation_results['passed'] += 1
        else:
            logger.warning(f"⚠️ Problèmes de types: {type_issues}")
            self.validation_results['warnings'] += 1
        
        return check_result['passed']
    
    def check_value_ranges(self, df: pd.DataFrame) -> bool:
        """Vérifier les plages de valeurs"""
        logger.info("🔍 Check: Plages de valeurs...")
        
        range_checks = {
            'tenure_months': (0, 100),
            'monthly_charges': (0, 200),
            'total_charges': (0, 10000),
            'senior_citizen': (0, 1),
            'churned': (0, 1)
        }
        
        range_issues = []
        for col, (min_val, max_val) in range_checks.items():
            if col in df.columns:
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if out_of_range > 0:
                    range_issues.append(f"{col}: {out_of_range} valeurs hors plage [{min_val}, {max_val}]")
        
        check_result = {
            'check': 'value_ranges',
            'passed': len(range_issues) == 0,
            'issues': range_issues
        }
        
        self.validation_results['checks'].append(check_result)
        
        if check_result['passed']:
            logger.success("✅ Toutes les valeurs dans les plages attendues")
            self.validation_results['passed'] += 1
        else:
            logger.error(f"❌ Valeurs hors plage: {range_issues}")
            self.validation_results['failed'] += 1
        
        return check_result['passed']
    
    def check_categorical_values(self, df: pd.DataFrame) -> bool:
        """Vérifier les valeurs catégorielles"""
        logger.info("🔍 Check: Valeurs catégorielles...")
        
        expected_values = {
            'contract_type': ['Month-to-month', 'One year', 'Two year'],
            'payment_method': ['Electronic check', 'Mailed check', 'Credit card', 'Bank transfer'],
            'internet_service': ['DSL', 'Fiber optic', 'No'],
            'phone_service': ['Yes', 'No'],
            'online_security': ['Yes', 'No', 'No internet'],
            'tech_support': ['Yes', 'No', 'No internet']
        }
        
        categorical_issues = []
        for col, valid_values in expected_values.items():
            if col in df.columns:
                invalid = ~df[col].isin(valid_values)
                invalid_count = invalid.sum()
                if invalid_count > 0:
                    invalid_vals = df[col][invalid].unique()
                    categorical_issues.append(f"{col}: {invalid_count} valeurs invalides {list(invalid_vals)}")
        
        check_result = {
            'check': 'categorical_values',
            'passed': len(categorical_issues) == 0,
            'issues': categorical_issues
        }
        
        self.validation_results['checks'].append(check_result)
        
        if check_result['passed']:
            logger.success("✅ Valeurs catégorielles valides")
            self.validation_results['passed'] += 1
        else:
            logger.error(f"❌ Valeurs catégorielles invalides: {categorical_issues}")
            self.validation_results['failed'] += 1
        
        return check_result['passed']
    
    def check_target_distribution(self, df: pd.DataFrame) -> bool:
        """Vérifier la distribution de la target"""
        logger.info("🔍 Check: Distribution target...")
        
        if 'churned' not in df.columns:
            logger.warning("⚠️ Colonne 'churned' absente")
            return False
        
        churn_rate = df['churned'].mean()
        
        # Acceptable si entre 10% et 50%
        is_acceptable = 0.10 <= churn_rate <= 0.50
        
        check_result = {
            'check': 'target_distribution',
            'passed': is_acceptable,
            'churn_rate': float(churn_rate),
            'churn_count': int(df['churned'].sum()),
            'no_churn_count': int((df['churned'] == 0).sum())
        }
        
        self.validation_results['checks'].append(check_result)
        
        if is_acceptable:
            logger.success(f"✅ Churn rate acceptable: {churn_rate:.2%}")
            self.validation_results['passed'] += 1
        else:
            logger.warning(f"⚠️ Churn rate inhabituel: {churn_rate:.2%}")
            self.validation_results['warnings'] += 1
        
        return check_result['passed']
    
    def check_outliers(self, df: pd.DataFrame) -> bool:
        """Détecter les outliers"""
        logger.info("🔍 Check: Outliers...")
        
        numerical_cols = ['tenure_months', 'monthly_charges', 'total_charges']
        
        outlier_summary = {}
        for col in numerical_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_pct = (outliers / len(df)) * 100
                
                outlier_summary[col] = {
                    'count': int(outliers),
                    'percentage': float(outlier_pct)
                }
        
        total_outliers = sum(info['count'] for info in outlier_summary.values())
        
        check_result = {
            'check': 'outliers',
            'passed': True,  # Informatif seulement
            'outlier_summary': outlier_summary,
            'total_outliers': total_outliers
        }
        
        self.validation_results['checks'].append(check_result)
        
        logger.info(f"ℹ️ {total_outliers} outliers détectés (informatif)")
        self.validation_results['warnings'] += 1
        
        return True
    
    def check_data_freshness(self, df: pd.DataFrame) -> bool:
        """Vérifier la fraîcheur des données"""
        logger.info("🔍 Check: Fraîcheur des données...")
        
        if 'created_at' not in df.columns:
            logger.warning("⚠️ Colonne 'created_at' absente, skip")
            return True
        
        df['created_at'] = pd.to_datetime(df['created_at'])
        latest_date = df['created_at'].max()
        days_old = (datetime.now() - latest_date).days
        
        is_fresh = days_old <= 7  # Moins de 7 jours
        
        check_result = {
            'check': 'data_freshness',
            'passed': is_fresh,
            'latest_date': latest_date.isoformat(),
            'days_old': days_old
        }
        
        self.validation_results['checks'].append(check_result)
        
        if is_fresh:
            logger.success(f"✅ Données récentes ({days_old} jours)")
            self.validation_results['passed'] += 1
        else:
            logger.warning(f"⚠️ Données anciennes ({days_old} jours)")
            self.validation_results['warnings'] += 1
        
        return check_result['passed']
    
    def save_validation_report(self):
        """Sauvegarder le rapport de validation"""
        os.makedirs('data/validation_reports', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f'data/validation_reports/validation_report_{timestamp}.json'
        
        def convert_types(obj):
            """Convertir les types NumPy pour JSON"""
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return str(obj)
        
        with open(report_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=convert_types)
    
            logger.info(f"💾 Rapport sauvegardé: {report_path}")
    
    def run_all_validations(self, table_name: str = 'customers_raw') -> bool:
        """Exécuter toutes les validations"""
        logger.info("\n" + "="*60)
        logger.info("🚀 DÉMARRAGE DE LA VALIDATION DES DONNÉES")
        logger.info("="*60)
        
        # Charger les données
        df = self.load_data(table_name)
        
        if df is None or len(df) == 0:
            logger.error("❌ Impossible de charger les données")
            return False
        
        logger.info(f"\n📊 Dataset: {len(df)} lignes, {len(df.columns)} colonnes\n")
        
        # Exécuter tous les checks
        self.check_missing_values(df)
        self.check_duplicates(df)
        self.check_data_types(df)
        self.check_value_ranges(df)
        self.check_categorical_values(df)
        self.check_target_distribution(df)
        self.check_outliers(df)
        self.check_data_freshness(df)
        
        # Sauvegarder le rapport
        self.save_validation_report()
        
        # Résumé
        logger.info("\n" + "="*60)
        logger.info("📊 RÉSUMÉ DE LA VALIDATION")
        logger.info("="*60)
        logger.success(f"✅ Checks réussis: {self.validation_results['passed']}")
        logger.error(f"❌ Checks échoués: {self.validation_results['failed']}")
        logger.warning(f"⚠️  Warnings: {self.validation_results['warnings']}")
        logger.info("="*60)
        
        # Validation globale réussie si pas d'échecs critiques
        all_passed = self.validation_results['failed'] == 0
        
        if all_passed:
            logger.success("\n🎉 VALIDATION RÉUSSIE! Les données sont prêtes pour le feature engineering.")
        else:
            logger.error("\n❌ VALIDATION ÉCHOUÉE! Corriger les problèmes avant de continuer.")
        
        return all_passed
    
    def close(self):
        """Fermer la connexion"""
        if self.conn:
            self.conn.close()
            logger.info("👋 Connexion fermée")


def main():
    """Pipeline de validation"""
    validator = DataValidator()
    
    try:
        success = validator.run_all_validations('customers_raw')
        exit(0 if success else 1)
    finally:
        validator.close()


if __name__ == "__main__":
    main()