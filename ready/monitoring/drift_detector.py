"""
Drift Detection pour le monitoring MLOps
Détecte les changements dans les données et les prédictions
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from loguru import logger
import os
from pathlib import Path
import json
from snowflake.connector import connect
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()


class DriftDetector:
    """Détecteur de drift pour le monitoring ML"""
    
    def __init__(self):
        """Initialisation"""
        self.conn = connect(
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database=os.getenv('SNOWFLAKE_DATABASE'),
            schema=os.getenv('SNOWFLAKE_SCHEMA'),
            role=os.getenv('SNOWFLAKE_ROLE')
        )
        
        self.drift_threshold = 0.1  # Seuil Kolmogorov-Smirnov
        self.report_dir = Path('monitoring/reports')
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ DriftDetector initialisé")
    
    def load_reference_data(self):
        """Charge les données de référence (training)"""
        query = "SELECT * FROM customer_features LIMIT 5000"
        df = pd.read_sql(query, self.conn)
        logger.info(f"📊 Données de référence: {len(df)} lignes")
        return df
    
    def load_production_data(self, days=7):
        """
        Charge les données de production récentes
        En réel, tu les récupérerais depuis une table de prédictions
        """
        # Simulation : on prend les mêmes données avec un peu de bruit
        query = "SELECT * FROM customer_features LIMIT 2000"
        df = pd.read_sql(query, self.conn)
        
        # Ajouter du bruit pour simuler le drift
        np.random.seed(42)
        df['monthly_charges'] = df['monthly_charges'] * np.random.uniform(1.0, 1.2, len(df))
        df['tenure_months'] = df['tenure_months'] * np.random.uniform(0.8, 1.0, len(df))
        
        logger.info(f"📊 Données production: {len(df)} lignes")
        return df
    
    def kolmogorov_smirnov_test(self, ref_data, prod_data, feature):
        """
        Test Kolmogorov-Smirnov pour détecter le drift
        Retourne (statistic, p_value)
        """
        try:
            ref_values = ref_data[feature].dropna()
            prod_values = prod_data[feature].dropna()
            
            statistic, p_value = stats.ks_2samp(ref_values, prod_values)
            
            return statistic, p_value
        except Exception as e:
            logger.warning(f"⚠️ Erreur KS test pour {feature}: {e}")
            return None, None
    
    def detect_feature_drift(self, ref_data, prod_data):
        """Détecte le drift pour chaque feature numérique"""
        
        numerical_features = ref_data.select_dtypes(include=[np.number]).columns
        numerical_features = [col for col in numerical_features if col not in ['customer_id', 'churned']]
        
        drift_results = []
        
        logger.info(f"🔍 Analyse de {len(numerical_features)} features...")
        
        for feature in numerical_features:
            statistic, p_value = self.kolmogorov_smirnov_test(ref_data, prod_data, feature)
            
            if statistic is None:
                continue
            
            # Drift détecté si statistic > threshold OU p_value < 0.05
            drift_detected = statistic > self.drift_threshold or p_value < 0.05
            
            result = {
                'feature': feature,
                'ks_statistic': float(statistic),
                'p_value': float(p_value),
                'drift_detected': drift_detected,
                'ref_mean': float(ref_data[feature].mean()),
                'prod_mean': float(prod_data[feature].mean()),
                'change_pct': float(((prod_data[feature].mean() - ref_data[feature].mean()) / 
                                   ref_data[feature].mean() * 100))
            }
            
            drift_results.append(result)
            
            if drift_detected:
                logger.warning(f"⚠️ DRIFT détecté: {feature} "
                             f"(KS={statistic:.4f}, p={p_value:.4f}, "
                             f"change={result['change_pct']:.1f}%)")
            else:
                logger.info(f"✅ {feature}: OK")
        
        return pd.DataFrame(drift_results)
    
    def detect_prediction_drift(self, ref_data, prod_data):
        """Détecte le drift dans les prédictions (target)"""
        
        if 'churned' not in ref_data.columns or 'churned' not in prod_data.columns:
            logger.warning("⚠️ Colonne 'churned' non trouvée")
            return None
        
        ref_churn_rate = float(ref_data['churned'].mean())
        prod_churn_rate = float(prod_data['churned'].mean())
        
        change = abs(prod_churn_rate - ref_churn_rate)
        change_pct = (prod_churn_rate - ref_churn_rate) / ref_churn_rate * 100
        
        drift_detected = bool(change > 0.05)  # Si diff > 5%
        
        result = {
            'ref_churn_rate': float(ref_churn_rate),
            'prod_churn_rate': float(prod_churn_rate),
            'absolute_change': float(change),
            'change_pct': float(change_pct),
            'drift_detected': bool(drift_detected)
        }
        
        if drift_detected:
            logger.warning(f"⚠️ PREDICTION DRIFT: "
                         f"Churn rate: {ref_churn_rate:.1%} → {prod_churn_rate:.1%} "
                         f"({change_pct:+.1f}%)")
        else:
            logger.info(f"✅ Prediction drift: OK")
        
        return result
    
    def generate_drift_report(self, drift_df, prediction_drift):
        """Génère un rapport détaillé"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.report_dir / f'drift_report_{timestamp}.json'
        
        # Calculer statistiques globales
        total_features = len(drift_df)
        drifted_features = int(drift_df['drift_detected'].sum())
        drift_percentage = (drifted_features / total_features * 100) if total_features > 0 else 0
        
        # Top 5 features avec le plus de drift - convertir en types Python natifs
        top_drifted = []
        for _, row in drift_df.nlargest(5, 'ks_statistic').iterrows():
            top_drifted.append({
                'feature': str(row['feature']),
                'ks_statistic': float(row['ks_statistic']),
                'change_pct': float(row['change_pct'])
            })
        
        # Convertir le DataFrame en records avec types Python natifs
        all_features = []
        for _, row in drift_df.iterrows():
            all_features.append({
                'feature': str(row['feature']),
                'ks_statistic': float(row['ks_statistic']),
                'p_value': float(row['p_value']),
                'drift_detected': bool(row['drift_detected']),
                'ref_mean': float(row['ref_mean']),
                'prod_mean': float(row['prod_mean']),
                'change_pct': float(row['change_pct'])
            })
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_features_analyzed': int(total_features),
                'features_with_drift': int(drifted_features),
                'drift_percentage': float(drift_percentage),
                'overall_drift_status': 'ALERT' if drift_percentage > 20 else 'WARNING' if drift_percentage > 10 else 'OK'
            },
            'prediction_drift': prediction_drift,
            'top_drifted_features': top_drifted,
            'all_features': all_features
        }
        
        # Sauvegarder
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.success(f"📄 Rapport sauvegardé: {report_path}")
        
        return report
    
    def visualize_drift(self, ref_data, prod_data, drift_df):
        """Crée des visualisations du drift"""
        
        # Top 5 features avec drift
        top_drifted = drift_df.nlargest(5, 'ks_statistic')
        
        if len(top_drifted) == 0:
            logger.info("✅ Aucun drift significatif à visualiser")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(top_drifted.iterrows()):
            if idx >= 5:
                break
            
            feature = row['feature']
            ax = axes[idx]
            
            # Distributions
            ax.hist(ref_data[feature].dropna(), bins=30, alpha=0.5, 
                   label='Reference', color='blue', density=True)
            ax.hist(prod_data[feature].dropna(), bins=30, alpha=0.5, 
                   label='Production', color='red', density=True)
            
            ax.set_title(f"{feature}\nKS={row['ks_statistic']:.3f}, "
                        f"Change={row['change_pct']:.1f}%",
                        fontsize=10)
            ax.legend()
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
        
        # Summary dans le dernier subplot
        ax = axes[5]
        ax.axis('off')
        
        summary_text = f"""
        DRIFT DETECTION SUMMARY
        
        Total features: {len(drift_df)}
        Drifted features: {drift_df['drift_detected'].sum()}
        Drift rate: {drift_df['drift_detected'].mean()*100:.1f}%
        
        Status: {'⚠️ ALERT' if drift_df['drift_detected'].mean() > 0.2 else '✅ OK'}
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=12, 
               verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        
        # Sauvegarder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_path = self.report_dir / f'drift_visualization_{timestamp}.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.success(f"📊 Visualisation sauvegardée: {viz_path}")
    
    def run_drift_detection(self):
        """Lance la détection complète de drift"""
        
        logger.info("\n" + "="*60)
        logger.info("🔍 DÉTECTION DE DRIFT - DÉMARRAGE")
        logger.info("="*60)
        
        # 1. Charger les données
        ref_data = self.load_reference_data()
        prod_data = self.load_production_data()
        
        # 2. Détecter drift sur les features
        drift_df = self.detect_feature_drift(ref_data, prod_data)
        
        # 3. Détecter drift sur les prédictions
        prediction_drift = self.detect_prediction_drift(ref_data, prod_data)
        
        # 4. Générer le rapport
        report = self.generate_drift_report(drift_df, prediction_drift)
        
        # 5. Visualiser
        self.visualize_drift(ref_data, prod_data, drift_df)
        
        # 6. Afficher le résumé
        self.print_summary(report)
        
        logger.info("="*60)
        logger.success("✅ DÉTECTION DE DRIFT TERMINÉE")
        logger.info("="*60)
        
        return report
    
    def print_summary(self, report):
        """Affiche un résumé du rapport"""
        
        summary = report['summary']
        
        print("\n" + "="*60)
        print("📊 RÉSUMÉ DU DRIFT DETECTION")
        print("="*60)
        print(f"Features analysées:      {summary['total_features_analyzed']}")
        print(f"Features avec drift:     {summary['features_with_drift']}")
        print(f"Taux de drift:           {summary['drift_percentage']:.1f}%")
        print(f"Status:                  {summary['overall_drift_status']}")
        print("="*60)
        
        if report['prediction_drift']:
            pred = report['prediction_drift']
            print(f"\n📈 Prediction Drift:")
            print(f"Churn rate (ref):        {pred['ref_churn_rate']:.1%}")
            print(f"Churn rate (prod):       {pred['prod_churn_rate']:.1%}")
            print(f"Changement:              {pred['change_pct']:+.1f}%")
            print(f"Drift détecté:           {'⚠️ OUI' if pred['drift_detected'] else '✅ NON'}")
        
        print("\n🔝 Top 5 features avec le plus de drift:")
        for feat in report['top_drifted_features']:
            print(f"  • {feat['feature']:<25} "
                  f"KS={feat['ks_statistic']:.3f}  "
                  f"Change={feat['change_pct']:+.1f}%")
        
        print("="*60 + "\n")
    
    def close(self):
        """Ferme la connexion"""
        self.conn.close()
        logger.info("👋 Connexion fermée")


def main():
    """Point d'entrée principal"""
    try:
        detector = DriftDetector()
        report = detector.run_drift_detection()
        detector.close()
        
        print("\n💡 Prochaines étapes:")
        print("  1. Consulter le rapport: monitoring/reports/")
        print("  2. Si drift élevé (>20%) → Réentraîner le modèle")
        print("  3. Automatiser ce script (daily cron)")
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        raise


if __name__ == "__main__":
    main()