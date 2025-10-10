"""
Script d'entraînement du modèle de prédiction de churn
Avec MLflow tracking et model registry
"""

import os
import pandas as pd
import numpy as np
from snowflake.connector import connect
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report, 
                            confusion_matrix)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
from loguru import logger
from dotenv import load_dotenv
import json
from datetime import datetime


from pathlib import Path
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)


class ChurnModelTrainer:
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

        
        # MLflow setup
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME', 'churn-prediction'))
        
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
        logger.info("✅ ChurnModelTrainer initialisé")

    def load_features(self):
        """Charge les features depuis Snowflake"""
        query = "SELECT * FROM customer_features"
        
        logger.info("📥 Chargement des features...")
        df = pd.read_sql(query, self.conn)
        
        # Séparer X et y
        X = df.drop(['customer_id', 'churned'], axis=1)
        y = df['churned']
        
        logger.info(f"✅ Features chargées: {X.shape}")
        
        return X, y

    def split_data(self, X, y):
        """Split train/test avec stratification"""
        test_size = float(os.getenv('TRAIN_TEST_SPLIT', 0.2))
        random_state = int(os.getenv('RANDOM_STATE', 42))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )
        
        logger.info(f"✅ Split: Train={len(X_train)}, Test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test

    def get_model_configs(self):
        """Retourne les configurations des modèles à tester"""
        return {
            'RandomForest': {
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5
                }
            },
            'XGBoost': {
                'model': XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss'
                ),
                'params': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1
                }
            },
            'LightGBM': {
                'model': LGBMClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    num_leaves=31,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                ),
                'params': {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.1
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                ),
                'params': {
                    'n_estimators': 150,
                    'max_depth': 5,
                    'learning_rate': 0.1
                }
            }
        }

    def evaluate_model(self, model, X_test, y_test):
        """Évalue un modèle et retourne les métriques"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics, y_pred

    def train_and_log_model(self, name, config, X_train, X_test, y_train, y_test):
        """Entraîne un modèle et log dans MLflow"""
        logger.info(f"\n🚀 Entraînement: {name}")
        
        with mlflow.start_run(run_name=name):
            # 1. Log des paramètres
            mlflow.log_params(config['params'])
            mlflow.log_param('model_type', name)
            mlflow.log_param('train_size', len(X_train))
            mlflow.log_param('test_size', len(X_test))
            
            # 2. Entraînement
            model = config['model']
            model.fit(X_train, y_train)
            
            # 3. Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            mlflow.log_metric('cv_f1_mean', cv_scores.mean())
            mlflow.log_metric('cv_f1_std', cv_scores.std())
            
            # 4. Évaluation test set
            metrics, y_pred = self.evaluate_model(model, X_test, y_test)
            
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # 5. Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Log top 10 features
                for idx, row in feature_importance.head(10).iterrows():
                    mlflow.log_metric(f"importance_{row['feature']}", row['importance'])
            
            # 6. Log du modèle
            mlflow.sklearn.log_model(model, "model")
            
            # 7. Log artifacts
            conf_matrix = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            with open('classification_report.txt', 'w') as f:
                f.write(report)
            mlflow.log_artifact('classification_report.txt')
            os.remove('classification_report.txt')
            
            # Afficher résultats
            logger.info(f"✅ {name} - F1: {metrics['f1_score']:.4f}, AUC: {metrics['roc_auc']:.4f}")
            
            # Sauvegarder si meilleur
            if metrics['f1_score'] > self.best_score:
                self.best_score = metrics['f1_score']
                self.best_model = model
                self.best_model_name = name
                logger.info(f"🏆 Nouveau meilleur modèle: {name}")
            
            return model, metrics

    def register_best_model(self):
        """Enregistre le meilleur modèle dans MLflow Registry"""
        if self.best_model is None:
            logger.warning("❌ Aucun modèle à enregistrer")
            return
        
        model_name = os.getenv('MODEL_NAME', 'churn_predictor')
        
        # Créer une nouvelle run pour le meilleur modèle
        with mlflow.start_run(run_name=f"PRODUCTION_{self.best_model_name}"):
            mlflow.sklearn.log_model(
                self.best_model,
                "model",
                registered_model_name=model_name
            )
            
            mlflow.log_param('production_model', self.best_model_name)
            mlflow.log_metric('best_f1_score', self.best_score)
            
            logger.success(f"✅ Modèle enregistré: {model_name}")

    def save_model_metadata(self):
        """Sauvegarde les métadonnées du modèle"""
        metadata = {
            'model_name': self.best_model_name,
            'best_score': self.best_score,
            'trained_at': datetime.now().isoformat(),
            'features': list(self.X_train.columns),
            'metrics': {
                'f1_score': self.best_score,
                'accuracy': self.best_accuracy
            }
        }
        
        os.makedirs('artifacts', exist_ok=True)
        with open('artifacts/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("✅ Métadonnées sauvegardées")

    def close(self):
        self.conn.close()


def main():
    """Pipeline principal d'entraînement"""
    try:
        trainer = ChurnModelTrainer()
        
        # 1. Charger les features
        X, y = trainer.load_features()
        
        # 2. Split train/test
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        trainer.X_train = X_train  # Pour metadata
        
        # 3. Obtenir les configs des modèles
        model_configs = trainer.get_model_configs()
        
        # 4. Entraîner tous les modèles
        results = {}
        logger.info("\n" + "="*50)
        logger.info("🏁 Début de l'entraînement de tous les modèles")
        logger.info("="*50)
        
        for name, config in model_configs.items():
            model, metrics = trainer.train_and_log_model(
                name, config, X_train, X_test, y_train, y_test
            )
            results[name] = metrics
            trainer.models[name] = model
        
        # 5. Afficher le comparatif
        print("\n" + "="*70)
        print("📊 COMPARAISON DES MODÈLES")
        print("="*70)
        print(f"{'Modèle':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC':<12}")
        print("-"*70)
        
        for name, metrics in results.items():
            print(f"{name:<20} "
                  f"{metrics['accuracy']:<12.4f} "
                  f"{metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} "
                  f"{metrics['f1_score']:<12.4f} "
                  f"{metrics['roc_auc']:<12.4f}")
        
        print("="*70)
        print(f"\n🏆 MEILLEUR MODÈLE: {trainer.best_model_name} (F1={trainer.best_score:.4f})")
        print("="*70)
        
        # 6. Enregistrer le meilleur modèle
        trainer.best_accuracy = results[trainer.best_model_name]['accuracy']
        trainer.register_best_model()
        trainer.save_model_metadata()
        
        trainer.close()
        logger.success("\n🎉 Entraînement terminé avec succès!")
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        raise


if __name__ == "__main__":
    main()