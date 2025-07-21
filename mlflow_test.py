"""
MLflow Testing Script - No GPU Required
=====================================

This script demonstrates MLflow capabilities using scikit-learn for testing purposes.
It includes:
- Dataset logging
- Parameter logging
- Metric logging
- Model logging
- Artifact logging
- System metrics logging

Run this to test MLflow functionality without needing GPU resources.
"""

import mlflow
import mlflow.sklearn
import mlflow.data
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import time

# Import your config manager if available
try:
    from config_manager import get_config_manager
    config_available = True
except ImportError:
    config_available = False
    print("Config manager not available, using default MLflow settings")

# Import utils functions
try:
    from utils import log_configurations_to_mlflow, setup_run_name
    utils_available = True
except ImportError:
    utils_available = False
    print("Utils not available, using inline functions")


class MLflowTester:
    """
    Test MLflow capabilities with scikit-learn models
    """
    
    def __init__(self):
        self.experiment_name = "mlflow-test-experiment"
        self.run_name = f"test-run-{int(time.time())}"
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        # Set tracking URI if configured
        if config_available:
            try:
                config_manager = get_config_manager()
                mlflow_config = config_manager.get_section("mlflow")
                mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "http://localhost:5000"))
                print(f"--- ✅ MLflow tracking URI set to: {mlflow_config.get('tracking_uri')} ---")
            except Exception as e:
                print(f"--- ⚠️ Using default MLflow settings: {e} ---")
                mlflow.set_tracking_uri("http://localhost:5000")
        else:
            mlflow.set_tracking_uri("http://localhost:5000")
            
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                print(f"--- ✅ Created new experiment: {self.experiment_name} (ID: {experiment_id}) ---")
            else:
                experiment_id = experiment.experiment_id
                print(f"--- ✅ Using existing experiment: {self.experiment_name} (ID: {experiment_id}) ---")
            
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            print(f"--- ⚠️ Experiment setup issue: {e} ---")
            
    def create_sample_datasets(self):
        """Create sample datasets for testing"""
        print("--- 📊 Creating sample datasets ---")
        
        # Create synthetic classification dataset
        X_synthetic, y_synthetic = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=42
        )
        
        # Create DataFrame for easier handling
        feature_names = [f"feature_{i}" for i in range(X_synthetic.shape[1])]
        synthetic_df = pd.DataFrame(X_synthetic, columns=feature_names)
        synthetic_df['target'] = y_synthetic
        
        # Load iris dataset
        iris = load_iris()
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        iris_df['target'] = iris.target
        
        return synthetic_df, iris_df
        
    def log_datasets_to_mlflow(self, synthetic_df, iris_df):
        """Log datasets to MLflow"""
        print("--- 📝 Logging datasets to MLflow ---")
        
        try:
            # Log synthetic dataset
            synthetic_dataset = mlflow.data.from_pandas(
                synthetic_df, 
                source="synthetic_classification_dataset",
                name="synthetic_data"
            )
            mlflow.log_input(synthetic_dataset, context="training")
            
            # Log iris dataset
            iris_dataset = mlflow.data.from_pandas(
                iris_df,
                source="sklearn_iris_dataset", 
                name="iris_data"
            )
            mlflow.log_input(iris_dataset, context="testing")
            
            print("--- ✅ Datasets logged successfully ---")
            
        except Exception as e:
            print(f"--- ⚠️ Dataset logging failed: {e} ---")
            
    def log_parameters(self, model_params):
        """Log parameters to MLflow"""
        print("--- ⚙️ Logging parameters ---")
        
        try:
            # Log model parameters
            for key, value in model_params.items():
                mlflow.log_param(key, value)
                
            # Log additional metadata
            mlflow.log_param("script_name", "mlflow_test.py")
            mlflow.log_param("test_timestamp", datetime.now().isoformat())
            mlflow.log_param("python_version", "3.12")
            mlflow.log_param("purpose", "mlflow_capability_testing")
            
            print("--- ✅ Parameters logged successfully ---")
            
        except Exception as e:
            print(f"--- ⚠️ Parameter logging failed: {e} ---")
            
    def train_and_log_model(self, X_train, X_test, y_train, y_test, model_type="random_forest"):
        """Train model and log to MLflow"""
        print(f"--- 🏋️ Training {model_type} model ---")
        
        try:
            # Choose model based on type
            if model_type == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                model_params = {
                    "model_type": "RandomForestClassifier",
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42
                }
            else:
                model = LogisticRegression(
                    random_state=42,
                    max_iter=1000
                )
                model_params = {
                    "model_type": "LogisticRegression", 
                    "random_state": 42,
                    "max_iter": 1000
                }
                
            # Log parameters
            self.log_parameters(model_params)
            
            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("training_time_seconds", training_time)
            mlflow.log_metric("training_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=f"test_model_{model_type}"
            )
            
            print(f"--- ✅ Model trained and logged - Accuracy: {accuracy:.4f} ---")
            
            return model, {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "training_time": training_time
            }
            
        except Exception as e:
            print(f"--- ❌ Model training/logging failed: {e} ---")
            return None, None
            
    def create_and_log_artifacts(self, metrics, model_type):
        """Create and log artifacts"""
        print("--- 📁 Creating and logging artifacts ---")
        
        try:
            # Create metrics summary
            metrics_summary = {
                "model_type": model_type,
                "performance_metrics": metrics,
                "timestamp": datetime.now().isoformat(),
                "notes": "Test run for MLflow capabilities"
            }
            
            # Save metrics as JSON
            with open("metrics_summary.json", "w") as f:
                json.dump(metrics_summary, f, indent=2)
            mlflow.log_artifact("metrics_summary.json")
            
            # Create a simple plot
            plt.figure(figsize=(10, 6))
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            plt.bar(metric_names, metric_values)
            plt.title(f"Model Performance Metrics - {model_type}")
            plt.ylabel("Score")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("performance_metrics.png")
            mlflow.log_artifact("performance_metrics.png")
            plt.close()
            
            # Create a simple text report
            report = f"""
Model Performance Report
=======================
Model Type: {model_type}
Timestamp: {datetime.now().isoformat()}

Performance Metrics:
{'-' * 20}
"""
            for metric, value in metrics.items():
                if isinstance(value, float):
                    report += f"{metric.title()}: {value:.4f}\n"
                else:
                    report += f"{metric.title()}: {value}\n"
                    
            with open("model_report.txt", "w") as f:
                f.write(report)
            mlflow.log_artifact("model_report.txt")
            
            print("--- ✅ Artifacts created and logged ---")
            
        except Exception as e:
            print(f"--- ⚠️ Artifact logging failed: {e} ---")
            
    def run_complete_test(self):
        """Run complete MLflow test"""
        print("=== 🚀 STARTING MLFLOW CAPABILITY TEST ===")
        
        try:
            # Start MLflow run
            with mlflow.start_run(run_name=self.run_name) as run:
                # Enable system metrics logging
                try:
                    mlflow.system_metrics.enable_system_metrics_logging()
                    print("--- ✅ System metrics logging enabled ---")
                except Exception as e:
                    print(f"--- ⚠️ System metrics logging failed: {e} ---")
                    
                # Create datasets
                synthetic_df, iris_df = self.create_sample_datasets()
                
                # Log datasets
                self.log_datasets_to_mlflow(synthetic_df, iris_df)
                
                # Prepare data for training
                X = synthetic_df.drop('target', axis=1)
                y = synthetic_df['target']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Test Random Forest
                print("\n--- 🌲 Testing Random Forest ---")
                rf_model, rf_metrics = self.train_and_log_model(
                    X_train_scaled, X_test_scaled, y_train, y_test, "random_forest"
                )
                
                if rf_metrics:
                    self.create_and_log_artifacts(rf_metrics, "random_forest")
                    
                # Log run summary
                mlflow.log_param("test_status", "completed")
                mlflow.log_param("models_tested", "random_forest")
                mlflow.log_metric("total_datasets", 2)
                
                print(f"\n--- ✅ MLflow test completed successfully ---")
                print(f"--- 🔗 Run ID: {run.info.run_id} ---")
                print(f"--- 📊 Experiment: {self.experiment_name} ---")
                
        except Exception as e:
            print(f"--- ❌ MLflow test failed: {e} ---")
            mlflow.log_param("test_status", "failed")
            mlflow.log_param("error_message", str(e))
            
    def run_simple_test(self):
        """Run a simple test with minimal features"""
        print("=== 🧪 RUNNING SIMPLE MLFLOW TEST ===")
        
        try:
            with mlflow.start_run(run_name=f"simple-{self.run_name}"):
                # Simple iris classification
                iris = load_iris()
                X_train, X_test, y_train, y_test = train_test_split(
                    iris.data, iris.target, test_size=0.2, random_state=42
                )
                
                # Train simple model
                model = LogisticRegression(random_state=42)
                model.fit(X_train, y_train)
                
                # Log basic metrics
                accuracy = accuracy_score(y_test, model.predict(X_test))
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_param("model_type", "LogisticRegression")
                mlflow.log_param("dataset", "iris")
                
                print(f"--- ✅ Simple test completed - Accuracy: {accuracy:.4f} ---")
                
        except Exception as e:
            print(f"--- ❌ Simple test failed: {e} ---")


def main():
    """Main function to run MLflow tests"""
    print("MLflow Capability Testing Script")
    print("=" * 40)
    
    # Create tester instance
    tester = MLflowTester()
    
    # Run tests
    print("\n1. Running complete MLflow test...")
    tester.run_complete_test()
    
    print("\n2. Running simple MLflow test...")
    tester.run_simple_test()
    
    print("\n=== 🎉 ALL TESTS COMPLETED ===")
    print("Check your MLflow UI to see the logged runs, metrics, and artifacts!")


if __name__ == "__main__":
    main()
