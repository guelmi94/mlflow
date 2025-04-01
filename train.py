import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
import xgboost as xgb
import mlflow
import mlflow.xgboost
import argparse

def load_data(data_path='data/DSA-2025_clean_data.tsv'):
    """Load and prepare data for training."""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, sep='\t')
    print(f"Data shape: {df.shape}")
    
    # Define features and target
    X = df[['chol', 'crp', 'phos']]
    y = df['readmission']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test, experiment_name="xgboost_readmission_model"):
    """Train XGBoost model with grid search and log each model to MLflow without registration."""
    # Set up MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Define parameter grid for grid search
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'objective': ['binary:logistic']
    }
    
    # Create parameter combinations
    grid = list(ParameterGrid(param_grid))
    print(f"Training {len(grid)} models with different hyperparameters")
    
    # Define number of cross-validation folds
    n_folds = 5
    print(f"Using {n_folds}-fold cross-validation for each model")
    
    # Create ROC AUC scorer for cross-validation
    roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
    
    # Train models with each hyperparameter combination
    for i, params in enumerate(grid):
        print(f"Training model {i+1}/{len(grid)} with params: {params}")
        
        # Start MLflow run for this model
        with mlflow.start_run(run_name=f"model_{i+1}") as run:
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Initialize model with current parameters
            model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=n_folds, 
                scoring=roc_auc_scorer,
                n_jobs=-1
            )
            
            # Log cross-validation results
            mlflow.log_metric("cv_roc_auc_mean", np.mean(cv_scores))
            mlflow.log_metric("cv_roc_auc_std", np.std(cv_scores))
            
            for fold_idx, cv_score in enumerate(cv_scores):
                mlflow.log_metric(f"cv_roc_auc_fold_{fold_idx+1}", cv_score)
            
            # Train the model on the full training set
            model.fit(X_train, y_train)
            
            # Make predictions on test set
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Log test metrics
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_f1_score", f1)
            mlflow.log_metric("test_roc_auc", roc_auc)
            
            # Log feature importance
            feature_importance = model.feature_importances_
            for j, feature in enumerate(X_train.columns):
                mlflow.log_metric(f"feature_importance_{feature}", feature_importance[j])
            
            # Log model (without registering it)
            model_name = f"model_depth_{params['max_depth']}_lr_{params['learning_rate']}"
            mlflow.xgboost.log_model(model, model_name)
            
            print(f"Model trained and logged with run_id: {run.info.run_id}")
            print(f"Cross-validation ROC AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            print(f"Test metrics - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    
    # Summary of all models
    print("\n==== Grid Search Summary ====")
    print(f"Total models trained: {len(grid)}")
    print(f"Each model was evaluated with {n_folds}-fold cross-validation")
    print("All models have been logged to MLflow without automatic registration")

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost models with grid search and log to MLflow")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="data/DSA-2025_clean_data.tsv",
        help="Path to the data file"
    )
    parser.add_argument(
        "--experiment_name", 
        type=str, 
        default="xgboost_readmission_model",
        help="MLflow experiment name"
    )
    
    args = parser.parse_args()
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(args.data_path)
    
    # Train models and log to MLflow
    train_model(X_train, X_test, y_train, y_test, args.experiment_name)
    
    print(f"You can view and compare all models in the MLflow UI")

if __name__ == "__main__":
    main()