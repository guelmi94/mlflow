# Readmission Prediction Model with MLflow and FastAPI

This project implements an XGBoost model for predicting patient readmission based on cholesterol, C-reactive protein, and phosphorus levels. The model is trained and tracked using MLflow, and served via a FastAPI REST API.

## Project Structure

```
mlflow_xgboost_project/
├── data/
│   └── DSA-2025_clean_data.tsv
├── train.py (loads data, trains model with grid search, logs to MLflow)
├── model_wrapper.py (Python wrapper for model prediction)
├── api.py (FastAPI application)
├── MLproject (MLflow project file)
├── conda.yaml (environment dependencies)
└── README.md
```

## Setup Instructions

1. Download the data:

```bash
mkdir -p mlflow_xgboost_project/data
cp mlflow/DSA-2025_clean_data.tsv mlflow_xgboost_project/data/
cd mlflow_xgboost_project
```

2. Start the MLflow tracking server (if not already running on SSPcloud):

```bash
mlflow server --host 0.0.0.0 --port 5000
```

3. Train the model using MLflow:

```bash
mlflow run . -P experiment_name=xgboost_readmission_model
```

The command will output a run ID which you can use to refer to this specific model version or  just use the registered model name 'xgboost'.

4. Serve the model via FastAPI:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## Using the API

The API provides the following endpoints:

- `GET /`: Basic information about the API
- `GET /health`: Health check endpoint
- `POST /predict`: Single patient prediction
- `POST /predict/batch`: Batch prediction for multiple patients

### Example API Requests

#### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"chol": 200, "crp": 3.5, "phos": 4.2}'
```

#### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"patients": [{"chol": 200, "crp": 3.5, "phos": 4.2}, {"chol": 180, "crp": 2.8, "phos": 3.9}]}'
```

## Interactive API Documentation

FastAPI provides automatic interactive API documentation. After starting the API server, you can access:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Viewing Model Results

To view the logged model metrics and parameters, open the MLflow UI:

```bash
# If MLflow server is running locally
open http://localhost:5000
```
The criteria in order to choose the final model were :
1. Accuracy
2. F1 score
3. Time of training
In the end, the chosen model (model_5) was registered as 'xgboost' since it was the smallest time consuming for its high Accuracy and relatively low F1 score