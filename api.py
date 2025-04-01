import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Union, Any
import pandas as pd
from model_wrapper import ReadmissionPredictor
import uvicorn

app = FastAPI(
    title="Readmission Prediction API",
    description="API for predicting patient readmission using XGBoost model",
    version="1.0.0",
    root_path="/proxy/8000"  # Add this line
)

# Initialize model (this will be loaded on startup)
predictor = None

# Define input data model for single prediction
class PatientData(BaseModel):
    chol: float = Field(..., description="Cholesterol level", example=200.0)
    crp: float = Field(..., description="C-reactive protein level", example=3.5)
    phos: float = Field(..., description="Phosphorus level", example=4.2)
    
    @validator('chol')
    def validate_chol(cls, v):
        if v <= 0:
            raise ValueError("Cholesterol must be positive")
        return v
    
    @validator('crp')
    def validate_crp(cls, v):
        if v < 0:
            raise ValueError("CRP cannot be negative")
        return v
    
    @validator('phos')
    def validate_phos(cls, v):
        if v <= 0:
            raise ValueError("Phosphorus must be positive")
        return v

# Define batch prediction input model
class BatchPredictionInput(BaseModel):
    patients: List[PatientData]

# Define prediction result models
class PredictionResult(BaseModel):
    prediction: int = Field(..., description="Predicted readmission (0 or 1)")
    probability: float = Field(..., description="Probability of readmission")

class BatchPredictionResult(BaseModel):
    predictions: List[PredictionResult]
    summary: Dict[str, Any] = Field(
        ..., 
        description="Summary statistics about the predictions"
    )

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global predictor
    try:
        # Try to load the latest model from registry
        predictor = ReadmissionPredictor.from_latest()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fall back to a specific run ID if needed
        # predictor = ReadmissionPredictor(run_id="YOUR_RUN_ID")

@app.get("/")
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Readmission Prediction API",
        "description": "Use /predict for single predictions or /predict/batch for batch predictions",
        "status": "active"
    }

@app.post("/predict", response_model=PredictionResult)
async def predict(patient: PatientData):
    """
    Predict readmission for a single patient.
    
    Returns:
        PredictionResult: Prediction result with class and probability
    """
    global predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to dict for prediction
        patient_dict = patient.dict()
        
        # Get prediction
        result = predictor.predict(patient_dict)
        
        # Format response
        return PredictionResult(
            prediction=result['prediction'][0],
            probability=result['probability'][0]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResult)
async def predict_batch(batch_input: BatchPredictionInput):
    """
    Predict readmission for multiple patients.
    
    Returns:
        BatchPredictionResult: Prediction results with classes and probabilities
    """
    global predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(batch_input.patients) == 0:
        raise HTTPException(status_code=400, detail="Empty batch")
    
    try:
        # Convert to dataframe
        patients_data = [p.dict() for p in batch_input.patients]
        df = pd.DataFrame(patients_data)
        
        # Get predictions
        results = predictor.predict(df)
        
        # Prepare individual prediction results
        predictions = []
        for i in range(len(results['prediction'])):
            predictions.append(
                PredictionResult(
                    prediction=results['prediction'][i],
                    probability=results['probability'][i]
                )
            )
        
        # Prepare summary statistics
        positive_count = sum(1 for p in results['prediction'] if p == 1)
        negative_count = sum(1 for p in results['prediction'] if p == 0)
        avg_probability = sum(results['probability']) / len(results['probability'])
        
        summary = {
            "total_patients": len(predictions),
            "positive_predictions": positive_count,
            "negative_predictions": negative_count,
            "average_probability": avg_probability
        }
        
        return BatchPredictionResult(
            predictions=predictions,
            summary=summary
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Check if the API and model are running properly."""
    global predictor
    if predictor is None:
        return {"status": "error", "message": "Model not loaded"}
    
    return {"status": "ok", "message": "API is running and model is loaded"}

# Run the API server when the script is executed directly
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, root_path="/proxy/8000")