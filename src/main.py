"""FastAPI app that serves the diabetes progression model.

Run from the project root with: uvicorn src.main:app --reload
"""
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.predict import load_artifacts, predict_data

app = FastAPI(
    title="Diabetes Progression API",
    description="Predicts one year diabetes progression from patient measurements.",
    version="1.0.0",
)


class PatientData(BaseModel):
    """Measurements for one patient. Ranges reject values that are not realistic."""
    age: float = Field(ge=18, le=100, description="Age in years")
    sex: int = Field(ge=1, le=2, description="Sex coded as 1 or 2")
    bmi: float = Field(ge=10, le=60, description="Body mass index")
    blood_pressure: float = Field(ge=40, le=200, description="Average blood pressure in mm Hg")
    total_cholesterol: float = Field(ge=50, le=400, description="Total serum cholesterol")
    ldl: float = Field(ge=20, le=300, description="Low density lipoproteins")
    hdl: float = Field(ge=10, le=120, description="High density lipoproteins")
    cholesterol_hdl_ratio: float = Field(ge=1, le=12, description="Total cholesterol divided by HDL")
    log_triglycerides: float = Field(ge=2, le=8, description="Log of serum triglycerides")
    blood_sugar: float = Field(ge=40, le=200, description="Blood sugar level")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 59, "sex": 2, "bmi": 32.1, "blood_pressure": 101,
                "total_cholesterol": 157, "ldl": 93.2, "hdl": 38,
                "cholesterol_hdl_ratio": 4, "log_triglycerides": 4.86, "blood_sugar": 87,
            }
        }
    }

    def to_row(self) -> list:
        """Return the values in the column order the model was trained on."""
        return [self.age, self.sex, self.bmi, self.blood_pressure,
                self.total_cholesterol, self.ldl, self.hdl,
                self.cholesterol_hdl_ratio, self.log_triglycerides, self.blood_sugar]


class BatchRequest(BaseModel):
    """Up to 100 patients in one request."""
    patients: list[PatientData] = Field(min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    """Prediction for one patient."""
    progression_score: float
    risk_level: str


class BatchResponse(BaseModel):
    """Predictions for a batch of patients."""
    count: int
    predictions: list[PredictionResponse]


def _predict(rows: list) -> list:
    """Run the model and turn failures into clear HTTP errors."""
    try:
        return predict_data(rows)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/", status_code=status.HTTP_200_OK)
async def health():
    """Health check that also reports whether the model is loaded."""
    try:
        load_artifacts()
        return {"status": "healthy", "model_loaded": True}
    except FileNotFoundError:
        return {"status": "healthy", "model_loaded": False}


@app.get("/model-info")
async def model_info():
    """Return the model type, input features and test set metrics."""
    try:
        _, meta = load_artifacts()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return meta


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    """Predict the progression score and risk level for one patient."""
    return _predict([patient.to_row()])[0]


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """Predict progression for several patients at once."""
    results = _predict([p.to_row() for p in request.patients])
    return {"count": len(results), "predictions": results}
