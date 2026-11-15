# FastAPI Lab 1: Diabetes Progression API

Base lab: API_Labs/FastAPI_Labs from [raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps)

Author: Sathyasai Rajan

This project serves a machine learning model as a REST API with FastAPI. You send a patient's measurements and the API returns a predicted diabetes progression score along with a low, moderate or high risk level.

## What I changed from the original lab

1. Changed the problem from Iris classification to regression. The model predicts a one year disease progression score for diabetes patients using the scikit-learn diabetes data.
2. The API takes real measurements like age, BMI and blood pressure instead of scaled values, and each field has a realistic range. Bad input such as an age of 200 gets a clear 422 error.
3. Added a risk level to each prediction based on where the score falls in the training data.
4. Added a batch endpoint that accepts up to 100 patients in one request.
5. Added a model info endpoint that returns the model type, input features and test metrics.
6. The original loaded the model file from disk on every request. Now it loads once and is reused.
7. The original only worked when run from inside the src folder because of relative paths. Paths are now based on the file location, so everything runs from the project root.
8. Fixed the run command in the original README, which said uvicorn app:main instead of uvicorn main:app.
9. Added 14 unit tests covering the data, the model and every endpoint, including invalid input.
10. Added a Dockerfile that trains the model during the build and serves the API.
11. Added a GitHub Actions workflow that trains the model and runs all tests on every push.

## Results

The random forest model gets an R squared of 0.46 and a mean absolute error of 43 on the test set. This is in line with what is usually reported for this dataset, since progression is hard to predict from these 10 measurements alone.

## Endpoints

GET / checks that the API is running and the model is loaded.

GET /model-info returns the model type, features and test metrics.

POST /predict returns a prediction for one patient.

POST /predict/batch returns predictions for several patients.

Example request for POST /predict:

```json
{
  "age": 59,
  "sex": 2,
  "bmi": 32.1,
  "blood_pressure": 101,
  "total_cholesterol": 157,
  "ldl": 93.2,
  "hdl": 38,
  "cholesterol_hdl_ratio": 4,
  "log_triglycerides": 4.86,
  "blood_sugar": 87
}
```

Example response:

```json
{
  "progression_score": 225.4,
  "risk_level": "high"
}
```

## How to run

With Docker, which needs Docker Desktop running:

```bash
docker compose up --build
```

Or with Python directly, from the project root:

```bash
pip install -r requirements.txt
python -m src.train
uvicorn src.main:app --reload
```

Then open http://localhost:8000/docs to try the endpoints in the browser.

## Tests

```bash
python -m src.train
pytest -v
```
