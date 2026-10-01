from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# --------------------------------------------------
# 1. Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0"
)


# --------------------------------------------------
# 2. Load trained machine learning model
# --------------------------------------------------

model = joblib.load("models/churn_model.pkl")


# --------------------------------------------------
# 3. Define the input data structure
# --------------------------------------------------

class Customer(BaseModel):
    Geography: str
    CreditScore: int
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


# --------------------------------------------------
# 4. Prediction function
# --------------------------------------------------

def predict_customer(customer):

    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0][1]

    return prediction, probability


# --------------------------------------------------
# 5. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API"
    }


# --------------------------------------------------
# 6. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: Customer):

    # Convert Pydantic model to dictionary
    customer = pd.DataFrame([data.model_dump()])

    # Make prediction
    prediction, probability = predict_customer(customer)

    # Convert prediction to human-readable text
    if prediction == 1:
        result = "Customer is likely to churn"
    else:
        result = "Customer is likely to stay"

    # Return prediction
    return {
        "churn_prediction": int(prediction),
        "result": result,
        "churn_probability": float(probability)
    }

