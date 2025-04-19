from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import shap
import logging
import uvicorn
import pandas as pd
import numpy as np
import great_expectations as ge

# Load Models and Vectorizer
models = {
    "LogisticRegression": joblib.load("logistic_model.pkl"),
    "XGBoost": joblib.load("xgboost_model.pkl")
}
vectorizer = joblib.load("vectorizer.pkl")

# Load or create background data
try:
    background_data = joblib.load("background_data.pkl")
except:
    print("Background data not found. Creating new background data sample...")
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)
    X_background = df['text']
    X_background_tfidf = vectorizer.transform(X_background)
    background_data = shap.sample(X_background_tfidf, 100)
    joblib.dump(background_data, "background_data.pkl")
    print("Background data created and saved.")

# Setup Logging
logging.basicConfig(level=logging.INFO)

# FastAPI App
app = FastAPI()

# Pydantic Model
class SpamRequest(BaseModel):
    model_choice: str
    text: str

@app.get("/")
def home():
    return {"message": "Welcome to Spam Detector API"}

# Predict Route
@app.post("/predict")
def predict_spam(request: SpamRequest):
    try:
        model = models.get(request.model_choice)
        if model is None:
            return JSONResponse(content={"error": "Invalid model choice"}, status_code=400)

        text_transformed = vectorizer.transform([request.text])
        prediction = model.predict(text_transformed)[0]
        probability = model.predict_proba(text_transformed)[0][1]

        return {
            "prediction": "spam" if prediction == 1 else "ham",
            "probability": round(float(probability) * 100, 2)
        }
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# SHAP Explanation Route
@app.post("/explain")
def explain_spam(request: SpamRequest):
    try:
        model = models.get(request.model_choice)
        if model is None:
            return JSONResponse(content={"error": "Invalid model choice"}, status_code=400)

        text_transformed = vectorizer.transform([request.text])
        text_transformed_dense = text_transformed.toarray()
        background_dense = background_data.toarray()

        if request.model_choice == "LogisticRegression":
            explainer = shap.LinearExplainer(model, background_dense)
            shap_values = explainer.shap_values(text_transformed_dense)[0]
        elif request.model_choice == "XGBoost":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(text_transformed_dense)[0]
        else:
            return JSONResponse(content={"error": "Unsupported model for explanation"}, status_code=400)

        feature_names = vectorizer.get_feature_names_out()

        return {
            "shap_values": shap_values.tolist(),
            "feature_names": feature_names.tolist()
        }
    except Exception as e:
        logging.error(f"SHAP explanation error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Validate Input Route (NEW)
@app.post("/validate")
def validate_input(request: SpamRequest):
    try:
        text = request.text

        # Basic manual checks first
        if not isinstance(text, str):
            return {"valid": False}
        if text.strip() == "":
            return {"valid": False}
        if len(text.strip()) < 5:
            return {"valid": False}
        if len(text.strip()) > 5000:
            return {"valid": False}

        # If all checks passed
        return {"valid": True}

    except Exception as e:
        logging.error(f"Validation error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
