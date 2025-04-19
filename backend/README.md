#  Spam Detection API 

A real-time Spam Detection API using XGBoost, built with FastAPI and containerized using Docker. This project simulates a **risk modeling task**, where input text is classified as **spam** or **ham**. It includes model evaluation, API performance monitoring, input validation, and is fully deployable to a cloud platform.

# Live Demo

- Swagger UI: http://0.0.0.0:8000/docs

-----------------------

# Project Overview

This project simulates a Risk Modeling machine learning use case. The API takes in text input and predicts whether it's spam  or ham (not spam) using a trained XGBoost classifier.

# Features

- Risk prediction using a pre-trained XGBoost model
- Response includes prediction, probability, memory usage, and time taken
- RESTful API built with FastAPI
- Containerized using Docker

-----------------------

# Prediction Task

- Selected Task: Risk Modeling (Spam Classification)
- Model Used: XGBoost Classifier and Logistic regression
- Reason: Robust with tabular data, handles imbalance, great performance
- Dataset: Public spam message dataset taken from kaggle (https://www.kaggle.com/datasets/bagavathypriya/spam-ham-dataset/data)
- Preprocessing: TF-IDF Vectorization + StandardScaler

-----------------------

# API Users

- Target Users: Email services, text classifiers, internal IT filters
- Expected Daily Requests: 500 - 1000
- User Requirements:**
  - Real-time response (< 500ms per request)
  - Lightweight response in JSON format
  - Simple POST endpoint for integration

-----------------------

# Model Evaluation

- Algorithm: XGBoost and logistic regression
- Vectorization: TF-IDF
- Scaler: StandardScaler (with_mean=False)
- Metrics Used:
  
  - F1-score (primary)
  - ROC-AUC
  - Precision & Recall
  - Brier Score


- Why not Accuracy? Accuracy is misleading in imbalanced datasets like spam classification.

-------------------------

# API Endpoints

# GET
 json
{ "message": "Welcome to the Spam Detection Model" }


# POST /predict 
* input:
json
{ "text": "Going for dinner.msg you after" }


*Output:
json
{
  "text": "Going for dinner.msg you after.",
  "prediction": "ham",
  "spam_probability": 0.0198,
  "response_time_seconds": 0.0017,
  "memory_usage_mb": 0.09
}


-----------------------

#  API Service Performance

Performance metrics are logged and returned in the API response.

- Response Time: 0.015s per request
- Memory Usage: 3-4 MB per prediction
- Monitoring Tools:
  - Built-in logging (FastAPI + Python logging)

---

# User Interaction

- Input Format:
  - Text input via JSON POST
  - Testable via Swagger UI
- Output Format:
  - JSON structured response with prediction, probability, and performance metrics

-----------------------------


# Deployment

- Containerization: Docker

# Docker Commands
bash
docker build -t spam-detector-api.
docker run -p 8000:8000 spam-detector-api


------------------------------

#  Example Use Case

- Detect spam in email pipelines
- Real-time filtering in SMS apps

----------------------------------

## Author

Divyansh Janu  
Docker Hub(divyanshjanu/spam_model) | LinkedIn(www.linkedin.com/in/divyansh-janu-91446a25b) | Docker image[docker pull divyanshjanu/spam_model:1.0]

------------------------------


