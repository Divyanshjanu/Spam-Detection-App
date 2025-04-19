#  Spam Detection App (FastAPI + Streamlit)

A real-time Spam Detection Web App using XGBoost and Logistic Regression, built with **FastAPI** (backend) and **Streamlit** (frontend), fully containerized and deployed to **Google Cloud Run**.

---

# 🔗 Live Demo Links

- **Backend (FastAPI Base URL):**  
  [Backend API](https://app-backend-1027897761252.northamerica-northeast1.run.app)

- **Backend (Swagger API Docs):**  
  [Swagger Docs](https://app-backend-1027897761252.northamerica-northeast1.run.app/docs)

- **Frontend (Streamlit App):**  
  [Spam Detector Frontend](https://app-frontend-1027897761252.northamerica-northeast1.run.app)

---

#  Project Overview

This project simulates a **Risk Modeling** machine learning use case:
Classifying a text as **Spam** or **Ham** based on pre-trained models.

- **Backend:** FastAPI service for prediction, explanation (SHAP), and validation.
- **Frontend:** Streamlit UI for easy interaction.
- **Models:** XGBoost and Logistic Regression trained on SMS Spam Dataset.

---

#  Features

- Predict Spam vs Ham.
- SHAP-based feature importance visualized using Plotly Beeswarm Plot.
- Great Expectations used for input validation.
- OpenAPI Schema (Data Contract) exported.
- Deployed using Docker and Google Cloud Run.

---

#  Prediction Task

- **Task:** Spam Detection (Risk Modeling)
- **Models:** XGBoost and Logistic Regression
- **Dataset:** Public Kaggle SMS Spam Collection Dataset
- **Preprocessing:**
  - TF-IDF Vectorization
- **Metrics Evaluated:**
  - F1-Score
  - Precision
  - Recall
  - ROC-AUC
  - Brier Score

---

#  API Endpoints

| Method | Endpoint | Description |
|:------|:---------|:------------|
| GET   | `/`         | Home Route |
| POST  | `/validate` | Validate Input Message |
| POST  | `/predict`  | Predict Spam or Ham |
| POST  | `/explain`  | SHAP Explainability |

Example Request to `/predict`:
```json
{
  "model_choice": "XGBoost",
  "text": "Congratulations! You've won a $1000 Walmart gift card."
}
```

Example Response:
```json
{
  "prediction": "spam",
  "probability": 98.76
}
```

---

#  Deployment Overview

- **Dockerized** both Backend and Frontend separately.
- **Deployed** using **Google Cloud Run**.
- Used **Docker Buildx** for correct architecture (`linux/amd64`).
- Pushed container images to Google Container Registry.

---

#  Key Technologies Used

- Python 3.9
- FastAPI
- Streamlit
- XGBoost
- Logistic Regression
- Great Expectations (Validation)
- SHAP (Explainability)
- Plotly Express (Visualization)
- Docker
- Google Cloud Run

---

#  Target Audience

- Email service providers
- SMS filtering systems
- Enterprise anti-spam systems
- Real-time fraud detection apps

---

#  Performance

- Backend Response Time: ~150ms
- Frontend Response (Cloud Run): ~2s
- Memory Footprint: Very lightweight


---

#  How To Run Locally

```bash
# Clone Repo
$ git clone https://github.com/Divyanshjanu/spam-detection-app.git

# Backend
$ cd backend
$ docker build -t app-backend .
$ docker run -p 8000:8000 app-backend

# Frontend
$ cd ../frontend
$ docker build -t app-frontend .
$ docker run -p 8501:8501 app-frontend
```

---

#  Project Timeline

- Day 1: Backend Setup with FastAPI
- Day 2: Streamlit Frontend Development
- Day 3: Great Expectations Validation + SHAP Explainability
- Day 4: Dockerization
- Day 5: Google Cloud Deployment
- Day 6: Final Testing and Reflection

---

#  Reflection

> "In A3 reflection, I realized I underplanned A2. This time, I broke down my work day-by-day using Agile principles. Result: A fully functioning app with faster iteration, proper validation, and explainable predictions!"


