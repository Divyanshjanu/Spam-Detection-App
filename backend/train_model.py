# train_model.py

import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Step 1: Load Dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)
df = df[['label', 'text']]  # Only keep necessary columns

# Step 2: Preprocessing Function
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Step 3: Define Features and Target
X = df['clean_text']
y = df['label'].map({'ham': 0, 'spam': 1})

# Step 4: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=7000)
X_tfidf = vectorizer.fit_transform(X)

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Step 7: Train XGBoost Classifier
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Step 8: Save Models and Vectorizer
joblib.dump(lr_model, "logistic_model.pkl")
joblib.dump(xgb_model, "xgboost_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print(" Logistic Regression and XGBoost models trained and saved successfully!")
