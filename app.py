import streamlit as st
import pickle
import joblib
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Paths to models and vectorizer

MODEL_DIR = "models"
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

MODEL_PATHS = {
    "logistic_regression": os.path.join(MODEL_DIR, "final_logistic_regression_model.pkl"),
    "svm": os.path.join(MODEL_DIR, "svm_model_final.pkl"),
    "random_forest": os.path.join(MODEL_DIR, "random_forest_final.pkl"),
    "xgboost": os.path.join(MODEL_DIR, "xgboost_model_final.pkl")
}

# BERT folder
BERT_DIR = os.path.join(MODEL_DIR, "bangla_bert_fake_news_final")

# Load TF-IDF vectorizer

with open(VECTORIZER_PATH, "rb") as f:
    VECTORIZER = pickle.load(f)


# Load models

MODELS = {}
for name, path in MODEL_PATHS.items():
    try:
        with open(path, "rb") as f:
            MODELS[name] = pickle.load(f)
    except (pickle.UnpicklingError, EOFError):
        # fallback to joblib if pickle fails
        MODELS[name] = joblib.load(path)


# Load BERT model
tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR)
bert_model.eval()  # set to evaluation mode


# Helper function for BERT prediction

def predict_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    return pred


# UI

st.title("Bangla Fake News Detection")
st.write("Enter Bangla text below to predict if it is Real or Fake:")

# Input box
user_input = st.text_area("Enter text here:", height=150)

# Map 0/1 to Fake/Real
label_map = {0: "Fake", 1: "Real"}

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text!")
    else:
        # Transform input using TF-IDF for traditional ML models
        X_input = VECTORIZER.transform([user_input])

        st.subheader("Predictions:")

        # Traditional ML models
        for model_name, model in MODELS.items():
            pred = model.predict(X_input)[0]
            st.write(f"**{model_name}**: {label_map.get(pred, pred)}")

        # BERT model
        bert_pred = predict_bert(user_input)
        st.write(f"**BERT**: {label_map.get(bert_pred, bert_pred)}")

