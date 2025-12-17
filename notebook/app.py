import streamlit as st
import pickle
import joblib
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# ------------------------------
# Paths
# ------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(CURRENT_DIR, "tfidf_vectorizer.pkl")

MODEL_PATHS = {
    "Logistic Regression": os.path.join(CURRENT_DIR, "logistic_regression_model.pkl"),
    "SVM": os.path.join(CURRENT_DIR, "svm_model.pkl"),
    "Random Forest": os.path.join(CURRENT_DIR, "random_forest.pkl"),
    "XGBoost": os.path.join(CURRENT_DIR, "xgboost_model.pkl")
}

BERT_DIR = os.path.join(CURRENT_DIR, "bangla_bert_fake_news_v3")

# ------------------------------
# Load TF-IDF vectorizer
# ------------------------------
try:
    with open(VECTORIZER_PATH, "rb") as f:
        VECTORIZER = pickle.load(f)
except FileNotFoundError:
    st.error(f"TF-IDF vectorizer not found at {VECTORIZER_PATH}")
    st.stop()

# ------------------------------
# Load ML models
# ------------------------------
MODELS = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        MODELS[name] = joblib.load(path)
    else:
        st.warning(f"{name} model not found at {path}")

# ------------------------------
# Load BERT
# ------------------------------
if os.path.exists(BERT_DIR):
    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR)
    bert_model.eval()
else:
    bert_model = None

# ------------------------------
# BERT prediction helper
# ------------------------------
def predict_bert(text):
    if bert_model is None:
        return None, None
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred = int(np.argmax(logits.numpy(), axis=1)[0])
    return pred, probs

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Bangla Fake News Detection")
st.write("Enter Bangla text below to predict if it is Real or Fake:")

user_input = st.text_area("Enter text here:", height=150)
label_map = {0: "Fake", 1: "Real"}

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text!")
    else:
        X_input = VECTORIZER.transform([user_input])
        st.subheader("Predictions with Confidence:")

        # Store all probabilities to calculate overall confidence
        all_probs = []

        # Traditional ML models
        for model_name, model in MODELS.items():
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_input)[0]  # [fake_prob, real_prob]
                all_probs.append(probs)
                pred = int(model.predict(X_input)[0])
                st.write(
                    f"**{model_name}** predicts: **{label_map[pred]}** "
                    f"(Fake: {probs[0]*100:.2f}%, Real: {probs[1]*100:.2f}%)"
                )
            else:
                # fallback if no probability
                pred = int(model.predict(X_input)[0])
                st.write(f"**{model_name}** predicts: **{label_map[pred]}**")

        # BERT
        if bert_model is not None:
            pred, probs = predict_bert(user_input)
            if pred is not None:
                all_probs.append(probs)
                st.write(
                    f"**BERT** predicts: **{label_map[pred]}** "
                    f"(Fake: {probs[0]*100:.2f}%, Real: {probs[1]*100:.2f}%)"
                )

        # ------------------------------
        # Overall confidence
        if all_probs:
            overall = np.mean(np.array(all_probs), axis=0)  # average across models
            st.subheader("Overall Confidence:")
            st.write(f"Fake: {overall[0]*100:.2f}% | Real: {overall[1]*100:.2f}%")











