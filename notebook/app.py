import streamlit as st
import os
import pickle
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import altair as alt



CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


VECTORIZER_PATH = os.path.join(CURRENT_DIR, "tfidf_vectorizer.pkl")
MODEL_PATHS = {
    "Logistic Regression": os.path.join(CURRENT_DIR, "logistic_regression_model.pkl"),
    "SVM": os.path.join(CURRENT_DIR, "svm_model.pkl"),
    "Random Forest": os.path.join(CURRENT_DIR, "random_forest.pkl"),
    "XGBoost": os.path.join(CURRENT_DIR, "xgboost_model.pkl")
}


FRIEND_VECTORIZER_PATH = os.path.join(CURRENT_DIR, "final_vectorizer.pkl")
FRIEND_MODELS = {
    "LightGBM": os.path.join(CURRENT_DIR, "frontend_lightgbm.pkl"),
    "Passive Aggressive": os.path.join(CURRENT_DIR, "frontend_passive_aggressive.pkl")
}

# BERT
BERT_DIR = os.path.join(CURRENT_DIR, "bangla_bert_fake_news_v3")


with open(VECTORIZER_PATH, "rb") as f:
    VECTORIZER = pickle.load(f)


MODELS = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        MODELS[name] = joblib.load(path)



friend_vectorizer = joblib.load(FRIEND_VECTORIZER_PATH)


friend_models = {name: joblib.load(path) for name, path in FRIEND_MODELS.items()}


# Load BERT

if os.path.exists(BERT_DIR):
    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR)
    bert_model.eval()
else:
    bert_model = None


# BERT prediction helper

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


# Streamlit UI

st.title("Bangla Fake News Detection")
st.write("Enter Bangla text below to predict if it is Real or Fake:")

user_input = st.text_area("Enter text here:", height=150)
label_map = {0: "Fake", 1: "Real"}

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text!")
    else:
        all_probs = []

        
        X_input = VECTORIZER.transform([user_input])
        st.subheader("Models Predictions:")
        for model_name, model in MODELS.items():
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_input)[0]
                all_probs.append(probs)
                pred = int(model.predict(X_input)[0])
                st.write(f"**{model_name}** → {label_map[pred]} "
                         f"(Fake: {probs[0]*100:.2f}%, Real: {probs[1]*100:.2f}%)")
            else:
                pred = int(model.predict(X_input)[0])
                st.write(f"**{model_name}** → {label_map[pred]}")

        
        
        X_friend = friend_vectorizer.transform([user_input])
        for model_name, model in friend_models.items():
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_friend)[0]
                all_probs.append(probs)
                pred = int(model.predict(X_friend)[0])
                st.write(f"**{model_name}** → {label_map[pred]} "
                         f"(Fake: {probs[0]*100:.2f}%, Real: {probs[1]*100:.2f}%)")
            else:
                pred = int(model.predict(X_friend)[0])
                st.write(f"**{model_name}** → {label_map[pred]}")

        
        # BERT model
       
        if bert_model is not None:
            pred, probs = predict_bert(user_input)
            if pred is not None:
                all_probs.append(probs)
                st.subheader("BERT Prediction:")
                st.write(f"**BERT** → {label_map[pred]} "
                         f"(Fake: {probs[0]*100:.2f}%, Real: {probs[1]*100:.2f}%)")
                
# Overall Confidence

        if all_probs:
         overall = np.mean(np.array(all_probs), axis=0)
    
    # Display numeric confidence
        st.subheader("Overall Confidence Across All Models:")
        st.write(f"Fake: {overall[0]*100:.2f}% | Real: {overall[1]*100:.2f}%")
    
    # Create DataFrame
        confidence_df = pd.DataFrame({
        "Label": ["Fake", "Real"],
        "Confidence": overall * 100
    })
    
    
    
    # Altair chart 
    
        chart = alt.Chart(confidence_df).mark_bar().encode(
        x=alt.X("Label", sort=None),
        y=alt.Y("Confidence", title="Confidence (%)"),
        color=alt.Color("Label", scale=alt.Scale(domain=["Fake","Real"], range=["#f44336","#4CAF50"])),
        tooltip=[alt.Tooltip("Label"), alt.Tooltip("Confidence", format=".2f")]
    )
    
    # Add text labels above bars
        text = chart.mark_text(
        align='center',
        baseline='bottom',
        dy=-5,
        color='black'
    ).encode(
        text=alt.Text("Confidence", format=".2f")
    )
    
    # Combine bars and text
        final_chart = chart + text
    
        st.subheader("Overall Confidence Visualization (Altair)")
        st.altair_chart(final_chart, use_container_width=True)


                

           















