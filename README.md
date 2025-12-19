# Bangla Fake News Detection Using Machine Learning & Natural Language Processing.
## Project Overview
This is a machine learning  application designed to identify whether a Bangla news article is **Fake** or **Real**.  
Due to the rapid spread of misinformation on social media and online platforms, this system aims to assist users in verifying the authenticity of Bangla news content.
The application provides predictions using multiple models and shows both **individual model results** and an **overall confidence score**.

---

## Objectives
- Detect fake and real Bangla news articles 
- Provide probability-based predictions
- Build an easy-to-use web interface for users

---
## Models Used
The system uses the following models:

- Logistic Regression  
- Support Vector Machine (SVM)  
- Passive Aggressive Classifier  
- XGBoost
- Naive Bayes
- Complement Naive Bayes
- Random Forest
- LightGBM
- Hierarchical Attention Network (HAN - Deep learning model)
- Bangla BERT (Transformer-based Deep Learning model)

---
## Technologies Used
- Python  3.14.0
- Streamlit  
- Scikit-learn  
- Hugging Face Transformers  
- PyTorch  
- XGBoost  
- LightGBM  
- NumPy
- Pandas  

---
## System Features
- Bangla text input support
- Multiple model predictions
- Fake vs Real probability output
- Overall confidence calculation (average of all model probabilities)
- An easy-to-use web application for Bangla fake news detection.

---
## Project Architecture
1. User inputs Bangla news text  
2. Text is preprocessed  
3. Input is passed to multiple trained models  
4. Each model outputs prediction probabilities  
5. Final confidence score is calculated and displayed  

---
## How to Run the Project
1. Install required libraries   
2. Run the Streamlit application  
   python -m streamlit run notebook\app.py
## Developed by

Habiba Alam Raisa (2231272642)
Rifat Imtiaze Ruddro (2311474642)

# The processed dataset is too large, you can download it from here: https://drive.google.com/file/d/1Dmm50Edrz64ldr1wfWXZQl_x4zKZkD0_/view?usp=drive_link 
 























