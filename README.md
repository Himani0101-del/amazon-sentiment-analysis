# amazon-sentiment-analysis
Sentiment Analysis with Ensemble Learning (Naive Bayes + XGBoost) This project classifies Amazon customer reviews as Positive or Not Positive using a soft voting ensemble of Naive Bayes and XGBoost. It includes text preprocessing, sentiment labeling with TextBlob, TF-IDF feature extraction, model training, evaluation, and full dataset deployment.
#  Sentiment Analysis with Ensemble Learning

This project classifies Amazon customer reviews as **Positive** or **Not Positive** using a soft voting ensemble of Naive Bayes and XGBoost. It includes text preprocessing, sentiment labeling with TextBlob, TF-IDF feature extraction, model training, evaluation, and full dataset deployment.

---

## 🔍 Project Overview

- Cleaned and preprocessed review text  
- Labeled sentiment using TextBlob polarity  
- Extracted features using TF-IDF 
- Converted to binary sentiment: Positive vs Not Positive  
- Balanced training data
- Trained Naive Bayes and XGBoost models  
- Combined them using soft voting ensemble  
- Evaluated performance on test set  
- Deployed predictions on full dataset



##  Files Included

| File | Description |
|------|-------------|
| `sentiment_model.ipynb` | Full training and evaluation notebook |
| `ensemble_model.joblib` | Trained soft voting ensemble model |
| `vectorizer.joblib` | TF-IDF vectorizer |
| `label_encoder.joblib` | Label encoder for sentiment classes |




##  Model Performance

- Accuracy: 78%
- Precision/Recall: Balanced across both classes
- F1 Scores:  
  - Positive: 0.79  
  - Not Positive: 0.77

## Load the model and vectorizer:

import joblib
model = joblib.load('ensemble_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')
encoder = joblib.load('label_encoder.joblib')
