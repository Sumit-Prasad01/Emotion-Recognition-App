# Emotion Classification from Text
This project focuses on detecting human emotions from text using both classical Machine Learning models and Deep Learning techniques (LSTM). It involves data preprocessing, exploratory data analysis (EDA), encoding, and training multiple models to classify text into predefined emotion categories.


---

## 📊 Dataset

- **Source**: `train.txt`  
- **Format**: Each line contains a text and its corresponding emotion label, separated by a semicolon (`;`).
- **Emotions Covered**: joy, sadness, anger, fear, love, surprise, etc.

---

## 🔍 Key Features

- **EDA**: Word clouds, length distribution plots, and frequency analysis.
- **Preprocessing**:
  - Lowercasing
  - Removing special characters and stopwords
  - Stemming
- **Encoding**: Emotions encoded using `LabelEncoder`.
- **Modeling Approaches**:
  - Classical ML: Logistic Regression, Random Forest
  - Deep Learning: LSTM-based sequential model with Keras
- **Evaluation**: Accuracy, Confusion Matrix, and Classification Report
- **Web App**: Real-time emotion detection using Streamlit

---

## 🧠 Models Used

| Model              | Description                         |
|-------------------|-------------------------------------|
| Logistic Regression | Baseline ML model for classification |
| Random Forest      | Ensemble model for better accuracy   |
| LSTM               | Deep learning for sequence modeling  |

---

