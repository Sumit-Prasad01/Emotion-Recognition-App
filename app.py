import streamlit as st
import numpy as np
import re
import pickle
import nltk
from nltk.stem import PorterStemmer
import tensorflow
from tensorflow.keras.models import load_model



# Download stopwords (once)
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# ============================ Load Models ============================================
model = load_model("models/LSTM_Model.keras") 
tfidf_vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
label_encoder = pickle.load(open('models/lb1.pkl', 'rb'))

# ============================ Text Preprocessing =====================================
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)     # Remove non-letter characters
    text = text.lower().split()               # Lowercase and split
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

# ============================ Prediction Function ====================================
def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    probs = model.predict(input_vectorized)[0]              
    predicted_index = np.argmax(probs)                     
    predicted_emotion = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(np.max(probs))                       

    return predicted_emotion, confidence

# ============================ Streamlit App UI =======================================
st.set_page_config(page_title="Emotion Detector", layout="centered")

st.markdown("## 💬 Six Human Emotions Detection App")
st.markdown("---")
st.markdown("**Emotion classes:** Joy, Fear, Anger, Love, Sadness, Surprise")
st.markdown("---")

user_input = st.text_input("✍️ Enter your text here:")

if st.button("🔍 Predict Emotion"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        predicted_emotion, confidence = predict_emotion(user_input)

        st.success(f"### 🎯 Predicted Emotion: `{predicted_emotion}`")

        if confidence is not None:
            st.metric(label="Prediction Confidence", value=f"{confidence * 100:.2f} %")
        else:
            st.info("Confidence score not available for this model.")

        st.markdown("---")
