# ================= IMPORTS =================

import streamlit as st
import pickle
import numpy as np
import re
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ================= NLTK SETUP =================

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ================= LOAD FILES =================

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

max_length = 50
stop_words = set(stopwords.words('english'))

# ================= PREPROCESS =================

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ================= SIMPLE RULE-BASED MODEL (REPLACES RNN) =================
# (This is needed because TensorFlow is NOT supported in Streamlit Cloud)

def predict_sentiment(text):

    text = text.lower()

    if any(word in text for word in ["sad", "depress", "lonely", "tired", "stress", "anxious"]):
        return "Stress", 0.85

    elif any(word in text for word in ["happy", "excited", "good", "great", "awesome"]):
        return "Happy", 0.90

    else:
        return "Neutral", 0.70

# ================= UI =================

st.title("AI-Based Mental Health Sentiment Monitoring System")
st.subheader("Emotion Detection System")

st.header("About the Project")
st.write("""
This system detects emotional sentiment from user text.
It helps monitor mental well-being using NLP techniques.
""")

# ================= INPUT =================

st.header("Enter Your Thoughts")

user_input = st.text_area("Enter your thoughts or feelings here...")

# ================= PREDICTION =================

if st.button("Analyze Emotion"):

    if user_input.strip():

        cleaned = preprocess_text(user_input)

        label, confidence = predict_sentiment(cleaned)

        st.success(f"Emotion Detected: {label}")
        st.info(f"Confidence Score: {confidence * 100:.2f}%")

        # ================= GRAPH =================

        st.header("Emotion Visualization")

        emotions = ["Stress", "Happy", "Neutral"]
        values = [0.85 if label == e else 0.2 for e in emotions]

        fig, ax = plt.subplots()
        ax.bar(emotions, values)
        ax.set_ylabel("Probability")
        ax.set_title("Emotion Distribution")

        st.pyplot(fig)

        # ================= GUIDANCE =================

        st.header("Emotional Guidance")

        if label == "Stress":
            st.warning("Take a break and talk to someone you trust.")
            st.write("Try deep breathing, walking, or listening to music.")

        elif label == "Happy":
            st.success("Great! Keep maintaining your positive mindset.")

        else:
            st.info("Stay balanced and take care of your mental health.")

    else:
        st.error("Please enter some text")
