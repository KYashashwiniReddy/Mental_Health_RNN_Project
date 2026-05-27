# ================= IMPORT LIBRARIES =================

import streamlit as st
import pickle
import numpy as np
import re
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ================= NLTK SETUP =================

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ================= LOAD MODEL & FILES =================

model = load_model("mental_health_rnn_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# ================= PARAMETERS =================

max_length = 50
stop_words = set(stopwords.words('english'))

# ================= PREPROCESS FUNCTION =================

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ================= UI =================

st.title("AI-Based Mental Health Sentiment Monitoring System")
st.subheader("Emotion Detection using Simple RNN")

st.header("About the Project")
st.write("""
This AI system detects emotional sentiment from text using NLP and RNN.
It helps understand mental health patterns and emotional states.
""")

# ================= INPUT =================

st.header("Enter Your Thoughts")

user_input = st.text_area("Enter your thoughts or feelings here...")

# ================= PREDICTION =================

if st.button("Analyze Emotion"):

    if user_input.strip():

        # preprocess
        cleaned = preprocess_text(user_input)

        # tokenize
        seq = tokenizer.texts_to_sequences([cleaned])

        # padding
        padded = pad_sequences(seq, maxlen=max_length, padding='post')

        # prediction
        pred = model.predict(padded)

        idx = np.argmax(pred)
        label = encoder.inverse_transform([idx])[0]
        confidence = np.max(pred) * 100

        # ================= OUTPUT =================

        st.success(f"Emotion Detected: {label}")
        st.info(f"Confidence: {confidence:.2f}%")

        # ================= GRAPH =================

        st.header("Prediction Probabilities")

        fig, ax = plt.subplots()
        ax.bar(encoder.classes_, pred[0])
        ax.set_ylabel("Probability")
        ax.set_xlabel("Emotion")
        ax.set_title("Emotion Distribution")
        plt.xticks(rotation=45)

        st.pyplot(fig)

        # ================= GUIDANCE =================

        st.header("Emotional Guidance")

        if label.lower() in ["stress", "anxiety", "depression", "sadness"]:

            st.warning("Take a break and talk to someone you trust.")
            st.write("Try breathing exercises, walk, or music therapy.")

        else:

            st.success("You seem emotionally stable. Keep it up!")
            st.write("Keep doing activities that make you happy.")

    else:
        st.error("Please enter some text")
