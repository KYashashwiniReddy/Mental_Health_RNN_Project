import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
MAX_LEN = 100

# ================= THEME STYLE =================
st.set_page_config(
    page_title="Mental Health AI",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}

h1, h2, h3 {
    color: #38bdf8;
}

.stButton>button {
    background: linear-gradient(90deg,#06b6d4,#3b82f6);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-size: 16px;
}

.stTextArea textarea {
    border-radius: 10px;
    background-color: #1e293b;
    color: white;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #1e293b;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================

@st.cache_resource
def load_resources():
    model = load_model("mental_health_rnn_model.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model, tokenizer, encoder

model, tokenizer, encoder = load_resources()

# ================= PREPROCESS =================

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# ================= PREDICT =================

def predict_sentiment(text):
    processed = preprocess_text(text)

    seq = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    pred = model.predict(padded, verbose=0)[0]

    idx = np.argmax(pred)
    label = encoder.inverse_transform([idx])[0]
    confidence = pred[idx]

    probs = dict(zip(encoder.classes_, pred))

    return label, confidence, probs, processed

# ================= SIDEBAR =================

st.sidebar.title("🧠 Mental Health AI")
menu = st.sidebar.radio("Navigate", ["🏠 Home", "ℹ About", "🔍 Prediction"])

# ================= HOME =================

if menu == "🏠 Home":

    st.markdown("<h1>AI Mental Health Monitoring System</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    Detect emotions from text using AI-powered sentiment analysis.
    Get insights, confidence scores, and wellness suggestions.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("AI Model", "RNN")
    col2.metric("Input Type", "Text")
    col3.metric("Output", "Emotion")

# ================= ABOUT =================

if menu == "ℹ About":

    st.markdown("<h2>About This Project</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <b>Emotional AI:</b> Detects human emotions from text<br><br>

    <b>NLP Uses:</b> Mental health tracking, sentiment analysis<br><br>

    <b>RNN Role:</b> Learns sequence patterns in sentences
    </div>
    """, unsafe_allow_html=True)

# ================= PREDICTION =================

if menu == "🔍 Prediction":

    st.markdown("<h2>Emotion Detection</h2>", unsafe_allow_html=True)

    st.markdown("### Enter your thoughts")

    user_input = st.text_area("Write here...", height=150)

    if st.button("Analyze Emotion 🚀"):

        if user_input.strip() == "":
            st.warning("Please enter text")
        else:

            emotion, confidence, probs, processed = predict_sentiment(user_input)

            st.markdown("### Result")

            st.success(f"Emotion: {emotion}")

            st.metric("Confidence", f"{confidence*100:.2f}%")

            st.markdown("### AI Suggestion")

            st.info("Take care of your mental well-being ❤️")

            # ================= CHART =================

            df = pd.DataFrame({
                "Emotion": list(probs.keys()),
                "Probability": list(probs.values())
            })

            fig = px.bar(
                df,
                x="Emotion",
                y="Probability",
                color="Probability",
                color_continuous_scale="blues"
            )

            st.plotly_chart(fig, use_container_width=True)

# ================= FOOTER =================

st.sidebar.markdown("---")
st.sidebar.write("Built with ❤️ using Streamlit + AI")
