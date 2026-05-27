# app.py

# ================= IMPORT LIBRARIES =================

import streamlit as st
import pickle
import numpy as np
import re
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# ================= DOWNLOAD NLTK FILES =================

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ================= LOAD MODEL =================

model = load_model("mental_health_rnn_model.h5")

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

# ================= PARAMETERS =================

max_length = 50

# ================= TEXT PREPROCESSING =================

stop_words = set(stopwords.words('english'))

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text)

    tokens = [word for word in tokens if word not in stop_words]

    text = " ".join(tokens)

    return text

# ================= STREAMLIT UI =================

st.title("AI-Based Mental Health Sentiment Monitoring System")

st.subheader(
    "Emotion Detection using Simple Recurrent Neural Networks"
)

# ================= ABOUT PROJECT =================

st.header("About the Project")

st.write("""
This application uses Artificial Intelligence and Natural Language Processing (NLP)
to analyze emotional sentiment from user text.

Emotional AI helps identify emotional patterns, monitor mental well-being,
and support early emotional intervention.

The Simple RNN model learns text sequentially and remembers previous words
through hidden states, helping it understand emotional context.
""")

# ================= USER INPUT =================

st.header("Enter Your Thoughts")

st.write("Sample Sentences:")
st.write("- I feel stressed and anxious today")
st.write("- I am feeling very happy and excited")
st.write("- I feel lonely and tired")

user_input = st.text_area(
    "Enter your thoughts or feelings here..."
)

# ================= PREDICTION =================

if st.button("Analyze Emotion"):

    if user_input.strip() != "":

        cleaned_text = preprocess_text(user_input)

        sequence = tokenizer.texts_to_sequences([cleaned_text])

        padded_sequence = pad_sequences(
            sequence,
            maxlen=max_length,
            padding='post'
        )

        prediction = model.predict(padded_sequence)

        predicted_index = np.argmax(prediction)

        predicted_sentiment = encoder.inverse_transform(
            [predicted_index]
        )[0]

        confidence = np.max(prediction) * 100

        # ================= OUTPUT =================

        st.header("Prediction Result")

        st.success(
            f"Emotion Detected: {predicted_sentiment}"
        )

        st.info(
            f"Confidence Score: {confidence:.2f}%"
        )

        # ================= VISUALIZATION =================

        st.header("Sentiment Confidence Graph")

        labels = encoder.classes_

        probabilities = prediction[0]

        fig, ax = plt.subplots()

        ax.bar(labels, probabilities)

        ax.set_xlabel("Sentiment")

        ax.set_ylabel("Probability")

        ax.set_title("Emotion Probability Distribution")

        plt.xticks(rotation=45)

        st.pyplot(fig)

        # ================= EMOTIONAL GUIDANCE =================

        st.header("Emotional Wellness Guidance")

        if predicted_sentiment.lower() in [
            "anxiety",
            "depression",
            "stress",
            "sadness"
        ]:

            st.warning(
                "Take a short break and talk with someone you trust."
            )

            st.write(
                "Try deep breathing, light exercise, journaling, or listening to calming music."
            )

        else:

            st.success(
                "Keep maintaining your positive mindset and healthy routine."
            )

            st.write(
                "Continue activities that make you feel happy and motivated."
            )

    else:

        st.error("Please enter some text.")