# Advanced Sentiment Analyzer - Streamlit Deployable Version (Fixed Session Bug)

import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import pandas as pd
from io import StringIO
from transformers import pipeline
from deep_translator import GoogleTranslator
import speech_recognition as sr
import tempfile
import sqlite3
import hashlib
import os

# Configure Streamlit page
st.set_page_config(page_title="Advanced Sentiment Analyzer", layout="wide")

# Load AI Model
sentiment_pipeline = pipeline("sentiment-analysis")

# Connect to SQLite database
DB_FILE = "sentiment_results.db"
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS results (username TEXT, text TEXT, language TEXT, sentiment TEXT, score REAL)''')

# Utility Functions
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed):
    return make_hashes(password) == hashed

def add_user(username, password):
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, make_hashes(password)))
    conn.commit()

def login_user(username, password):
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, make_hashes(password)))
    return c.fetchall()

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']
    score = result['score']
    emoji = "😊" if label == "POSITIVE" else ("😠" if label == "NEGATIVE" else "😐")
    return f"{emoji} {label.title()}", score

def create_wordcloud(text):
    wc = WordCloud(width=700, height=400, background_color='white').generate(text)
    return wc

def store_result(username, text, lang, sentiment, score):
    c.execute("INSERT INTO results (username, text, language, sentiment, score) VALUES (?, ?, ?, ?, ?)", (username, text, lang, sentiment, score))
    conn.commit()

# Auth UI
st.sidebar.title("Login Area")
menu = ["Login", "SignUp"]
choice = st.sidebar.selectbox("Select Action", menu)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

if choice == "SignUp":
    new_user = st.sidebar.text_input("Create Username")
    new_password = st.sidebar.text_input("Create Password", type='password')
    if st.sidebar.button("Create Account"):
        add_user(new_user, new_password)
        st.sidebar.success("Account created. Login now.")

elif choice == "Login":
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type='password')
    if st.sidebar.button("Login"):
        result = login_user(username, password)
        if result:
            st.session_state.logged_in = True
            st.session_state.username = username
        else:
            st.error("Invalid Credentials. Try Again.")

# Post-login app features
if st.session_state.logged_in:
    username = st.session_state.username
    st.success(f"Welcome {username} 👋")
    st.title("Advanced Sentiment Analyzer")
    st.caption("Multi-language • Deep AI Analysis • Audio Input • Dashboard")

    st.sidebar.header("Choose Mode")
    mode = st.sidebar.radio("Input Method", ["Text", "Upload CSV", "Speech Input", "Dashboard"])
    show_wc = st.sidebar.checkbox("Show Word Cloud", True)

    if mode == "Text":
        user_text = st.text_area("Enter text for sentiment analysis")
        if st.button("Analyze Text"):
            if user_text.strip() != "":
                lang = detect_language(user_text)
                translated = translate_to_english(user_text)
                sentiment, score = analyze_sentiment(translated)
                st.metric("Sentiment", sentiment)
                st.progress(score)
                store_result(username, user_text, lang, sentiment, score)
                if show_wc:
                    st.image(create_wordcloud(user_text).to_array())
            else:
                st.warning("Please enter some text to analyze.")

    elif mode == "Upload CSV":
        file = st.file_uploader("Upload CSV with 'text' column", type="csv")
        if file:
            df = pd.read_csv(file)
            if 'text' in df.columns:
                df['Language'] = df['text'].apply(detect_language)
                df['Translated'] = df['text'].apply(translate_to_english)
                df['Sentiment'], df['Score'] = zip(*df['Translated'].apply(analyze_sentiment))

                for _, row in df.iterrows():
                    store_result(username, row['text'], row['Language'], row['Sentiment'], row['Score'])

                st.dataframe(df[['text', 'Language', 'Sentiment', 'Score']])
                st.bar_chart(df['Sentiment'].value_counts())
                if show_wc:
                    st.image(create_wordcloud(' '.join(df['text'])).to_array())
                st.download_button("Download Results", df.to_csv(index=False), "results.csv")
            else:
                st.error("CSV must have 'text' column")

    elif mode == "Speech Input":
        audio = st.file_uploader("Upload audio (wav/mp3)", type=["wav", "mp3"])
        if audio:
            recognizer = sr.Recognizer()
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(audio.read())
                with sr.AudioFile(tmp.name) as source:
                    audio_data = recognizer.record(source)
                    try:
                        text = recognizer.recognize_google(audio_data)
                        st.success("Transcribed Text:")
                        st.write(text)
                        translated = translate_to_english(text)
                        sentiment, score = analyze_sentiment(translated)
                        st.metric("Sentiment", sentiment)
                        st.progress(score)
                        store_result(username, text, detect_language(text), sentiment, score)
                        if show_wc:
                            st.image(create_wordcloud(text).to_array())
                    except:
                        st.error("Could not process audio")

    elif mode == "Dashboard":
        st.subheader("Sentiment Analysis History")
        c.execute("SELECT text, language, sentiment, score FROM results WHERE username = ?", (username,))
        data = c.fetchall()
        if data:
            df = pd.DataFrame(data, columns=["Text", "Language", "Sentiment", "Score"])
            st.dataframe(df)
            st.bar_chart(df['Sentiment'].value_counts())
            st.line_chart(df['Score'])
        else:
            st.info("No sentiment data found.")

else:
    st.info("Please log in from the sidebar to continue.")
