# app.py
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

# Page config
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

# Load pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Connect DB
conn = sqlite3.connect("sentiment_results.db", check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS results (username TEXT, text TEXT, language TEXT, sentiment TEXT, score REAL)''')

# Auth utils
def make_hashes(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_hashes(password, hashed):
    return make_hashes(password) == hashed

def add_user(username, password):
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, make_hashes(password)))
    conn.commit()

def login_user(username, password):
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, make_hashes(password)))
    return c.fetchall()

# NLP utils
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
    emoji = "😊" if result['label'] == "POSITIVE" else ("😠" if result['label'] == "NEGATIVE" else "😐")
    return f"{emoji} {result['label'].title()}", result['score']

def create_wordcloud(text):
    return WordCloud(width=700, height=400, background_color='white').generate(text)

def store_result(username, text, lang, sentiment, score):
    c.execute("INSERT INTO results (username, text, language, sentiment, score) VALUES (?, ?, ?, ?, ?)", (username, text, lang, sentiment, score))
    conn.commit()

# Login UI
st.sidebar.title("🔐 Login Area")
menu = ["Login", "SignUp"]
choice = st.sidebar.selectbox("Select Action", menu)

if choice == "SignUp":
    new_user = st.sidebar.text_input("Create Username")
    new_pass = st.sidebar.text_input("Create Password", type='password')
    if st.sidebar.button("Create Account"):
        add_user(new_user, new_pass)
        st.sidebar.success("Account created! Please log in.")

elif choice == "Login":
    user = st.sidebar.text_input("Username")
    passwd = st.sidebar.text_input("Password", type='password')
    if st.sidebar.button("Login"):
        result = login_user(user, passwd)
        if result:
            st.success(f"Welcome {user} 👋")
            st.title("📊 Advanced Sentiment Analyzer")

            mode = st.sidebar.radio("Choose Mode", ["Text", "CSV Upload", "Speech Input", "Dashboard"])
            show_wc = st.sidebar.checkbox("Show Word Cloud", True)

            if mode == "Text":
                user_input = st.text_area("Enter your text")
                if st.button("Analyze"):
                    lang = detect_language(user_input)
                    translated = translate_to_english(user_input)
                    sentiment, score = analyze_sentiment(translated)
                    st.metric("Sentiment", sentiment)
                    st.progress(score)
                    store_result(user, user_input, lang, sentiment, score)
                    if show_wc:
                        st.image(create_wordcloud(user_input).to_array())

            elif mode == "CSV Upload":
                file = st.file_uploader("Upload CSV with 'text' column", type="csv")
                if file:
                    df = pd.read_csv(file)
                    if 'text' in df.columns:
                        df['Lang'] = df['text'].apply(detect_language)
                        df['Translated'] = df['text'].apply(translate_to_english)
                        df['Sentiment'], df['Score'] = zip(*df['Translated'].apply(analyze_sentiment))
                        for _, r in df.iterrows():
                            store_result(user, r['text'], r['Lang'], r['Sentiment'], r['Score'])
                        st.dataframe(df)
                        st.bar_chart(df['Sentiment'].value_counts())
                        if show_wc:
                            st.image(create_wordcloud(' '.join(df['text'])).to_array())
                    else:
                        st.error("CSV must contain 'text' column")

            elif mode == "Speech Input":
                audio = st.file_uploader("Upload Audio (wav/mp3)", type=["wav", "mp3"])
                if audio:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp.write(audio.read())
                        recognizer = sr.Recognizer()
                        with sr.AudioFile(tmp.name) as src:
                            audio_data = recognizer.record(src)
                            try:
                                text = recognizer.recognize_google(audio_data)
                                st.write("🎙 Transcribed:", text)
                                translated = translate_to_english(text)
                                sentiment, score = analyze_sentiment(translated)
                                st.metric("Sentiment", sentiment)
                                st.progress(score)
                                store_result(user, text, detect_language(text), sentiment, score)
                                if show_wc:
                                    st.image(create_wordcloud(text).to_array())
                            except:
                                st.error("Could not recognize speech")

            elif mode == "Dashboard":
                st.subheader("📈 Sentiment History")
                c.execute("SELECT text, language, sentiment, score FROM results WHERE username = ?", (user,))
                data = c.fetchall()
                if data:
                    df = pd.DataFrame(data, columns=["Text", "Language", "Sentiment", "Score"])
                    st.dataframe(df)
                    st.bar_chart(df['Sentiment'].value_counts())
                    st.line_chart(df['Score'])
                else:
                    st.info("No data found for your account.")

        else:
            st.error("Invalid credentials. Try again.")

