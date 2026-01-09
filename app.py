import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Sentiment Classification App")

text = st.text_area("Text input:")

if st.button("Predict"):
    X_input = vectorizer.transform([text])
    prediction = model.predict(X_input)[0]
    st.write("Predicted Result:", prediction)
