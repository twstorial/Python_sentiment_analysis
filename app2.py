
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# 1 Load mô hình và tokenizer

MODEL_PATH = "./lstm_sentiment_model.keras"
TOKENIZER_PATH = "./tokenizer.joblib"
MAX_LEN = 200  # phải trùng với file train model

@st.cache_resource  # Cache để load nhanh hơn
def load_lstm_model():
    model = load_model(MODEL_PATH)
    return model

@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        tok = pickle.load(f)
    return tok


model = load_lstm_model()
tokenizer = load_tokenizer()

# 2 Hàm tiền xử lý văn bản

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  
    text = re.sub(r"[^a-z\s]", "", text)                 
    text = re.sub(r"\s+", " ", text).strip()             
    return text

# 3  Streamlit

st.title(" Sentiment Analysis – LSTM Model")
st.write("Nhập vào đoạn text và mô hình sẽ dự đoán cảm xúc.")

user_input = st.text_area("Nhập văn bản cần phân tích:", height=200)

if st.button("Phân tích cảm xúc"):
    if user_input.strip() == "":
        st.warning("Vui lòng nhập nội dung!")
    else:
        clean = clean_text(user_input)

        seq = tokenizer.texts_to_sequences([clean])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

        pred = model.predict(padded)[0]

        classes = ["negative", "neutral", "positive"]
        label = classes[np.argmax(pred)]

        st.subheader("Kết quả dự đoán")
        st.write(f"**Dự đoán:** `{label}`")

        st.write("### Xác suất")
        st.json({
            "negative": float(pred[0]),
            "neutral": float(pred[1]),
            "positive": float(pred[2]),
        })

        st.write("###  Văn bản đã tiền xử lý")
        st.code(clean)
