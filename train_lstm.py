import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from joblib import dump

# ----------------------------
# Đọc và xử lý dữ liệu
# ----------------------------

print(" Đọc dữ liệu")

DATA_PATH = "./data/train_df.csv"
data = pd.read_csv(DATA_PATH)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# đặt theo key trong dataset
# train_df có key là text
# IMDB_Dataset có key là review
data["clean_text"] = data["text"].apply(clean_text)

# Chuẩn hóa nhãn: negative=0, neutral=1, positive=2
label_map = {"negative": 0, "neutral": 1, "positive": 2}
data["label_id"] = data["sentiment"].map(label_map)

# ----------------------------
# 2. Chia dữ liệu train/test
# ----------------------------

X = data["clean_text"].values
y = data["label_id"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 3. Tokenizer + Sequence
# ----------------------------

print(" tạo tokenizer")

MAX_WORDS = 20000
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<UNK>")
tokenizer.fit_on_texts(X_train)

dump(tokenizer, "tokenizer.joblib")
print("Tokenizer đã được lưu!")

# Convert text -> sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding
MAX_LEN = 200  # độ dài tối ưu cho review IMDB
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)

# One-hot labels
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# ----------------------------
# 4. Xây dựng mô hình LSTM
# ----------------------------

print(" Đang xây dựng mô hình LSTM")

model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),

    LSTM(128, return_sequences=False),
    Dropout(0.5),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(3, activation="softmax")  # 3 labels
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
# 5. Train model
# ----------------------------

print(" Bắt đầu huấn luyện")

es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(
    X_train_pad, y_train_cat,
    validation_split=0.1,
    epochs=10,
    batch_size=64,
    callbacks=[es]
)

# ----------------------------
# 6. Evaluate
# ----------------------------

loss, acc = model.evaluate(X_test_pad, y_test_cat)
print(f"\n Accuracy trên test: {acc:.4f}")

# ----------------------------
# 7. Lưu mô hình
# ----------------------------

model.save("lstm_sentiment_model.keras")
print("Đã lưu")
