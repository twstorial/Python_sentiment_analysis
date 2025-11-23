# Sentiment Analysis App

Tạo virtual environment tên "venv" trên máy local:
```bash
    python3 -m venv venv
```
Activate môi trường ảo venv này:
```bash
    venv\Scripts\activate.bat
```
install thêm các module vào folder venv này dựa trên file requirements.txt:
```bash
    pip install -r requirements.txt
```
Train mô hình:
```bash
    python train_lstm.py
```
train_lstm.py sẽ tạo file tokenizer là tokenizer.joblib và file mô hình lstm_sentiment_model.keras  
Chạy app bằng streamlit:
```bash
    streamlit run app2.py
```
