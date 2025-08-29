# fraud-detection-streamlit
A Credit Card Fraud Detection App using Python, Machine Learning, and Streamlit.

# 💳 Credit Card Fraud Detection using Machine Learning & Streamlit

## 📌 Project Overview
This project is a **Credit Card Fraud Detection Web App** built using **Python, Machine Learning, and Streamlit**.  
It predicts whether a transaction is **Legitimate** or **Fraudulent** based on historical transaction data.

## 🗂️ Dataset
- Dataset: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- Features: Transaction details (V1–V28), normalized amount  
- Target: `Class` (0 = Legitimate, 1 = Fraudulent)

## ⚙️ Tech Stack
- Python 🐍
- Scikit-learn 🤖
- Streamlit 🎨
- Pandas & NumPy 📊
- Matplotlib 📈

## 🚀 Features
- **Manual Input**: Enter transaction values to predict fraud.  
- **CSV Upload**: Upload transaction data and get fraud predictions.  
- **Dataset Visualization**: View fraud distribution and charts.  

## 🖥️ How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/Pavan-Kumar-IT/fraud-detection-streamlit.git
   cd fraud-detection-streamlit
##Install dependencies:
bash
pip install -r requirements.txt

Run the Streamlit app:
bash 
streamlit run app.py

Open the link shown in the terminal (e.g., http://localhost:8501)
