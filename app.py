import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("creditcard.csv")
    scaler = StandardScaler()
    data['normAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1,1))
    
    # Drop safely (only if exists)
    for col in ['Time', 'Amount']:
        if col in data.columns:
            data = data.drop(col, axis=1)
    return data

data = load_data()

# -------------------------------
# Train Model
# -------------------------------
@st.cache_data
def train_model():
    X = data.drop('Class', axis=1)
    y = data['Class']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

model = train_model()

# -------------------------------
# App Layout
# -------------------------------
st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("💳 Credit Card Fraud Detection")
st.write("This app predicts whether a transaction is **Legitimate** or **Fraudulent**.")

choice = st.radio("Choose input method:", ["Manual Input", "Upload CSV", "View Dataset & Charts"])

# -------------------------------
# Manual Input
# -------------------------------
if choice == "Manual Input":
    st.subheader("Enter Transaction Details")
    input_data = {}
    for col in data.drop('Class', axis=1).columns:
        input_data[col] = st.number_input(col, value=0.0)
    
    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.write("### Prediction:", "🚨 Fraudulent" if prediction==1 else "✅ Legitimate")

# -------------------------------
# CSV Upload
# -------------------------------
elif choice == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file with transaction data", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Add normalized Amount if column exists
        if 'Amount' in df.columns:
            df['normAmount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
        
        # Drop columns safely
        for col in ['Time', 'Amount']:
            if col in df.columns:
                df = df.drop(col, axis=1)

        # Drop target variable if exists
        if 'Class' in df.columns:
            df = df.drop('Class', axis=1)

        # Predictions
        predictions = model.predict(df)
        df['Prediction'] = ["Fraudulent" if p==1 else "Legitimate" for p in predictions]
        
        st.write("### Predictions")
        st.dataframe(df)

        # Show chart
        st.subheader("Fraud vs Legitimate Transactions")
        chart_data = df['Prediction'].value_counts()
        st.bar_chart(chart_data)

# -------------------------------
# View Dataset & Charts
# -------------------------------
elif choice == "View Dataset & Charts":
    st.subheader("Dataset Preview")
    st.dataframe(data.head(100))

    st.subheader("Fraudulent vs Legitimate Transactions in Dataset")
    chart_data = data['Class'].map({0:'Legitimate', 1:'Fraudulent'}).value_counts()
    st.bar_chart(chart_data)

    st.subheader("Fraud Distribution Pie Chart")
    fig, ax = plt.subplots()
    ax.pie(chart_data, labels=chart_data.index, autopct='%1.1f%%', colors=['#4CAF50', '#FF5252'])
    st.pyplot(fig)

