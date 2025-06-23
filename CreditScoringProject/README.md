# 💼 Credit Risk Evaluation App

A professional and interactive Machine Learning web application to predict **creditworthiness** of loan applicants using logistic regression and random forest classifiers. Built with **Streamlit**, it features a clean UI with glassmorphism, animations, tab-based navigation, and real-time predictions.

---

## 🚀 Features

- 🔍 Predict credit risk using Logistic Regression or Random Forest
- 📈 Simulate credit score out of 850
- 🧊 Glassmorphism UI with animated score meter
- 🧠 Trained on the **German Credit Dataset**
- 💬 Clean, interactive form with instant results
- 🎯 Dark Mode enabled for modern UX


---

## 📊 Machine Learning Models Used

- **Logistic Regression**
- **Random Forest Classifier**

Both trained on preprocessed and scaled features from the German Credit dataset using `scikit-learn`.

---

## 📁 Folder Structure

📦 Credit-Risk-App
┣ 📜 app.py
┣ 📜 german_credit_full.csv
┣ 📜 README.md
┗ 📦 .streamlit
┗ 📜 config.toml


---

## 🧪 How to Run

1. **Clone the repo**:
   
   git clone https://github.com/your-username/credit-risk-app.git
   cd credit-risk-app

2. **Install dependencies:**:

pip install -r requirements.txt

3. **Run the app:**:

streamlit run app.py

📦 Requirements

1.pandas

2.numpy

3.scikit-learn

4.streamlit

5.plotly

You can install all with:

pip install pandas numpy scikit-learn streamlit plotly

📚 Dataset
Dataset used: German Credit Risk Dataset

Attributes: Age, Sex, Job, Housing, Saving/Checking accounts, Credit Amount, Duration, Purpose

Label: Risk (Good / Bad)

💡 Credits & Acknowledgements
Built during ML Internship (2025)

UI Design: Inspired by fintech dashboard UIs

Dataset Source: UCI Repository - German Credit Data

🔐 Disclaimer
This project is for educational/demo purposes and should not be used for real-world credit decisions.

