# 📊 Customer Purchase Prediction & Segmentation Platform

An end-to-end Data Science + Machine Learning + Analytics project that predicts customer purchase behavior, performs customer segmentation using RFM analysis, and provides interactive business insights via Streamlit dashboard.

---

## 🚀 Project Overview

This project helps businesses:

- Predict whether a customer will make a purchase  
- Identify high-value customers (Top 20%)  
- Segment customers using RFM (Recency, Frequency, Monetary)  
- Analyze customer behavior through EDA & visualizations  
- Provide product insights & recommendations  

---

## 🧠 Key Features

### 🔹 1. Purchase Prediction (Machine Learning)
- Logistic Regression  
- Random Forest (Tuned + Calibrated)  
- XGBoost (Tuned + Calibrated)  
- Hyperparameter tuning using RandomizedSearchCV  
- Evaluation using Accuracy, ROC-AUC, Classification Report  

### 🔹 2. Customer Scoring
- Predicts purchase probability  
- Ranks customers  
- Identifies Top 20% customers  

### 🔹 3. RFM Analysis
- Recency, Frequency, Monetary scoring  
- KMeans clustering  

### 🔹 4. Recommendations
- Popular products  
- Customer purchase history  
- Spending insights  

### 🔹 5. Streamlit Dashboard
- Prediction  
- Customer Scoring  
- RFM Segmentation  
- Recommendations  
- EDA & Visualizations  
- Feature Importance  

### 🔹 6. Data Cleaning
- Removes duplicates  
- Handles missing values  
- Standardizes data  

---

## 🏗️ Project Structure

```
project/
├── data/
├── code/
├── app/
└── README.md
```

---

## ⚙️ Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit

---

## ▶️ How to Run

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
python code/train_and_rfm.py
streamlit run app/app_vis_added.py
```

---

## 👨‍💻 Author

Lingesh Kabila Kumar
