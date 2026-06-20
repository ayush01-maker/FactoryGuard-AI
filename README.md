# 🏭 FactoryGuard AI

## IoT Predictive Maintenance Engine using Machine Learning

FactoryGuard AI is an intelligent predictive maintenance system designed to identify potential machine failures before they occur. By analyzing real-time sensor data, the system enables proactive maintenance, reduces unplanned downtime, and improves operational efficiency in manufacturing environments.

---

## 👥 Team Members

* Ayush Bhaskar
* Ansha T V
* Mande Indu

---

## 📌 Project Overview

Modern manufacturing plants rely heavily on machine health monitoring to prevent costly equipment failures.

FactoryGuard AI leverages Machine Learning and IoT sensor data to predict machine failures in advance using operational parameters such as:

* Air Temperature
* Process Temperature
* Rotational Speed
* Torque
* Tool Wear

The system provides early failure warnings and supports maintenance teams in making informed decisions before catastrophic breakdowns occur.

---

## 🎯 Project Objective

To predict machine failures up to **24 hours before occurrence**, allowing scheduled preventive maintenance and minimizing production downtime.

---

## ✨ Key Features

### Data Preprocessing

* Dataset cleaning and preparation
* Removal of irrelevant attributes
* Handling missing values
* Feature standardization

### Feature Engineering

* Rolling Mean Features
* Rolling Standard Deviation Features
* Exponential Moving Average (EMA)
* Lag Features
* Time-Series Pattern Extraction

### Machine Learning Models

* Random Forest (Baseline Model)
* XGBoost (Production Model)

### Failure Prediction

* Real-time machine health assessment
* Failure probability estimation
* Risk categorization

### Explainable AI

* SHAP (SHapley Additive exPlanations)
* Feature Importance Analysis
* Prediction Interpretability

### Interactive Dashboard

* Streamlit-based web application
* Sensor monitoring interface
* Risk visualization
* Explainability reports

---

## 🛠 Technology Stack

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-Learn
* XGBoost
* Imbalanced-Learn (SMOTE)

### Explainability

* SHAP

### Visualization

* Matplotlib
* Plotly

### Deployment

* Streamlit

---

## 📂 Project Structure

```text
FactoryGuard-AI/
│
├── Dataset/
│   └── ai4i2020.csv
│
├── models/
│   ├── xgboost_model.pkl
│   ├── random_forest.pkl
│   ├── label_encoder.pkl
│   └── feature_columns.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
│   └── shap_analysis.py
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
│
└── README.md
---
