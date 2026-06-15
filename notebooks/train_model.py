import os
import re
import joblib
import pandas as pd

from preprocessing import load_data
from feature_engineering import create_features

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    average_precision_score
)

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = load_data("Dataset/ai4i2020.csv")

print("Original Shape:", df.shape)

# =====================================================
# FEATURE ENGINEERING
# =====================================================

print("\nCreating Features...")

df = create_features(df)

print("Feature Engineered Shape:", df.shape)

# =====================================================
# ENCODE TYPE COLUMN
# =====================================================

encoder = LabelEncoder()

df["Type"] = encoder.fit_transform(df["Type"])

# =====================================================
# FEATURES AND TARGET
# =====================================================

TARGET = "Machine failure"

X = df.drop(TARGET, axis=1)
y = df[TARGET]
print("NaN Before Cleaning:", X.isna().sum().sum())

X = X.fillna(0)

print("NaN After Cleaning:", X.isna().sum().sum())

# Clean feature names for XGBoost
X.columns = [
    re.sub(r"[^A-Za-z0-9_]", "_", str(col))
    for col in X.columns
]

# Remove duplicate columns if any
X = X.loc[:, ~X.columns.duplicated()]

print("\nTotal Features:", X.shape[1])

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples :", X_test.shape[0])

# =====================================================
# HANDLE CLASS IMBALANCE USING SMOTE
# =====================================================
print("\nMissing Values:")
print(X.isna().sum().sum())

print("\nApplying SMOTE...")

smote = SMOTE(
    sampling_strategy="auto",
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# =====================================================
# RANDOM FOREST BASELINE
# =====================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train_smote,
    y_train_smote
)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Results")

print("Accuracy :", round(
    accuracy_score(y_test, rf_pred), 4))

print("Precision:", round(
    precision_score(y_test, rf_pred), 4))

print("Recall   :", round(
    recall_score(y_test, rf_pred), 4))

print("F1 Score :", round(
    f1_score(y_test, rf_pred), 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    rf_pred
))

# =====================================================
# XGBOOST PRODUCTION MODEL
# =====================================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print(
    f"scale_pos_weight = {scale_pos_weight:.2f}"
)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

# Use NumPy arrays to avoid feature name issues
xgb_model.fit(
    X_train_smote.values,
    y_train_smote.values
)

xgb_pred = xgb_model.predict(
    X_test.values
)

xgb_prob = xgb_model.predict_proba(
    X_test.values
)[:, 1]

pr_auc = average_precision_score(
    y_test,
    xgb_prob
)

print("\nXGBoost Results")

print("Accuracy :", round(
    accuracy_score(y_test, xgb_pred), 4))

print("Precision:", round(
    precision_score(y_test, xgb_pred), 4))

print("Recall   :", round(
    recall_score(y_test, xgb_pred), 4))

print("F1 Score :", round(
    f1_score(y_test, xgb_pred), 4))

print("PR-AUC   :", round(
    pr_auc, 4))

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        xgb_pred
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        xgb_pred
    )
)

# =====================================================
# SAVE MODELS
# =====================================================

print("\nSaving Models...")

joblib.dump(
    rf_model,
    os.path.join(
        MODEL_DIR,
        "random_forest.pkl"
    )
)

joblib.dump(
    xgb_model,
    os.path.join(
        MODEL_DIR,
        "xgboost_model.pkl"
    )
)

joblib.dump(
    encoder,
    os.path.join(
        MODEL_DIR,
        "label_encoder.pkl"
    )
)

joblib.dump(
    list(X.columns),
    os.path.join(
        MODEL_DIR,
        "feature_columns.pkl"
    )
)

print("\nModels Saved Successfully!")

print("\nSaved Files:")
print("random_forest.pkl")
print("xgboost_model.pkl")
print("label_encoder.pkl")
print("feature_columns.pkl")

print("\nLocation:")
print(MODEL_DIR)