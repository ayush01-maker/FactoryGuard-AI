import os
import re
import joblib
import shap
import matplotlib.pyplot as plt

from preprocessing import load_data
from feature_engineering import create_features

# ==================================
# Paths
# ==================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

# ==================================
# Load Model & Artifacts
# ==================================

model = joblib.load(
    os.path.join(MODEL_DIR, "xgboost_model.pkl")
)

encoder = joblib.load(
    os.path.join(MODEL_DIR, "label_encoder.pkl")
)

feature_columns = joblib.load(
    os.path.join(MODEL_DIR, "feature_columns.pkl")
)

# ==================================
# Load & Process Data
# ==================================

df = load_data("Dataset/ai4i2020.csv")

df = create_features(df)

# Encode Type
df["Type"] = encoder.transform(df["Type"])

# Features only
X = df.drop("Machine failure", axis=1)

# Clean feature names exactly as in training
X.columns = [
    re.sub(r"[^A-Za-z0-9_]", "_", str(col))
    for col in X.columns
]

# Remove duplicates
X = X.loc[:, ~X.columns.duplicated()]

# Match training feature order
X = X.reindex(
    columns=feature_columns,
    fill_value=0
)

print("Feature Count:", X.shape[1])

# ==================================
# SHAP Analysis
# ==================================

explainer = shap.TreeExplainer(model)

# Use a subset for faster execution
sample_X = X.sample(
    min(500, len(X)),
    random_state=42
)

shap_values = explainer.shap_values(
    sample_X.values
)

# ==================================
# SHAP Summary Plot
# ==================================

plt.figure(figsize=(10, 6))

shap.summary_plot(
    shap_values,
    sample_X,
    show=False
)

plt.savefig(
    "shap_summary.png",
    bbox_inches="tight"
)

print("SHAP summary saved as shap_summary.png")

# ==================================
# SHAP Bar Plot
# ==================================

plt.figure(figsize=(10, 6))

shap.summary_plot(
    shap_values,
    sample_X,
    plot_type="bar",
    show=False
)

plt.savefig(
    "shap_feature_importance.png",
    bbox_inches="tight"
)

print("SHAP feature importance saved as shap_feature_importance.png")