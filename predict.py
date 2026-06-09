import os
import sys
import joblib
import pandas as pd
import re

# ==========================================
# FIX IMPORT PATH
# ==========================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from feature_engineering import create_features

# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(CURRENT_DIR)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

# ==========================================
# LOAD MODEL FILES
# ==========================================

model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "xgboost_model.pkl"
    )
)

encoder = joblib.load(
    os.path.join(
        MODEL_DIR,
        "label_encoder.pkl"
    )
)

feature_columns = joblib.load(
    os.path.join(
        MODEL_DIR,
        "feature_columns.pkl"
    )
)

# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_failure(
    machine_type,
    air_temp,
    process_temp,
    rpm,
    torque,
    tool_wear
):

    encoded_type = encoder.transform(
        [machine_type]
    )[0]

    # Create small dataframe
    df = pd.DataFrame({
        "Type": [encoded_type] * 12,
        "Air temperature [K]": [air_temp] * 12,
        "Process temperature [K]": [process_temp] * 12,
        "Rotational speed [rpm]": [rpm] * 12,
        "Torque [Nm]": [torque] * 12,
        "Tool wear [min]": [tool_wear] * 12
    })

    # Apply feature engineering
    df = create_features(df)

    # Use latest row
    input_df = df.iloc[[-1]].copy()

    # Clean column names
    input_df.columns = [
        re.sub(
            r"[^A-Za-z0-9_]",
            "_",
            str(col)
        )
        for col in input_df.columns
    ]

    # Match training columns
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Predict
    probability = model.predict_proba(
        input_df
    )[0][1]

    prediction = (
        1 if probability >= 0.50
        else 0
    )

    return prediction, probability


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    pred, prob = predict_failure(
        machine_type="L",
        air_temp=298.1,
        process_temp=308.6,
        rpm=1551,
        torque=42.8,
        tool_wear=0
    )

    print("\nPrediction:", pred)
    print(
        "Probability:",
        round(prob * 100, 2),
        "%"
    )