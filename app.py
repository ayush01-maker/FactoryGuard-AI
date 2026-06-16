from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# =====================================
# Load Model
# =====================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(
    os.path.join(MODEL_DIR, "xgboost_model.pkl")
)

encoder = joblib.load(
    os.path.join(MODEL_DIR, "label_encoder.pkl")
)

feature_columns = joblib.load(
    os.path.join(MODEL_DIR, "feature_columns.pkl")
)

# =====================================
# Health Check
# =====================================

@app.route("/")
def home():

    return jsonify({
        "status": "FactoryGuard API Running"
    })

# =====================================
# Prediction Endpoint
# =====================================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    machine_type = encoder.transform(
        [data["Type"]]
    )[0]

    input_data = {
        "Type": machine_type,
        "Air_temperature_K": data["Air_temperature_K"],
        "Process_temperature_K": data["Process_temperature_K"],
        "Rotational_speed_rpm": data["Rotational_speed_rpm"],
        "Torque_Nm": data["Torque_Nm"],
        "Tool_wear_min": data["Tool_wear_min"]
    }

    df = pd.DataFrame([input_data])

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    probability = float(
        model.predict_proba(df.values)[0][1]
    )

    prediction = int(
        probability > 0.5
    )

    return jsonify({
        "prediction": prediction,
        "failure_probability": round(
            probability,
            4
        )
    })

# =====================================
# Run Server
# =====================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )