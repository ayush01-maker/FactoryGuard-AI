import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_failure

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="FactoryGuard AI",
    page_icon="🏭",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

/* Main App Background */
.stApp {
    background: linear-gradient(
        135deg,
        #f4f7fc 0%,
        #e8eef7 100%
    );
}

/* Dropdown Box */
div[data-baseweb="select"] > div {
    background-color: #334155 !important;
    color: white !important;
    border: 1px solid #475569 !important;
}

/* Dropdown Text */
div[data-baseweb="select"] span {
    color: white !important;
}
            
/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1e293b;
}

[data-testid="stSidebar"] * {
    color: white;
}

/* Main Title */
.dashboard-title {
    text-align: center;
    color: #0f172a;
    font-size: 42px;
    font-weight: 700;
}

/* Subtitle */
.dashboard-subtitle {
    text-align: center;
    color: #475569;
    font-size: 18px;
    margin-bottom: 20px;
}

/* Metric Cards */
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.1);
}

/* Buttons */
.stButton > button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #1d4ed8;
}

/* Success Box */
.stSuccess {
    border-radius: 10px;
}

/* Warning Box */
.stWarning {
    border-radius: 10px;
}

/* Error Box */
.stError {
    border-radius: 10px;
}

/* KPI Metrics */
[data-testid="metric-container"] {
    background: black;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}

/* Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================

st.markdown(
    "<div class='dashboard-title'>🏭 FactoryGuard AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='dashboard-subtitle'>IoT Predictive Maintenance Engine</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("⚙ Machine Inputs")

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temp = st.sidebar.slider(
    "Air Temperature (K)",
    290.0,
    330.0,
    298.1
)

process_temp = st.sidebar.slider(
    "Process Temperature (K)",
    300.0,
    340.0,
    308.6
)

rpm = st.sidebar.slider(
    "Rotational Speed (RPM)",
    1000,
    3000,
    1551
)

torque = st.sidebar.slider(
    "Torque (Nm)",
    0.0,
    100.0,
    42.8
)

tool_wear = st.sidebar.slider(
    "Tool Wear (min)",
    0,
    300,
    0
)

# ==================================================
# LIVE SENSOR STATUS
# ==================================================

st.subheader("📊 Live Sensor Status")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Air Temp", f"{air_temp:.1f} K")
c2.metric("Process Temp", f"{process_temp:.1f} K")
c3.metric("RPM", rpm)
c4.metric("Torque", f"{torque:.1f} Nm")
c5.metric("Tool Wear", f"{tool_wear} min")

# ==================================================
# SENSOR VISUALIZATION
# ==================================================

sensor_df = pd.DataFrame({
    "Sensor": [
        "Air Temp",
        "Process Temp",
        "RPM",
        "Torque",
        "Tool Wear"
    ],
    "Value": [
        air_temp,
        process_temp,
        rpm,
        torque,
        tool_wear
    ]
})

fig = px.bar(
    sensor_df,
    x="Sensor",
    y="Value",
    title="Current Sensor Readings"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==================================================
# PREDICTION
# ==================================================

st.subheader("🤖 Predictive Maintenance")

if st.button("Predict Failure"):

    prediction, probability = predict_failure(
        machine_type,
        air_temp,
        process_temp,
        rpm,
        torque,
        tool_wear
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Failure Probability",
            f"{probability*100:.2f}%"
        )

    with col2:

        if prediction == 1:
            st.error(
                "⚠ Machine Failure Predicted"
            )
        else:
            st.success(
                "✅ Machine Operating Normally"
            )

    # ==========================================
    # GAUGE CHART
    # ==========================================

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Failure Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red"},
                "steps": [
                    {"range": [0, 20], "color": "lightgreen"},
                    {"range": [20, 50], "color": "yellow"},
                    {"range": [50, 80], "color": "orange"},
                    {"range": [80, 100], "color": "red"}
                ]
            }
        )
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )


    # ==========================================
    # RISK ASSESSMENT
    # ==========================================

    st.subheader("🚨 Risk Assessment")

    if probability >= 0.80:

        st.error("🔴 HIGH RISK")

        st.write(
            "Immediate maintenance recommended. Failure likely."
        )

    elif probability >= 0.50:

        st.warning("🟠 MEDIUM RISK")

        st.write(
            "Monitor machine closely and schedule maintenance."
        )

    elif probability >= 0.20:

        st.info("🟡 LOW-MEDIUM RISK")

        st.write(
            "Machine healthy but showing minor warning signs."
        )

    else:

        st.success("🟢 LOW RISK")

        st.write(
            "Machine operating under normal conditions."
        )

# ==================================================
# MODEL PERFORMANCE
# ==================================================

st.markdown("---")
st.subheader("📈 Model Performance")

m1, m2, m3, m4 = st.columns(4)

m2.metric("Precision", "58.26%")
m3.metric("Recall", "98.53%")
m4.metric("PR-AUC", "94.91%")

st.info(
    "High Recall ensures that machine failures are rarely missed."
)

# ==================================================
# SHAP EXPLAINABILITY
# ==================================================

st.markdown("---")
st.subheader("🔍 Explainability")

shap_path = "shap_summary.png"

if os.path.exists(shap_path):

    st.image(
        shap_path,
        caption="SHAP Feature Importance"
    )

else:

    st.info(
        "Run shap_analysis.py to generate SHAP visualization."
    )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "FactoryGuard AI | IoT Predictive Maintenance Dashboard | Powered by XGBoost + SMOTE + SHAP"
)