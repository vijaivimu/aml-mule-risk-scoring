import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# --------------------------------------------
# Load model + data
# --------------------------------------------
@st.cache_resource
def load_data():
    features = pd.read_csv("data/features_account_level.csv")
    feature_cols = [c for c in features.columns if c not in ("is_mule", "account_id")]
    model = joblib.load("models/xgb_calibrated.pkl")
    raw_model = model.estimator
    return features, feature_cols, raw_model

features, feature_cols, model = load_data()
X = features[feature_cols]

# Safety patches for SHAP
if not hasattr(np, "int"): np.int = int
if not hasattr(np, "bool"): np.bool = bool

# SHAP explainer
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Optimal threshold (from Practical 5)
BEST_THRESHOLD = 0.50

# --------------------------------------------
# Rule logic
# --------------------------------------------
def apply_rules(row):
    rules = {
        "total_inflow_amount > 150k": row["total_inflow_amount"] > 150000,
        "num_unique_receivers > 30": row["num_unique_receivers"] > 30,
        "proportion_txn_below_threshold > 0.95": row["proportion_txn_below_threshold"] > 0.95,
        "txn_velocity_per_day > 20": row["txn_velocity_per_day"] > 20,
        "net_flow > 80000": row["net_flow"] > 80000
    }
    rule_count = sum(rules.values())
    return rule_count, rules


# --------------------------------------------
# UI
# --------------------------------------------
st.title("🕵️ AML Mule Account Detection — Streamlit UI")

# ============================================
# Model Info / Guidance
# ============================================
with st.expander("ℹ️ How this model works"):
    st.write("""
    ### Model
    We use a **calibrated XGBoost binary classifier** trained to detect mule accounts using synthetic but realistic transaction features.

    ### Features (examples)
    - Transaction volume (total_inflow_amount, total_outflow_amount)
    - Structuring indicators (proportion_txn_below_threshold)
    - Behavioral velocity (txn_velocity_per_day)
    - Transaction network (num_unique_receivers, num_unique_senders)

    ### Explainability
    SHAP values explain how each feature pushes the model towards **flagging** or **not flagging** this specific account.

    Blue bars = lowers risk  
    Red bars = increases risk

    ### Rule model
    The rule system is included for **comparison** and interpretability:
    If 2+ rules trigger, the account is flagged.

    ---
    """)

account_id_input = st.text_input("Enter Account ID", "")

if account_id_input:
    try:
        account_id = int(account_id_input)
        row_idx = features[features["account_id"] == account_id].index[0]
        row = X.iloc[row_idx]

        st.subheader(f"Account: {account_id}")

        # ---------------------
        # ML prediction
        # ---------------------
        prob = model.predict_proba(row.values.reshape(1, -1))[0, 1]
        ml_flag = int(prob >= BEST_THRESHOLD)

        # ---------------------
        # Rule-based decision
        # ---------------------
        rule_count, rule_details = apply_rules(row)
        rule_flag = int(rule_count >= 2)

        # ---------------------
        # Results
        # ---------------------
        st.write("### 🔍 Decisions")
        st.write(f"**ML Probability:** {prob:.4f}")
        st.write(f"**ML Flag (≥{BEST_THRESHOLD:.2f}):** {'🚨 FLAG' if ml_flag else '✔️ OK'}")
        st.write("---")
        st.write(f"**Rule-based flag:** {'🚨 FLAG' if rule_flag else '✔️ OK'}")
        st.write(f"**Rules triggered:** {rule_count} / 5")

        with st.expander("Show rule contributions"):
            for r, val in rule_details.items():
                st.write(f"- {r}: {'✔️ True' if val else '❌ False'}")

        # ---------------------
        # SHAP Explanation — Waterfall
        # ---------------------
        st.write("### 🔥 SHAP Local Explanation (Top 10)")
        fig, ax = plt.subplots(figsize=(10,6))
        shap.plots.waterfall(shap_values[row_idx], max_display=10, show=False)
        st.pyplot(fig)

        # ---------------------
        # Feature values table
        # ---------------------
        st.write("### 📊 Feature Snapshot")
        st.dataframe(row.to_frame().rename(columns={row_idx: "value"}))

    except Exception as e:
        st.error("Account ID not found in dataset.")
        st.text(str(e))