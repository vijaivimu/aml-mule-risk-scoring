import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AML Mule Account Detection",
    page_icon="🕵️",
    layout="wide"
)

# ============================================
# Load data + model
# ============================================
@st.cache_data
def load_data():
    features = pd.read_csv("data/features_account_level.csv")
    feature_cols = [c for c in features.columns if c not in ("is_mule","account_id")]
    model = joblib.load("models/xgb_calibrated.pkl")
    return features, feature_cols, model

features, feature_cols, model = load_data()


# ============================================
# Sidebar — recent account history
# ============================================
st.sidebar.title("🔁 Recent Accounts")

if "history" not in st.session_state:
    st.session_state.history = []

if st.session_state.history:
    st.sidebar.write(st.session_state.history)
else:
    st.sidebar.write("No history yet")


# ============================================
# Header
# ============================================
st.title("🕵️ AML Mule Account Detection — Streamlit UI")

account_id = st.number_input("Enter Account ID", min_value=0, value=1440, step=1)


# ============================================
# Add to session history
# ============================================
if account_id not in st.session_state.history:
    st.session_state.history.append(account_id)
    st.session_state.history = st.session_state.history[-5:]   # keep last 5


# ============================================
# Fetch record
# ============================================
row = features[features["account_id"] == account_id]

if row.empty:
    st.error("❌ Account not found.")
    st.stop()

X_row = row[feature_cols]


# ============================================
# ML prediction
# ============================================
raw_model = model.estimator          # calibrated model wrapper
prob = raw_model.predict_proba(X_row)[0, 1]
flag = (prob >= 0.50)

st.subheader(f"Account: {account_id}")

st.markdown("### 🔍 Decisions")
st.write(f"**ML Probability:** `{prob:.4f}`")
st.write(f"**ML Flag (≥0.50):** {'🚨 Flagged' if flag else '✓ OK'}")


# ============================================
# Rules (same as Practical 7)
# ============================================
rule_1 = (X_row["total_inflow_amount"].iloc[0] > 150000)
rule_2 = (X_row["num_unique_receivers"].iloc[0] > 30)
rule_3 = (X_row["proportion_txn_below_threshold"].iloc[0] > 0.95)
rule_4 = (X_row["txn_velocity_per_day"].iloc[0] > 20)
rule_5 = (X_row["net_flow"].iloc[0] > 80000)

rule_list = [
    ("total_inflow_amount > 150k", rule_1),
    ("num_unique_receivers > 30", rule_2),
    ("proportion_txn_below_threshold > 0.95", rule_3),
    ("txn_velocity_per_day > 20", rule_4),
    ("net_flow > 80000", rule_5),
]

rule_score = sum(int(r) for _, r in rule_list)
rule_flag = (rule_score >= 2)

st.write(f"**Rule-based flag:** {'🚨 Flagged' if rule_flag else '✓ OK'}")
st.write(f"**Rules triggered:** `{rule_score} / 5`")

with st.expander("🔎 Show rule contributions"):
    for name, value in rule_list:
        icon = "✓ True" if value else "❌ False"
        st.write(f"- {name}: {icon}")


# ============================================
# Model explanation (Waterfall)
# ============================================
st.markdown("## 🔥 SHAP Local Explanation (Top 10)")

explainer = shap.TreeExplainer(raw_model)
shap_values = explainer.shap_values(X_row)

fig, ax = plt.subplots(figsize=(8,6))
shap.plots.waterfall(
    shap_values[0],
    max_display=10,
    show=False
)
st.pyplot(fig)


# ============================================
# Feature snapshot
# ============================================
st.markdown("## 📊 Feature Snapshot")
st.dataframe(X_row.T.rename(columns={X_row.index[0]: "value"}))


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


st.success("✔ Inspection complete")