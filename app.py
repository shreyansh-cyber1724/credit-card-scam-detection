import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

# Load saved model artifact
@st.cache_resource
def load_artifact():
    return joblib.load("credit_card_fraud_syllabus_model.joblib")

artifact = load_artifact()
model = artifact["model"]
model_name = artifact["model_name"]
threshold = float(artifact["threshold"])
feature_cols = artifact["feature_columns"]

# Helper function matching notebook logic
def get_score(model, df_row):
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(df_row)[:, 1][0])
    z = model.decision_function(df_row)
    return float((z - z.min()) / (z.max() - z.min() + 1e-12))[0]

# --- UI Layout ---
st.title("💳 Credit Card Fraud Detection System")
st.markdown(f"**CSET211 Showcase** | Active Model: `{model_name}` | F2-Optimized Threshold: `{threshold:.4f}`")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔍 Single Transaction Simulator", "📂 Batch CSV Scoring", "📊 Project Architecture"])

# TAB 1: Interactive Single Transaction Demo
with tab1:
    st.subheader("Test a Transaction")
    try:
        samples = pd.read_csv("demo_transactions.csv")
        sample_labels = [
            f"Sample {i+1} — Amount: ${row['Amount']:.2f} (Actual: {'FRAUD' if row['Actual_Class']==1 else 'Legitimate'})"
            for i, row in samples.iterrows()
        ]
        selected_idx = st.selectbox("Load a held-out test transaction:", range(len(sample_labels)), format_func=lambda x: sample_labels[x])
        selected_row = samples.iloc[selected_idx].to_dict()
    except FileNotFoundError:
        st.warning("`demo_transactions.csv` not found. Using default zeroed features.")
        selected_row = {col: 0.0 for col in feature_cols}

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Key Editable Parameters")
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=float(selected_row.get("Amount", 100.0)), step=10.0)
        time_sec = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=float(selected_row.get("Time", 36000.0)), step=600.0)
        v14 = st.slider("V14 (Top PCA Component)", -20.0, 10.0, float(selected_row.get("V14", 0.0)))
        v17 = st.slider("V17 (Top PCA Component)", -25.0, 10.0, float(selected_row.get("V17", 0.0)))
        v12 = st.slider("V12 (Top PCA Component)", -20.0, 10.0, float(selected_row.get("V12", 0.0)))

        # Recompute engineered features from Module II
        input_data = {col: selected_row.get(col, 0.0) for col in feature_cols}
        input_data["Amount"] = amount
        input_data["Time"] = time_sec
        input_data["V14"] = v14
        input_data["V17"] = v17
        input_data["V12"] = v12
        if "Amount_log" in feature_cols:
            input_data["Amount_log"] = np.log1p(max(amount, 0.0))
        if "Hour" in feature_cols:
            input_data["Hour"] = float((time_sec // 3600) % 24)

    with col2:
        st.markdown("#### Model Prediction")
        row_df = pd.DataFrame([input_data]).reindex(columns=feature_cols)
        score = get_score(model, row_df)
        is_fraud = score >= threshold

        st.metric(label="Fraud Probability / Risk Score", value=f"{score:.4f}", delta=f"Threshold: {threshold:.4f}", delta_color="off")
        st.progress(min(max(score, 0.0), 1.0))

        if is_fraud:
            st.error("🚨 **DECISION: POTENTIAL FRAUD — FLAG FOR REVIEW**")
        else:
            st.success("✅ **DECISION: LIKELY LEGITIMATE**")

        with st.expander("View Full 32-Feature Input Vector"):
            st.dataframe(row_df.T.rename(columns={0: "Value"}), use_container_width=True)

# TAB 2: Batch Upload
with tab2:
    st.subheader("Upload Batch Transactions (CSV)")
    uploaded_file = st.file_uploader("Upload a CSV matching the dataset schema", type=["csv"])
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        if "Amount" in batch_df.columns and "Amount_log" not in batch_df.columns:
            batch_df["Amount_log"] = np.log1p(batch_df["Amount"].clip(lower=0))
        if "Time" in batch_df.columns and "Hour" not in batch_df.columns:
            batch_df["Hour"] = ((batch_df["Time"] // 3600) % 24).astype(float)

        X_batch = batch_df.reindex(columns=feature_cols)
        if hasattr(model, "predict_proba"):
            batch_scores = model.predict_proba(X_batch)[:, 1]
        else:
            z = model.decision_function(X_batch)
            batch_scores = (z - z.min()) / (z.max() - z.min() + 1e-12)

        batch_df["Fraud_Score"] = np.round(batch_scores, 4)
        batch_df["Prediction"] = np.where(batch_scores >= threshold, "FRAUD", "Legitimate")

        fraud_count = (batch_df["Prediction"] == "FRAUD").sum()
        st.info(f"Analyzed **{len(batch_df)}** transactions. Flagged **{fraud_count}** potential fraud cases.")
        st.dataframe(batch_df[["Time", "Amount", "Fraud_Score", "Prediction"] + [c for c in batch_df.columns if c not in ["Time", "Amount", "Fraud_Score", "Prediction"]]], use_container_width=True)

# TAB 3: Viva / Showcase Summary
with tab3:
    st.subheader("CSET211 Methodology Overview")
    st.markdown("""
    - **Module I (EDA):** Analyzed severe class imbalance (~0.17% fraud) and skewed transaction amounts.
    - **Module II (Preprocessing):** Engineered `Amount_log` and `Hour`, applied median imputation and `StandardScaler` inside leakage-free Scikit-Learn pipelines.
    - **Module III (Modeling):** Evaluated 6 classifiers (Logistic Regression, KNN, Gaussian NB, LinearSVC, Decision Tree, Random Forest) using **PR-AUC** and **F2-Score**.
    - **Threshold Tuning:** Selected the F2-maximizing decision threshold strictly on the validation set to prioritize high fraud recall without leaking test data.
    """)