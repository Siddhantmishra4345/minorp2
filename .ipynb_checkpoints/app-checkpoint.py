import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
from data_loader import get_splits, LABELS, load_data, DATA_ROOT
from pipeline import selective_engine, evaluate_model
from utils.metrics import compute_ece, compute_tps, plot_reliability
from utils.drift import ks_drift, psi_score

st.set_page_config(page_title="Trustworthy HAR Pipeline", layout="wide")

st.title("🎯 Trustworthy Human Activity Recognition (HAR) Pipeline")
st.markdown("---")

@st.cache_resource
def load_or_train_models():
    if not os.path.exists("models/lr_model.joblib"):
        st.warning("Models not found. Training models now, please wait...")
        from train_models import train_and_save
        train_and_save()
    
    if os.path.exists("models/lr_model.joblib"):
        lr_model = joblib.load("models/lr_model.joblib")
        xgb_model = joblib.load("models/xgb_model.joblib")
        le = joblib.load("models/label_encoder.joblib")
        return lr_model, xgb_model, le
    return None, None, None

@st.cache_data
def get_data():
    return get_splits()

lr_model, xgb_model, le = load_or_train_models()
X_train, y_train, X_cal, y_cal, X_eval, y_eval = get_data()

if X_train is None:
    st.error(f"Dataset not found at {DATA_ROOT}. Please ensure the UCI HAR dataset is available.")
    st.stop()

# Set best thresholds found in the notebook
BEST_TAU_HIGH = 0.55
BEST_TAU_LOW = 0.50

# --- (A) Problem & Objective ---
st.header("🔷 (A) Problem & Objective")
st.write("""
**The Problem**: Standard Machine Learning models suffer from *overconfidence* - they often predict incorrect classes with very high probabilities (>90%). 
This makes them unreliable for safety-critical or high-stakes applications.

**The Objective**: Achieve high reliability via a **Selective Prediction Engine**. Instead of forcing a prediction on every input, the system should:
1. **ACCEPT** predictions it is truly confident about.
2. **DEFER** ambiguous cases to a more complex fallback model.
3. **REJECT** highly uncertain cases for human review.
""")

# --- (B) Pipeline Architecture ---
st.header("🔷 (B) Pipeline Architecture")
st.markdown("""
**Input → Logistic Regression (Primary)** → **Selective Engine**
* `conf ≥ tau_high` → **ACCEPT** (Use LR Prediction)
* `tau_low ≤ conf < tau_high` → **DEFER** (Use XGBoost Fallback)
* `conf < tau_low` → **REJECT** (Abstain)

Parallel Monitoring:
* **Drift Detection (KS + PSI)**: Monitors incoming data distributions.
* **Explainability (SHAP)**: Provides heuristic insight into feature importance.
""")

# --- (C) Interactive Demo ---
st.header("🔷 (C) Interactive Demo")
st.write("Select a sample from the evaluation dataset to see the pipeline in action.")

col1, col2 = st.columns([1, 2])
with col1:
    uploaded_file = st.file_uploader("Optional: Upload CSV with 561 features", type=["csv"])
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file, header=None)
            if custom_df.shape[1] == 561:
                sample_idx = 0
                sample_X = custom_df.iloc[[0]].copy()
                true_label = "Unknown (Custom Upload)"
                st.success("Successfully loaded custom sample from CSV.")
            else:
                st.error(f"CSV must have exactly 561 features. Found {custom_df.shape[1]}.")
                sample_idx = st.number_input("Sample Index (0 to len(X_eval)-1)", min_value=0, max_value=len(X_eval)-1, value=0)
                sample_X = X_eval.iloc[[sample_idx]].copy()
                true_label = LABELS[y_eval.iloc[sample_idx]]
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            sample_idx = st.number_input("Sample Index (0 to len(X_eval)-1)", min_value=0, max_value=len(X_eval)-1, value=0)
            sample_X = X_eval.iloc[[sample_idx]].copy()
            true_label = LABELS[y_eval.iloc[sample_idx]]
    else:
        sample_idx = st.number_input("Sample Index (0 to len(X_eval)-1)", min_value=0, max_value=len(X_eval)-1, value=0)
        sample_X = X_eval.iloc[[sample_idx]].copy()
        true_label = LABELS[y_eval.iloc[sample_idx]]
        
    noise_level = st.slider("Optional: Add Gaussian Noise (σ)", min_value=0.0, max_value=0.5, value=0.0, step=0.05)

if noise_level > 0:
    np.random.seed(42)
    sample_X += np.random.normal(0, noise_level, sample_X.shape)

# Predict LR
lr_proba = lr_model.predict_proba(sample_X)[0]
lr_conf = np.max(lr_proba)
lr_pred_idx = np.argmax(lr_proba)
lr_pred = LABELS[lr_pred_idx + 1]

# Predict XGB
xgb_proba = xgb_model.predict_proba(sample_X)[0]
xgb_conf = np.max(xgb_proba)
xgb_pred_idx = np.argmax(xgb_proba)
xgb_pred = LABELS[le.inverse_transform([xgb_pred_idx])[0]]

# Decision
if lr_conf >= BEST_TAU_HIGH:
    decision = "ACCEPT"
    final_pred = lr_pred
    final_conf = lr_conf
    color = "green"
elif lr_conf >= BEST_TAU_LOW:
    decision = "DEFER"
    final_pred = xgb_pred
    final_conf = xgb_conf
    color = "orange"
else:
    decision = "REJECT"
    final_pred = "N/A (Human Review)"
    final_conf = lr_conf
    color = "red"

with col2:
    st.markdown(f"**True Activity:** `{true_label}`")
    st.markdown(f"**Predicted Activity:** `{final_pred}`")
    st.markdown(f"**Final Confidence:** `{final_conf:.4f}`")
    st.markdown(f"**System Decision:** <span style='color:{color}; font-weight:bold; font-size:1.2em;'>{decision}</span>", unsafe_allow_html=True)

# --- (D) Decision Explanation ---
st.header("🔷 (D) Decision Explanation")
st.write(f"**Why was this decision made?**")
st.write(f"- LR Confidence: `{lr_conf:.4f}`")
st.write(f"- Decision Rule: `tau_high = {BEST_TAU_HIGH}`, `tau_low = {BEST_TAU_LOW}`")
if decision == "ACCEPT":
    st.success(f"LR Confidence ({lr_conf:.4f}) >= tau_high ({BEST_TAU_HIGH}). System ACCEPTED the primary model's prediction.")
elif decision == "DEFER":
    st.warning(f"LR Confidence ({lr_conf:.4f}) is between {BEST_TAU_LOW} and {BEST_TAU_HIGH}. System DEFERRED to XGBoost fallback.")
    st.write(f"Fallback (XGBoost) prediction used: `{xgb_pred}` with confidence `{xgb_conf:.4f}`.")
else:
    st.error(f"LR Confidence ({lr_conf:.4f}) < tau_low ({BEST_TAU_LOW}). System REJECTED the prediction.")

# --- (E) Model Comparison Panel ---
st.header("🔷 (E) Model Comparison Panel")
c1, c2 = st.columns(2)
c1.info(f"**Logistic Regression**\n\nPrediction: `{lr_pred}`\n\nConfidence: `{lr_conf:.4f}`")
c2.info(f"**XGBoost (Fallback)**\n\nPrediction: `{xgb_pred}`\n\nConfidence: `{xgb_conf:.4f}`")

# --- (F) Metrics Dashboard ---
st.header("🔷 (F) Metrics Dashboard (Evaluation Set)")
@st.cache_data
def compute_dashboard_metrics():
    # We compute metrics for the whole eval set
    lr_pb = lr_model.predict_proba(X_eval)
    lr_cf = np.max(lr_pb, axis=1)
    lr_pd = lr_model.predict(X_eval)
    
    xgb_pb = xgb_model.predict_proba(X_eval)
    xgb_cf = np.max(xgb_pb, axis=1)
    xgb_pd = le.inverse_transform(np.argmax(xgb_pb, axis=1))
    
    res = selective_engine(lr_cf, lr_pd, xgb_cf, xgb_pd, y_eval, BEST_TAU_HIGH, BEST_TAU_LOW)
    return res

metrics = compute_dashboard_metrics()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Coverage", f"{metrics['coverage']*100:.1f}%", help="Percentage of samples not rejected.")
m2.metric("Selective Accuracy", f"{metrics['sel_acc']*100:.2f}%", help="Accuracy on accepted/deferred samples.")
m3.metric("ECE (Selective)", f"{metrics['ece_sel']:.4f}", help="Expected Calibration Error on covered samples.")
m4.metric("TPS", f"{metrics['tps']:.4f}", help="Trustworthy Prediction Score: Coverage × Sel_Acc / (1 + ECE).")

# --- (G) Confidence Visualization ---
st.header("🔷 (G) Confidence Visualization")
fig, ax = plt.subplots(figsize=(10, 2))
ax.barh([0], [lr_conf], color='skyblue', edgecolor='black')
ax.axvline(BEST_TAU_LOW, color='red', linestyle='--', label='tau_low')
ax.axvline(BEST_TAU_HIGH, color='green', linestyle='--', label='tau_high')
ax.set_xlim(0, 1)
ax.set_yticks([])
ax.set_title("LR Confidence vs Thresholds")
ax.legend()
st.pyplot(fig)

# --- (H) Noise Experiment ---
st.header("🔷 (H) Noise Experiment")
st.write("Models often remain overconfident even when data is noisy. This demonstrates the need for Drift Detection.")
if noise_level > 0:
    st.write(f"Current Noise Level: **σ = {noise_level}**")
    # Quick metrics for noisy
    X_noisy = X_eval + np.random.normal(0, noise_level, X_eval.shape)
    lr_pb_n = lr_model.predict_proba(X_noisy)
    acc_n = (lr_model.predict(X_noisy) == y_eval).mean()
    ece_n = compute_ece(y_eval, lr_pb_n)
    st.write(f"- LR Accuracy on noisy data: `{acc_n*100:.2f}%`")
    st.write(f"- LR ECE on noisy data: `{ece_n:.4f}`")
else:
    st.write("Increase the noise slider in the Interactive Demo to see its effect on accuracy and calibration.")

# --- (I) Drift Detection Panel ---
st.header("🔷 (I) Drift Detection Panel")
st.write("Parallel monitoring using Kolmogorov-Smirnov (KS) and Population Stability Index (PSI).")
if st.button("Run Drift Detection (Training vs Current Eval + Noise)"):
    with st.spinner("Computing Drift..."):
        X_test_drift = X_eval.copy()
        if noise_level > 0:
            np.random.seed(42)
            X_test_drift += np.random.normal(0, noise_level, X_test_drift.shape)
        
        ks = ks_drift(X_train, X_test_drift, ks_threshold=0.10)
        psi = psi_score(X_train, X_test_drift)
        
        c1, c2 = st.columns(2)
        c1.write(f"**KS Test (Features Drifted)**: {ks['drift_fraction']*100:.1f}%")
        c2.write(f"**Mean PSI**: {psi['mean_psi']:.4f}")
        st.markdown(f"**System Status**: `{psi['verdict']}`")
        st.info("Drift detection acts as an early warning system. It does not block individual predictions.")

# --- (J) Explainability (SHAP) ---
st.header("🔷 (J) Explainability (SHAP)")
st.write("Top contributing features for the fallback model (XGBoost) on this sample.")
if st.button("Generate SHAP Explanation"):
    with st.spinner("Computing SHAP values..."):
        try:
            # Extract base estimator from CalibratedClassifierCV
            base_xgb = xgb_model.estimator if hasattr(xgb_model, 'estimator') else xgb_model.base_estimator
            explainer = shap.TreeExplainer(base_xgb)
            shap_vals = explainer.shap_values(sample_X)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            # Plot for the predicted class
            pred_c = xgb_pred_idx
            # shap_vals has shape (1, n_features, n_classes) or is a list of arrays
            if isinstance(shap_vals, list):
                vals = shap_vals[pred_c][0]
            else:
                vals = shap_vals[0, :, pred_c] if len(shap_vals.shape) == 3 else shap_vals[0]
            
            top_indices = np.argsort(np.abs(vals))[::-1][:10]
            
            ax.barh(range(10), vals[top_indices][::-1], color='steelblue')
            ax.set_yticks(range(10))
            ax.set_yticklabels([f"Feature {i}" for i in top_indices][::-1])
            ax.set_xlabel("SHAP Value")
            ax.set_title(f"Top 10 features for class {xgb_pred}")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error computing SHAP values: {e}")

# --- (K) Key Findings ---
st.header("🔷 (K) Key Findings")
st.success("""
1. **Accuracy vs Reliability**: High accuracy does not mean high reliability. Overconfident errors are dangerous.
2. **Selective Prediction**: A small rejection rate (abstaining on uncertain cases) significantly improves the reliability on the remaining cases.
3. **Calibration**: Calibration aligns the confidence scores with empirical accuracy, making thresholds like `tau_high` meaningful.
""")

# --- (L) Limitations ---
st.header("🔷 (L) Limitations")
st.warning("""
- **Confidence ≠ Guarantee**: Confidence scores are statistical, not guarantees of correctness.
- **TPS is Heuristic**: The Trustworthy Prediction Score is a custom heuristic for this project.
- **Shared Feature Space**: The fallback model (XGBoost) shares the same feature space as the primary model. Severe distribution shifts will degrade both models.
""")
