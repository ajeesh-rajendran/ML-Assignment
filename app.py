import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score,
    classification_report, confusion_matrix
)

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Classification",
    page_icon="🚗",
    layout="wide"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .metric-card h1 {
        margin: 0.3rem 0 0 0;
        font-size: 1.8rem;
    }
    .stSelectbox > div > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Constants ──────────────────────────────────────────────────────────────────
MODELS_DIR = "Models"

# ─── Helper Functions ───────────────────────────────────────────────────────────

@st.cache_resource
def load_preprocessing_artifacts():
    """Load saved preprocessing artifacts."""
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    label_encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.joblib"))
    target_encoder = joblib.load(os.path.join(MODELS_DIR, "target_encoder.joblib"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))
    return scaler, label_encoders, target_encoder, feature_names


@st.cache_resource
def load_model(model_path):
    """Load a saved model from disk."""
    return joblib.load(model_path)


def get_available_models():
    """List all .joblib model files in the Models directory (excluding preprocessing artifacts)."""
    excluded = {"scaler.joblib", "label_encoders.joblib", "target_encoder.joblib", "feature_names.joblib"}
    if not os.path.exists(MODELS_DIR):
        return []
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib") and f not in excluded]
    return sorted(models)


def preprocess_test_data(df, label_encoders, scaler, target_encoder, feature_names):
    """Preprocess uploaded test data to match training format."""
    df = df.copy()

    # Drop Car_ID if present
    if "Car_ID" in df.columns:
        df = df.drop("Car_ID", axis=1)

    # Drop Price_USD if present
    if "Price_USD" in df.columns:
        df = df.drop("Price_USD", axis=1)

    # Check for target column
    if "Price_Category" not in df.columns:
        st.error("❌ The uploaded CSV must contain a **Price_Category** column for evaluation.")
        return None, None

    # Separate target
    y_true = df["Price_Category"]
    df = df.drop("Price_Category", axis=1)

    # Encode categorical columns using the saved label encoders
    for col, le in label_encoders.items():
        if col in df.columns:
            # Handle unseen labels by mapping to a default
            df[col] = df[col].map(
                lambda val, encoder=le: encoder.transform([val])[0]
                if val in encoder.classes_ else -1
            )

    # Ensure correct column order
    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        st.warning(f"⚠️ Missing columns in uploaded data: {missing_cols}")
        for c in missing_cols:
            df[c] = 0

    df = df[feature_names]

    # Scale features
    X_scaled = scaler.transform(df)

    # Encode target
    y_encoded = target_encoder.transform(y_true)

    return X_scaled, y_encoded


def compute_metrics(model, X_test, y_test, target_encoder):
    """Compute all 6 evaluation metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC Score": roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="weighted"),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "MCC Score": matthews_corrcoef(y_test, y_pred),
    }

    class_names = target_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return metrics, report, cm, y_pred, class_names


# ─── Main App ───────────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">🚗 Car Price Classification</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload test data, select a model, and view evaluation results</p>', unsafe_allow_html=True)

# Check if Models folder exists
if not os.path.exists(MODELS_DIR):
    st.error("⚠️ **Models folder not found!** Please run the notebook first to train and save models.")
    st.stop()

available_models = get_available_models()
if not available_models:
    st.error("⚠️ **No models found!** Please run the notebook first to train and save models.")
    st.stop()

# Load preprocessing artifacts
try:
    scaler, label_encoders, target_encoder, feature_names = load_preprocessing_artifacts()
except Exception as e:
    st.error(f"⚠️ **Error loading preprocessing artifacts:** {e}\n\nPlease run the notebook first.")
    st.stop()

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # Model selection dropdown
    st.subheader("🤖 Select Model")
    model_display_names = {f: f.replace(".joblib", "").replace("_", " ") for f in available_models}
    selected_file = st.selectbox(
        "Choose a classification model:",
        available_models,
        format_func=lambda x: model_display_names[x]
    )

    st.divider()

    # CSV Upload
    st.subheader("📁 Upload Test Data")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with test data",
        type=["csv"],
        help="The CSV must contain the same columns as the training data, including 'Price_Category' for evaluation."
    )

    st.divider()
    st.markdown("**Expected columns:**")
    st.code(", ".join(feature_names + ["Price_Category"]), language=None)

# ─── Main Content ───────────────────────────────────────────────────────────────
if uploaded_file is not None:
    # Read uploaded CSV
    test_df = pd.read_csv(uploaded_file)

    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(test_df.head(10), use_container_width=True)
    st.caption(f"Shape: {test_df.shape[0]} rows × {test_df.shape[1]} columns")

    # Preprocess
    X_test, y_test = preprocess_test_data(test_df, label_encoders, scaler, target_encoder, feature_names)

    if X_test is not None and y_test is not None:
        # Load selected model
        model = load_model(os.path.join(MODELS_DIR, selected_file))
        model_name = model_display_names[selected_file]

        # Compute metrics
        metrics, report, cm, y_pred, class_names = compute_metrics(model, X_test, y_test, target_encoder)

        st.divider()

        # ─── Evaluation Metrics ─────────────────────────────────────────────
        st.subheader(f"📊 Evaluation Metrics — {model_name}")

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        metric_items = list(metrics.items())
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]

        for i, (col, (metric_name, metric_val)) in enumerate(
            zip([col1, col2, col3, col4, col5, col6], metric_items)
        ):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, {colors[i]} 0%, {colors[i]}CC 100%);">
                    <h3>{metric_name}</h3>
                    <h1>{metric_val:.4f}</h1>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # ─── Classification Report & Confusion Matrix ───────────────────────
        tab1, tab2 = st.tabs(["📝 Classification Report", "🔲 Confusion Matrix"])

        with tab1:
            st.text(report)

        with tab2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names, ax=ax
            )
            ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

else:
    # Show instructions when no file is uploaded
    st.info("👈 **Upload a test CSV file** from the sidebar to get started.")

    st.markdown("### How to use this app:")
    st.markdown("""
    1. **Select a model** from the dropdown in the sidebar
    2. **Upload a CSV** file containing your test data
    3. The CSV must include a `Price_Category` column (the true labels) for evaluation
    4. View the **evaluation metrics**, **classification report**, and **confusion matrix**
    """)

    st.markdown("### Available Models:")
    for m in available_models:
        st.markdown(f"- 🤖 **{model_display_names[m]}**")
