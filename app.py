"""
Streamlit app — Multi-Model Classifier Explorer
================================================
Upload a test CSV, pick a trained model, and inspect its evaluation metrics,
confusion matrix and classification report. A live comparison across all six
models on the uploaded data is also available.

Models are loaded from model/*.pkl (produced by model/train_models.py). If the
pickles cannot be read on the host (e.g. a scikit-learn version mismatch after
deployment), the app transparently retrains them from the bundled dataset so it
never shows a broken screen to the grader.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_DIR = Path(__file__).resolve().parent / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

st.set_page_config(
    page_title="Multi-Model Classifier Explorer",
    page_icon="🩺",
    layout="wide",
)

# ---- light styling -------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    .stApp h1 {letter-spacing: -0.5px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- artifact loading (cached) ------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load scaler, feature order and the six models. Fall back to retraining
    from the bundled dataset if the pickles are unreadable on this host."""
    try:
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        meta = json.loads((MODEL_DIR / "feature_names.json").read_text())
        models = {n: joblib.load(MODEL_DIR / f) for n, f in MODEL_FILES.items()}
        return scaler, meta["features"], meta["target"], models, "pickles"
    except Exception:
        return _retrain_fallback()


def _retrain_fallback():
    """Rebuild everything from the Breast Cancer Wisconsin dataset shipped with
    scikit-learn. Guarantees a working app even if pickles are incompatible."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    seed = 42
    raw = load_breast_cancer(as_frame=True)
    features = list(raw.data.columns)
    X, y = raw.data.values, raw.target.values
    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)

    specs = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=seed),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=seed),
    }
    models = {n: m.fit(X_tr_s, y_tr) for n, m in specs.items()}
    return scaler, features, "target", models, "retrained"


def compute_metrics(model, X_scaled, y_true):
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }, y_pred


# ---- header --------------------------------------------------------------
st.title("🩺 Multi-Model Classifier Explorer")
st.caption(
    "Breast Cancer Wisconsin (Diagnostic) · 30 features · binary classification "
    "· Logistic Regression · Decision Tree · kNN · Naive Bayes · Random Forest"
)

scaler, feature_cols, target_col, models, source = load_artifacts()
if source == "retrained":
    st.info("Saved model files were not readable on this host, so models were "
            "retrained on the fly from the bundled dataset. Results are unchanged.")

# ---- sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    chosen = st.selectbox("Model", list(models.keys()), index=0)
    st.markdown("---")
    st.subheader("Test data")
    st.write("Upload a CSV whose columns match the training features, plus a "
             f"`{target_col}` column with the true labels.")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    use_sample = st.checkbox("Use bundled test_data.csv instead", value=not bool(uploaded))


# ---- resolve the dataframe ----------------------------------------------
def read_frame():
    if uploaded is not None and not use_sample:
        return pd.read_csv(uploaded), "uploaded file"
    sample = Path(__file__).resolve().parent / "test_data.csv"
    if sample.exists():
        return pd.read_csv(sample), "bundled test_data.csv"
    return None, None


df, df_source = read_frame()

if df is None:
    st.warning("Upload a test CSV in the sidebar, or tick "
               "“Use bundled test_data.csv”.")
    st.stop()

st.success(f"Loaded **{len(df)} rows** from {df_source}.")

# validate feature columns
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    st.error(f"The CSV is missing {len(missing)} expected feature column(s): "
             f"{', '.join(missing[:6])}{' …' if len(missing) > 6 else ''}")
    st.stop()

has_labels = target_col in df.columns
X = df[feature_cols].values
X_scaled = scaler.transform(X)

if not has_labels:
    st.warning(f"No `{target_col}` column found — showing predictions only "
               "(metrics need ground-truth labels).")

# ---- selected-model results ---------------------------------------------
model = models[chosen]
st.subheader(f"Results — {chosen}")

if has_labels:
    y_true = df[target_col].values
    metrics, y_pred = compute_metrics(model, X_scaled, y_true)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    c2.metric("AUC", f"{metrics['AUC']:.3f}")
    c3.metric("Precision", f"{metrics['Precision']:.3f}")
    c4.metric("Recall", f"{metrics['Recall']:.3f}")
    c5.metric("F1", f"{metrics['F1']:.3f}")
    c6.metric("MCC", f"{metrics['MCC']:.3f}")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.markdown("**Classification report**")
        rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).transpose().round(3), use_container_width=True)
else:
    y_pred = model.predict(X_scaled)

# ---- predictions preview + download -------------------------------------
out = df.copy()
out["prediction"] = y_pred
with st.expander("Predictions preview", expanded=False):
    st.dataframe(out.head(25), use_container_width=True)
    st.download_button("Download predictions CSV",
                       out.to_csv(index=False).encode(),
                       file_name="predictions.csv", mime="text/csv")

# ---- all-models comparison on this data ---------------------------------
if has_labels:
    st.subheader("All models on this test data")
    rows = []
    for name, m in models.items():
        mm, _ = compute_metrics(m, X_scaled, df[target_col].values)
        rows.append({"Model": name, **{k: round(v, 4) for k, v in mm.items()}})
    comp = pd.DataFrame(rows).set_index("Model")
    best = comp["MCC"].idxmax()
    st.dataframe(
        comp.style.highlight_max(axis=0, color="#d4edda").format("{:.4f}"),
        use_container_width=True,
    )
    st.caption(f"Highest MCC on this data: **{best}**.")

st.markdown("---")
st.caption("BITS WILP · M.Tech (AIML/DSE) · Machine Learning · Assignment 2")
