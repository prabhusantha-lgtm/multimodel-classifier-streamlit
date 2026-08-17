# BITS ML Assignment 2 - Streamlit classifier app
# Loads the five trained models and evaluates them on an uploaded test CSV.

import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

st.set_page_config(page_title="Multi-Model Classifier Explorer", layout="wide")

st.markdown(
    "<style>.block-container{padding-top:2rem;max-width:1200px;}"
    'div[data-testid="stMetricValue"]{font-size:1.6rem;}</style>',
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_models():
    # Load the saved pickles first. If they can't be read (e.g. a scikit-learn
    # version mismatch after deploy), rebuild them from the dataset instead.
    try:
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        meta = json.loads((MODEL_DIR / "feature_names.json").read_text())
        models = {n: joblib.load(MODEL_DIR / f) for n, f in MODEL_FILES.items()}
        return scaler, meta["features"], meta["target"], models, "pickles"
    except Exception:
        return rebuild_models()


def rebuild_models():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier

    data = load_breast_cancer(as_frame=True)
    feats = list(data.data.columns)
    X, y = data.data.values, data.target.values
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, stratify=y,
                                       random_state=42)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)

    clfs = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }
    for c in clfs.values():
        c.fit(Xtr, ytr)
    return scaler, feats, "target", clfs, "rebuilt"


def get_metrics(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    scores = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba),
        "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
    }
    return scores, pred


st.title("Multi-Model Classifier Explorer")
st.caption("Breast Cancer Wisconsin (Diagnostic) | 5 models | 6 metrics")

scaler, feature_cols, target_col, models, source = load_models()
if source == "rebuilt":
    st.info("Saved models couldn't be loaded on this host, so they were "
            "retrained from the dataset. Results are the same.")

# sidebar controls
st.sidebar.header("Controls")
chosen = st.sidebar.selectbox("Model", list(models.keys()))
st.sidebar.write(f"Upload a test CSV: feature columns plus a `{target_col}` column.")
uploaded = st.sidebar.file_uploader("Test CSV", type=["csv"])
use_bundled = st.sidebar.checkbox("Use bundled test_data.csv", value=not bool(uploaded))

# choose the data source
if uploaded is not None and not use_bundled:
    df = pd.read_csv(uploaded)
    src = "uploaded file"
elif (BASE / "test_data.csv").exists():
    df = pd.read_csv(BASE / "test_data.csv")
    src = "bundled test_data.csv"
else:
    st.warning("Upload a CSV or tick 'Use bundled test_data.csv'.")
    st.stop()

st.success(f"Loaded {len(df)} rows from {src}.")

# make sure the expected feature columns are present
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    st.error(f"CSV is missing {len(missing)} feature column(s): {missing[:5]} ...")
    st.stop()

has_labels = target_col in df.columns
X = scaler.transform(df[feature_cols].values)
if not has_labels:
    st.warning(f"No `{target_col}` column found - showing predictions only.")

# results for the selected model
model = models[chosen]
st.subheader(f"Results: {chosen}")

if has_labels:
    y = df[target_col].values
    scores, pred = get_metrics(model, X, y)

    cols = st.columns(6)
    for col, name in zip(cols, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
        col.metric(name, f"{scores[name]:.3f}")

    left, right = st.columns(2)
    with left:
        st.write("**Confusion matrix**")
        cm = confusion_matrix(y, pred)
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.write("**Classification report**")
        rep = classification_report(y, pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).transpose().round(3), use_container_width=True)
else:
    pred = model.predict(X)

# predictions preview + download
out = df.copy()
out["prediction"] = pred
with st.expander("Predictions"):
    st.dataframe(out.head(25), use_container_width=True)
    st.download_button("Download predictions CSV", out.to_csv(index=False).encode(),
                       "predictions.csv", "text/csv")

# all five models on the same test data
if has_labels:
    st.subheader("All models on this test data")
    y = df[target_col].values
    table = []
    for name, m in models.items():
        s, _ = get_metrics(m, X, y)
        table.append({"Model": name, **{k: round(v, 4) for k, v in s.items()}})
    comp = pd.DataFrame(table).set_index("Model")
    st.dataframe(comp.style.highlight_max(axis=0, color="#d4edda").format("{:.4f}"),
                 use_container_width=True)
    st.caption(f"Best MCC on this data: {comp['MCC'].idxmax()}")
