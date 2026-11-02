"""Streamlit app: evaluate pre-trained classifiers on an uploaded test CSV (no retraining)."""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
)

from common import METRIC_COLUMNS, compute_metrics

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Classifier Evaluation Lab", page_icon="🧪", layout="wide")


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))


@st.cache_data
def load_meta():
    with open(os.path.join(MODEL_DIR, "meta.json")) as f:
        return json.load(f)


meta = load_meta()
features, target, classes = meta["features"], meta["target"], meta["classes"]
class_to_idx = {c: i for i, c in enumerate(classes)}

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Controls")
uploaded = st.sidebar.file_uploader("Upload test CSV", type=["csv"])
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
show_all = st.sidebar.checkbox("Compare all five models", value=True)
st.sidebar.caption(
    f"CSV must contain the target column **{target}** and all {len(features)} feature columns."
)

# ---------------------------------------------------------------- main
st.title("🧪 Classifier Evaluation Lab")
st.write(
    "Upload a **test** CSV, pick a pre-trained model and inspect its metrics. "
    "Models are loaded from disk; nothing is retrained."
)

if uploaded is None:
    st.info("Upload a CSV in the sidebar to begin. You can use `test_data.csv` from the repo.")
    st.stop()

df = pd.read_csv(uploaded)

missing = [c for c in features + [target] if c not in df.columns]
if missing:
    st.error(f"Missing columns in uploaded file: {missing}")
    st.stop()

df = df.dropna(subset=[target])
unknown = set(df[target].astype(str)) - set(classes)
if unknown:
    st.error(f"Target contains labels the models were not trained on: {sorted(unknown)}")
    st.stop()

X = df[features]
y_true = df[target].astype(str).map(class_to_idx).to_numpy()

with st.expander(f"Preview of uploaded data ({len(df)} rows)"):
    st.dataframe(df.head(20), width="stretch")

if len(np.unique(y_true)) < 2:
    st.warning("Uploaded data has only one class; AUC and MCC are undefined.")
    st.stop()


def evaluate(name):
    pipe = load_model(name)
    y_pred = pipe.predict(X)
    y_proba = pipe.predict_proba(X)
    return y_pred, y_proba, compute_metrics(y_true, y_pred, y_proba)


y_pred, y_proba, metrics = evaluate(model_name)

# ---------------------------------------------------------------- metrics
st.subheader(f"Results: {model_name}")
cols = st.columns(len(METRIC_COLUMNS))
for col, key in zip(cols, METRIC_COLUMNS):
    col.metric(key, f"{metrics[key]:.4f}")

left, right = st.columns(2)

with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="crest", cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.markdown("**Classification report**")
    report = classification_report(
        y_true, y_pred, labels=range(len(classes)), target_names=classes,
        output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

if len(classes) == 2:
    st.markdown("**ROC curve**")
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    fig2, ax2 = plt.subplots(figsize=(4.5, 3.6))
    ax2.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.3f}")
    ax2.plot([0, 1], [0, 1], "--", color="grey")
    ax2.set_xlabel("False positive rate")
    ax2.set_ylabel("True positive rate")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

# ---------------------------------------------------------------- comparison
if show_all:
    st.subheader("All models on the uploaded data")
    rows = []
    for name in MODEL_FILES:
        _, _, m = evaluate(name)
        rows.append({"ML Model Name": name, **m})
    comp = pd.DataFrame(rows).set_index("ML Model Name")
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8cf"),
                 width="stretch")
    st.bar_chart(comp[["Accuracy", "F1", "MCC"]])
