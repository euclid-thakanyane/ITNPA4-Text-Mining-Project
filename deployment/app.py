"""Researchly – research abstract classifier (ITNPA4-B34, Question 2.3).
Run:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from preprocess import clean_text

HERE = Path(__file__).parent
st.set_page_config(page_title="Researchly – Abstract Classifier", page_icon="📄", layout="centered")


@st.cache_resource
def load_model():
    model = joblib.load(HERE / "model.joblib")
    meta = json.loads((HERE / "model_meta.json").read_text())
    return model, meta


model, meta = load_model()
LABELS = {"ai": "AI / machine-learning method", "classical": "Classical (econometric / statistical) method"}

st.title("📄 Researchly – Research Abstract Classifier")
st.caption("SAnalytics expert system prototype: paste a research abstract and the model predicts its research-method category.")

with st.sidebar:
    st.header("Deployed model")
    st.write(f"**{meta['model']}**")
    st.write("Pipeline: spaCy denoising + lemmatisation → TF-IDF → classifier")
    tm = meta["test_metrics"]
    st.metric("Test accuracy", f"{tm['test_accuracy']:.3f}")
    st.metric("Test macro-F1", f"{tm['test_f1_macro']:.3f}")
    st.metric("Test ROC-AUC", f"{tm['test_roc_auc']:.3f}")

EXAMPLE = ("This paper proposes a long short-term memory (LSTM) recurrent neural network to forecast monthly "
           "inflation and compares it with random forest and support vector regression models.")
text = st.text_area("Research abstract", value=st.session_state.get("text", ""), height=220,
                    placeholder="Paste an abstract here…")
c1, c2 = st.columns([1, 1])
predict = c1.button("Predict", type="primary", use_container_width=True)
if c2.button("Load example", use_container_width=True):
    st.session_state["text"] = EXAMPLE
    st.rerun()

if predict:
    if len(text.split()) < 5:
        st.warning("Please enter a longer abstract (at least 5 words).")
    else:
        cleaned = clean_text(text)
        proba = model.predict_proba([cleaned])[0]
        classes = list(model.classes_)
        pred = classes[proba.argmax()]
        st.success(f"Predicted category: **{pred}** – {LABELS.get(pred, pred)}  (confidence {proba.max():.1%})")
        dfp = pd.DataFrame({"category": classes, "probability": proba}).set_index("category")
        st.subheader("Class probabilities")
        st.bar_chart(dfp)
        st.dataframe(dfp.style.format({"probability": "{:.3f}"}), use_container_width=True)
        with st.expander("Denoised text fed to the model"):
            st.write(cleaned)
