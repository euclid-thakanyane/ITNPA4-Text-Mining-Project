# Researchly – Abstract Classifier: deployment (Questions 1 & 2)

This folder contains everything needed to run the best-performing configuration from Question 2
(**GridSearchCV-tuned TF-IDF + Logistic Regression**, test macro-F1 = 0.869, accuracy = 0.916, ROC-AUC = 0.946)
as a web interface for testing.

| File | Purpose |
|---|---|
| `app.py` | Streamlit web interface (text box → **Predict** → predicted category + class probabilities) |
| `preprocess.py` | The same spaCy denoising/lemmatisation used during training (avoids training/serving skew) |
| `model.joblib` | Trained scikit-learn pipeline (TF-IDF vectoriser + Logistic Regression), refitted on all 1 125 cleaned abstracts |
| `model_meta.json` | Model name, tuned hyper-parameters and held-out test metrics shown in the sidebar |
| `requirements.txt` | Pinned runtime dependencies (incl. the spaCy `en_core_web_sm` model wheel) |
| `Dockerfile` | Optional container build |

## Option 1 – run locally (3 commands)

Requires Python 3.10–3.13.

```bash
cd deployment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501>, paste an abstract (or click **Load example**) and press **Predict**.

## Option 2 – Docker

```bash
cd deployment
docker build -t researchly .
docker run -p 8501:8501 researchly
```

## Option 3 – Streamlit Community Cloud

1. Push this repository to GitHub.
2. On <https://share.streamlit.io> choose **New app**, select the repository, branch `main` and main file `deployment/app.py`.
3. Streamlit Cloud installs `deployment/requirements.txt` automatically and gives a public URL.

## Notes

* `model.joblib` was saved with scikit-learn 1.9.1 – keep that version (pinned in `requirements.txt`) to load it without warnings.
* To retrain, run `notebooks/Q1_Q2_Researchly_Expert_System.ipynb` top to bottom; its last cells overwrite `model.joblib` and `model_meta.json`.
* Labels: `ai` = abstract describes an AI / machine-learning method; `classical` = classical econometric / statistical method.
