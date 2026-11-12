# ITNPA4-B34 – Researchly: Research Papers Expert System & Research Topic Analysis

This is a text-mining project with two parts, based on the brief by Matanga (2026).

* **Part A (Questions 1–2): Researchly expert system.** Classifies research abstracts (`datasets/research_abstracts.json`, 1 138 abstracts) by research-method category (`ai` vs `classical`). It covers denoising, feature analysis, a baseline, five ML models and GridSearchCV tuning, and deploys the best model as a Streamlit web app.
* **Part B (Questions 3–4): topic analysis.** Analyses the five-article corpus (`datasets/research_corpus.zip`). It covers spaCy denoising, TextRank and LSA summarisation, gensim LDA and a pyLDAvis intertopic distance map.

## Key results

| | Baseline (TF-IDF + MultinomialNB) | Final (tuned TF-IDF + Logistic Regression) |
|---|---|---|
| Test accuracy | 0.809 | **0.916** |
| Test macro-F1 | 0.524 | **0.869** |
| Test F1 (ai) | 0.157 | **0.791** |
| 5-fold CV macro-F1 | 0.468 | **0.858** |

LDA: **3 topics**, chosen by mean C_v coherence over 3 seeds (0.486; final model C_v 0.498). The topics are measurement reliability, the research/design process, and qualitative rigour through triangulation.

## Repository structure

```
docs/          PROJECT_DOCUMENTATION.md (Q1–4 write-up), figures/, lda_intertopic_distance_map.html
datasets/      original datasets + cleaned CSVs
notebooks/     Q1_Q2_Researchly_Expert_System.ipynb, Q3_Q4_Topic_Analysis.ipynb (executed, outputs included)
deployment/    Streamlit app, trained model, requirements, Dockerfile, README (deployment steps)
results/       JSON files with every computed metric
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace Q1_Q2_Researchly_Expert_System.ipynb
jupyter nbconvert --to notebook --execute --inplace Q3_Q4_Topic_Analysis.ipynb
```

## Deploy the web app

```bash
cd deployment && pip install -r requirements.txt && streamlit run app.py
```

See [`deployment/README.md`](deployment/README.md) for the Docker and Streamlit Community Cloud options.

## Note on the corpus

The brief describes `research_corpus.zip` as AI research articles, but the supplied files are research-methodology papers (mixed methods, reflexivity, developmental research, Cronbach's alpha, triangulation). The pipeline does not depend on the domain, and the topics are interpreted for the corpus as supplied.
