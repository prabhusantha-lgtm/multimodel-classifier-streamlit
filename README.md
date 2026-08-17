# Multi-Model Classifier Explorer — Breast Cancer Wisconsin (Diagnostic)

BITS WILP · M.Tech (AIML / DSE) · Machine Learning · Assignment 2

An end-to-end classification workflow: six models trained on a real diagnostic
dataset, evaluated on six metrics, and served through an interactive Streamlit
app deployed on Streamlit Community Cloud.

---

## a. Problem statement

Given 30 numeric measurements computed from digitised images of fine-needle
aspirates of breast masses, classify each tumour as **malignant (0)** or
**benign (1)**. This is a binary classification problem where minimising false
negatives (calling a malignant tumour benign) is especially important, so recall
and MCC are watched alongside accuracy.

## b. Dataset description

- **Source:** Breast Cancer Wisconsin (Diagnostic) — UCI Machine Learning
  Repository (also distributed inside scikit-learn as `load_breast_cancer`).
- **Instances:** 569 (≥ 500 required ✔)
- **Features:** 30 continuous features (≥ 12 required ✔) — mean, standard error,
  and "worst" values of ten cell-nucleus properties (radius, texture, perimeter,
  area, smoothness, compactness, concavity, concave points, symmetry, fractal
  dimension).
- **Target:** binary — `0 = malignant` (212 cases), `1 = benign` (357 cases).
- **Train / test split:** 80 / 20, stratified, `random_state = 42`
  (455 train / 114 test rows). `test_data.csv` in this repo is exactly that
  114-row held-out test split (original unscaled features + true `target`).
- **Preprocessing:** a single `StandardScaler` fitted on the training split and
  reused everywhere. Linear and distance-based models (LR, kNN, NB) need it;
  tree-based models are scale-invariant and are unaffected by it.

## c. GitHub repository link

https://github.com/prabhusantha-lgtm/multimodel-classifier-streamlit

## d. Models used

Five classifiers were trained on the same dataset and evaluated on the 114-row
held-out test set: Logistic Regression, Decision Tree, kNN, Naive Bayes and
Random Forest (ensemble).

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9147 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |

*Metrics are computed on the test set; AUC uses predicted class-1 probabilities.*

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Top performer (Acc 0.982, AUC 0.995, MCC 0.962). After standardisation the two classes are close to linearly separable, so a linear decision boundary fits almost perfectly with balanced precision and recall and no overfitting. |
| Decision Tree | Weakest model (Acc 0.912, MCC 0.817) and the lowest AUC (0.915). A single tree has high variance and makes hard axis-aligned splits, so its probability estimates are coarse — reflected in the poor AUC despite a respectable accuracy. |
| kNN | Very strong (Acc 0.974, MCC 0.944) with **perfect recall (1.000)** — it misses no benign cases. Slightly lower precision (0.960) means a few malignant cases are called benign; scaling is essential here since kNN relies on Euclidean distance. |
| Naive Bayes | Solid AUC (0.987) but lower accuracy/MCC (0.930 / 0.849). Its feature-independence assumption is violated — the "mean / SE / worst" versions of each measurement are highly correlated — which caps its point-prediction quality even though the ranking of probabilities stays good. |
| Random Forest (Ensemble) | Robust and well-calibrated (AUC 0.994) with balanced precision/recall (0.958). The ensemble fixes the single tree's variance problem, but on this fairly linear dataset it is narrowly out-performed by the linear models. |
| **Overall winner for this dataset?** | **Logistic Regression** — highest accuracy, AUC, F1 and MCC of the five models. It is also the simplest, fastest and most interpretable of the top performers, which makes it the natural choice for this near-linearly-separable dataset. |

## Repository structure

```
bits-ml-a2/
├── app.py                     # Streamlit application
├── requirements.txt
├── README.md
├── test_data.csv              # 114-row held-out test split (features + target)
└── model/
    ├── train_models.py        # trains all 6 models, writes metrics + artifacts
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── scaler.pkl             # StandardScaler fitted on the training split
    ├── feature_names.json     # ordered feature columns the app expects
    └── metrics.csv            # comparison table (source of the table above)
```

## How to run locally

```bash
pip install -r requirements.txt
python model/train_models.py     # (re)generates models + test_data.csv
streamlit run app.py
```

## Streamlit app features

- **CSV upload** of test data (or one click to use the bundled `test_data.csv`).
- **Model selection dropdown** across all six models.
- **Evaluation metrics** (Accuracy, AUC, Precision, Recall, F1, MCC) shown live
  for the selected model on the uploaded data.
- **Confusion matrix** heatmap **and** full **classification report**.
- A live **all-models comparison table** on the uploaded data, highlighting the
  best value per metric.
- Downloadable predictions CSV.

The app loads the saved `*.pkl` models; if a host has an incompatible
scikit-learn version it transparently retrains from the bundled dataset so it
never fails to render.

## Live Streamlit app link

> **`<PASTE YOUR STREAMLIT APP URL HERE>`**
