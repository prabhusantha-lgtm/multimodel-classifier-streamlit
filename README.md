# Multi-Model Classifier Explorer — Breast Cancer Wisconsin (Diagnostic)

An end-to-end classification workflow: five models trained on a real diagnostic
dataset, evaluated on six metrics, and served through an interactive Streamlit
app deployed on Streamlit Community Cloud.

## 1. Problem statement

Given 30 numeric measurements computed from digitised images of fine-needle
aspirates of breast masses, classify each tumour as **malignant (0)** or
**benign (1)**. This is a binary classification problem where minimising false
negatives (calling a malignant tumour benign) matters most, so recall and MCC
are watched alongside accuracy.

## 2. Dataset description

- **Source:** Breast Cancer Wisconsin (Diagnostic) — UCI Machine Learning Repository
- **Instances:** 569 (≥ 500 required)
- **Features:** 30 continuous (≥ 12 required) — ten cell-nucleus properties
  (radius, texture, perimeter, area, smoothness, compactness, concavity,
  concave points, symmetry, fractal dimension), each reported as mean, standard
  error and "worst" value.
- **Target:** binary — `0 = malignant` (212 cases), `1 = benign` (357 cases).
- **Train / test split:** 80 / 20, stratified, `random_state = 42`
  (455 train / 114 test). `test_data.csv` is the 114-row held-out test split.

## 3. GitHub repository link

https://github.com/prabhusantha-lgtm/multimodel-classifier-streamlit

## 4. Models used

Five classifiers were trained on the same dataset and evaluated on the 114-row
held-out test set: Logistic Regression, Decision Tree, kNN, Naive Bayes and
Random Forest (ensemble).

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9147 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9932 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

*Metrics are computed on the test set; AUC uses predicted class-1 probabilities.*

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall (Acc 0.982, AUC 0.995, MCC 0.962). Once the features are standardised the two classes are almost linearly separable, so a linear boundary fits well. Precision and recall stay balanced and there is no sign of overfitting. |
| Decision Tree | Weakest model here (Acc 0.912, MCC 0.817) and the lowest AUC (0.915). A single tree is high-variance and splits on hard thresholds, so its probability estimates are coarse. That explains the poor AUC even though the accuracy is acceptable. |
| kNN | Strong overall (Acc 0.956, MCC 0.905) at k=5. Recall is 0.972 and precision 0.959, so a few cases go wrong in each direction — it no longer catches every benign case as it did at higher k. Scaling is essential here because kNN works on Euclidean distance. |
| Naive Bayes | Good AUC (0.987) but lower accuracy and MCC (0.930 / 0.849). Its feature-independence assumption does not hold, since the mean, SE and worst versions of each measurement are strongly correlated. That caps the point predictions even though the probability ranking stays decent. |
| Random Forest (Ensemble) | Robust and well-calibrated (Acc 0.956, AUC 0.993). The ensemble removes the single tree's variance problem, but on this near-linear dataset the linear models still edge it out. |
| **Overall winner for this dataset?** | **Logistic Regression** — highest accuracy, AUC, F1 and MCC of the five, and also the simplest and most interpretable. A natural fit for a dataset that turns out to be close to linearly separable. |

## 5. Live Streamlit app link

https://multimodel-classifier-app-apcaudju5dpcfy3appwyksr.streamlit.app/
