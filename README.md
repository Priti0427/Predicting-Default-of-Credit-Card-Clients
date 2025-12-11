# Credit Card Default Prediction – CS613 Machine Learning Project

Course: **CS613 – Machine Learning**  
Project: **Predicting Default of Credit Card Clients**

## 1. Project Overview

Credit card defaults cause significant financial losses to banks and other lenders.  
The goal of this project is to build machine learning models that **predict whether a credit card client will default in the next month**, using historical billing, payment, and demographic information.

We:

- Use the **UCI “Default of Credit Card Clients”** dataset.
- Benchmark **classic statistical models** (Logistic Regression) against **modern ML models** (Decision Trees, SVMs, and Ensembles).
- Focus on both **predictive performance** and **interpretability**, with an eye toward **real-world credit risk use-cases**.

Ultimately, this work is inspired by and extends prior research that shows:

- Neural networks and ensembles can outperform traditional models in terms of metrics like **AUC** and **F1**.
- Logistic regression remains popular in finance due to its **interpretability** and regulatory acceptance.

## 2. Problem Statement

> **Task:** Given a client’s historical credit and payment behavior, predict whether they will **default on payment in the next month**.

**Why this matters:**

- Helps banks **identify high-risk customers** before issuing or renewing credit.
- Supports **IFRS 9**-style tasks like estimating **Probability of Default (PD)** and Expected Credit Loss.
- Reduces credit losses and improves financial stability and decision-making.

## 3. Dataset

**Source:**

- UCI Machine Learning Repository – _Default of Credit Card Clients_

**Link:**

- [(official UCI link)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

**Size & Structure:**

- **30,000 observations**
- **25 columns** total:
  - **24 features**
  - **1 target**: `default payment next month` (0 = No default, 1 = Default)

**Feature groups:**

- **Demographics**
  - `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
- **Credit Limit**
  - `LIMIT_BAL`
- **Repayment Status (last 6 months)**
  - `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6`
- **Bill Amounts (last 6 months)**
  - `BILL_AMT1` … `BILL_AMT6`
- **Previous Payments (last 6 months)**
  - `PAY_AMT1` … `PAY_AMT6`

**Target variable:**

- `default payment next month`
  - `0` – No default
  - `1` – Default

**Class balance (imbalanced):**

- ~**78%** non-default
- ~**22%** default

We explicitly treat this as an **imbalanced classification** problem.

## 4. Related Work

### Yeh & Lien (2009)

> _The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients_

- Compared: KNN, Logistic Regression, Discriminant Analysis, Naïve Bayes, ANN, Classification Trees.
- Introduced the **Sorting Smoothing Method (SSM)** to estimate the _“real”_ probability of default from binary labels.
- Found that **Artificial Neural Networks (ANN)** provided the best **probability calibration**, with:
  - Regression slope `B ≈ 0.998` (close to 1)
  - Intercept `A ≈ 0.0145` (close to 0)
  - `R² ≈ 0.965`
- Concluded that ANN was uniquely capable of accurately estimating the true PD on this dataset.

### Bhandary & Ghosh (2025) / Wahab et al. (2024) – Modern ML Comparisons

Recent work revisiting the Taiwan dataset shows:

- **Deep Neural Networks (DNNs)** and **tree ensembles** (Random Forest, XGBoost) outperform traditional models like Logistic Regression and SVM in:
  - **Accuracy**
  - **F1 score**
  - **G-mean**
  - **AUC**
- **DNN/XGBoost** typically achieve AUC around **0.77**, vs ~0.73 for Logistic Regression.
- Feature importance analysis highlights:
  - **BillSum / bill amounts** and **Age** as key predictors.
  - Education and marital status as relatively weak predictors.

These studies motivate our decision to:

- Use **Logistic Regression** as a **baseline** (standard in credit scoring).
- Compare against **Decision Trees** and **SVMs**.
- Extend to **ensemble models** (e.g., Random Forest, Voting Classifier, AdaBoost) for potential performance gains.
- Explore **interpretability** and the question of whether ensembles can be made explainable in a financial context.

---

## 5. Methods

### 5.1. Basic Approach

We follow a standard **supervised machine learning pipeline**:

1. **Data Loading & Cleaning**

   - Load data from `.xls` file.
   - Drop/rename columns as needed (e.g., `ID`).
   - Confirm no missing values or handle them if present.

2. **Exploratory Data Analysis (EDA)**

   - `df.head()`, `df.info()`, basic summary stats.
   - Target distribution (default vs non-default).
   - Histograms for key numeric variables (e.g., `LIMIT_BAL`, `AGE`, `BILL_AMT*`, `PAY_AMT*`).
   - Correlation heatmap between numerical features and the target.

3. **Preprocessing & Feature Engineering**

   - Encode categorical variables:
     - `SEX` → binary or one-hot.
     - `EDUCATION`, `MARRIAGE`, `PAY_*` → one-hot or category remapping.
   - **Standardize numeric features** (e.g., using `StandardScaler` in a `Pipeline`).
   - Optionally:
     - Handle class imbalance (class weights / resampling).
     - Create derived features (e.g., total bill, total payment, utilization ratios).

4. **Model Training**

We plan to train and compare:

- **Logistic Regression (LR)**

  - Baseline model.
  - Popular in credit risk due to interpretability.
  - Potentially with `class_weight='balanced'`.

- **Decision Tree (DT)**

  - Intuitive, rule-based model.
  - Good for capturing non-linear relationships.

- **Support Vector Machine (SVM)**

  - Handles high-dimensional boundaries.
  - Can capture complex margins (with kernel tricks).

- **Ensemble Methods**
  - **Random Forest**
  - **Voting Classifier** (e.g., LR + DT+SVM)
  - Motivation: prior work shows tree ensembles often outperform single models.

5. **Model Evaluation**

We evaluate using:

- **Accuracy**
- **Precision / Recall**
- **F1 Score**
- **ROC Curve & AUC**
- **Confusion Matrix**

Given class imbalance, we will pay particular attention to **Recall** and **AUC**, rather than Accuracy alone.

## 6. Implementation Details

### 6.1. Code & Environment

- Language: **Python**
- Typical libraries:
  - `pandas`, `numpy`
  - `scikit-learn` for models & preprocessing
  - `matplotlib` / `seaborn` for plots
- Development:
  - **Google Colab** notebook .

### 6.2. Basic Usage

1. Upload the dataset file:

   - `default of credit card clients.xls`

2. Open the notebook:

   - `ml_final_project.ipynb`

3. Run cells in order:

   - Data loading
   - EDA
   - Preprocessing
   - Model training
   - Model evaluation
   - Ensemble models & explainability

## 7. Results & Observations

Planned content:

- **Baseline performance**:
  - Logistic Regression on cleaned + scaled data.
- **Comparative performance**:
  - Decision Tree vs SVM vs Logistic Regression.
  - (Later) Ensemble models vs single models.
- **Key observations** (to fill in):
  - Which model achieved the best **AUC**?
  - How large is the gap between Logistic Regression and more complex models?
  - What features showed up as most important (e.g., credit limit, repayment status, bill amounts)?
  - How does class imbalance affect performance?

Example structure to fill in:

Logistic Regression:

- AUC: TBD
- F1: TBD

Decision Tree:

- AUC: TBD
- F1: TBD

SVM:

- AUC: TBD
- F1: TBD

Ensemble (e.g., Random Forest):

- AUC: TBD
- F1: TBD

Main findings:

- TBD

8. Future Work & Research Extensions

Based on both prior work and our project direction, potential extensions include:

Ensemble & Advanced Models

Implement and compare Random Forest, XGBoost, Gradient Boosting, and Deep Neural Networks (DNNs).

Study whether combining models improves AUC, F1, and calibration.

Probability Calibration

Inspired by Yeh & Lien’s “Sorting Smoothing Method”.

Evaluate how well each model’s predicted probability matches true default rates.

Explore calibration methods (e.g., Platt scaling, isotonic regression).

Cost-Sensitive Learning

Assign different costs to:

False Negatives (missed defaulters)

False Positives (rejecting good customers)

Tune decision thresholds to minimize expected financial loss, not just maximize AUC.

Explainability in Finance

Investigate whether ensemble models can be made explainable enough for practical credit decisioning.

Use tools like feature importance, SHAP, or LIME.

Compare interpretability vs performance trade-offs.

Fairness & Bias

Analyze model performance across groups (e.g., by SEX, AGE buckets).

Check for disparities in error rates.

Explore basic bias-mitigation strategies (removal of sensitive features, reweighting, etc.).

Richer Model Architectures

Compare with Deep Neural Networks (DNNs) and potentially sequence models (e.g., treating 6-month history as a short time series).

Explore LSTM/GRU for temporal modelling of monthly bills and payments.

Data Diversity

Combine this dataset with other credit risk datasets (where available) to improve generalization.

Compare model behavior across different populations or time periods.

9. References

Yeh & Lien (2009)
I-Cheng Yeh & Che-hui Lien,
“The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients”,
Expert Systems with Applications, 36(2):2473–2480, 2009.

Bhandary & Ghosh (2025) – (Credit card default prediction with DNN and XGBoost)
(Exact citation to be finalized by team.)

Wahab et al. (2024)
“A Comparative Study of Machine Learning Models for Credit Card Default Prediction”,
Journal of Risk and Financial Management, 2024.

10. Project Structure

```
.
├── README.md
├── data/
│   └── default_of_credit_card_clients.xls   # (not committed if large / private)
├── notebooks/
│   └── Copy_of_ml_project_Priti.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
└── reports/
    ├── slides/
    └── final_report.pdf


This can be adjusted as the project evolves.
```

Team: **Devdeep · Dennis · Priti**  
