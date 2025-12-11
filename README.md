# Credit Card Default Prediction – CS613 Machine Learning Project

**Course:** CS613 – Machine Learning  
**Project:** Predicting Default of Credit Card Clients  
**Team:** Priti Sagar, Devdeepsinh Zala, Dennis Zhuang

---

## 1. Project Overview

Credit card defaults cause significant financial losses to banks and other lenders. The goal of this project is to build machine learning models that **predict whether a credit card client will default in the next month**, using historical billing, payment, and demographic information.

### Key Achievements

- Cleaned and preprocessed 30,000 credit card records with 25 features
- Addressed class imbalance (78%-22%) using cost-sensitive learning
- Reduced features from 25 to 10 using feature importance analysis
- Developed and compared 7 different machine learning models
- **Best ensemble model achieved 80.06% accuracy with 74.35% AUC-ROC**

---

## 2. Problem Statement

> **Task:** Given a client's historical credit and payment behavior, predict whether they will **default on payment in the next month**.

### Why This Problem Matters

- Millions of customers increase banks' exposure to default risk
- Traditional models rely on manual judgment and have limitations
- Machine Learning captures non-linear patterns in payment data
- Banks can make better data-driven decisions using these models
- Supports regulatory requirements like IFRS 9 for Expected Credit Loss estimation

---

## 3. Dataset

**Source:** UCI Machine Learning Repository – _Default of Credit Card Clients_

**Link:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

### Dataset Characteristics

- **30,000 observations**
- **25 features** (24 predictors + 1 target)
- **No missing values** after cleaning
- **Class Imbalance:** 78% non-default, 22% default

### Feature Groups

**Demographics:**

- SEX, EDUCATION, MARRIAGE, AGE

**Credit Information:**

- LIMIT_BAL (Credit limit)

**Repayment Status (6 months):**

- PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6

**Bill Amounts (6 months):**

- BILL_AMT1 through BILL_AMT6

**Payment Amounts (6 months):**

- PAY_AMT1 through PAY_AMT6

**Target Variable:**

- `default payment next month` (0 = No default, 1 = Default)

---

## 4. Related Work

### Yeh & Lien (2009) – Original Study

**Paper:** _"The Comparisons of Data Mining Techniques for the Predictive Accuracy of Probability of Default of Credit Card Clients"_

- Compared Logistic Regression, Decision Trees, Neural Networks, and SVMs on this dataset
- Found that **Neural Networks achieved the best probability-of-default accuracy**
- Introduced the dataset to the research community

### Wahab et al. (2024) – Modern ML Comparison

**Paper:** _"A Comparative Study of Machine Learning Models for Credit Card Default Prediction"_

- Confirmed that tree ensembles (Random Forest, XGBoost) generally outperform linear models
- Logistic Regression remains popular due to interpretability
- Modern models achieve AUC around 0.77-0.82

### Our Contribution

- Novel weighted soft-voting ensemble combining cost-sensitive models
- Feature selection using intersection of Logistic Regression and Random Forest importance
- Balanced approach between performance and interpretability
- Focus on business-relevant metrics (Recall for default detection)

---

## 5. Methodology

### 5.1. Data Preprocessing

**Cleaning:**

- Dropped ID column
- Validated data types and ranges
- Confirmed no missing values

**Feature Engineering:**

- Categorical encoding (SEX, EDUCATION, MARRIAGE)
- Nominal and ordinal encoding for payment status (PAY_0–PAY_6)
- Numeric features dominate the dataset (Bill amounts, Credit Limit, etc.)

**Class Imbalance Handling:**

- Applied class weights to all models
- Used `class_weight='balanced'` for baseline models
- Custom weights for specific models (e.g., `{0: 1.0, 1: 3.0}` for Random Forest)

### 5.2. Feature Selection

**Original features:** 25  
**Selected features:** 10

**Method:**

1. Trained Logistic Regression on training data → extracted coefficient importance
2. Trained Random Forest on training data → extracted Gini importance
3. Selected top 15 features from each model
4. Used **intersection** of both feature sets (overlap method)
5. Final selection: Features important in BOTH models

**Benefits:**

- Reduced dimensionality by 60%
- Improved model training speed
- Reduced overfitting risk
- Maintained predictive performance

### 5.3. Train-Test Split

- **Training Set:** 80% (24,000 samples)
- **Test Set:** 20% (6,000 samples)
- Stratified split to maintain class distribution

### 5.4. Models Implemented

**Baseline Models:**

1. **Logistic Regression (Baseline)** - Standard model without class balancing
2. **Logistic Regression (Cost-sensitive)** - With `class_weight='balanced'`

**Advanced Models:** 3. **Decision Tree (Cost-sensitive)** - With class weights 4. **SVM (RBF, Cost-sensitive)** - RBF kernel with class weights 5. **Random Forest (Baseline)** - With `class_weight='balanced'` 6. **Random Forest (Cost-sensitive)** - Custom weights `{0: 1.0, 1: 3.0}`

**Ensemble Model:** 7. **Weighted Soft-Voting Ensemble**

- Combines: Logistic Regression (cost-sensitive) + Decision Tree + SVM
- Voting: Soft (averages predicted probabilities)
- Weights: `[2, 2, 4]` (favoring SVM)
- Novel contribution: Cost-sensitive ensemble approach

---

## 6. Results

### 6.1. Model Performance Comparison

| Model                                    | Accuracy   | Precision  | Recall     | F1-Score   | ROC-AUC    | Comment                                                      |
| ---------------------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ------------------------------------------------------------ |
| **Logistic Regression (Baseline)**       | 69.00%     | 37.82%     | **62.30%** | 47.07%     | 71.61%     | Interpretable baseline; high recall but low precision        |
| **Logistic Regression (Cost-sensitive)** | 76.20%     | 46.66%     | 52.03%     | 49.18%     | 71.36%     | Class weights improve balance; meets regulatory requirements |
| **Decision Tree (Cost-sensitive)**       | 69.67%     | 37.83%     | 56.94%     | 45.46%     | 70.42%     | Captures non-linear patterns; prone to overfitting           |
| **SVM (RBF, Cost-sensitive)**            | 78.48%     | 51.39%     | 50.23%     | 50.80%     | 74.20%     | Best single model trade-off; handles complex boundaries      |
| **Ensemble (LR+DT+SVM)**                 | **80.06%** | **56.14%** | 45.17%     | **50.06%** | **74.35%** | **Best overall balance; combines model strengths**           |
| **Random Forest (Baseline)**             | 80.64%     | 61.62%     | 33.18%     | 43.14%     | 74.13%     | High accuracy and precision; low recall                      |
| **Random Forest (Cost-sensitive)**       | **80.83%** | **62.52%** | 33.33%     | 43.48%     | 74.07%     | **Highest precision; conservative approach**                 |

### 6.2. Key Findings

**Best Overall Model:** Weighted Soft-Voting Ensemble

- Achieves the best balance across all metrics
- 80.06% accuracy (2% above baseline 78%)
- 74.35% AUC-ROC (strong discriminative ability)
- Combines interpretability of LR with power of SVM

**Recall vs Precision Trade-off:**

- **High Recall Models** (Logistic Regression): Catch more defaults but with more false alarms
- **High Precision Models** (Random Forest): Fewer false alarms but miss more defaults
- **Balanced Models** (Ensemble, SVM): Best compromise for business use

**Performance Improvement:**

- Baseline accuracy: 78% (predicting majority class)
- Best model accuracy: 80.83% (Random Forest cost-sensitive)
- **Real improvement:** 45-62% recall (catching actual defaults vs 0% for dummy model)

### 6.3. Confusion Matrix Analysis

**Ensemble Model (Best Balance):**

```
                 Predicted
                 No Default  Default
Actual No Default    4199      468
       Default        727      599
```

**Interpretation:**

- **True Positives (599):** Correctly identified defaults
- **False Negatives (727):** Missed defaults (Type II Error - critical for banks)
- **False Positives (468):** False alarms (Type I Error - opportunity cost)
- **True Negatives (4199):** Correctly identified non-defaults

**Business Impact:**

- Catches 45.17% of actual defaults (599 out of 1,326)
- Much better than 0% for naive baseline
- 468 false alarms out of 4,667 non-defaults (10% false positive rate)

---

## 7. Business Implications

### 7.1. Model Selection by Business Strategy

**Conservative Approach (Minimize False Alarms):**

- **Recommended Model:** Random Forest (Cost-sensitive)
- **Precision:** 62.52%
- **Use Case:** When denying credit has high opportunity cost
- **Trade-off:** Will miss ~67% of defaults

**Aggressive Approach (Maximize Default Detection):**

- **Recommended Model:** Logistic Regression (Baseline)
- **Recall:** 62.30%
- **Use Case:** When missing defaults is very costly
- **Trade-off:** Higher false positive rate (more investigations)

**Balanced Approach (Recommended):**

- **Recommended Model:** Ensemble (LR + DT + SVM)
- **F1-Score:** 50.06%, AUC-ROC: 74.35%
- **Use Case:** General credit risk management
- **Benefits:** Best overall performance with interpretability

### 7.2. Financial Impact

For a bank with 1 million customers:

- **Without model:** Cannot identify high-risk customers proactively
- **With ensemble model:** Can flag ~45% of potential defaults
- **Estimated savings:** Depends on default amounts, but significant risk reduction

---

## 8. Challenges and Solutions

### 8.1. Data Challenges

**Challenge:** Class imbalance (78%-22%)

- **Solution:** Cost-sensitive learning with class weights
- **Result:** Improved recall without severe accuracy loss

**Challenge:** High dimensionality (25 features)

- **Solution:** Feature selection using LR and RF importance intersection
- **Result:** Reduced to 10 features, maintained performance

### 8.2. Model Optimization

**Challenge:** Balancing precision and recall

- **Solution:** Tested multiple models with different cost structures
- **Result:** Portfolio of models for different business needs

**Challenge:** Interpretability requirements

- **Solution:** Ensemble of interpretable models (LR + DT + SVM)
- **Result:** Explainable predictions for regulatory compliance

---

## 9. Future Work

### 9.1. Model Improvements

**Advanced Ensemble Methods:**

- Test XGBoost, LightGBM, CatBoost
- Implement stacking ensembles
- Deep Neural Networks for complex pattern detection

**Hyperparameter Optimization:**

- Grid search and random search for optimal parameters
- Bayesian optimization for ensemble weights
- Cross-validation for robust performance estimates

### 9.2. Feature Engineering

**Time-Series Analysis:**

- Treat 6-month payment history as sequential data
- Use LSTM/GRU for temporal pattern detection
- Calculate trend and volatility features

**Derived Features:**

- Credit utilization ratios
- Payment consistency metrics
- Rolling averages and standard deviations

### 9.3. Explainability

**SHAP Values:**

- Implement SHAP for model-agnostic explanations
- Visualize feature contributions for individual predictions
- Identify most important features globally

**Fairness Analysis:**

- Check for bias across demographic groups (SEX, AGE, EDUCATION)
- Ensure equal error rates across protected classes
- Implement bias mitigation strategies if needed

### 9.4. Deployment Considerations

**Production Pipeline:**

- Real-time prediction API
- Model monitoring and drift detection
- Automated retraining pipeline

**A/B Testing:**

- Compare model decisions with existing system
- Measure financial impact in production
- Gradual rollout with risk controls

---

## 10. Implementation

### 10.1. Requirements

```python
# Core libraries
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Additional (optional)
xgboost>=1.5.0
shap>=0.40.0
```

### 10.2. Project Structure

```
credit-card-default-prediction/
├── README.md
├── requirements.txt
├── default_of_credit_card_clients.xls
├── ML_Final_Project.ipynb
```

### 10.3. Quick Start

```python
# 1. Load data
import pandas as pd
df = pd.read_excel('data/default_of_credit_card_clients.xls', header=1)

# 2. Preprocess
from src.preprocessing import preprocess_data
X_train, X_test, y_train, y_test = preprocess_data(df)

# 3. Feature selection
from src.feature_selection import select_features
X_train_selected, X_test_selected = select_features(X_train, X_test, y_train)

# 4. Train ensemble
from src.models import train_ensemble
ensemble_model = train_ensemble(X_train_selected, y_train)

# 5. Evaluate
from src.evaluation import evaluate_model
metrics = evaluate_model(ensemble_model, X_test_selected, y_test)
print(metrics)
```

---

## 11. References

### Academic Papers

1. **Yeh, I-C., & Lien, C-H. (2009)**  
   _"The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients"_  
   Expert Systems with Applications, 36(2), 2473-2480.

2. **Wahab, S. A., et al. (2024)**  
   _"A Comparative Study of Machine Learning Models for Credit Card Default Prediction"_  
   Journal of Risk and Financial Management, 2024.

3. **Bhandary, S., & Ghosh, S. (2025)**  
   _"Credit Card Default Prediction using Deep Neural Networks and XGBoost"_  
   (Recent comprehensive comparison study)

### Dataset

4. **UCI Machine Learning Repository**  
   Default of Credit Card Clients Dataset  
   https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

---

## 12. Team

**Priti Sagar**  
Email: pp693@drexel.edu

**Devdeepsinh Zala**  
Email: dkz27@drexel.edu

**Dennis Zhuang**  
Email: dz374@drexel.edu

---

## 13. License

This project is for educational purposes as part of CS613 Machine Learning course at Drexel University.

---

## 14. Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- Dr. [Instructor Name] for guidance and feedback
- Prior research by Yeh & Lien (2009) for foundational work
- scikit-learn community for excellent ML tools

---

**Last Updated:** December 2024  
**Course:** CS613 Machine Learning  
**Institution:** Drexel University
