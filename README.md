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

---

## 3. Dataset

**Source:** UCI Machine Learning Repository – _Default of Credit Card Clients_

**Link:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

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

- One Hot Encoding MARRIAGE column and Ordinal encoding on EDUCATION column
- Numeric features dominate the dataset (Bill amounts, Credit Limit, etc.)

**Class Imbalance Handling:**

- Applied class weights to all models
- Used `class_weight='balanced'` for baseline models
- Custom weights for specific models (e.g., `{0: 1.0, 1: 3.0}` for Random Forest)

### 5.2. Feature Selection

**Original features:** 25

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

**Advanced Models:**

3. **Decision Tree (Cost-sensitive)** - With class weights
4. **SVM (RBF, Cost-sensitive)** - RBF kernel with class weights
5. **Random Forest (Baseline)** - With `class_weight='balanced'`
6. **Random Forest (Cost-sensitive)** - Custom weights `{0: 1.0, 1: 3.0}`

**Ensemble Model:**

7. **Weighted Soft-Voting Ensemble**
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
- 80.06% accuracy
- 74.35% AUC-ROC (strong discriminative ability)
- Combines interpretability of LR with power of SVM and decision Tree

**Recall vs Precision Trade-off:**

- **High Recall Models** (Logistic Regression): Catch more defaults but with more false alarms
- **High Precision Models** (Random Forest): Fewer false alarms but miss more defaults
- **Balanced Models** (Ensemble, SVM): Best compromise for business use

**Performance Improvement:**

- Baseline accuracy: 78% (predicting majority class)
- Best model accuracy: 80.83% (Random Forest cost-sensitive)
- **Real improvement:** 45-62% recall (catching actual defaults vs 0% for dummy model)

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

## 8. How to Run and Reproduce Results

### 8.1. Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or VS Code
- Internet connection (for cloning repository)

### 8.2. Quick Start Guide

#### **Step 1: Clone the Repository**

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-default-prediction.git
cd credit-card-default-prediction
```

#### **Step 2: Install Required Packages**

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl jupyter
```

#### **Step 3: Run the Notebook**

1. In Jupyter or vs-code, click on `ML_Final_Project.ipynb` to open it
2. Click **"Kernel"** → **"Restart & Run All"**
3. Wait 5-7 minutes for all cells to execute
4. Results will be displayed inline

### 8.2. Alternative: Google Colab

It can be done without install anything locally:

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **"File"** → **"Upload notebook"**
3. Upload `ML_Final_Project.ipynb`
4. Upload `default_of_credit_card_clients.xls` using the file upload icon (📁) in the left sidebar
5. Click **"Runtime"** → **"Run all"**

**Note:** All required libraries are pre-installed in Google Colab!

### 8.3. Verifying Results

To verify the reproduced results, check that final model comparison table shows:

- **Ensemble Model:** ~80.06% accuracy, ~74.35% AUC
- **Random Forest (Cost-sensitive):** ~80.83% accuracy, ~62.52% precision
- **Logistic Regression (Baseline):** ~69.00% accuracy, ~62.30% recall

Small variations (±0.5%) are normal due to random state differences.

---

## 9. Project Structure

```
credit-card-default-prediction/
├── README.md
├── requirements.txt                       # Python dependencies
├── default_of_credit_card_clients.xls    # Dataset (30,000 records)
├── ML_Final_Project.ipynb                # Main notebook with all analysis and results
└──CS613_Final_Presentation.pdf
```

---

## 10. References

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

## 11. Team

**Priti Sagar**  
Email: pp693@drexel.edu

**Devdeepsinh Zala**  
Email: dkz27@drexel.edu

**Dennis Zhuang**  
Email: dz374@drexel.edu

---

## 12. License

This project is for educational purposes as part of CS613 Machine Learning course at Drexel University.

---

## 13. Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- Dr. Matthew Burlick for guidance and feedback
- Prior research by Yeh & Lien (2009) for foundational work
- scikit-learn community for excellent ML tools
