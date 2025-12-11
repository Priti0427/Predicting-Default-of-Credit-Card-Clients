# Data Dictionary: Default of Credit Card Clients Dataset

## Dataset Overview

- **Source:** UCI Machine Learning Repository
- **Dataset Name:** Default of Credit Card Clients Dataset
- **Total Records:** 30,000
- **Total Features:** 24 (23 predictors + 1 target)
- **Task Type:** Binary Classification
- **Class Distribution:**
  - Non-default (0): 23,364 (77.88%)
  - Default (1): 6,636 (22.12%)

---

## Target Variable

| Variable Name                | Description                                           | Data Type | Values                        |
| ---------------------------- | ----------------------------------------------------- | --------- | ----------------------------- |
| `default payment next month` | Whether the client will default on payment next month | Binary    | 0 = No default<br>1 = Default |

---

## Feature Variables

### 1. Demographic Information

| Variable Name | Description                       | Data Type   | Values                                                                                               | Notes                                   |
| ------------- | --------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `ID`          | Unique identifier for each client | Integer     | 1 to 30,000                                                                                          | Not used for modeling                   |
| `LIMIT_BAL`   | Credit limit (NT dollar)          | Integer     | 10,000 - 1,000,000                                                                                   | Amount of given credit                  |
| `SEX`         | Gender                            | Categorical | 1 = Male<br>2 = Female                                                                               | not Encoded       |
| `EDUCATION`   | Education level                   | Categorical | 1 = Graduate school<br>2 = University<br>3 = High school<br>4 = Others<br>5 = Unknown<br>6 = Unknown | Categories 5, 6 sometimes merged with 4 |
| `MARRIAGE`    | Marital status                    | Categorical | 1 = Married<br>2 = Single<br>3 = Others                                                              | Category 0 exists but undocumented      |
| `AGE`         | Age in years                      | Integer     | 21 - 79                                                                                              | Continuous variable                     |

---

### 2. Repayment Status (Past 6 Months)

**Description:** Payment status for the past 6 months (April to September 2005)

| Variable Name        | Description                        | Data Type   | Values                                                                                                                                                                                           |
| -------------------- | ---------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `PAY_0` (or `PAY_1`) | Repayment status in September 2005 | Categorical | -2 = No consumption<br>-1 = Pay duly<br>0 = Revolving credit used<br>1 = Payment delay 1 month<br>2 = Payment delay 2 months<br>...<br>8 = Payment delay 8 months<br>9 = Payment delay 9+ months |
| `PAY_2`              | Repayment status in August 2005    | Categorical | Same as above                                                                                                                                                                                    |
| `PAY_3`              | Repayment status in July 2005      | Categorical | Same as above                                                                                                                                                                                    |
| `PAY_4`              | Repayment status in June 2005      | Categorical | Same as above                                                                                                                                                                                    |
| `PAY_5`              | Repayment status in May 2005       | Categorical | Same as above                                                                                                                                                                                    |
| `PAY_6`              | Repayment status in April 2005     | Categorical | Same as above                                                                                                                                                                                    |

**Note:** Higher positive values indicate longer payment delays. Negative values indicate on-time or early payment behavior.

---

### 3. Bill Statement Amount (Past 6 Months)

**Description:** Amount of bill statement (NT dollar) for the past 6 months

| Variable Name | Description                             | Data Type | Range                 | Notes                                        |
| ------------- | --------------------------------------- | --------- | --------------------- | -------------------------------------------- |
| `BILL_AMT1`   | Bill statement amount in September 2005 | Integer   | -165,580 to 964,511   | Can be negative (overpayment/credit balance) |
| `BILL_AMT2`   | Bill statement amount in August 2005    | Integer   | -69,777 to 983,931    | Can be negative                              |
| `BILL_AMT3`   | Bill statement amount in July 2005      | Integer   | -157,264 to 1,664,089 | Can be negative                              |
| `BILL_AMT4`   | Bill statement amount in June 2005      | Integer   | -170,000 to 891,586   | Can be negative                              |
| `BILL_AMT5`   | Bill statement amount in May 2005       | Integer   | -81,334 to 927,171    | Can be negative                              |
| `BILL_AMT6`   | Bill statement amount in April 2005     | Integer   | -339,603 to 961,664   | Can be negative                              |

---

### 4. Previous Payment Amount (Past 6 Months)

**Description:** Amount of previous payment (NT dollar) for the past 6 months

| Variable Name | Description                   | Data Type | Range          | Notes               |
| ------------- | ----------------------------- | --------- | -------------- | ------------------- |
| `PAY_AMT1`    | Amount paid in September 2005 | Integer   | 0 to 873,552   | Actual payment made |
| `PAY_AMT2`    | Amount paid in August 2005    | Integer   | 0 to 1,684,259 | Actual payment made |
| `PAY_AMT3`    | Amount paid in July 2005      | Integer   | 0 to 896,040   | Actual payment made |
| `PAY_AMT4`    | Amount paid in June 2005      | Integer   | 0 to 621,000   | Actual payment made |
| `PAY_AMT5`    | Amount paid in May 2005       | Integer   | 0 to 426,529   | Actual payment made |
| `PAY_AMT6`    | Amount paid in April 2005     | Integer   | 0 to 528,666   | Actual payment made |

---

## Data Preprocessing Steps

### 1. Categorical Encoding

| Original Feature | Encoding Method  | Result                                            |
| ---------------- | ---------------- | ------------------------------------------------- |
| `SEX`            | Binary encoding  | `SEX_MALE` (0 or 1)                               |
| `EDUCATION`      | One-hot encoding | `EDUCATION_2`, `EDUCATION_3`, `EDUCATION_4`, etc. |
| `MARRIAGE`       | One-hot encoding | `MARRIAGE_2`, `MARRIAGE_3`, etc.                  |

**Note:** One-hot encoding uses `drop_first=True` to avoid multicollinearity.

### 2. Feature Scaling

All numerical features are standardized using StandardScaler:

- **Method:** Z-score normalization
- **Formula:** `z = (x - μ) / σ`
- **Features scaled:** `LIMIT_BAL`, `AGE`, `PAY_0` through `PAY_6`, `BILL_AMT1` through `BILL_AMT6`, `PAY_AMT1` through `PAY_AMT6`

### 3. Feature Selection

Top features selected based on combined importance from:

- Logistic Regression coefficients (absolute values)
- Random Forest feature importance (Gini importance)

**Selection Method:** Intersection of top N features from both models

---

## Data Quality Notes

### Missing Values

- **No missing values** in the dataset (all 30,000 records are complete)

### Data Anomalies

1. **Education codes 0, 5, 6:** Not defined in original documentation
   - Common practice: Merge with category 4 (Others)
2. **Marriage code 0:** Not defined in original documentation
   - Common practice: Merge with category 3 (Others)
3. **Negative bill amounts:** Represent overpayment or credit balance (valid)
4. **PAY_0 values:** Sometimes labeled as `PAY_1` in different versions

### Class Imbalance

- **Imbalanced dataset:** 77.88% non-default vs 22.12% default
- **Handling method:** `class_weight='balanced'` parameter in models

---

## Currency Information

- **Currency:** New Taiwan Dollar (NT$)
- **Approximate conversion (as of dataset creation, 2005):**
  - 1 USD ≈ 32 NT$
  - Example: Credit limit of 320,000 NT$ ≈ 10,000 USD

---

## Temporal Information

- **Data collection period:** April 2005 - September 2005
- **Prediction target:** Default in October 2005
- **Historical window:** 6 months of payment history

---

## Key Statistics

| Feature     | Mean    | Std Dev | Min      | Max       |
| ----------- | ------- | ------- | -------- | --------- |
| `LIMIT_BAL` | 167,484 | 129,747 | 10,000   | 1,000,000 |
| `AGE`       | 35.49   | 9.22    | 21       | 79        |
| `PAY_0`     | -0.02   | 1.12    | -2       | 8         |
| `BILL_AMT1` | 51,223  | 73,636  | -165,580 | 964,511   |
| `PAY_AMT1`  | 5,664   | 16,563  | 0        | 873,552   |

---






