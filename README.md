# Medical Insurance Cost Analysis

A machine learning project for predicting medical insurance charges and classifying smoker status using US patient data.

**Contributors:** Evans Frimpong, Aniket Shinde

---

## 📋 Table of Contents

- [Dataset](#dataset)
- [Problem Definition](#problem-definition)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Task 1: Linear Regression](#task-2-linear-regression-and-interpretation)
- [Task 2: Classification Model Comparison](#task-3-classification-model-comparison)
- [Task 3: Precision-Recall Evaluation](#task-4-precision-recall-evaluation)

---

## Dataset

**Source:** [Kaggle – Medical Insurance Cost Dataset](https://www.kaggle.com/)

The dataset contains **1,338 rows** of US medical insurance beneficiary records with the following features:

| Feature    | Type        | Description                          |
|------------|-------------|--------------------------------------|
| `age`      | Numerical   | Age of the beneficiary               |
| `sex`      | Categorical | Gender of the beneficiary            |
| `bmi`      | Numerical   | Body Mass Index                      |
| `children` | Numerical   | Number of dependent children         |
| `smoker`   | Categorical | Whether the patient is a smoker      |
| `region`   | Categorical | US region of the beneficiary         |
| `charges`  | Numerical   | Medical insurance charges (target)   |

---

## Problem Definition

Two modelling goals were defined:

- **Regression:** Predict continuous medical insurance charges.
- **Classification:** Predict whether a patient is a smoker based on their health and demographic data.

---

## Exploratory Data Analysis

Key observations from the EDA:

- The distribution of `charges` is **left-skewed**, with most charges falling in the **$10,000–$30,000** range.
- The dataset is **imbalanced for smoker status** — the majority of patients are non-smokers (~1,100 vs ~200 smokers).
- **Smokers consistently exhibit higher charges** compared to non-smokers.

---

## Task 1: Linear Regression and Interpretation

A **Multiple Linear Regression** model was built to predict medical charges.

### Features Used

- **Numerical:** `age`, `bmi`, `children`
- **Binary:** `smoker` (encoded: 1 = smoker, 0 = non-smoker)
- **One-hot encoded:** `region` (reference: northeast), `sex` (reference: female)

**Train/Test Split:** 80% training (1,070 records) / 20% testing (268 records), with a fixed random seed.

### Model Performance

| Metric    | Value      |
|-----------|------------|
| R²        | 0.8704     |
| RMSE      | $3,936.51  |

An R² of **0.87** means the model explains **87% of the variance** in charges — a strong fit for real-world healthcare data. The RMSE of ~$3,937 is reasonable given the charge range of $1,121–$63,770.

### Key Coefficient Interpretations

| Feature       | Coefficient   | Interpretation                                              |
|---------------|---------------|-------------------------------------------------------------|
| `smoker`      | +$24,154.54   | Strongest predictor; smoking dramatically increases charges |
| `children`    | +$606.02      | Each additional child increases charges by ~$606            |
| `bmi`         | +$335.63      | Each BMI unit increase raises charges by ~$335              |
| Intercept     | $453.53       | Baseline for reference category (female, non-smoker, NE)   |

---

## Task 2: Classification Model Comparison

Three classifiers were trained to predict **smoker status** using all available features. The dataset was split 80/20 with **stratification** to preserve class balance.

### Results

| Model                  | Accuracy | Precision | Recall | F1 Score |
|------------------------|----------|-----------|--------|----------|
| Logistic Regression    | 0.9963   | 1.0000    | 0.9722 | 0.9859   |
| Naive Bayes            | 0.9851   | 0.9800    | 0.9423 | 0.9608   |
| Linear Discriminant Analysis (LDA) | 0.9963 | 1.0000 | 0.9722 | 0.9859 |

### Model Notes

- **Logistic Regression:** Near-perfect performance (99.6% accuracy). The near-linear separability driven by `charges` suits this model well.
- **Naive Bayes:** Strong performance (98.5%) despite the violated feature independence assumption (age, BMI, and charges are correlated).
- **LDA:** Matched Logistic Regression exactly, confirming a near-linear classification boundary.

**Best Model:** Logistic Regression and LDA tied. **Logistic Regression is preferred** for deployment due to its direct probability interpretability and regularization flexibility.

---

## Task 3: Precision-Recall Evaluation

The **Precision-Recall curve** was generated for the best model (Logistic Regression).

- **Average Precision Score:** 1.000

### False Positive vs. False Negative

In this healthcare context, **False Negatives are more costly** — missing a smoker means:

- No smoking cessation counselling or targeted health monitoring.
- Incorrect premium calculation, creating financial risk for the insurer.
- A critical comorbidity risk factor being overlooked.

### Optimal Threshold Selection

The decision threshold was optimized by **maximizing the F1 Score**.

| Metric at Optimal Threshold | Value  |
|-----------------------------|--------|
| Optimal Threshold           | 0.30   |
| Precision                   | 1.0000 |
| Recall                      | 0.9722 |
| F1 Score                    | 0.9859 |

At the optimal threshold, the model correctly identifies **97.2% of all smokers** while maintaining **100% precision** — every flagged smoker is truly a smoker. This strongly supports deployment in health profiling or insurance risk-assessment contexts.

---

## Technologies Used

- Python
- scikit-learn
- pandas / NumPy
- Matplotlib / Seaborn
