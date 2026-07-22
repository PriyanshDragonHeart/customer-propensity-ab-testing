# Customer Propensity Modeling & A/B Testing Pipeline

> **Production-style Machine Learning pipeline for customer reactivation using propensity modeling, CatBoost, randomized A/B testing, and uplift measurement.**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-yellow)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Optimization-green)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

# Overview

Customer reactivation campaigns are expensive when sent to every inactive customer. Most customers never respond, resulting in unnecessary marketing costs and poor campaign efficiency.

This project builds an **end-to-end customer propensity modeling pipeline** that identifies customers most likely to reactivate, ranks them using propensity scores, randomly assigns Treatment and Control groups within each propensity decile, and measures incremental campaign impact through A/B testing.

The repository follows a production-style workflow covering data preparation, feature engineering, statistical feature selection, model optimization, customer scoring, campaign generation, and uplift measurement.

---

# Business Problem

A brokerage has millions of dormant customers.

Instead of sending marketing campaigns to every inactive customer, the objective is to answer:

> **"Which customers are most likely to reactivate if contacted?"**

After ranking customers by predicted probability, the campaign is scientifically validated using randomized experiments.

---

# Solution Architecture

```text
                                      Customer Data
                                             │
        ┌────────────────────────────────────┴────────────────────────────────────┐
        │                                                                         │
 Demographics                  Trading Behaviour                  Portfolio Information
        │                               │                                  │
        └───────────────────────────────┴──────────────────────────────────┘
                                             │
                                   Feature Engineering
                 ┌─────────────────────────────────────────────────┐
                 │ Log Transformations                             │
                 │ Entry Age                                       │
                 │ Never Traded Flag                               │
                 │ Missing Value Handling                          │
                 │ Data Cleaning                                   │
                 └─────────────────────────────────────────────────┘
                                             │
                                   Feature Selection
                 ┌─────────────────────────────────────────────────┐
                 │ Chi-Square Test                                │
                 │ Mutual Information                             │
                 │ Correlation Analysis                           │
                 │ Variance Inflation Factor                      │
                 └─────────────────────────────────────────────────┘
                                             │
                               Hyperparameter Optimization
                                       (Optuna)
                                             │
                                             ▼
                                 CatBoost Classifier
                                             │
                    ┌────────────────────────┴──────────────────────┐
                    │                                               │
            Propensity Score                              Feature Importance
                    │
                    ▼
            Customer Ranking
                    │
                    ▼
         Propensity Decile Creation
                    │
                    ▼
 Randomized Treatment / Control Assignment
                    │
                    ▼
          Marketing Campaign Execution
                    │
                    ▼
        Campaign Conversion Measurement
                    │
                    ▼
      Uplift Analysis • Incremental Lift • ROI
```

---

# Input Dataset

Each customer record contains demographic, account and historical trading behaviour.

### Numerical Features

- Customer Age
- Account Vintage
- Portfolio Value
- Total Margin
- Cash Orders
- Derivative Orders
- Commodity Orders
- Currency Orders
- Recency

### Categorical Features

- Customer Category
- Trading Vertical
- Self Dealer Status
- Trading Member Count
- Subscription Plan
- Dealing Zone

### Target

```
target = 1
```

Customer reactivated during campaign window.

```
target = 0
```

Customer remained inactive.

---

# Pipeline

## 1. Exploratory Data Analysis

- Missing value analysis
- Class imbalance analysis
- Distribution plots
- Correlation analysis
- Outlier detection

---

## 2. Feature Engineering

Created engineered variables including:

- Log transformed monetary variables
- Log transformed trading behaviour
- Entry Age
- Never Traded Indicator
- Recency Features

---

## 3. Statistical Feature Selection

Multiple statistical techniques were combined.

### Chi-Square Test

Evaluated categorical feature relevance.

### Mutual Information

Measured nonlinear dependency with target.

### Correlation Analysis

Removed redundant variables.

### Variance Inflation Factor (VIF)

Detected multicollinearity.

Only statistically useful features were retained for model development.

---

## 4. Hyperparameter Optimization

Model tuning performed using **Optuna**.

Optimized parameters included:

- Learning Rate
- Tree Depth
- Iterations
- L2 Regularization
- Random Strength
- Bagging Temperature
- Border Count

Objective:

```
Maximize ROC-AUC
```

---

## 5. Final Model

Algorithm:

```
CatBoost Classifier
```

CatBoost was selected because it:

- Handles categorical variables natively
- Performs well on imbalanced tabular datasets
- Requires minimal preprocessing
- Produces interpretable feature importance
- Supports probability estimation for propensity scoring

---

## 6. Customer Scoring

The trained model generates a **propensity score** for every customer.

```
0.00 ---------------------> 1.00
Low Probability      High Probability
```

Higher scores indicate a greater likelihood of customer reactivation.

---

## 7. Propensity Deciles

Customers are ranked by predicted probability and divided into:

```
Decile 10
Highest propensity customers

↓

Decile 1
Lowest propensity customers
```

This allows campaign performance to be analyzed across customer quality segments.

---

## 8. Randomized A/B Testing

Within every propensity decile:

```
50%

Treatment

50%

Control
```

Randomization ensures unbiased estimation of campaign effectiveness.

---

## 9. Campaign Evaluation

Following campaign execution, performance is measured using:

- Conversion Rate
- Absolute Uplift
- Relative Uplift
- Incremental Conversions
- Campaign ROI
- Lift by Propensity Decile

---

# Inputs

```
Customer Transaction History

Customer Demographics

Portfolio Information

Trading Behaviour

Account Information

Campaign Response
```

---

# Outputs

The pipeline produces:

### Model Outputs

- Trained CatBoost Model
- Feature Importance
- Hyperparameter Configuration
- Customer Propensity Scores

### Campaign Outputs

- Ranked Customer List
- Treatment Group
- Control Group
- Propensity Deciles
- Campaign File

### Evaluation Outputs

- ROC Curve
- Precision Recall Curve
- Confusion Matrix
- Feature Importance Plot
- Uplift Summary
- Campaign Performance Report

---

# Business Impact

The pipeline enables marketing teams to:

- Prioritize customers with highest reactivation probability
- Reduce campaign wastage
- Improve marketing efficiency
- Measure true campaign impact using randomized experiments
- Quantify incremental lift before scaling campaigns

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- CatBoost
- Optuna
- Matplotlib
- Seaborn
- Statsmodels
- Jupyter Notebook

---

# Reproducibility

Run the notebooks sequentially:

```
generate_dummy_data.ipynb

↓

01_eda.ipynb

↓

02_feature_engineering.ipynb

↓

03_feature_selection.ipynb

↓

04_hyperparameter_tuning.ipynb

↓

05_train_model.ipynb

↓

06_feature_ablation.ipynb

↓

07_final_model.ipynb

↓

08_score_customers.ipynb

↓

09_create_deciles.ipynb

↓

10_campaign_analysis.ipynb
```

Each notebook generates intermediate artifacts used by the next stage, enabling complete end-to-end reproducibility.

---

# Repository Highlights

✔ Production-style ML workflow

✔ Statistical feature selection

✔ Automated hyperparameter optimization

✔ Propensity scoring

✔ Customer ranking

✔ Randomized A/B testing

✔ Uplift analysis

✔ Business-focused evaluation

✔ Fully reproducible pipeline

---

## Author

**Priyansh Mahajan**

Machine Learning • Data Science • Experimentation • Customer Analytics
