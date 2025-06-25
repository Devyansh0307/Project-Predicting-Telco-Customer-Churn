# Project-Predicting-Telco-Customer-Churn
# Predicting Customer Churn with Machine Learning

## Project Overview

This project aims to predict customer churn for a fictional telecommunications company. By analyzing customer data, we can identify the key factors that lead to churn and build a predictive model to identify at-risk customers. This allows the business to take proactive steps—such as offering targeted promotions or addressing service issues—to improve customer retention and reduce revenue loss.

**Business Problem:** Customer churn is a critical metric for subscription-based businesses. The goal is to develop a highly accurate model that predicts churn, and more importantly, to derive actionable insights from this model to inform business strategy.

## Tech Stack

*   **Python:** The core programming language for the analysis.
*   **Pandas:** For data manipulation, cleaning, and preprocessing.
*   **Matplotlib & Seaborn:** For data visualization and exploratory data analysis (EDA).
*   **Scikit-learn:** For model building, training, and evaluation.

## Methodology

1.  **Data Loading & Initial Exploration:** The dataset was loaded and inspected to understand its structure, data types, and identify initial issues.
2.  **Data Cleaning & Preprocessing:**
    *   Handled missing values in the `TotalCharges` column.
    *   Corrected the data type of `TotalCharges` from `object` to `float`.
    *   Dropped the non-predictive `customerID` column.
3.  **Feature Engineering:**
    *   Converted the binary `Churn` target variable into `0` and `1`.
    *   Applied one-hot encoding to all categorical features to make them suitable for the model.
    *   Used `StandardScaler` to scale numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) to ensure they have a similar influence on the model.
4.  **Modeling:**
    *   The data was split into an 80% training set and a 20% testing set.
    *   A `RandomForestClassifier` was chosen for its high performance and ability to provide feature importance scores.
5.  **Evaluation:** The model's performance was assessed using Accuracy, Confusion Matrix, and a detailed Classification Report (Precision, Recall, F1-Score).

## Key Findings & Visualizations

The model revealed several key drivers of customer churn. The plot below shows the top 10 most important features in predicting whether a customer will leave.

*(Here, you would embed the feature importance plot you generated)*

The most significant factors are:
*   **Contract Type:** Customers on a `Month-to-Month` contract are far more likely to churn than those on long-term contracts.
*   **Tenure:** New customers with low tenure are at a high risk of churning.
*   **Internet Service (Fiber Optic):** Customers with Fiber Optic service show a higher churn rate, which may indicate issues with this specific service (e.g., price, reliability, installation).

## Model Performance

The final model achieved the following results on the unseen test data:
*   **Accuracy:** ~80%

## Actionable Business Insights

Based on the model's findings, the following strategic recommendations can be made:
1.  **Develop Retention Campaigns:** Proactively target customers on month-to-month contracts with special offers to convert them to one or two-year plans.
2.  **Enhance Customer Onboarding:** Create a robust onboarding program for the first 1-3 months to improve the experience for new customers and build loyalty.
3.  **Investigate Fiber Optic Service:** Conduct a deep-dive analysis into the Fiber Optic service line. Are customers leaving due to high prices, connection issues, or poor customer support?

