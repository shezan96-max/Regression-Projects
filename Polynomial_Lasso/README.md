# Polynomial + Lasso Regression Projects

This folder contains implementations of **Polynomial Regression with Lasso Regularization** using the **Student Performance Dataset**.

## Project Overview

- **Dataset:** Student Performance Dataset
- **Features:**
  - Numeric: Hours Studied, Previous Scores, Sleep Hours, Sample Question Papers Practiced
  - Categorical: Extracurricular Activities
- **Target:** Performance Index
- **Task:** Predict the final performance index of students based on features

## Techniques Used

1. **Preprocessing:**
   - StandardScaler for numeric features
   - OneHotEncoder for categorical features
2. **Polynomial Regression:**
   - Polynomial features (degree=2) added to capture non-linear relationships
3. **Lasso Regression:**
   - Regularization to reduce overfitting and perform feature selection
   - Fixed alpha = 0.1 (for demonstration)
   - Optionally, LassoCV can be used for optimized alpha selection

## Evaluation Metrics

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

## Visualization

- Scatter plot of Actual vs Predicted values

## Notebook

- `Student_Performance.ipynb`

---

**Note:** This project demonstrates how to combine preprocessing pipelines, polynomial feature expansion, and Lasso regularization to predict student performance in a multi-feature dataset.

