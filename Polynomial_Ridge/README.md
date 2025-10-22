# 🏠 Polynomial Ridge Regression on California Housing Dataset

## 📘 Overview
This project demonstrates a **complete end-to-end regression pipeline** using the **California Housing Dataset**.  
We combine **Polynomial Feature Expansion**, **Feature Selection (RFE)**, and **Ridge Regularization** to build a robust predictive model for estimating median house values.

---

## 🎯 Objectives
- Preprocess mixed-type data (numeric + categorical)
- Select the most impactful features using **RFE (Recursive Feature Elimination)**
- Capture nonlinear relationships with **Polynomial Features**
- Prevent overfitting with **Ridge Regularization**
- Evaluate model performance using MSE, RMSE, and R² metrics

---

## ⚙️ Technologies Used
- **Python 3.x**
- **pandas**, **numpy**
- **scikit-learn**
- **matplotlib** *(optional for visualization)*

---

## 🧩 Pipeline Workflow

### 1. Data Preprocessing
- Missing values handled using:
  - Mean imputation for numeric features  
  - Most frequent imputation for categorical features  
- One-hot encoding for the `ocean_proximity` column  

### 2. Feature Selection
- **RFE (Recursive Feature Elimination)** used with `LinearRegression`  
- Selected top 8 important features  

### 3. Polynomial Feature Expansion
- Degree = 2  
- Expanded features to capture complex relationships  

### 4. Feature Scaling
- Applied **StandardScaler** to normalize features  

### 5. Model Training
- Trained a **Ridge Regression** model with `alpha = 0.1`

---

## 📊 Model Evaluation

| Metric | Description | Value (Example) |
|--------|--------------|----------------|
| **MSE** | Mean Squared Error | 3.21e+09 |
| **RMSE** | Root Mean Squared Error | 56674.23 |
| **R² Score** | Coefficient of Determination | 0.82 |


