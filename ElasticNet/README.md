# 🚗 Car Price Prediction using ElasticNet Regression

This project predicts car prices using an **ElasticNetCV Regression model**, which combines L1 (Lasso) and L2 (Ridge) regularization to produce robust and generalizable predictions.

---

## 📘 Project Overview
- **Algorithm Used:** ElasticNetCV  
- **Dataset:** `CarPrice_Assignment.csv`  
- **Goal:** Predict car prices based on car specifications like engine size, body type, horsepower, and mileage.  
- **Framework:** Scikit-Learn (Python)  
- **Pipeline Steps:** Includes preprocessing, encoding, scaling, and modeling.

---

## ⚙️ Pipeline Steps
1. **Preprocessing**
   - Categorical columns are one-hot encoded with unseen categories ignored.
   - Numeric columns are scaled using `StandardScaler`.
2. **Modeling**
   - `ElasticNetCV` is used with 5-fold cross-validation.
   - Hyperparameters tuned:
     - `alphas = np.logspace(-2, 2, 10)` (0.01 → 100)
     - `l1_ratio = [0.1, 0.3, 0.5, 0.7, 1]`
3. **Evaluation**
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - R² Score

---

## 📊 Results
| Metric | Value |
|--------|-------|
| **MSE** | *replace with your output* |
| **RMSE** | *replace with your output* |
| **R² Score** | *replace with your output* |

---

## 🧠 Model Insights
- **ElasticNetCV** combines:
  - **L1 Regularization (Lasso):** Feature selection by shrinking some coefficients to zero.
  - **L2 Regularization (Ridge):** Reduces overfitting by penalizing large coefficients.
- This ensures a balance between bias and variance, producing a stable and generalizable model.

---

## 🏃‍♂️ How to Run
```bash
# Clone the repository
git clone https://github.com/shezan96-max/Regression-Projects.git
cd Regression-Projects/ElasticNet

# Run the main script
python main.py

