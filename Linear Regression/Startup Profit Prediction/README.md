# Startup Profit Prediction

## 📝 Project Overview
This project predicts the **profit of a startup** based on various investment features using **Linear Regression** and **Pipeline-based preprocessing** in Python (Scikit-learn).  

- Dataset: `startup.csv`  
- Objective: Predict the **Profit** of a startup given features like R&D Spend, Administration, Marketing Spend, and State.

---

## 📊 Dataset
| Column | Type | Description |
|--------|------|-------------|
| R&D Spend | Numeric | Investment in R&D |
| Administration | Numeric | Administration costs |
| Marketing Spend | Numeric | Marketing costs |
| State | Categorical | State in which startup operates |
| Profit | Numeric | Target variable |

- Missing values handled using **SimpleImputer**  
- Categorical variable `State` encoded using **OneHotEncoder**  
- Numeric features scaled using **StandardScaler**  

---

## ⚙️ Workflow
1. Load dataset and explore (head, info, describe)  
2. Drop unnecessary columns (if any)  
3. Handle missing values using **SimpleImputer**  
4. Feature engineering (if applicable)  
5. Split dataset into **train** and **test** sets  
6. Create **ColumnTransformer** for preprocessing numeric & categorical features  
7. Build **Pipeline**: Preprocessing + Linear Regression  
8. Train model on training data  
9. Predict on test data  
10. Evaluate using **MSE, RMSE, R², MAE**  
11. Visualize Actual vs Predicted Profit  

---

## 📈 Model Evaluation

| Metric | Value |
|--------|-------|
| MSE | 82,010,363 |
| RMSE | 9,056 (approx.) |
| R² | 0.899 |


> The model explains ~90% of the variation in Profit, which is a strong baseline for regression.

---

## 🔧 How to Run
1. Clone this repo:  
```bash
git clone https://github.com/shezan96-max/Regression-Projects/Linear-Regression/Startup-Profit-Prediction.git

