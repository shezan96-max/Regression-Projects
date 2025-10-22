# Superstore Sales Prediction

## 📝 Project Overview
This project predicts **sales of products in a superstore** using historical sales data and **Linear Regression** with **Pipeline-based preprocessing** in Python (Scikit-learn).  

- Dataset: `train.csv`  
- Objective: Predict the **Sales** for each order based on features like Order Date, Ship Mode, Customer Segment, Product Category, and more.

---

## 📊 Dataset
| Column | Type | Description |
|--------|------|-------------|
| Order Date | DateTime | Date of the order |
| Ship Date | DateTime | Date the order was shipped |
| Ship Mode | Categorical | Shipping method |
| Segment | Categorical | Customer segment |
| Country / City / State / Region | Categorical | Location details |
| Category / Sub-Category | Categorical | Product categories |
| Sales | Numeric | Target variable (to predict) |

**Data Preprocessing:**
- Dropped unnecessary columns: Row ID, Order ID, Customer ID, Customer Name, Product ID, Product Name, Postal Code  
- Converted `Order Date` and `Ship Date` to datetime objects  
- Extracted date features: `Order_Year`, `Order_Month`, `Order_DayOfWeek`, `Order_DayName`  
- Missing values handled using **SimpleImputer**  
- Categorical features encoded using **OneHotEncoder** (drop first)  
- Numeric features scaled using **StandardScaler**  

---

## ⚙️ Workflow
1. Load dataset and explore (`head()`, `info()`, `describe()`)  
2. Drop irrelevant columns  
3. Convert date columns to datetime format  
4. Extract date-based features  
5. Split dataset into **train** and **test** sets  
6. Define **categorical** and **numeric** features  
7. Build **ColumnTransformer** for preprocessing  
8. Create **Pipeline**: Preprocessing + Linear Regression  
9. Train model on training data  
10. Predict on test data  
11. Evaluate model using **MSE, RMSE, R², MAE**  
12. Visualize Actual vs Predicted Sales  

---

## 📈 Model Evaluation

| Metric | Value |
|--------|-------|
| MSE | (from your output, e.g., 550,122) |
| RMSE | (calculated: 741.70 approx) |
| R² | (from your output, e.g., 0.177) |
| MAE | (from your output, e.g., 239.65) |

> The model provides a baseline prediction. Further improvements can be made with advanced regression algorithms or additional feature engineering.

---

## 🔧 How to Run
1. Clone this repo:  
```bash
git clone https://github.com/shezan96-max/Regression-Projects/Linear-Regression/Superstore-Sales-Prediction.git

