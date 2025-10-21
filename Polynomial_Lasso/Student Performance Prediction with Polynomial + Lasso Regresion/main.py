import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler,OneHotEncoder,PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt



df = pd.read_csv('Student_Performance.csv')

print(df.head())
print(df.info())
print(df.describe())
print("Missing Values :",df.isnull().sum())

X = df.drop('Performance Index',axis=1)
y = df['Performance Index']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)

cat_cols = ['Extracurricular Activities']
num_cols = ['Hours Studied','Previous Scores','Sleep Hours','Sample Question Papers Practiced']


cat_transformer = Pipeline(steps=[
    ('onehot',OneHotEncoder(drop='first',sparse_output=False)),
    ('scaler',StandardScaler())
])
num_transformer = Pipeline(steps=[
    ('Scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('cat', cat_transformer, cat_cols),
    ('num', num_transformer, num_cols)

])

model_pipeline = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('poly', PolynomialFeatures(degree=2,include_bias=False)),
    ('model', Lasso(alpha=0.1,random_state=42))
])


model_pipeline.fit(X_train,y_train)
y_pred = model_pipeline.predict(X_test)

# Evaluation Metrics
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

print("MSE :",mse)
print("RMSE :",rmse)
print("R2 :",r2)


# For multi-feature datasets: show Actual vs Predicted
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Lasso Regression: Actual vs Predicted')
plt.show()