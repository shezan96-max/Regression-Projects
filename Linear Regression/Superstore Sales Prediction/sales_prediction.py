import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,root_mean_squared_error 


df = pd.read_csv('train.csv')

print(df.head())
print(df.info())
print(df.describe())
print("Missing Values :",df.isnull().sum())

df = df.drop(['Row ID', 'Order ID', 'Customer ID', 'Customer Name', 
              'Product ID', 'Product Name', 'Postal Code'], axis=1)

df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)

df['Order_Year'] = df['Order Date'].dt.year
df['Order_Month'] = df['Order Date'].dt.month
df['Order_DayOfWeek'] = df['Order Date'].dt.dayofweek

df['Order_DayName'] = df['Order Date'].dt.day_name()

X = df.drop('Sales', axis=1)
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

categorical_features = ['Ship Mode','Segment','Country','Region','Category','Sub-Category','Order_DayName']
numeric_features = ['Order_Year','Order_Month','Order_DayOfWeek']

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))
])

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('cat', cat_transformer, categorical_features),
    ('num', num_transformer, numeric_features)
])

model_pipeline = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model', LinearRegression())
])

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

print("MSE :",mean_squared_error(y_test,y_pred))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)
print("R2 :",r2_score(y_test,y_pred))
print("MAE :",mean_absolute_error(y_test,y_pred))

plt.figure(figsize=(12,8))
plt.scatter(y_test,y_pred, color='blue', alpha=0.6)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()