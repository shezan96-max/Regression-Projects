import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

df = pd.read_csv('housing.csv')

X = df.drop('median_house_value', axis=1)
y = df['median_house_value']


num_cols = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']
cat_cols = ['ocean_proximity']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(drop='first'))

])

preprocessor = ColumnTransformer(transformers=[
    ('numeric', numeric_transformer, num_cols),
    ('categorical',categorical_transformer, cat_cols)
])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selector', RFE(estimator=LinearRegression(),n_features_to_select=8)),
    ('poly',PolynomialFeatures(degree=2,include_bias=False)),
    ('scaler', StandardScaler()),
    ('model',Ridge(alpha=0.1))
])


model_pipeline.fit(X_train,y_train)
y_pred = model_pipeline.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

print("MSE :",mse)
print("RMSE :",rmse)
print("R2 :",r2)
