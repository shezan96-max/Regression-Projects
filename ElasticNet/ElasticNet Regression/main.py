# ===============================
# Car Price Prediction using ElasticNetCV
# ===============================

# Import libraries
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv('CarPrice_Assignment.csv')

# Separate features and target
X = df.drop('price', axis=1)  # all columns except 'price'
y = df['price']               # target variable

# -------------------------------
# Train-Test Split
# -------------------------------
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Identify categorical & numerical columns
# -------------------------------
cat_cols = ['CarName','fueltype','aspiration','doornumber','carbody','drivewheel',
            'enginelocation','enginetype','cylindernumber','fuelsystem']

num_cols = ['car_ID','symboling','wheelbase','carlength','carwidth','carheight',
            'curbweight','enginesize','boreratio','stroke','compressionratio',
            'horsepower','peakrpm','citympg','highwaympg']

# Drop 'car_ID' as it's just an identifier
num_cols.remove('car_ID')

# -------------------------------
# Categorical Transformer
# -------------------------------
# OneHotEncoder for categorical features + scaling
# handle_unknown='ignore' ensures unseen categories in test set do not crash the pipeline
cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')),
    ('scaler', StandardScaler())
])

# -------------------------------
# Numerical Transformer
# -------------------------------
# StandardScaler to normalize numeric features
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# -------------------------------
# ColumnTransformer
# -------------------------------
# Apply transformers to respective columns
preprocessor = ColumnTransformer(transformers=[
    ('cat', cat_transformer, cat_cols),
    ('num', num_transformer, num_cols)
])

# -------------------------------
# Suppress convergence warnings
# -------------------------------
# ElasticNetCV may throw ConvergenceWarning if optimization takes too long
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -------------------------------
# Build Pipeline
# -------------------------------
# Steps:
# 1. Preprocessing (categorical + numeric)
# 2. ElasticNetCV model with cross-validation
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', ElasticNetCV(
        cv=5,                            # 5-fold cross-validation
        alphas=np.logspace(-2, 2, 10),  # test alpha from 0.01 to 100
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 1],  # mix of L1 and L2
        random_state=42,
        max_iter=10000,                  # high iteration to ensure convergence
        n_jobs=-1                        # use all CPU cores for faster training
    ))
])

# -------------------------------
# Fit the model
# -------------------------------
model_pipeline.fit(X_train, y_train)

# -------------------------------
# Make Predictions
# -------------------------------
y_pred = model_pipeline.predict(X_test)

# -------------------------------
# Evaluate Model
# -------------------------------
mse = mean_squared_error(y_test, y_pred)      # Mean Squared Error
rmse = np.sqrt(mse)                            # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)                 # R² score

print("MSE :", mse)
print("RMSE :", rmse)
print("R2 :", r2)

# -------------------------------
# Optional: Visualize Predictions
# -------------------------------
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices (ElasticNetCV)")
plt.show()
