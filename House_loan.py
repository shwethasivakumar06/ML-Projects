import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Load California Housing dataset
House_price = fetch_california_housing()

# Create a DataFrame for the data
House_price_dataframe = pd.DataFrame(House_price.data, columns=House_price.feature_names)

# Show the first few rows of the data
print(House_price_dataframe.head())

# Add the target column (price)
House_price_dataframe['price'] = House_price.target
print(House_price_dataframe.head())

# Check for any missing values
print(House_price_dataframe.isnull().sum())

# Summary statistics of the data
print(House_price_dataframe.describe())

# Correlation heatmap
correlation = House_price_dataframe.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt='1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
plt.show()

# Splitting the data into features and target variable
X = House_price_dataframe.drop('price', axis=1)
Y = House_price_dataframe['price']
print(X.head())  # Show the feature set
print(Y.head())  # Show the target variable

# Train-test split (no need for stratify in regression)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=123)

# Model initialization and training
Model = XGBRegressor()
Model.fit(X_train, Y_train)

# Evaluate the model using R2 score and mean absolute error
train_score_r2 = metrics.r2_score(Y_train, Model.predict(X_train))
train_score_mae = metrics.mean_absolute_error(Y_train, Model.predict(X_train))

print(f"R2 Score on Training Data: {train_score_r2}")
print(f"Mean Absolute Error on Training Data: {train_score_mae}")

# You can also evaluate on the test data
test_score_r2 = metrics.r2_score(Y_test, Model.predict(X_test))
test_score_mae = metrics.mean_absolute_error(Y_test, Model.predict(X_test))

print(f"R2 Score on Test Data: {test_score_r2}")
print(f"Mean Absolute Error on Test Data: {test_score_mae}")
