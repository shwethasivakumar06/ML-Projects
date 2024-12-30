# Rock Vs Mine Prediction
Data can be sourced from various places depending on the nature of the problem:

- Internal Data: Data generated within your organization (e.g., customer data, sales data, or website traffic).
- External Data: Public datasets, third-party services, open data repositories, or APIs. For instance:
  - Kaggle: A platform with many open datasets for different problems.
  - UCI Machine Learning Repository: A collection of datasets for various types of problems.
  - Government Databases: Many governments provide open access to a variety of datasets (e.g., from data.gov).
  - Web Scraping: Collect data from websites using scraping techniques (ensure legality and ethical considerations).We use tools like BeautifulSoup and Scarpy.
  - APIs: Access structured data from APIs like Twitter (for tweets), Google Maps (for geographical data), or others (e.g., financial data).

**After Downloading the Data from a third-party Organization, We Should open Google Collab and make sure that ur csv file for the Dataset as well as your Google Collab file are in the same folder.**

This project implements a machine learning model to classify objects as either "Rock" or "Mine" based on sonar data.

## Description

The script performs the following steps:

1. **Importing Dependencies**: It imports libraries like NumPy, Pandas, and scikit-learn for data processing, model training, and evaluation.
2. **Data Collection and Processing**:
   - Loads the sonar dataset into a Pandas DataFrame.
   - Analyzes the dataset with statistical summaries and counts of target labels.
   - Separates the features (X) and labels (Y).
3. **Training and Testing Data**:
   - Splits the dataset into training and testing subsets.
4. **Model Training**:
   - Trains a logistic regression model using the training data.
5. **Accuracy Evaluation**:
   - Computes and prints the accuracy for both training and testing data.

## Code

https://colab.research.google.com/drive/1-igmy956lkzI5LUeE4wEcF_iqvanPCT_

Importing The Dependencies
```python

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

```
**Data Collection and Processing**

Loading a Dataset to a Pandas Dataframe
```python
Sonar_data=pd.read_csv("/content/Sonar dataset.csv",header=None)
Sonar_data.head()
```
![image](https://github.com/user-attachments/assets/f17e83f8-3a8d-4c8f-b8a9-0276068bfc63)
```python
#Number of rows and columns
Sonar_data.shape
```
![image](https://github.com/user-attachments/assets/9ab4755d-1e58-4d74-9e58-a90aaf462f03)
```python
#Describe Gives Statistics measures of the data
Sonar_data.describe()
```
![image](https://github.com/user-attachments/assets/c5d892af-3d63-4ed5-8470-5339f0ac5928)
```python
#Value_counts is used to find the number of rows for a particular value
Sonar_data[60].value_counts()
```
![image](https://github.com/user-attachments/assets/95be5525-4236-441f-8412-66bc711098a2)
```python
Sonar_data.groupby(60).mean()
```
![image](https://github.com/user-attachments/assets/236787f5-0568-4ccf-afbd-90ddcc4ab46f)
```python
#Seperating Data and Labels
X=Sonar_data.drop(columns=60,axis=1)
Y=Sonar_data[60]
print(X)
print(Y)
```
![image](https://github.com/user-attachments/assets/07b19ba9-fa82-4ea8-b0e4-b558647d2c1d)

**Training and Test Data**
```python
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.05,stratify=Y,random_state=1)
print(X.shape,X_train.shape,X_test.shape)

print(X_train)
print(Y_train)
```
![image](https://github.com/user-attachments/assets/691a9fcc-2635-4201-ad54-cec041f8f246)

**Model Training (Logistic Regression Model)**
```python
Model=LogisticRegression()

#Training the logistic Regression model with training Data
Model.fit(X_train, Y_train)

#Accuracy score for the Project
train_accuracy=accuracy_score(Model.predict(X_train),Y_train)
print("Train_accuracy",train_accuracy)
test_accuracy=accuracy_score(Model.predict(X_test),Y_test)
print("Test_accuracy",test_accuracy)
```
![image](https://github.com/user-attachments/assets/c159bb30-1b92-4e11-a598-e69808184538)
