## Support Vector Model
This is One of the most important model of the Supervised Machine learning algorithm.

It is commonly used for classification and regression tasks. It works by finding a hyperplane that best separates data points into different classes in a high-dimensional space.
### Importing the Dependencies
```pyython
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```
### Data Collection and Analysis
Make sure that the Dataset file is bought to the Collab Folder
```python
Diabetics_Data=pd.read_csv("/content/diabetes.csv")
```
**In Order to know the function and parameters of a particular Function**
```python
pd.read_csv?
```
**Printing the first five rows of the Dataset**
```python
print(Diabetics_Data.head())
```
![image](https://github.com/user-attachments/assets/d8877067-014d-4bdb-8eb1-f007fa22cc0a)
**Number of Rows and Columns in the Dataframe**
```python
Diabetics_Data.shape
```
![image](https://github.com/user-attachments/assets/c3aca70f-646d-4a00-8ffb-9a8fd265e4a1)

**Then try to get the statistical measure of the data and the value counts in case of the outcomes**
```python
Diabetics_Data.describe()
Diabetics_Data["Outcome"].value_counts()
```
**Grouping the Datasets based on the labels is another important aspect**
```python
Diabetics_Data.groupby("Outcome").mean()
```
![image](https://github.com/user-attachments/assets/b31ec7a9-ef0b-4b99-bc0d-bfa749df19ce)

**Seperating the Data and the Labels**
```python
X=Diabetics_Data.drop(columns="Outcome",axis=1)
Y=Diabetics_Data["Outcome"]
print(X)
print(Y)
```
## Data Standardization
We Use this if the columns consist of different range of Numbers. So we have to standardize and transform it.
```python
scaler=StandardScaler()
scaler.fit_transform(X)
```
**Then we split the data into train and test and then do the model fit**
```python
Model=svm.SVC(kernel="linear")
Model.fit(X_train,Y_train)
```
## Checking the accuracy and Inputing the Data
```python
Accuracy=accuracy_score(Model.predict(X_train),Y_train)
print(Accuracy)
Input=(1,85,66,29,0,26.6,0.351,31)
Input_array=np.asarray(Input)
Input_array_reshaped=Input_array.reshape(1,-1)
data=scaler.transform(Input_array_reshaped)
Predict=Model.predict(data)
print(Predict)
```
![image](https://github.com/user-attachments/assets/55f11a7b-a6a3-4479-ab53-00527730bc4a)
![image](https://github.com/user-attachments/assets/2e250b0f-3de2-41f6-9186-b75371739426)





