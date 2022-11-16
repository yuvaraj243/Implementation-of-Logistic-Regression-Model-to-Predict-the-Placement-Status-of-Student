# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries. 
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset. 
5.Predict the values of array. 
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: YUVARAJ.V
RegisterNumber:  212220220056
*/

import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```
## Output:

![ss1](https://user-images.githubusercontent.com/115924983/202195278-7cf1829f-f2d7-4aaf-97f5-8fe45b2abfb5.png)
![ss2](https://user-images.githubusercontent.com/115924983/202195350-757aa957-6804-46c0-bccc-326264d386e9.png)
![ss3](https://user-images.githubusercontent.com/115924983/202195442-49a1f0b9-4f83-409d-a18d-573b6acfe8f1.png)
![ss4](https://user-images.githubusercontent.com/115924983/202195479-a96c05ae-e6f4-4f46-962b-ad5519ec9148.png)
![ss5](https://user-images.githubusercontent.com/115924983/202195516-05260260-a6ee-4da9-b5fe-398b890667d1.png)
![ss6](https://user-images.githubusercontent.com/115924983/202195545-6b8fe0c7-d0f2-42d2-8185-7d446276a72e.png)
![ss7](https://user-images.githubusercontent.com/115924983/202195575-604c2b2d-370f-4677-968e-be97be7d3f44.png)
![ss8](https://user-images.githubusercontent.com/115924983/202195605-8aff0075-135e-42d6-bb10-92eb5b109c07.png)
![ss9](https://user-images.githubusercontent.com/115924983/202195633-7d9df9a0-2a3f-42a6-a026-8ae91e3dc236.png)
![ss10](https://user-images.githubusercontent.com/115924983/202195689-8e3e3668-e707-439f-bfd3-4b962743570b.png)
![ss11](https://user-images.githubusercontent.com/115924983/202195748-c898e8c2-95a8-4c40-9829-fa13264f9df7.png)
![ss12](https://user-images.githubusercontent.com/115924983/202195766-06a18358-e435-4f99-87fc-6323abe67843.png)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
