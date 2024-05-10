#  Exp:02 Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for the marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Kanishka.V.S
RegisterNumber: 212222230061
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
X=df.iloc[:,0:1]
Y=df.iloc[:,-1]
X
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
m=lr.coef_
m[0]
b=lr.intercept_
b
```

## Output:
![image](https://github.com/kanishka2305/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497357/a18699e6-a5ca-4980-94e5-4c869fc402c6)
![image](https://github.com/kanishka2305/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497357/96acf869-f98e-468d-930b-f0f1dc455930)
![image](https://github.com/kanishka2305/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497357/1a683fab-aca5-4876-865f-9ed8ebe0e093)
![image](https://github.com/kanishka2305/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497357/8e998bb8-e7fb-4803-92f5-d2c309a44432)
![image](https://github.com/kanishka2305/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497357/e3f494ae-8dd7-4bce-bab7-e1e79d90b8a9)
![image](https://github.com/kanishka2305/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497357/1f5a138b-2dfe-4bf9-bca0-a47e097cb832)
![image](https://github.com/kanishka2305/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497357/9e41884a-0090-4164-9909-74e85e1e16e7)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
