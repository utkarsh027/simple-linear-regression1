import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Salary_Data.csv')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values



#splitting the dataset into triningset and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)"""
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
#predicting test set result
y_pred=regressor.predict(x_test)
#visualising the training set result
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.tiltle('salary vs experience(training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.tiltle('salary vs experience(test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()