# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 20:53:52 2022

@author: Kedar Pandya
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%

#import the dataset

data = pd.read_csv('Salary_Data.csv')
print(data.head(10))
#%%

# use the code below to visualize all the null values int the data set 
# sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')

# create the X and y

X = data.iloc[:,:-1].values
y = data.iloc[:,1].values

print('\nThe features of the dataset are:\n',X)
print('\nThe classes of the dataset are:\n',y)

#%%

# split the data set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('\nThe training set of the features is:\n',X_train)
print('\nThe training set of the classes is:',y_train)
print('\nThe test set of the features is:',X_test)
print('\nThe test set of the class is:',y_test)

#%%

# training the simple linear regression

from sklearn.linear_model import LinearRegression
sreg = LinearRegression()
sreg.fit(X_train,y_train)

#%%

# predicting the model

y_pred = sreg.predict(X_test)

#%%

# visualizing the training set result

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, sreg.predict(X_train), color = 'blue')
plt.title('Training Set')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#%%

# visualizing the test set result

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Test Set')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
#%%

# to print the value of an arbitary year for eg. what would be the salary of an employee
# that has 21 years of experience

print(sreg.predict([[21]]))

#%%

# this code will tell you what is the value of the intercept and the coefficient in the 
# in the equation Y = aX + b

print(sreg.intercept_)
print(sreg.coef_)