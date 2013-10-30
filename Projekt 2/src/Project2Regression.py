# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 12:40:57 2013

@author: jesperplantener
"""


import numpy as np
import xlrd
import sklearn.linear_model as lm
from pylab import *

#Converts Present and Absent into numbers.
def convert(s):
    if s == "Present":
        return 1
    else:
        return 0

#Load dataset
doc = xlrd.open_workbook('../../dataset.xls').sheet_by_index(0)

size = 463
noAttributes = 9

#Get attributes and classnames
attributeNames = doc.row_values(0,1,noAttributes+1)
classLabels = doc.col_values(noAttributes+1,1,size)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(2)))


y = np.mat([classDict[value] for value in classLabels]).T
X = np.mat(np.empty((size-1,noAttributes)))
for i, col_id in enumerate(range(1,noAttributes+1)):
    if(attributeNames[col_id-1] == "famhist"):
        temp12 = [convert(i2) for i2 in doc.col_values(col_id,1,size)]
        X[:,i] = np.mat(temp12).T
    else:
        X[:,i] = np.mat(doc.col_values(col_id,1,size)).T

M = len(attributeNames) 
N = len(y)

print(X)

#Split dataset into features and data vector
ldl_idx = attributeNames.index('ldl')
y = X[:,ldl_idx]

X_cols = range(0,ldl_idx) + range(ldl_idx+1,len(attributeNames))
X_rows = range(0,len(y))
X = X[ix_(X_rows,X_cols)]

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)

# Predict ldl value
y_est = model.predict(X)
residual = y_est-y

# Display scatter plot
figure()
subplot(2,1,1)
plot(y, y_est, '.')
xlabel('ldl value (true)'); ylabel('ldl value (estimated)');
subplot(2,1,2)
hist(residual,40)

show()