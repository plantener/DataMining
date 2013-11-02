# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 12:40:57 2013

@author: jesperplantener
"""


import numpy as np
import xlrd
import sklearn.linear_model as lm
from pylab import *
from methods import *
from scipy.stats import zscore
import scipy

#Converts Present and Absent into numbers.
def convert(s):
    if s == "Present":
        return 1
    else:
        return 0

#Load dataset
doc = xlrd.open_workbook('../../dataset_sorted.xls').sheet_by_index(0)

size = 463
noAttributes = 9

#Get attributes and classnames
attributeNames = doc.row_values(0,1,noAttributes+1)
attributeNamesCHD = doc.row_values(0,1,noAttributes+1+1)

classLabels = doc.col_values(noAttributes+1,1,size)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(2)))


y = np.mat([classDict[value] for value in classLabels]).T

X = np.mat(np.empty((size-1,noAttributes)))
XCHD =np.mat(np.empty((size-1,noAttributes+1)))

for i, col_id in enumerate(range(1,noAttributes+1+1)):
    if(i < len(attributeNames) and attributeNames[i] == "famhist"):
        temp12 = [convert(i2) for i2 in doc.col_values(col_id,1,size)]
        if i < noAttributes:
            X[:,i] = np.mat(temp12).T
        XCHD[:,i] = np.mat(temp12).T
    else:
        if i < noAttributes:
            X[:,i] = np.mat(doc.col_values(col_id,1,size)).T
        XCHD[:,i] = np.mat(doc.col_values(col_id,1,size)).T

M = len(attributeNames) 
N = len(y)
C = len(classNames)

XStandardized = zscore(X, ddof=1)


#X = sortByChd(X,y)

#X.sort(key=lambda x: x[1])

print(X)

#Split dataset into features and data vector
#ldl_idx = attributeNames.index('ldl')
#y = X[:,ldl_idx]

#X_cols = range(0,ldl_idx) + range(ldl_idx+1,len(attributeNames))
#X_rows = range(0,len(y))
#X = X[ix_(X_rows,X_cols)]

# Fit ordinary least squares regression model
#model = lm.LinearRegression()
#model.fit(X,y)

# Predict ldl value
#y_est = model.predict(X)
#residual = y_est-y

# Display scatter plot
#figure()
#subplot(2,1,1)
#plot(y, y_est, '.')
#xlabel('ldl value (true)'); ylabel('ldl value (estimated)');
#subplot(2,1,2)
#hist(residual,40)

#show()

#linearRegression(X,y)


#forwardSelection(X,y,N,M,5,attributeNames)

#forwardSelection(XStandardized,y,N,M,5,attributeNames)

#X = XStandardized

#X = scipy.delete(X,8,1) # Age
#X = scipy.delete(X,7,1) # Alcohol
#X = scipy.delete(X,6,1) # Obesity
#X = scipy.delete(X,5,1) # TypeA
#X = scipy.delete(X,4,1) # Famhist
#X = scipy.delete(X,3,1) # Adiposity
#X = scipy.delete(X,2,1) # LDL
#X = scipy.delete(X,1,1) # Tobacco
#X = scipy.delete(X,0,1) # SBP

linearRegression(X,y,attributeNames)

#artificialNeuralNetwork(XStandardized,y,N,noAttributes)

#artificialNeuralNetworkByPC(XStandardized,y,N)

#decisionTree(X,y,attributeNames,classNames)

kNearestNeighbours(X,y,N,C,99)
