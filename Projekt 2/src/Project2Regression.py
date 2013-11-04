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

XPC = getTwoPrincipalComponents(XStandardized)

(XWithoutLDL,yWithoutLDL,attributeNamesWithoutAttr) = removeAttribute(X,y,2,attributeNames)

XWithoutLDLStandardized = zscore(XWithoutLDL, ddof=1)

(X_train,y_train),(X_test,y_test) = getTestAndTrainingSet(X,y)
(X_train_std,y_train_std),(X_test_std,y_test_std) = getTestAndTrainingSet(XStandardized,y)

#linearRegression(X,y,attributeNames,'chd')
#linearRegression(X,y,attributeNamesWithoutAttr,'ldl')
#linearRegression(X,y,attributeNames,'ldl')
#print(y)
#forwardSelection(X,y,N,5,attributeNames,classNames)
#forwardSelection(XWithoutLDL,yWithoutLDL,N,5,attributeNamesWithoutAttr,classNames)
#forwardSelection(XStandardized,y,N,5,attributeNames,classNames)

#artificialNeuralNetwork(X,y,N,noAttributes)

#forwardSelection(XWithoutLDLStandardized,yWithoutLDL,N,5,attributeNamesWithoutAttr,classNames)
#artificialNeuralNetworkByPC(XStandardized,y,N)

Xad = np.copy(X)

#Xad = scipy.delete(Xad,8,1) # Age
#Xad = scipy.delete(Xad,7,1) # Alcohol
Xad = scipy.delete(Xad,6,1) # Obesity
#Xad = scipy.delete(Xad,5,1) # TypeA
#Xad = scipy.delete(Xad,4,1) # Famhist
Xad = scipy.delete(Xad,3,1) # Adiposity
#Xad = scipy.delete(Xad,2,1) # LDL
Xad = scipy.delete(Xad,1,1) # Tobacco
Xad = scipy.delete(Xad,0,1) # SBP

(X_train_ad,y_train_ad),(X_test_ad,y_test_ad) = getTestAndTrainingSet(XStandardized,y)


#artificialNeuralNetwork(Xad, y, N, noAttributes-4)


# Classification
s1 = "X not modified."
s2 = "Attributes of X selected according to result of forward selection."
s3 = "X represented by two most important principal components."

predictLinearRegression(X,y)

#logisticRegression(X,y,s=s1)
#logisticRegression(Xad,y,s=s2)
#logisticRegression(XPC,y,s=s3)

#decisionTree(X,y,attributeNames,classNames,s=s1)
#decisionTree(Xad,y,attributeNames,classNames,s=s2)
#decisionTree(XPC,y,attributeNames,classNames,s=s3)

#kNearestNeighbours(X,y,N,C,s=s1)
#kNearestNeighbours(Xad,y,N,C,s=s2)
#kNearestNeighbours(XPC,y,N,C,s=s3)

#plotKNearestNeighbours(classNames, X, y, C)
#plotKNearestNeighbours(classNames, XStandardized, y, C, DoPrincipalComponentAnalysis = True,s=s3,neighbours=30)