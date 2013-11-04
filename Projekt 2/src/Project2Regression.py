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


linearRegression(X,y,attributeNames,'ldl')
forwardSelection(X,y,N,M,5,attributeNames,classNames)
artificialNeuralNetwork(X,y,N,noAttributes)

forwardSelection(XStandardized,y,N,M,5,attributeNames,classNames)
artificialNeuralNetworkByPC(XStandardized,y,N)

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

artificialNeuralNetwork(Xad, y, N, noAttributes-4)


# Classification

logisticRegression(X,y)
logisticRegression(Xad,y)
decisionTree(X,y,attributeNames,classNames)
kNearestNeighbours(X,y,N,C)
decisionTree(X, y, attributeNames, classNames)
plotKNearestNeighbours(classNames, X, y, C)
plotKNearestNeighbours(classNames, X, y, C, DoPrincipalComponentAnalysis = True)