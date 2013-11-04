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
XPC = getPrincipalComponents(XStandardized)

(XWithoutLDL,yWithoutLDL) = removeAttribute(X,y,2)

#forwardSelection(XWithoutLDL,yWithoutLDL,N,M,5,attributeNames,classNames)


(X_train,y_train),(X_test,y_test) = getTestAndTrainingSet(X,y)
(X_train_std,y_train_std),(X_test_std,y_test_std) = getTestAndTrainingSet(XStandardized,y)


#linearRegression(X,y,attributeNames,'ldl')
#print(y)
#forwardSelection(X,y,N,M,5,attributeNames,classNames)
#artificialNeuralNetwork(X,y,N,noAttributes)

#forwardSelection(XStandardized,y,N,M,5,attributeNames,classNames)
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

X2PC = np.copy(XPC)
X2PC = X2PC[:,0:1]

#X2PC = scipy.delete(X2PC,8,1) # PC9
#X2PC = scipy.delete(X2PC,7,1) # PC8
#X2PC = scipy.delete(X2PC,6,1) # PC7
#X2PC = scipy.delete(X2PC,5,1) # PC6
#X2PC = scipy.delete(X2PC,4,1) # PC5
#X2PC = scipy.delete(X2PC,3,1) # PC4
#X2PC = scipy.delete(X2PC,2,1) # PC3


(X_train_ad,y_train_ad),(X_test_ad,y_test_ad) = getTestAndTrainingSet(Xad,y)


#artificialNeuralNetwork(Xad, y, N, noAttributes-4)


# Classification
s1 = "X not modified."
s2 = "Attributes of X selected according to result of forward selection."
s3 = "X represented by principal components"
s4 = "X represented by two most important principal components."
#
K = 4

#artificialNeuralNetwork(X,y,N,noAttributes, K=K, s=s1)
#artificialNeuralNetwork(Xad,y,N,noAttributes-4, K=K, s=s2)
#artificialNeuralNetwork(XPC,y,N,noAttributes, K=K, s=s3)
#artificialNeuralNetwork(X2PC,y,N,2, K=K, s=s4)


#logisticRegression(X,y,s=s1)
#logisticRegression(Xad,y,s=s2)
#logisticRegression(XPC,y,s=s3)
#logisticRegression(X2PC,y,s=s4)

decisionTree(X,y,attributeNames,classNames,"Decision_Tree_X.gvz",s=s1)
decisionTree(Xad,y,attributeNames,classNames,"Decision_Tree_Xad.gvz",s=s2)
decisionTree(XPC,y,attributeNames,classNames,"Decision_Tree_XPC.gvz",s=s3)
decisionTree(X2PC,y,attributeNames,classNames,"Decision_Tree_X2PC.gvz",s=s4)
#
#kNearestNeighbours(X,y,N,C,s=s1)
#kNearestNeighbours(Xad,y,N,C,s=s2)
#kNearestNeighbours(XPC,y,N,C,s=s3)
#kNearestNeighbours(X2PC,y,N,C,s=s4)
#
#plotKNearestNeighbours(classNames, X, y, C, s=s1, k=15)
#plotKNearestNeighbours(classNames, Xad, y, C, s=s2, k=14)
#plotKNearestNeighbours(classNames, XPC, y, C, s=s3, k=24)
#plotKNearestNeighbours(classNames, X2PC, y, C, DoPrincipalComponentAnalysis = True,s=s4,neighbours=30)