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

class ResultHolder:
    resultNo = 0    
    results = []
    
    def init(__self__):
        results = []
        resultNo = 0
        
    def addResult(__self__, res):
        results.add(res)
        resultNo += 1
        
    def printMeanResult(__self__):
        sum = 0.0
        for e in results:
            sum += e
        return double(sum) / double(resultNo)

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

#XPC = getTwoPrincipalComponents(XStandardized)

(XWithoutLDL,yWithoutLDL,attributeNamesWithoutAttr) = removeAttribute(X,y,2,attributeNames)
#forwardSelection(XWithoutLDL,yWithoutLDL,N,M,5,attributeNames,classNames)

XWithoutLDLStandardized = zscore(XWithoutLDL, ddof=1)

(X_train,y_train),(X_test,y_test) = getTestAndTrainingSet(X,y)
(X_train_PC,y_train_PC),(X_test_PC,y_test_PC) = getTestAndTrainingSet(XStandardized,y)

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

X2PC = np.copy(XPC)
X2PC = X2PC[:,0:2]

#X2PC = scipy.delete(X2PC,8,1) # PC9
#X2PC = scipy.delete(X2PC,7,1) # PC8
#X2PC = scipy.delete(X2PC,6,1) # PC7
#X2PC = scipy.delete(X2PC,5,1) # PC6
#X2PC = scipy.delete(X2PC,4,1) # PC5
#X2PC = scipy.delete(X2PC,3,1) # PC4
#X2PC = scipy.delete(X2PC,2,1) # PC3


(X_train_ad,y_train_ad),(X_test_ad,y_test_ad) = getTestAndTrainingSet(Xad,y)
(X_train_2PC,y_train_2PC),(X_test_2PC,y_test_2PC) = getTestAndTrainingSet(X2PC,y)

(_,_,attributeNamesXad) = removeAttribute(X,y,6,attributeNames)
(_,_,attributeNamesXad) = removeAttribute(X,y,3,attributeNamesXad)
(_,_,attributeNamesXad) = removeAttribute(X,y,1,attributeNamesXad)
(_,_,attributeNamesXad) = removeAttribute(X,y,0,attributeNamesXad)

(_,_,attributeNamesX2PC) = removeAttribute(X,y,8,attributeNames)
(_,_,attributeNamesX2PC) = removeAttribute(X,y,7,attributeNames)
(_,_,attributeNamesX2PC) = removeAttribute(X,y,6,attributeNames)
(_,_,attributeNamesX2PC) = removeAttribute(X,y,5,attributeNames)
(_,_,attributeNamesX2PC) = removeAttribute(X,y,4,attributeNames)
(_,_,attributeNamesX2PC) = removeAttribute(X,y,3,attributeNames)
(_,_,attributeNamesX2PC) = removeAttribute(X,y,2,attributeNames)


#artificialNeuralNetwork(Xad, y, N, noAttributes-4)


# Classification
s1 = "X not modified."
s2 = "Attributes of X selected according to result of forward selection."
s3 = "X represented by principal components"
s4 = "X represented by two most important principal components."
#

#K=4
#artificialNeuralNetwork(X,y,N,noAttributes, K=K, s=s1)
#artificialNeuralNetwork(Xad,y,N,noAttributes-4, K=K, s=s2)
#artificialNeuralNetwork(XPC,y,N,noAttributes, K=K, s=s3)
#artificialNeuralNetwork(X2PC,y,N,2, K=K, s=s4)


#predictLinearRegression(X,y)

#logisticRegression(X,y,X_train, y_train, X_test, y_test, s=s1)
#logisticRegression(Xad,y,X_train_ad, y_train_ad, X_test_ad, y_test_ad, s=s2)
#logisticRegression(XPC,y,X_train_PC, y_train_PC, X_test_PC, y_test_PC, s=s3)
#logisticRegression(X2PC,y,X_train_2PC, y_train_2PC, X_test_2PC, y_test_2PC, s=s4)
#
#decisionTree(X,y,attributeNames,classNames,"Decision_Tree_X.gvz",s=s1)#X,X_train,y_train,X_test,y_test)
#decisionTree(Xad,y,attributeNamesXad,classNames,"Decision_Tree_Xad.gvz",s=s1)#,X_train_ad,y_train_ad,X_test_ad,y_test_ad)
#decisionTree(XPC,y,attributeNames,classNames,"Decision_Tree_XPC.gvz",s=s1)#,X_train_PC,y_train_PC,X_test_PC,y_test_PC)
#decisionTree(X2PC,y,attributeNamesX2PC,classNames,"Decision_Tree_X2PC.gvz",s=s1)#,X_train_2PC,y_train_2PC,X_test_2PC,y_test_2PC)
#
(XK,e1) = kNearestNeighbours(X,y,C,s=s1)
(XKad,e2) = kNearestNeighbours(Xad,y,C,s=s2)
(XKPC,e3) = kNearestNeighbours(XPC,y,C,s=s3)
(XK2PC,e4) = kNearestNeighbours(X2PC,y,C,s=s4)
#
plotKNearestNeighbours(classNames, X, y, C, s=s1, K=15)
plotKNearestNeighbours(classNames, Xad, y, C, s=s2, K=14)
plotKNearestNeighbours(classNames, XPC, y, C, s=s3, K=24)
plotKNearestNeighbours(classNames, X2PC, y, C, DoPrincipalComponentAnalysis = True,s=s4,K=30)
