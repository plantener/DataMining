# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:21:30 2013

@author: Jacob
"""

import numpy as np
import xlrd
from pylab import *
from scipy.stats import zscore
from methods import *

def convert(s):
    if s == "Present":
        return 1
    else:
        return 0
        


#Open data
doc = xlrd.open_workbook('..\\..\\dataset.xlsx').sheet_by_index(0)

#Get Attributes
attributeNames = doc.row_values(0,1,10)

#Create classes
classLabels = doc.col_values(10,1,463)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(2)))

y = np.mat([classDict[value] for value in classLabels]).T

yPositive = np.mat([[1] for i in range(160)])
yNegative = np.mat([[0] for i in range(463-160)])

N = len(y)
M = len(attributeNames)
C = len(classNames)


#Create matrix holding data
X = np.mat(np.empty((462,9)))
for i, col_id in enumerate(range(1,10)):
    if(attributeNames[col_id-1] == "famhist"):
        temp12 = [convert(i2) for i2 in doc.col_values(col_id,1,463)]
        X[:,i] = np.mat(temp12).T
    else:
        X[:,i] = np.mat(doc.col_values(col_id,1,463)).T
        
        
#Stanardize data       
XStandardized = zscore(X, ddof=1)

#Find all positive CHD
XPositive = X[y.A.ravel()==1,:]
XPositiveStd = zscore(XPositive,ddof=1)
XPositiveStd2 = XStandardized[y.A.ravel()==1,:]
#All negative CHD
XNegative = X[y.A.ravel()==0,:]
XNegativeStd = zscore(XNegative,ddof=1)

#Make histograms
histogram(X,attributeNames)

#Make boxplots
boxPlot(X,attributeNames)

#Plot alcohol and tobacco
plotTwoAttributes(1,7,X,y,classNames,attributeNames)
plotTwoAttributes(5,7,X,y,classNames,attributeNames)
plotTwoAttributes(3,6,X,y,classNames,attributeNames)

#Plot the variance explained by the principal components
computePrincipalComponents(XStandardized, "For both negative and positive CHD")
computePrincipalComponents(XPositiveStd, "Only for positive CHD")
computePrincipalComponents(XNegativeStd, "Only for negative CHD")

#Plot the data projected to the first two principal components
plotPrincipalComponents(0,1,XStandardized,y,classNames)
plotPrincipalComponents(0,1,XPositiveStd,yPositive,classNames)
plotPrincipalComponents(0,1,XNegativeStd,yPositive,classNames)

#Calculate directions of PCAs
print("For both positive and negative:")
print(getPCADirection(0,XStandardized))
print("For positive CHD")
print(getPCADirection(0,XPositiveStd))
print("For negative CHD:")
print(getPCADirection(0,XNegativeStd))
        

#Calculate correlation between attributes
corrcoef = corrcoef(X.T,y.T)

print("How the attributes correlate to CHD")
print([corrcoef[9]])