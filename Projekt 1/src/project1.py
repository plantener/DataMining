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
        
#Using CHD as attribute
size = 463
positive = 160
noAttributes = 9

#Open data
doc = xlrd.open_workbook('dataset.xls').sheet_by_index(0)

#Get Attributes
attributeNames = doc.row_values(0,1,noAttributes+1)

#Create classes
classLabels = doc.col_values(noAttributes+1,1,size)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(2)))

y = np.mat([classDict[value] for value in classLabels]).T

yPositive = np.mat([[1] for i in range(positive)])
yNegative = np.mat([[0] for i in range(size-positive)])

N = len(y)
M = len(attributeNames)
C = len(classNames)


#Create matrix holding data
X = np.mat(np.empty((size-1,noAttributes)))
for i, col_id in enumerate(range(1,noAttributes+1)):
    if(attributeNames[col_id-1] == "famhist"):
        temp12 = [convert(i2) for i2 in doc.col_values(col_id,1,size)]
        X[:,i] = np.mat(temp12).T
    else:
        X[:,i] = np.mat(doc.col_values(col_id,1,size)).T
        
        
#Stanardize data       
XStandardized = zscore(X, ddof=1)

#Find all positive CHD
XPositive = X[y.A.ravel()==1,:]
XPositiveStd = zscore(XPositive,ddof=1)
XPositiveStd2 = XStandardized[y.A.ravel()==1,:]
#All negative CHD
XNegative = X[y.A.ravel()==0,:]
XNegativeStd = zscore(XNegative,ddof=1)

#Calcuate mean and variance
print("Calculate statistics")
calculateStatistics(X,noAttributes)

#Make histograms
histogram(X,attributeNames,y)

#Make boxplots
boxPlot(X,attributeNames)

#Plot alcohol and tobacco
plotTwoAttributes(1,7,X,y,classNames,attributeNames)
plotTwoAttributes(5,7,X,y,classNames,attributeNames)
plotTwoAttributes(3,6,X,y,classNames,attributeNames)

#Plot the variance explained by the principal components
computePrincipalComponents(XStandardized, "For both negative and positive CHD")

#Plot the data projected to the first two principal components
plotPrincipalComponents(0,1,XStandardized,y,classNames)

#Calculate directions of PCAs
print("For both positive and negative:")
print(getPCADirections(XStandardized))
        

#Calculate correlation between attributes
corrcoef = corrcoef(X.T,y.T)

print("How the attributes correlate:")
print(corrcoef)

plot3DPrincipalComponents(X,y,classNames,0,1,2,attributeNames)