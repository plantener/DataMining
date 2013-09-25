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
attributeNames = doc.row_values(0,1,11)

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

#print(C)

#print(classDict)

#Create matrix holding data
off_set = 0
X = np.mat(np.empty((462,10)))
for i, col_id in enumerate(range(1,11)):
    #print(attributeNames[col_id-1])
    #print(doc.col_values(col_id,1,92))
    #print("i=",i)
    #print("col_id=",col_id)
    if(attributeNames[col_id-1] == "famhist"):
        temp12 = [convert(i2) for i2 in doc.col_values(col_id,1,463)]
        X[:,i] = np.mat(temp12).T
        #off_set = 1
    else:
        X[:,i-off_set] = np.mat(doc.col_values(col_id,1,463)).T
 
#Stanardize data       
#X = zscore(X, ddof=1)

#Find all positive CHD
XPositive = X[y.A.ravel()==1,:]
#All negative CHD
XNegative = X[y.A.ravel()==0,:]



        
        
#calculateSim(0,"Correlation")
        
#computePrincipalComponents(X)
#plotPrincipalComponents(0,1,X,y,classNames)
#plotTwoAttributes(3,6,X,y,classNames,attributeNames)

corrcoef(X.T)