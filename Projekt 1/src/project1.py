# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:21:30 2013

@author: Jacob
"""

import numpy as np
import xlrd
from pylab import *

def convert(s):
    if s == "Present":
        return 1
    else:
        return 0

#Open data
doc = xlrd.open_workbook('..\\..\\dataset.xlsx').sheet_by_index(0)

#Get Attributes
attributeNames = doc.row_values(0,1,10)
print(attributeNames)

#Create classes
classLabels = doc.col_values(10,1,463)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(2)))

y = np.mat([classDict[value] for value in classLabels]).T

N = len(y)
M = len(attributeNames)
C = len(classNames)

#print(C)

#print(classDict)

#Create matrix holding data
off_set = 0
X = np.mat(np.empty((462,9)))
for i, col_id in enumerate(range(1,10)):
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
      
      
#print(X[1,:])
        



#Plot the data
def plotTwoAttributesAgainstEachOther(attr1, attr2):
    f = figure()
    f.hold()
    title('Heart Disease')
    for c in range(C):
        class_mask = y.A.ravel()==c
        plot(X[class_mask,attr1], X[class_mask,attr2], 'o')
    legend(classNames)
    xlabel(attributeNames[attr1])
    ylabel(attributeNames[attr2])
    
    # Output result to screen
    show()


#Compute principal componentss
def computePrincipalComponents():
    Y = X - np.ones((462,1))*X.mean(0)
    
    U,S,V = linalg.svd(Y,full_matrices=False)
    
    rho = (S*S) / (S*S).sum()
    
    figure()
    plot(range(1,len(rho)+1),rho,'o-')
    title('Variance explained by principal components');
    xlabel('Principal component');
    ylabel('Variance explained');
    show()
    

#Plot principal components against each other
def plotPrincipalComponents(principal1, principal2):
    Y = X - np.ones((462,1))*X.mean(0)
    
    U,S,V = linalg.svd(Y,full_matrices=False)
    V = mat(V).T
    
    Z = Y * V
    
    # Plot PCA of the data
    f = figure()
    f.hold()
    title('Heart Disease data: PCA')
    for c in range(C):
        class_mask = y.A.ravel()==c
        plot(Z[class_mask,principal1], Z[class_mask,principal2], 'o')
    legend(classNames)
    xlabel('PC{0}'.format(principal1+1))
    ylabel('PC{0}'.format(principal2+1))
    
    # Output result to screen
    show()
    
    
#Calculate similarities
def calculateSim(row, similarity_measure):
    print("Id of interest: ",row)
    print(X[row])
    N, M = shape(X)
    noti = range(0,row) + range(row+1,N)
    sim = similarity(X[row,:], X[noti,:], similarity_measure)
    sim = sim.tolist()[0]
    sim_to_index = sorted(zip(sim,noti))
    
    for ms in range(3):
        im_id = sim_to_index[-ms-1][1]
        im_sim = sim_to_index[-ms-1][0]
        print("Id: ",im_id)
        print("Similarity: ", im_sim)
        print(X[im_id])
        
        
#calculateSim(0,"Correlation")
        
#computePrincipalComponents()
