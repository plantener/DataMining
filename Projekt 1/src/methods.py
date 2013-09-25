# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:50:50 2013

@author: Jacob
"""

import numpy as np
from pylab import *

#Plot the data
def plotTwoAttributes(attr1, attr2, X, y, classNames, attributeNames):
    C = len(classNames)
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
def computePrincipalComponents(X):
    Y = X - np.ones((len(X),1))*X.mean(0)
    
    U,S,V = linalg.svd(Y,full_matrices=False)
    
    rho = (S*S) / (S*S).sum()
    
    figure()
    plot(range(1,len(rho)+1),rho,'o-')
    title('Variance explained by principal components');
    xlabel('Principal component');
    ylabel('Variance explained');
    show()
    

#Plot principal components against each other
def plotPrincipalComponents(principal1, principal2, X, y, classNames):
    C = len(classNames)    
    Y = X - np.ones((len(X),1))*X.mean(0)
    
    U,S,V = linalg.svd(Y,full_matrices=False)
    V = mat(V).T
    
    Z = Y * V
    
    print(V[0])
    
    # Plot PCA of the data
    f = figure()
    f.hold()
    title('Heart Disease data: PCA')
    for c in range(C):
        class_mask = y.A.ravel()==c
        plot(Z[class_mask,principal1], Z[class_mask,principal2], 'o')
    #plot(Z[:,principal1], Z[:,principal2], 'o')
    legend(classNames)
    xlabel('PC{0}'.format(principal1+1))
    ylabel('PC{0}'.format(principal2+1))
    
    # Output result to screen
    show()
    
    
#Calculate similarities
def calculateSim(row, similarity_measure, X):
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