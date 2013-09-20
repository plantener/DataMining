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
attributeNames = doc.row_values(0,1,11)

#Create matrix holding data
X = np.mat(np.empty((462,10)))
for i, col_id in enumerate(range(1,11)):
    #print(attributeNames[col_id-1])
    #print(doc.col_values(col_id,1,92))
    print("i=",i)
    print("col_id=",col_id)
    if(attributeNames[col_id-1] == "famhist"):
        temp12 = [convert(i2) for i2 in doc.col_values(col_id,1,463)]
        X[:,i] = np.mat(temp12).T
    else:
        X[:,i] = np.mat(doc.col_values(col_id,1,463)).T
        
        
#Plot the data
attr1 = 8
attr2 = 7        
        
f = figure()
f.hold()
title('Heart Disease')
plot(X[:,attr1], X[:,attr2], 'o')
xlabel(attributeNames[attr1])
ylabel(attributeNames[attr2])

# Output result to screen
show()


#Compute principal componentss
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
principal1 = 0
principal2 = 1

#Y = X - np.ones((462,1))*X.mean(0)

#U,S,V = linalg.svd(Y,full_matrices=False)
V = mat(V).T

Z = Y * V

# Plot PCA of the data
f = figure()
f.hold()
title('Heart Disease data: PCA')
plot(Z[:,principal1], Z[:,principal2], 'o')
xlabel('PC{0}'.format(prin1+1))
ylabel('PC{0}'.format(prin2+1))

# Output result to screen
show()