# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:17:11 2013

@author: Jacob
"""

from pylab import *
import sklearn.linear_model as lm
from sklearn import cross_validation
from toolbox_02450 import feature_selector_lr, bmplot

def sortByChd(X,y):
    XPositive = X[y.A.ravel()==1,:]
    XNegative = X[y.A.ravel()==0,:]
    i = 0
    for e in XNegative:
        X[i] = e
        i = i+1
    for e in XPositive:
        X[i] = e
        i = i+1
    return X
    
    
    
def linearRegression(X,y):
    # Fit logistic regression model
    model = lm.logistic.LogisticRegression()
    model = model.fit(X, y.A.ravel())
    
    
    # Classify wine as CHD Negative/Positive (0/1)
    y_est = model.predict(X)
    y_est_chd_prob = model.predict_proba(X)[:, 1]
    
    # Evaluate classifier's misclassification rate over entire training data
    misclass_rate = sum(np.abs(np.mat(y_est).T - y)) / float(len(y_est))
    
    # Define a new data object
    #x = np.array([138.33, 3.64*2, 4.74, 25.41, 0, 53.10, 26.04, 17.04, 42.82])
    # Evaluate athe probability of x being possitive of CHD
    #x_class = model.predict_proba(x)[0,1]
    
    
    #print('\nProbability of given sample being positive for CHD: {0:.4f}'.format(x_class))
    
    print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))
    
    f = figure(); f.hold(True)
    class0_ids = nonzero(y==0)[0].tolist()[0]
    plot(class0_ids, y_est_chd_prob[class0_ids], '.y', c= 'red')
    class1_ids = nonzero(y==1)[0].tolist()[0]
    plot(class1_ids, y_est_chd_prob[class1_ids], '.r', c = 'blue')
    xlabel('Data object'); ylabel('Predicted prob. of having chd');
    legend(['Negative', 'Positive'])
    #ylim(-0.5,1.5)
    
    show()
    
    
    
    
def forwardSelection(X,y,N,M,K,attributeNames):
    # Add offset attribute
    X2 = np.concatenate((np.ones((X.shape[0],1)),X),1)
    attributeNames2 = [u'Offset']+attributeNames
    M2 = M+1
    
    ## Crossvalidation
    # Create crossvalidation partition for evaluation
    CV = cross_validation.KFold(N,K,shuffle=True)
    
    # Initialize variables
    Features = np.zeros((M2,K))
    #Error_train = np.empty((K,1))
    #Error_test = np.empty((K,1))
    Error_train_fs = np.empty((K,1))
    Error_test_fs = np.empty((K,1))
    Error_train_nofeatures = np.empty((K,1))
    Error_test_nofeatures = np.empty((K,1))
    
    k=0
    for train_index, test_index in CV:
        
        # extract training and test set for current CV fold
        X_train = X2[train_index]
        y_train = y[train_index]
        #X_test = X2[test_index]
        #y_test = y[test_index]
        internal_cross_validation = 10
        
        
        # Compute squared error with feature subset selection
        selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation)
        Features[selected_features,k]=1
        
        figure(k)
        subplot(1,2,1)
        plot(range(1,len(loss_record)), loss_record[1:])
        xlabel('Iteration')
        ylabel('Squared error (crossvalidation)')    
        
        subplot(1,3,3)
        bmplot(attributeNames2, range(1,features_record.shape[1]), -features_record[:,1:])
        clim(-1.5,0)
        xlabel('Iteration')
    
        print('Cross validation fold {0}/{1}'.format(k+1,K))
        print('Train indices: {0}'.format(train_index))
        print('Test indices: {0}'.format(test_index))
        print('Features no: {0}\n'.format(selected_features.size))
    
        k+=1
    
    
    # Display results
    print('\n')
    #print('Linear regression without feature selection:\n')
    #print('- Training error: {0}'.format(Error_train.mean()))
    #print('- Test error:     {0}'.format(Error_test.mean()))
    #print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
    #print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
    print('Linear regression with feature selection:\n')
    print('- Training error: {0}'.format(Error_train_fs.mean()))
    print('- Test error:     {0}'.format(Error_test_fs.mean()))
    print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))
    
    figure(k)
    subplot(1,3,2)
    bmplot(attributeNames2, range(1,Features.shape[1]+1), -Features)
    clim(-1.5,0)
    xlabel('Crossvalidation fold')
    ylabel('Attribute')
    
    # Inspect selected feature coefficients effect on the entire dataset and
    # plot the fitted model residual error as function of each attribute to
    # inspect for systematic structure in the residual
    f=2 # cross-validation fold to inspect
    ff=Features[:,f-1].nonzero()[0]
    m = lm.LinearRegression().fit(X2[:,ff], y)
    
    y_est= m.predict(X2[:,ff])
    residual=y-y_est
    
    figure(k+1)
    title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
    for i in range(0,len(ff)):
       subplot(2,ceil(len(ff)/2.0),i+1)
       plot(X2[:,ff[i]],residual,'.')
       xlabel(attributeNames2[ff[i]])
       ylabel('residual error')
    
    
    show()    
