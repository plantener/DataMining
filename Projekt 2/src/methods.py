# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:17:11 2013

@author: Jacob
"""

from pylab import *
import sklearn.linear_model as lm
from sklearn import cross_validation
from toolbox_02450 import feature_selector_lr, bmplot
import neurolab as nl
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import scipy




#Converting for legend
def convertToWord(n):
   if (n < 0.5):
        return "Negative CHD"
   else:
        return "Positive CHD"
        
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
    
    
    
def logisticRegression(X,y,s=""):
    print "Doing logistic regression for: "
    print s
    # Fit logistic regression model
    model = lm.logistic.LogisticRegression()
    model = model.fit(X, y.A.ravel())
    
    
    # Classify objects as CHD Negative/Positive (0/1)
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
    print "\n"
    
    f = figure(); f.hold(True)
    class0_ids = nonzero(y==0)[0].tolist()[0]
    plot(class0_ids, y_est_chd_prob[class0_ids], '.y', c= 'red')
    class1_ids = nonzero(y==1)[0].tolist()[0]
    plot(class1_ids, y_est_chd_prob[class1_ids], '.r', c = 'blue')
    xlabel('Data object'); ylabel('Predicted prob. of having chd');
    legend(['Negative', 'Positive'])
    title("Logistic regression")
    #ylim(-0.5,1.5)
    
    show()
    
def linearRegression(X,y,attributeNames,attribute):
   # Split dataset into features and target vector
    alcohol_idx = attributeNames.index(attribute)
    y = X[:,alcohol_idx]
    
    X_cols = range(0,alcohol_idx) + range(alcohol_idx+1,len(attributeNames))
    X_rows = range(0,len(y))
    U = X[ix_(X_rows,X_cols)]
    
    # Fit ordinary least squares regression model
    model = lm.LinearRegression()
    model.fit(U,y)
    
    # Predict alcohol content
    y_est = model.predict(U)
    residual = y_est-y
    
    # Display scatter plot
    figure()
    subplot(2,1,1)
    
    plot(y, y_est, '.')
    xlabel(attribute + ' value (true)'); ylabel(attribute + ' value (estimated)');
    subplot(2,1,2)
    hist(residual,40)
    
    show()

    
    
    
def forwardSelection(X,y,N,M,K,attributeNames, classNames):
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
        X_test = X2[test_index]
        y_test = y[test_index]
        internal_cross_validation = 10
        
        
        
        # Compute squared error without using the input data at all
        Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
        Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]

        # Compute squared error with feature subset selection
        selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation)
        Features[selected_features,k]=1
            # .. alternatively you could use module sklearn.feature_selection
        m = lm.LinearRegression().fit(X_train[:,selected_features], y_train)
        Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
        Error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]

        
        figure()
        subplot(1,2,1)
        plot(range(1,len(loss_record)), loss_record[1:])
        xlabel('Iteration')
        ylabel('Squared error (crossvalidation)')    
        
        subplot(1,3,3)
        bmplot(attributeNames2, range(1,features_record.shape[1]), -features_record[:,1:])
        clim(-1.5,0)
        xlabel('Iteration')
    
        print('Cross validation fold {0}/{1}'.format(k+1,K))
    
        k+=1
    
    
    # Display results
    print('\n')
    print('Linear regression with feature selection:\n')
    print('- Training error: {0}'.format(Error_train_fs.mean()))
    print('- Test error:     {0}'.format(Error_test_fs.mean()))
    print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))
    
    figure()
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
    
    figure()
    title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
    for i in range(0,len(ff)):
       subplot(2,ceil(len(ff)/2.0),i+1)
       for c in classNames:
           class_mask = (y_est==c)
           plot(X2[:,ff[i]],residual,'.')
       xlabel(attributeNames2[ff[i]])
       ylabel('residual error')
    
    
    show()    


def artificialNeuralNetwork(X,y,N,noAttributes,K=4):
    # Parameters for neural network classifier
    n_hidden_units = 1      # number of hidden units
    n_train = 2             # number of networks trained in each k-fold
    
    # These parameters are usually adjusted to: (1) data specifics, (2) computational constraints
    #learning_rate = 0.01    # rate of weights adaptation
    learning_goal = 2.0     # stop criterion 1 (train mse to be reached)
    max_epochs = 200        # stop criterion 2 (max epochs in training)
    
    # K-fold CrossValidation
    CV = cross_validation.KFold(N,K,shuffle=True)
    
    # Variable for classification error
    errors = np.zeros(K)
    error_hist = np.zeros((max_epochs,K))
    bestnet = list()
    k=0
    for train_index, test_index in CV:
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
        
        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index,:]
        X_test = X[test_index,:]
        y_test = y[test_index,:]
        
        best_train_error = 1e100
        for i in range(n_train):
            # Create randomly initialized network with 2 layers
            ann = nl.net.newff([[0, 1]]*noAttributes, [n_hidden_units, 1], [nl.trans.LogSig(),nl.trans.LogSig()])
            # train network
            train_error = ann.train(X_train, y_train, goal=learning_goal, epochs=max_epochs, show=round(max_epochs/8))
            if train_error[-1]<best_train_error:
                bestnet.append(ann)
                best_train_error = train_error[-1]
                error_hist[range(len(train_error)),k] = train_error
        
        y_est = bestnet[k].sim(X_test)
        y_est = (y_est>.5).astype(int)
        errors[k] = (y_est!=y_test).sum().astype(float)/y_test.shape[0]
        k+=1
        
    
    # Print the average classification error rate
    print('Error rate: {0}%'.format(100*mean(errors)))
    
    # Display exemplary networks learning curve (best network of each fold)
    figure(); hold(True)
    bn_id = argmax(error_hist[-1,:])
    error_hist[error_hist==0] = learning_goal
    for bn_id in range(K):
        plot(error_hist[:,bn_id]); xlabel('epoch'); ylabel('train error (mse)'); title('Learning curve (best for each CV fold)')
    
    plot(range(max_epochs), [learning_goal]*max_epochs, '-.')
    
    
    show()
    
    
def getTwoPrincipalComponents(X):
    Y = X - np.ones((len(X),1))*X.mean(0)
    
    U,S,V = linalg.svd(Y,full_matrices=False)
    
    return U[:,0:2]


def artificialNeuralNetworkByPC(X,y,N,K=4):
    U = getTwoPrincipalComponents(X)
    
    # Parameters for neural network classifier
    n_hidden_units = 1      # number of hidden units
    n_train = 2             # number of networks trained in each k-fold
    
    # These parameters are usually adjusted to: (1) data specifics, (2) computational constraints
    learning_goal = 2.0     # stop criterion 1 (train mse to be reached)
    max_epochs = 200        # stop criterion 2 (max epochs in training)
    
    # K-fold CrossValidation
    CV = cross_validation.KFold(N,K,shuffle=True)
    
    # Variable for classification error
    errors = np.zeros(K)
    error_hist = np.zeros((max_epochs,K))
    bestnet = list()
    k=0
    for train_index, test_index in CV:
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
        
        # extract training and test set for current CV fold
        X_train = U[train_index,:]
        y_train = y[train_index,:]
        X_test = U[test_index,:]
        y_test = y[test_index,:]
        
        best_train_error = 1e100
        for i in range(n_train):
            # Create randomly initialized network
            ann = nl.net.newff([[0, 1], [0, 1]], [n_hidden_units, 1], [nl.trans.LogSig(),nl.trans.LogSig()])
            # train network
            train_error = ann.train(X_train, y_train, goal=learning_goal, epochs=max_epochs, show=round(max_epochs/8))
            if train_error[-1]<best_train_error:
                bestnet.append(ann)
                best_train_error = train_error[-1]
                error_hist[range(len(train_error)),k] = train_error
        
        y_est = bestnet[k].sim(X_test)
        y_est = (y_est>.5).astype(int)
        errors[k] = (y_est!=y_test).sum().astype(float)/y_test.shape[0]
        k+=1
        
    
    # Print the average classification error rate
    print('Error rate: {0}%'.format(100*mean(errors)))
    
    
    # Display the decision boundary for the several crossvalidation folds.
    # (create grid of points, compute network output for each point, color-code and plot).
    grid_range = [-0.25, 0.25, -0.25, 0.25]; delta = 0.05; levels = 100
    a = arange(grid_range[0],grid_range[1],delta)
    b = arange(grid_range[2],grid_range[3],delta)
    A, B = meshgrid(a, b)
    values = np.zeros(A.shape)
    
    figure(figsize=(18,9)); hold(True)
    for k in range(K):
        subplot(2,2,k+1)
        cmask = (y==0).A.ravel(); plot(U[cmask,0], U[cmask,1],'.r')
        cmask = (y==1).A.ravel(); plot(U[cmask,0], U[cmask,1],'.b')
        title('Model prediction and decision boundary (kfold={0})'.format(k+1))
        xlabel('PC 1'); ylabel('PC 2');
        for i in range(len(a)):
            for j in range(len(b)):
                values[i,j] = bestnet[k].sim( np.mat([a[i],b[j]]) )[0,0]
        contour(A, B, values, levels=[.5], colors=['k'], linestyles='dashed')
        contourf(A, B, values, levels=linspace(values.min(),values.max(),levels), cmap=cm.RdBu)
        if k==0: colorbar(); legend(['Class A (y=0)', 'Class B (y=1)'])
    
    
    # Display exemplary networks learning curve (best network of each fold)
    figure(); hold(True)
    bn_id = argmax(error_hist[-1,:])
    error_hist[error_hist==0] = learning_goal
    colors = ['red','blue','green','orange']
    for bn_id in range(K):
        plot(error_hist[:,bn_id],c=colors[bn_id]); xlabel('epoch'); ylabel('train error (mse)'); title('Learning curve (best for each CV fold)')
    legend(['Round 1', 'Round 2', 'Round 3', 'Round 4'])
    plot(range(max_epochs), [learning_goal]*max_epochs, '-.')
    
    
    show()
    
    
def decisionTree(X,y,attributeNames,classNames,s=""):
    print "Doing decision tree for: "
    print s
    # Fit regression tree classifier, Gini split criterion, pruning enabled
    dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=100)
    dtc = dtc.fit(X,y)
    
    # Export tree graph for visualization purposes:
    # (note: you can use i.e. Graphviz application to visualize the file)
    out = tree.export_graphviz(dtc, out_file='tree_gini_CHD_data.gvz', feature_names=attributeNames)
    out.close()
    
    # Define a new data object (new type of wine) with the attributes given in the text
    #x = np.array([138.33, 3.64, 4.74, 25.41, 0, 53.10, 26.04, 17.04, 42.82])
    #x = np.array([138.33*2, 3.64*2, 4.74*2, 25.41*2, 1, 53.10*2, 26.04*2, 17.04*2, 42.82])
    #x = np.array([138.33, 3.64, 4.74, 25.41, 1, 20, 26.04, 17.04*1, 40])
     
    # Evaluate the classification tree for the new data object
    #x_class = dtc.predict(x)[0]
    
    # Print results
    #print '\nNew object attributes:'
    #for attr in attributeNames:
    #    print attr[0]
    #print '\nClassification result:'
    #if classNames[x_class] > 0.5:
    #    print "Positive"
    #else:
    #    print "Negative"
        
        

def kNearestNeighbours(X, y, N, C, L=40, s=""):    
    print "Doing k-nearest neighbours for: " 
    print s
    
    # Cross-validation not necessary. Instead, compute matrix of nearest neighbor
    # distances between each pair of data points ..
    knclassifier = KNeighborsClassifier(n_neighbors=L+1, warn_on_equidistant=False).fit(X, y)
    neighbors = knclassifier.kneighbors(X)
    # .. and extract matrix where each row contains class labels of subsequent neighbours
    # (sorted by distance)
    ndist, nid = neighbors[0], neighbors[1]
    nclass = y[nid].flatten().reshape(N,L+1)
    
    # Use the above matrix to compute the class labels of majority of neighbors
    # (for each number of neighbors l), and estimate the test errors.
    errors = np.zeros(L)
    nclass_count = np.zeros((N,C))
    for l in range(1,L+1):
        for c in range(C):
            nclass_count[:,c] = sum(nclass[:,1:l+1]==c,1).A.ravel()
        y_est = np.argmax(nclass_count,1);
        errors[l-1] = (y_est!=y.A.ravel()).sum()
    
        
    # Plot the classification error rate
    figure()
    plot(100*errors/N)
    xlabel('Number of neighbors')
    ylabel('Classification error rate (%)')
    
    figure()
    imshow(nclass, cmap='binary', interpolation='None'); xlabel("k'th neighbor"); ylabel('data point'); title("Neighbors class matrix");
    
    show()

def getTestAndTrainingSet(X,y,K=5):
    N = len(X)
    
    CV = cross_validation.KFold(N,K,shuffle=True)
    
    k=0
    
    for train_index, test_index in CV:
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
        
        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index,:]
        X_test = X[test_index,:]
        y_test = y[test_index,:]
        k+=1
        
        if(k==K):
            return (X_train,y_train),(X_test,y_test)
    
def plotKNearestNeighbours(classNames,X, y, C, K=5, attribute1 = 0, attribute2 = 1, DoPrincipalComponentAnalysis = False, s="", neighbours = 5, X_train = [], y_train = [], X_test = [], y_test = []):
    print "Plotting k-nearest neighbours for: "
    print s
    if DoPrincipalComponentAnalysis:
        U = getTwoPrincipalComponents(X)
        a1 = 0
        a2 = 1
    else:
        a1 = attribute1
        a2 = attribute2
        U = X
    (X_train,y_train),(X_test,y_test) = getTestAndTrainingSet(U,y,K)
    
    #figure();
    #hold(True);
    #styles = ['.b', '.r']
    #for c in range(C):
    #    class_mask = (y_train == c).A.ravel()
        #class_mask = (y_train == c)
    #    plot(X_train[class_mask,a1], X_train[class_mask,a2], styles[c])
    #styles = ['ob', 'or']
    
    
    
    # Distance metric (corresponds to 2nd norm, euclidean distance).
    # You can set dist=1 to obtain manhattan distance (cityblock distance).
    dist=2
    
    # Fit classifier and classify the test points
    knclassifier = KNeighborsClassifier(n_neighbors=K, p=dist);
    knclassifier.fit(X_train, y_train);
    y_est = knclassifier.predict(X_test);
    
    
    # Plot the classfication results
    
    # Plot the training data points (color-coded) and test data points.
    figure();
    hold(True);
    styles = ['.b', '.r']
    for c in range(C):
        class_mask = y_train.A.ravel()==c
        #class_mask = y_train == c
        plot(X_train[class_mask,a1], X_train[class_mask,a2], styles[c])    
    styles = ['ob', 'or']
    # Plot result of classification
    for c in range(C):
        class_mask = (y_est==c)
        plot(X_test[class_mask,a1], X_test[class_mask,a2], styles[c], markersize=10)
        plot(X_test[class_mask,a1], X_test[class_mask,a2], 'kx', markersize=8)
    title('Data classification Results - KNN');
    legend([convertToWord(i) for i in classNames])
    show()
    
    #Plot actual values of objects
    figure();
    hold(True);
    styles = ['.b', '.r']
    for c in range(C):
        class_mask = y_train.A.ravel()==c
        #class_mask = (y_train == c)
        plot(X_train[class_mask,a1], X_train[class_mask,a2], styles[c])
    styles = ['ob', 'or']
    for c in range(C):
        class_mask = y_test.A.ravel() == c
        plot(X_test[class_mask,a1], X_test[class_mask,a2], styles[c], markersize=10)
        plot(X_test[class_mask,a1], X_test[class_mask,a2], 'kx', markersize=8)
    title('Actual value of objects - KNN');
    legend([convertToWord(i) for i in classNames])
    show()


    cm = confusion_matrix(y_test.A.ravel(), y_est);
    accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
    figure();
    imshow(cm, cmap='binary', interpolation='None');
    colorbar()
    xticks(range(C)); yticks(range(C));
    xlabel('Predicted class'); ylabel('Actual class');
    title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy, error_rate));
    
    show()
    
def removeAttribute(X,y,attribute):
    yWithoutAttr = X[:,attribute]
    XWithoutAttr = np.copy(X)
    XWithoutAttr = scipy.delete(XWithoutAttr,attribute,1)
    return (XWithoutAttr, yWithoutAttr)
    