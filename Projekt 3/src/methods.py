
from sklearn.mixture import GMM
from pylab import *
from toolbox_02450 import clusterplot
from sklearn import cross_validation
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram


def gmm(X,y,M):
    #X = X[:,0:2]
    #y = y[:,:]
    #M=2
    #M += 3
    # Number of clusters
    
    K = 4
    
    cov_type = 'full'       # type of covariance, you can try out 'diag' as well
    
    reps = 1                # number of fits with different initalizations, best result will be kept
    
    
    
    # Fit Gaussian mixture model
    
    gmm = GMM(n_components=K, covariance_type=cov_type, n_init=reps, params='wmc').fit(X)
    
    cls = gmm.predict(X)    # extract cluster labels
    
    cds = gmm.means_        # extract cluster centroids (means of gaussians)
    
    covs = gmm.covars_      # extract cluster shapes (covariances of gaussians)
    
    
    
    if cov_type == 'diag':
    
        new_covs = np.zeros([K,M,M])
    
        count = 0
    
        for elem in covs:
    
            temp_m = np.zeros([M,M])
    
            for i in range(len(elem)):
    
                temp_m[i][i] = elem[i]
    
            new_covs[count] = temp_m
    
            count += 1
    
        covs = new_covs
    
    
    
    # Plot results:
    
    figure(figsize=(14,9))
    X2 = X[:,0:2]
    M=2
    
    #for c in range(C):
    #    class_mask = y.A.ravel()==c
    #    plot(X[class_mask,attr1], X[class_mask,attr2], 'o')
    ncolors = K
    y2 = np.asarray(y)

    hold(True)
    colors = [0]*ncolors
    for color in range(ncolors):
        colors[color] = cm.jet.__call__(color*255/(ncolors-1))[:3]
    for i,cs in enumerate(np.unique(y2)):
        plot(X[(y2==cs).ravel(),0], X[(y2==cs).ravel(),1], 'o', markeredgecolor='k', markerfacecolor=colors[i],markersize=6, zorder=2)
    for i,cr in enumerate(np.unique(cls)):
        plot(X[(cls==cr).ravel(),0], X[(cls==cr).ravel(),1], 'o', markersize=12, markeredgecolor=colors[i], markerfacecolor='None', markeredgewidth=3, zorder=1)
    if False:#centroids!='None':        
        for cd in range(centroids.shape[0]):
            plot(centroids[cd,0], centroids[cd,1], '*', markersize=22, markeredgecolor='k', markerfacecolor=colors[cd], markeredgewidth=2, zorder=3)

    #clusterplot(X2, clusterid=cls, centroids=cds, y=y, covars=covs)
    
    show()


def getPrincipalComponents(X):
    Y = X - np.ones((len(X),1))*X.mean(0)
    
    U,S,V = linalg.svd(Y,full_matrices=False)
    
    return U
    
def CVK(X,KRange,covar_type,reps):
    N, M = X.shape
    T = len(KRange)
    
    CVE = np.zeros((T,1))
    
    # K-fold crossvalidation
    CV = cross_validation.KFold(N,10,shuffle=True)

    for t,K in enumerate(KRange):
            print('Fitting model for K={0}\n'.format(K))
    
            # Fit Gaussian mixture model
            gmm = GMM(n_components=K, covariance_type=covar_type, n_init=reps, params='wmc').fit(X)
            
             # For each crossvalidation fold
            for train_index, test_index in CV:
    
                # extract training and test set for current CV fold
                X_train = X[train_index]
                X_test = X[test_index]
    
                # Fit Gaussian mixture model to X_train
                gmm = GMM(n_components=K, covariance_type=covar_type, n_init=reps, params='wmc').fit(X_train)
    
                # compute negative log likelihood of X_test
                CVE[t] += -gmm.score(X_test).sum()
                print CVE[t]
                
        # Plot results
    
    figure(1); hold(True)
    #plot(KRange, BIC)
    #plot(KRange, AIC)
    plot(KRange, 2*CVE)
    legend(['Crossvalidation'])
    xlabel('K')
    show()
    
    
def HCANDERSEN(X,y,Maxclust, Method = 'single', Metric = 'euclidean'):
    # Perform hierarchical/agglomerative clustering on data matrix
    
    Z = linkage(X, method=Method, metric=Metric)
    
    # Compute and display clusters by thresholding the dendrogram
    cls = fcluster(Z, criterion='maxclust', t=Maxclust)
    figure()
    clusterplot(X, cls.reshape(cls.shape[0],1), y=y)
    
    # Display dendrogram
    max_display_levels=50
    figure()
    dendrogram(Z, truncate_mode='level', p=max_display_levels)
    
    show()
