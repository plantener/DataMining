
from sklearn.mixture import GMM
from pylab import *
from toolbox_02450 import clusterplot


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