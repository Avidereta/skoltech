import numpy as np
from sklearn.linear_model.base import BaseEstimator
from numpy.testing import assert_approx_equal
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from scipy.misc import logsumexp

from utils import compute_labels, log_likelihood, log_likelihood_from_labels


class Random(BaseEstimator):
    def __init__(self, n_clusters, n_init=10):
        self.n_clusters = n_clusters
        self.n_init = n_init

    def fit(self, X):
        n_objects = X.shape[0]
        best_log_likelihood = float("-inf")
        for i in range(self.n_init):
            centers_idx = np.random.choice(n_objects, size=self.n_clusters, replace=False)
            mu = X[centers_idx, :]
            labels = compute_labels(X, mu)
            ll = log_likelihood_from_labels(X, labels)
            if ll > best_log_likelihood:
                best_log_likelihood = ll
                self.cluster_centers_ = mu.copy()
                self.labels_ = labels

class EM_init:
    def __init__(self, n_clusters, max_iter, min_covar = 0.001, tol = 0.001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.min_covar = min_covar
        self.tol = tol
        self.logs = {'mu': [], 'labels': [], 'w': [], 'sigma':[], 'log_likelihood': []}
        self.convergence = 1


    def estep(self, X, w, mu, sigma):

        """Compute Gamma_ik.

        @param X is n_objects x n_features matrix
        @param w is a vector of size n_clusters of the prior probabilities of the clusters
        mu is n_clusters x n_features matrix of the centers of the clusters
        sigma is n_clusters x n_features x n_features tensor with the covariance
                matrix of each cluster.
        """
        assert_approx_equal(np.sum(w), 1)
        n_objects, n_features = X.shape
        n_clusters = w.size
        log_gamma = np.zeros((n_objects, n_clusters))
        for cluster in range(n_clusters):
            log_gamma[:, cluster] = np.log(w[cluster])
            log_gamma[:, cluster] += multivariate_normal.logpdf(X, mu[cluster, :],
                                         sigma[cluster, :, :] + self.min_covar*np.eye(n_features))                                                            
        norm_coef = logsumexp(log_gamma, axis=1)
        for cluster in range(n_clusters):
            log_gamma[:, cluster] -= norm_coef

        return log_gamma
    
    def mstep(self, X, log_gamma):
        
        n_clusters = self.n_clusters
        n_objects, n_features = X.shape
        
        w = np.zeros(n_clusters)
        mu = np.zeros((n_clusters, n_features))
        sigma = np.zeros((n_clusters, n_features, n_features))
        
        for cluster in range(n_clusters):

            gamma_k = np.exp(log_gamma[:, cluster]) # Column of Gammas for this cluster
            N_k = np.sum(gamma_k)
            
            w[cluster] = N_k/n_objects
            mu[cluster, :] = np.dot(gamma_k.T, X)/N_k
            mu_k = mu[cluster, :]
            
            for i in range (n_objects):
                  sigma[cluster, :, :] += gamma_k[i] * np.outer((X[i] - mu_k), (X[i] - mu_k).T)/N_k
          
        return w, mu, sigma
    
    def save_logs(self, labels, w, mu, sigma, ll):
        self.logs['mu'].append(mu)
        self.logs['sigma'].append(sigma)
        self.logs['w'].append(w)
        self.logs['labels'].append(labels)
        self.logs['log_likelihood'].append(ll)
        
    def fit(self,X):
        
        n_objects = X.shape[0]
        n_features = X.shape[1]
        
        sigma = np.zeros((self.n_clusters, n_features, n_features))
        w = np.tile(1.0/self.n_clusters, self.n_clusters)


        centers_idx = np.random.choice(n_objects, size=self.n_clusters, replace=False)
        mu = X[centers_idx, :] 
            
        for cluster in range (self.n_clusters):
            sigma[cluster :, :] = np.eye(n_features)
        
        ll = log_likelihood(X, w, mu, sigma)
        
        for i in range(self.max_iter):
            
            ll_new = log_likelihood(X, w, mu, sigma)
            self.save_logs(compute_labels(X, mu), w, mu, sigma, ll)
            
            if i > 0 and abs(ll_new - ll) < self.tol:
                self.cluster_centers_ = mu.copy()
                self.labels_ = compute_labels(X, mu)
                self.covars_ = sigma.copy()
                self.w_ = w.copy()
                break
            else:
                log_gamma = self.estep(X,w, mu, sigma)
                w, mu, sigma = self.mstep(X,log_gamma)
                ll = ll_new
                i+=1
                if i == self.max_iter:
                    self.convergence = -1
                    
                    
class EM:
    def __init__(self, n_clusters, max_iter=100, n_init=10, min_covar=0.001, tol=0.001, logging = True):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.min_covar = min_covar
        self.tol = tol
        self.inits = []
        self.logs = {'mu': [], 'labels': [], 'w': [], 'sigma':[], 'log_likelihood': []}
        for i in range(self.n_init):
            self.inits.append(EM_init(self.n_clusters, self.max_iter, self.min_covar, self.tol))

    def fit(self, X):
        
        for i in range(self.n_init):
            self.inits[i].fit(X)
        
        convergent_inits =  []
        for i in range (len(self.inits)):
            if self.inits[i].convergence == 1: 
                convergent_inits.append(i)
            else:
                print  'Initialization', i, 'has no convergence'
        
        best_init = convergent_inits[np.argmax(self.inits[i].logs['log_likelihood'][-1] for i in convergent_inits)]
        self.cluster_centers_ = self.inits[best_init].cluster_centers_.copy()
        self.labels_ = self.inits[best_init].labels_.copy()
        self.covars_ = self.inits[best_init].covars_.copy()
        self.w_ = self.inits[best_init].w_.copy()
        self.logs = self.inits[best_init].logs.copy() 
        
        
        
        
        

class SoftKMeans_init:
    def __init__(self, n_clusters, max_iter, min_covar = 0.001, tol = 0.001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.min_covar = min_covar
        self.tol = tol
        self.logs = {'mu': [], 'labels': [], 'w': [], 'sigma':[], 'log_likelihood': []}
        self.convergence = 1
 
    def estep(self, X, w, mu, sigma):

        """Compute Gamma_ik.

        X is n_objects x n_features matrix
        w is a vector of size n_clusters of the prior probabilities of the clusters
        mu is n_clusters x n_features matrix of the centers of the clusters
        sigma is n_clusters x n_features x n_features tensor with the covariance
                matrix of each cluster.
        """
        assert_approx_equal(np.sum(w), 1)
        n_objects, n_features = X.shape
        n_clusters = w.size
        log_gamma = np.zeros((n_objects, n_clusters))
        for cluster in range(n_clusters):
            log_gamma[:, cluster] = np.log(w[cluster])
            log_gamma[:, cluster] += multivariate_normal.logpdf(X, mu[cluster, :],
                                                         sigma[cluster, :, :])                                                            
        norm_coef = logsumexp(log_gamma, axis=1)
        for cluster in range(n_clusters):
            log_gamma[:, cluster] -= norm_coef

        return log_gamma
    
    def mstep(self, X, log_gamma):
        
        n_clusters = self.n_clusters
        n_objects, n_features = X.shape
        
        w = np.zeros(n_clusters)
        mu = np.zeros((n_clusters, n_features))
        sigma = np.zeros((n_clusters, n_features, n_features))
        
        for cluster in range(n_clusters):

            gamma_k = np.exp(log_gamma[:, cluster]) # Column of Gammas for this cluster
            N_k = np.sum(gamma_k)
            
            w[cluster] = N_k/n_objects
            mu[cluster, :] = np.dot(gamma_k.T, X)/N_k
            mu_k = mu[cluster, :]
            sigma[cluster, :, :] += np.eye(n_features)
          
        return w, mu, sigma
    
    def save_logs(self, labels, w, mu, sigma, ll):
        self.logs['mu'].append(mu)
        self.logs['sigma'].append(sigma)
        self.logs['w'].append(w)
        self.logs['labels'].append(labels)
        self.logs['log_likelihood'].append(ll)
        
    def fit(self,X):
        
        n_objects = X.shape[0]
        n_features = X.shape[1]
        
        sigma = np.zeros((self.n_clusters, n_features, n_features))
        w = np.tile(1.0/self.n_clusters, self.n_clusters)


        centers_idx = np.random.choice(n_objects, size=self.n_clusters, replace=False)
        mu = X[centers_idx, :] 
            
        for cluster in range (self.n_clusters):
            sigma[cluster :, :] = np.eye(n_features)
        
        ll = log_likelihood(X, w, mu, sigma)
        
        for i in range(self.max_iter):
            
            ll_new = log_likelihood(X, w, mu, sigma)
            self.save_logs(compute_labels(X, mu), w, mu, sigma, ll)
            
            if i > 0 and abs(ll_new - ll) < self.tol:
                self.cluster_centers_ = mu.copy()
                self.labels_ = compute_labels(X, mu)
                self.covars_ = sigma.copy()
                self.w_ = w.copy()
                break
            else:
                log_gamma = self.estep(X,w, mu, sigma)
                w, mu, sigma = self.mstep(X,log_gamma)
                ll = ll_new
                i+=1
                if i == self.max_iter:
                    self.convergence = -1


class SoftKMeans:
    def __init__(self, n_clusters, max_iter=100, n_init=10, min_covar=0.001, tol=0.001, logging = True):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.min_covar = min_covar
        self.tol = tol
        self.logs = {'mu': [], 'labels': [], 'w': [], 'sigma':[], 'log_likelihood': []}
        
        self.inits = []
        for i in range(self.n_init):
            self.inits.append(SoftKMeans_init(self.n_clusters, self.max_iter, self.min_covar, self.tol))

    def fit(self, X):
        
        for i in range(self.n_init):
            self.inits[i].fit(X)
        
        convergent_inits =  []
        for i in range (len(self.inits)):
            if self.inits[i].convergence == 1: 
                convergent_inits.append(i)
            else:
                print  'Initialization', i, 'has no convergence'
        best_init = convergent_inits[np.argmax(self.inits[i].logs['log_likelihood'][-1] for i in convergent_inits)]
        self.cluster_centers_ = self.inits[best_init].cluster_centers_.copy()
        self.labels_ = self.inits[best_init].labels_.copy()
        self.covars_ = self.inits[best_init].covars_.copy()
        self.w_ = self.inits[best_init].w_.copy()
        self.logs = self.inits[best_init].logs.copy()
        
        
class KMeans_init:
    def __init__(self, n_clusters, max_iter, min_covar = 0.001, tol = 0.001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.min_covar = min_covar
        self.tol = tol
        self.logs = {'mu': [], 'labels': [], 'w': [], 'sigma':[], 'log_likelihood': []}
        self.convergence = 1
 
    def estep(self, X, w, mu, sigma):

        """Compute Gamma_ik.

        X is n_objects x n_features matrix
        w is a vector of size n_clusters of the prior probabilities of the clusters
        mu is n_clusters x n_features matrix of the centers of the clusters
        sigma is n_clusters x n_features x n_features tensor with the covariance
                matrix of each cluster.
        """
        assert_approx_equal(np.sum(w), 1)
        n_objects, n_features = X.shape
        n_clusters = w.size
        log_prob = np.zeros((n_objects, n_clusters))
        gamma = np.zeros((n_objects, n_clusters))
        for cluster in range(n_clusters):
            log_prob[:, cluster] = multivariate_normal.logpdf(X, mu[cluster, :],
                                                         sigma[cluster, :, :])                                                            
        norm_coef = logsumexp(log_prob, axis=1)
        for cluster in range(n_clusters):
            log_prob[:, cluster] -= norm_coef
        
        labels = np.argmax(log_prob, axis = 1)
        for i in range (len(labels)):
            gamma[i, labels[i]] = 1
        return gamma
    
    def mstep(self, X, gamma):
        
        n_clusters = self.n_clusters
        n_objects, n_features = X.shape
        
        w = np.tile(1.0/self.n_clusters, self.n_clusters)
        mu = np.zeros((n_clusters, n_features))
        sigma = np.zeros((n_clusters, n_features, n_features))
        
        for cluster in range(n_clusters):

            gamma_k = gamma[:, cluster] # Column of Gammas for this cluster
            N_k = np.sum(gamma_k)
            if N_k > 0:
                mu[cluster, :] = np.dot(gamma_k.T, X)/N_k
                mu_k = mu[cluster, :]
            sigma[cluster, :, :] += np.eye(n_features)
          
        return w, mu, sigma
    
    def save_logs(self, labels, w, mu, sigma, ll):
        self.logs['mu'].append(mu)
        self.logs['sigma'].append(sigma)
        self.logs['w'].append(w)
        self.logs['labels'].append(labels)
        self.logs['log_likelihood'].append(ll)
        
    def fit(self,X):
        
        n_objects = X.shape[0]
        n_features = X.shape[1]
        
        sigma = np.zeros((self.n_clusters, n_features, n_features))
        w = np.tile(1.0/self.n_clusters, self.n_clusters)


        centers_idx = np.random.choice(n_objects, size=self.n_clusters, replace=False)
        mu = X[centers_idx, :] 
            
        for cluster in range (self.n_clusters):
            sigma[cluster :, :] = np.eye(n_features)
        
        ll = log_likelihood(X, w, mu, sigma)
        
        for i in range(self.max_iter):
            
            ll_new = log_likelihood(X, w, mu, sigma)
            self.save_logs(compute_labels(X, mu), w, mu, sigma, ll)
            
            if i > 0 and abs(ll_new - ll) < self.tol:
                self.cluster_centers_ = mu.copy()
                self.labels_ = compute_labels(X, mu)
                self.covars_ = sigma.copy()
                self.w_ = w.copy()
                break
            else:
                gamma = self.estep(X,w, mu, sigma)
                w, mu, sigma = self.mstep(X,gamma)
                ll = ll_new
                i+=1
                if i == self.max_iter:
                    self.convergence = -1


class KMeans:
    def __init__(self, n_clusters, max_iter=100, n_init=10, min_covar=0.001, tol=0.001, logging = True):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.min_covar = min_covar
        self.tol = tol
        self.logs = {'mu': [], 'labels': [], 'w': [], 'sigma':[], 'log_likelihood': []}
        self.convergence = 1
        self.inits = []
        for i in range(self.n_init):
            self.inits.append(KMeans_init(self.n_clusters, self.max_iter, self.min_covar, self.tol))

    def fit(self, X):
        
        for i in range(self.n_init):
            self.inits[i].fit(X)
        
        convergent_inits =  []
        for i in range (len(self.inits)):
            if self.inits[i].convergence == 1: 
                convergent_inits.append(i)
            else:
                print  'Initialization', i, 'has no convergence'
        if len(convergent_inits) > 0:
            best_init = convergent_inits[np.argmax(self.inits[i].logs['log_likelihood'][-1] for i in convergent_inits)]
            self.cluster_centers_ = self.inits[best_init].cluster_centers_.copy()
            self.labels_ = self.inits[best_init].labels_.copy()
            self.covars_ = self.inits[best_init].covars_.copy()
            self.w_ = self.inits[best_init].w_.copy()
            self.logs = self.inits[best_init].logs.copy()
        else: 
            self.convergence = -1
            print '\nUnfortunately, there is not convergence at all'
