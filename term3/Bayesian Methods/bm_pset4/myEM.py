
import numpy as np
from sklearn.linear_model.base import BaseEstimator
from numpy.testing import assert_approx_equal
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from scipy.misc import logsumexp

from utils import compute_labels, log_likelihood, log_likelihood_from_labels
from abstract_clusterer import *

class EM(AbstractClusterer):
    """Look abstract_clusterer.py for description of params"""
    
    def __init__(self, n_clusters, max_iter, min_covar = 0.001, tol = 0.001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.min_covar = min_covar
        self.tol = tol
        self.logs = {'mu': [], 'labels': [], 'w': [], 'sigma':[], 'log_likelihood': []}
        self.convergence = 1


    def estep(self, X, w, mu, sigma):
        
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