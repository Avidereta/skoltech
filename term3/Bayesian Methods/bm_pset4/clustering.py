"""
Bayesian Methods

Implementation of algorithms for clusterization, where the samples are from a mixture of Gaussian
distributions with unknown parameters:
    1.EM-algorithm
    2.Soft K-Means, that is the EM-algorithm with fixing sigma to be identity matrix
    3.K-Means
    4.Random
    
For each algorithm * 2-4 two classes are implemented:
    1. *.py with whole implementation. 
        Look abstract_clusterer.py for description of params
    2. wrapper of * for several runs - class Clustering
       returns the best run according to log likelihoods

Note, that the tests and visualization for the algorithms are provided in the 
    separate file main.ipynb.
Also there are generated two datasets on which
    (a) All the methods work equally good;
    (b) Full EM-algorithm works much better than the others.
""" 

# coding=utf-8

import numpy as np
from sklearn.linear_model.base import BaseEstimator
from numpy.testing import assert_approx_equal
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from scipy.misc import logsumexp

from utils import compute_labels, log_likelihood, log_likelihood_from_labels
from myEM import EM
from mySoftKMeans import SoftKMeans
from myKMeans import KMeans


class UnknownAlgorithmException(Exception):
    """This exception means that the algorithm name provided doesn't refer to any known algorithm"""
                    
class Clustering:
    """Algorithm wrapper: it is run several times, best logs and other logs are saved"""
    
    def __init__(self, n_clusters, algorithm = "EM", max_iter=100, n_init=10, min_covar=0.001, tol=0.001, logging = True):
        """
        @param n_init: the number of runs of the algorithm with different random inits
        @param n_clusters: the number of clusters K
        @param max_iter: the maximal number of iterations
        @param min_covar: regularization for covariance matrix
        @param tol: if log-likelihood changed less than tol on the current iteration, 
            stop the optimization process
        @return: the results of the best run according to log-likelihood
        """
        
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.min_covar = min_covar
        self.tol = tol
        self.inits = []
        self.logs = {'mu': [], 'labels': [], 'w': [], 'sigma':[], 'log_likelihood': []}
       
        for i in range(self.n_init):
            if algorithm == "EM":
                self.inits.append(EM(self.n_clusters, self.max_iter, self.min_covar, self.tol))
            elif algorithm == "SoftKMeans":
                self.inits.append(SoftKMeans(self.n_clusters, self.max_iter, self.min_covar, self.tol))
            elif algorithm == "KMeans":
                self.inits.append(KMeans(self.n_clusters, self.max_iter, self.min_covar, self.tol))
            else: 
                raise UnknownAlgorithmException()
            
     

    def fit(self, X):
        """ Fit algorithm
        
        @return self.labels_: labeling of the data points (cluster index for each data point).
        @return self.w_: vector of prior probabilities w
        @return self.cluster centers_: K x d matrix of cluster centers
        @return self.covars_ : K x d x d tensor of cluster covariance matrices
        """
        
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
            
class Random(BaseEstimator):
    """
    Random Algorithm for clusterization
    """

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
