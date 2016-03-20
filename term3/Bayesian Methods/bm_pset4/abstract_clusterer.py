
class AbstractClusterer:
    def __init__(self, n_clusters, max_iter, min_covar = 0.001, tol = 0.001):
         """
        @param n_clusters: the number of clusters K
        @param max_iter: the maximal number of iterations
        @param min_covar: regularization for covariance matrix
        @param tol: if log-likelihood changed less than tol on the current iteration,
            stop the optimization process
        @type logs: dictionary
        @param logs:
            log likelihood: list of log-likelihoods on each iteration
            labels: list of labeling of the data points on each iteration
            w: list of the vectors of prior probabilities w on each iteration
            mu: list, for each iteration contains matrix of cluster centers
            sigma: list, for each iteration contains a tensor of cluster covariance matrices
        @param convergence: 1 if algorithm converges, -1 else
        """
        raise NotImplementedError()
        
    def estep(self, X, w, mu, sigma):
        """Compute Gamma_ik

        @param X: n_objects x n_features matrix
        @param w: a vector of size n_clusters of the prior probabilities of the clusters
        @param mu: n_clusters x n_features matrix of the centers of the clusters
        @param sigma: n_clusters x n_features x n_features tensor with the covariance
                matrix of each cluster.
        @return: n_objects x n_clusters matrix - logarithmic gammas

        """
        raise NotImplementedError()

    def mstep(self, X, log_gamma):
        """Compute Mu and Sigma of Normal distributions based on Gammas from E-step

        @param X: n_objects x n_features input matrix
        @param log_gamma: n_objects x n_clusters matrix of logarithmic gammas
        @return w: a vector of size n_clusters of the prior probabilities of the clusters
        @return mu: n_clusters x n_features matrix of the centers of the clusters
        @return sigma: n_clusters x n_features x n_features tensor with the covariance
                matrix of each cluster
        """
        raise NotImplementedError()

    def fit(self,X):
        """ Fit algorithm

        @return self.labels_: labeling of the data points (cluster index for each data point).
        @return self.w_: vector of prior probabilities w.
        @return self.cluster centers_: K x d matrix of cluster centers.
        @return self.covars_ : K x d x d tensor of cluster covariance matrices
        """
        raise NotImplementedError()
