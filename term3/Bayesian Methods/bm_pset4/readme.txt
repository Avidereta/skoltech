It is a small lib with implementations of algorithms for clusterization, 
where the samples are from a mixture of Gaussian distributions with unknown parameters:
    1. myEm.py: EM-algorithm 
    2. mySoftKMeans.py: Soft K-Means, that is the EM-algorithm with fixing sigma to be identity matrix
    3. myKMeans.py: K-Means
    4. Random in clustering.py

clustering.py class Clustering: wrapper for algorithms 1-3
abstract_clusterer.py: description of params in 1-3
    
Note, that the tests and visualization for the algorithms are provided in the 
    separate file main.ipynb. Also there are generated two datasets on which:
    (a) All the methods work equally good;
    (b) Full EM-algorithm works much better than the others.

Detailed description of assignment can be found in assignment.pdf
