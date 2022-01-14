import numpy as np
cimport numpy as np
from sklearn.neighbors import KDTree
from scipy.sparse import csr_matrix
from datetime import datetime
from collections import defaultdict, Counter


cdef extern from "quickmeanshiftpp.h":
    void generate_offsets_cy(int d, 
                             int base,
                             int * offsets)

    void quick_cluster(int n,
                  int d,
                  int base,
                  int iterations,
                  float bandwidth,
                  int * offsets,
                  float * X_shifted,
                  int * membership,
                  int * k_num)

cdef generate_offsets_np(d, 
                         base,
                         np.ndarray[np.int32_t, ndim=2, mode="c"] offsets):
    generate_offsets_cy(d, 
                        base,
                        <int *> np.PyArray_DATA(offsets))

cdef shift_np(n,
              d,
              base,
              iterations,
              bandwidth,
              np.ndarray[np.int32_t, ndim=2, mode="c"] offsets,
              np.ndarray[float, ndim=2, mode="c"] X_shifted,
              np.ndarray[np.int32_t, ndim=1, mode="c"] membership,
              np.ndarray[np.int32_t, ndim=1, mode="c"] k_num):
    quick_cluster(n,
             d,
             base,
             iterations,
             bandwidth,
             <int *> np.PyArray_DATA(offsets),
             <float *> np.PyArray_DATA(X_shifted),
             <int *> np.PyArray_DATA(membership),
             <int *> np.PyArray_DATA(k_num))




class QuickMeanShiftPP:
    """
    Parameters
    ----------
    
    bandwidth: Radius for binning points. Points are assigned to the bin 
               corresponding to floor division by bandwidth

    threshold: Stop shifting if the L2 norm between iterations is less than
               threshold

    iterations: Maximum number of iterations to run

    """

    def __init__(self, bandwidth, threshold=0.0001, iterations=None):
        self.bandwidth = bandwidth
        self.threshold = threshold
        self.iterations = iterations



    def fit_predict(self, X, return_modes=False):
        """
        Each shift has two steps: First, points are binned based on floor 
        division by bandwidth. Second, each bin is shifted to the 
        weighted mean of its 3**d neighbors. 
        Lastly, points that are in the same bin are clustered together.

        Parameters
        ----------
        X: Data matrix. Each row should represent a datapoint in 
           Euclidean space

        Returns
        ----------
        (n, ) cluster labels
        """
        
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, d = X.shape
        X_shifted = np.copy(X)

        membership = np.full(n, -1, dtype=np.int32)

        iteration = 0
        base =  3
        offsets = np.full((base**d, d), -1, dtype=np.int32)
        generate_offsets_np(d, base, offsets)
        k = np.full((1,), -1, dtype=np.int32);

        shift_np(n, d, base, self.iterations, self.bandwidth, offsets, X_shifted, membership, k)

        centres = X_shifted[0:np.ndarray.item(k),:];

        return membership, centres
