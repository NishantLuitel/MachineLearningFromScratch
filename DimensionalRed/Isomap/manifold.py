import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.spatial.distance import cdist
from sklearn.utils.graph_shortest_path import graph_shortest_path
import sys
sys.path.insert(0, '../../DimensionalRed/PCA/')
sys.path.insert(0, '../../DimensionalRed/MDS/')
sys.path.insert(0, '../../DimensionalRed/Kernal_PCA/')

from pca import PCA
from mds import MDS
from kpca import KPCA



class KNNgraph():
    def __init__(self,X,k = 5,max_iter = 100,dist_metric = 'euclidean'):
        
        assert k > 0 and k<len(X), "Value of k should lie between 0 and {0}".format(len(X)-1)
        
        self.X = X
        self.k = k
        self.max_iter = max_iter
        self.metric = dist_metric
     
    
    def find_neighbours(self):
        dist_mat = self._calc_distance()
        neighbours_list = [self._get_neighbours(i,dist_mat) for i in range(len(self.X))]
        
        return np.array(neighbours_list)
    
    def create_graph(self,format = 'adjacency_matrix'):
        """
        Returns Adjacency Matrix 
        """
        
        
        n = len(self.X)
        dist_mat = self._calc_distance()
        adj =  np.zeros((n, n)) 
        nb = self.find_neighbours()
        
        temp = np.full((n,n), False, dtype=bool)
        if format == 'adjacency_matrix':
            for i in range(n):
                for j in range(self.k):
                    nth = nb[i,j]
                    temp[i,nth] = True
                    
            adj[temp] = dist_mat[temp]
            #print(adj)
            return adj
        
               
    
    def _get_neighbours(self,i,dist_mat):
        current_data = self.X[i]
        dist_to_neighbours = dist_mat[i]
        index = dist_to_neighbours.argsort()
        
        allowed_indices = index<=self.k
        allowed_indices[i] = False # Falsify the distance of the point to itself
        
        #Calculate Neighbours
        comp_ind = np.arange(len(self.X))
        neighbours = comp_ind[allowed_indices]
        
        #Start from 1 since the position 0 represents the point i itself
        neighbours = index[1:self.k+1]
        
        return neighbours 
        
        
        
        
    def _calc_distance(self):
        dist = cdist(self.X, self.X, metric=self.metric)
        return dist
    
    
    
    
    



class Isomap():
    
    
    def __init__(self,X,num_dim=2,k = 5,dist_func = 'euclidean',eps = None):
        self.X = X
        self.d = num_dim
        self.k = k #K-neighbours
        self.metric = dist_func
        
        if eps!=None:
            self.distance_matrix = self._make_adjacency_eps(eps = eps)
        else:
            self.distance_matrix = self._make_adjacency_knn()
        print(self.distance_matrix)
        
        
    
    
    def _make_adjacency_eps(self, eps=1):
        
        
        """
        Step one of ISOMAP algorithm, make Adjacency and distance matrix
        Compute the WEIGHTED adjacency matrix A from the given data points.  Points
        are considered neighbors if they are within epsilon of each other.  Distance
        between points will be calculated using SciPy's cdist which will
        compute the D matrix for us. 
        INPUT
        ------
         data - (ndarray) the dataset which should be a numpy array
         dist_func - (str) the distance metric to use. See SciPy cdist for list of
                     options
         eps - (int/float) epsilon value to define the local region. I.e. two points
                           are connected if they are within epsilon of each other.
        OUTPUT
        ------
         short - (ndarray) Distance matrix, the shortest path from every point to
             every other point in the set, INF if not reachable. 
        """
    
        n, d = self.X.shape
        dist = cdist(self.X,self.X, metric=self.metric)
        adj =  np.zeros((n, n)) + np.inf
        bln = dist < eps
        print(bln)
        adj[bln] = dist[bln]
        print(adj)
        short = graph_shortest_path(adj)
        print(short)

        return short
    
    def _make_adjacency_knn(self):
    
        knn_graph = KNNgraph(self.X,self.k,dist_metric = self.metric)
        partial_adj = knn_graph.create_graph()
        
        m = knn_graph.create_graph()
        n = m.T
        partial_adj[m == 0] = n[m == 0]
        adj = partial_adj
        #print('adj,',m,n)
        
        short = graph_shortest_path(adj)
        return short
    
    def shortest_distance(self):
        return self._make_adjacency_knn()


    def project(self,technique='kpca'):


        """
        take an adjacency matrix and distance matrix and compute the ISOMAP
        algorithm

        Take the shortest path distance matrix. This follows from the algorithm in
        class, create a centering matrix and apply it to the distance matrix D. Then
        we can compute the C matrix which will be used for the eigen-decomposion
        Find out more 

        INPUT
        ------
          d - (ndarray) Distance matrix between nodes. Should be square.
          dim - (int) how many dimensions to reduce down too

        OUTPUT
        ------
          z - (ndarray) data projection into new reduced space. Each row maps back
              to one of the origional datapoints
        """

#         n, d = self.distance_matrix.shape

#         #Calculate data centering matrix
#         h = np.eye(n) - (1/n)*np.ones((n, n))
#         d_2 = self.distance_matrix**2
#         c = -1/(2*n) * h.dot(d_2).dot(d_2)

#         if technique == 'pca':
#             pca = PCA(-0.5*self.distance_matrix**2,self.d)
#             proj = pca.project()
            

    #         evals, evecs = linalg.eig(c)
    #         idx = evals.argsort()[::-1]
    #         evals = evals[idx]
    #         evecs = evecs[:, idx]
    #         evals = evals[:self.d] 
    #         evecs = evecs[:, :self.d]
    #         z = evecs.dot(np.diag(evals**(-1/2)))
        if technique == 'mds':
            mds = MDS(self.X,self.d,G=-0.5*self.distance_matrix**2)
            proj = mds.project()
            
        elif technique == 'kpca':
            kpca = KPCA(None,None,self.d,G =-0.5*self.distance_matrix**2 )
            proj = kpca.project()
            proj = proj.T

        return proj