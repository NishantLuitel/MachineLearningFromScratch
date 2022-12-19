import numpy as np

class PCA():
    def __init__(self,X,num_dim):
        
        assert num_dim<=X.shape[-1], "Projection dimension should be less than input dimension"
        
        self.X = X
        self.d = num_dim
    
    def __centre_data(self):
        """
        Calculate the centralized score for data
        """
        
        self.mean = np.mean(self.X,axis = 0)
        self.cX = self.X - self.mean
        return self.cX
    

    def project_debug(self):
        """
        Compute scores
        Output:
        
        scores: 
        """
        
        self.cX = self.__centre_data()
        self.cov = self.cX.T @ self.cX
        w,v = np.linalg.eig(self.cov)
        
        w = np.real_if_close(w, tol=1)
        v = np.real_if_close(v, tol=1)
        print(w)
        print(v)
        list_v = [(v[:,i],w[i]) for i in range(len(w))]
        list_v.sort(key = lambda x:x[1],reverse = True)
        v = np.array([list(j[0]) for j in list_v]).T
        print(v)
        
        self.v_reduced = v[:,:self.d]
        print(self.cX.shape,self.v_reduced.shape)
        self.scores = self.cX @self.v_reduced
        return self.scores
    
    def project(self):
        """
        Compute scores
        Output:

        scores: 
        """

        self.cX = self.__centre_data()
        self.cov = self.cX.T @ self.cX
        w,v = np.linalg.eig(self.cov)

        w = np.real_if_close(w, tol=1)
        v = np.real_if_close(v, tol=1)
        list_v = [(v[:,i],w[i]) for i in range(len(w))]
        list_v.sort(key = lambda x:x[1],reverse = True)
        v = np.array([list(j[0]) for j in list_v]).T

        self.v_reduced = v[:,:self.d]
        self.scores = self.cX @self.v_reduced
        return self.scores