import numpy as np

class RandomP():
    def __init__(self,X,k = None):
        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[-1]
        self.b = 0.4
        
        if k == None:
            self.k = self.__estimate_dimension(0.5)
        else:
            self.k = k
        print(self.k)
    
    def __estimate_dimension(self,epsilon):
        return int(
            np.log(self.n)*(4.+2*self.b)/((epsilon**2/2) - (epsilon**3/3))
        )
        return int(
            np.log(self.n)*(4.+2*self.b)/((epsilon**2/2))
        )
    
#         return int(
#             np.log(self.n)/(epsilon**2)
#         )
    
    def __get_independent_random_matrix(self):  
        i = 0
        while(1):
            self.rm = self.__get_random_matrix()          
            u,d,v = np.linalg.svd(self.rm)
            if np.amin(d) <  0.001:
                continue

            #print(i)
            i+=1
            return self.rm
        
    def __get_random_matrix(self):
        rm = np.zeros((self.d,self.k))
        
        for j in range(self.k):
            z = np.random.randn(1,self.d)
            z = z/np.linalg.norm(z)
            rm[:,j] = z
        #print("RM:",rm)
        return rm
        
        
    def project(self):
        self.R = self.__get_independent_random_matrix()
        self.scores = (self.R.T @ self.X.T)
        
        return self.scores.T