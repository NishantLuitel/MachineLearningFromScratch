import numpy as np

class RandomP():
    def __init__(self,X,k = None,epsilon = 0.5):
        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[-1]
        self.b = 0.4
        self.e = epsilon
        
        assert self.e<1. and self.e>0, "Range for epsiolon is (0,1)"
        
        if k == None:
            self.k = self.__estimate_dimension(self.e)
        else:
            self.k = k
        print(self.k)
    
    def __estimate_dimension(self,epsilon):
        return int(
            np.log(self.n)*(4.)/((epsilon**2/2) - (epsilon**3/3)) +1
        )
#         return int(
#             np.log(self.n)*(4.+2*self.b)/((epsilon**2/2))
#         )
    
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
        
    def __check_projection(self,scores):
        for i in range(len(scores)):
            for j in range(len(scores)):
                if i!=j:
                    print(len(scores[i,:]),len(self.X[i,:]))
                    s = (self.d/self.k)*np.linalg.norm(scores[i,:] - scores[j,:])**2
                    s_u = (1+self.e)*(np.linalg.norm(self.X[i,:] - self.X[j,:])**2)
                    s_l = (1-self.e)*(np.linalg.norm(self.X[i,:] - self.X[j,:])**2)
                    print(i,j,s_l,s,s_u)
                    if(s>s_u or s<s_l):
                        return False
        return True
    
    def project(self):
        i = 0
        while(1):
            if(i%10 == 0):
                print(i)
            self.R = self.__get_independent_random_matrix()
            print(self.R)
            self.R, r = np.linalg.qr(self.R)
            self.scores = (self.R.T @ self.X.T)
            #print(self.scores.T.shape)
            print(np.linalg.norm(self.R.T,axis = 1))

            check = self.__check_projection(self.scores.T)
            if check==False:
                #print("Moved to check")
                i+=1
                continue
            break
        
        return self.scores.T
    