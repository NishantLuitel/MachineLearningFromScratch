import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np

class MDS():
    def __init__(self,X,num_dim,G = None):
        
        assert num_dim<=X.shape[-1], "Projection dimension should be less than input dimension"
        
        self.X = X
        self.d = num_dim
        self.G = G   #Gram kernel Matrix
            
    
    def __centre_data(self):
        """
        Calculate the centralized score for data
        """
        
        self.mean = np.mean(self.X,axis = 0)
        self.cX = self.X - self.mean
        return self.cX
    
    def __eu_dissimilarity(self):
        """
        Using Euclidean distance 
        """
        self.X = self.__centre_data()
        
        X1 = np.zeros_like(self.X[0,:])
        num_points = self.X.shape[0]
        self.dis = np.zeros((num_points,num_points))
        diffed = self.X - X1
        
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                diff = self.X[i,:]-self.X[j,:]
                self.dis[i,j] = np.dot(diffed[i,:],diffed[i,:]) + np.dot(diffed[j,:],diffed[j,:]) - np.dot(diff,diff)

        return self.dis
    
    
    def project(self,metric='euclidean'):
        """
        Compute scores
        Output:

        scores: 
        
        """
        
        print(type(self.G))
        if(type(self.G) == np.ndarray):
            w,v = np.linalg.eig(self.G)
            #print(w)
            w = np.real_if_close(w, tol=1)
            v = np.real_if_close(v, tol=1)
            list_v = [(v[:,i],w[i]) for i in range(len(w))]
            list_v.sort(key = lambda x:x[1],reverse = True)
            v = np.array([list(j[0]) for j in list_v]).T
            w_ordered = np.sort(w)[::-1]
            #print(w_ordered)
            self.emb = v[:,:self.d] @ np.sqrt(np.diag(w_ordered[:self.d]))
            return self.emb
        
        
        else:
            if metric == 'euclidean':
                self.gram = self.__eu_dissimilarity()
            #    print(self.gram)
            w,v = np.linalg.eig(self.gram)
            #print(w)
            w = np.real_if_close(w, tol=1)
            v = np.real_if_close(v, tol=1)


            list_v = [(v[:,i],w[i]) for i in range(len(w))]
            list_v.sort(key = lambda x:x[1],reverse = True)
            v = np.array([list(j[0]) for j in list_v]).T
            w_ordered = np.sort(w)[::-1]
            #print(w_ordered)

            self.emb = v[:,:self.d] @ np.sqrt(np.diag(w_ordered[:self.d]))

            return self.emb

        
    
    def plot_scores_2d(self, colors, grid = True, dim_1 = 1, dim_2 = 2):
        if self.d < 2:
            warnings.warn("No hay suficientes componentes prinicpales")
            return
        
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=(15,10))
        plt.axhline(c = 'black', alpha = 0.2)
        plt.axvline(c = 'black', alpha = 0.2)
        plt.scatter(self.emb.T[dim_1 - 1,:], self.emb.T[dim_2 - 1,:], c = colors)
        plt.grid(grid)
        plt.title('Multi Dimensional Scaling')
        plt.xlabel('${}^a$ dimension'.format(dim_1))
        plt.ylabel('${}^a$ dimension'.format(dim_2))
        plt.show()