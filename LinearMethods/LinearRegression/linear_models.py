import numpy as np


class LinearRegression():
    def __init__(self,X,y,use_bias = True):
        self.X = X
        self.y = y
        self._check = False
        self.use_bias = use_bias
        
        if self.use_bias == True:
            temp = np.ones((self.X.shape[0],self.X.shape[1] + 1))
            temp[:,:-1] = self.X
            self.X = temp
  #          print(temp)
        
              
    def fit(self):  
        
        
        xt_x = self.X.T @ self.X
        xt_x_inv = np.linalg.pinv(xt_x)
        
        self.w = xt_x_inv @ self.X.T @ self.y
        self._check = True
    
    
    def weights(self):
        assert (self._check == True),"Model not fitted"
        
        return self.w
              
    def predict(self,data,plot =False):
        
        if self.use_bias == True:
            temp = np.ones((data.shape[0],data.shape[1] + 1))
            temp[:,:-1] = data
            data = temp 
        p = data @ self.w
        return p