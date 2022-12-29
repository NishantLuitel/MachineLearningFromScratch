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
    
    
class Ridge():
    def __init__(self,X,y,alpha = 0.5,no_iter = 100,use_bias = True):
        self.X = X
        self.y = y
        self._check = False
        self.use_bias = use_bias
        
        assert alpha>0, "Choose the regularizer constant(alpha) > 0"
        self.alpha = alpha
       
        
        if self.use_bias == True:
            temp = np.ones((self.X.shape[0],self.X.shape[1] + 1))
            temp[:,:-1] = self.X
            self.X = temp
  #          print(temp)
        
              
    def fit(self):  
        
        
        xt_x = self.X.T @ self.X
        xt_x_ridge = xt_x + self.alpha*np.identity(self.X.shape[1])
        
        xt_x_inv = np.linalg.inv(xt_x_ridge)
        
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
    
class Lasso():
    def __init__(self,X,y,alpha = 0.5,learning_rate = 1,no_iter = 1000,use_bias = True):
        self.X = X
        self.y = y
        self._check = False
        self.use_bias = use_bias
        self.N = self.X.shape[0]
        self.no_iter = no_iter
        
        
        assert alpha>0, "Choose the regularizer constant(alpha) > 0"
        self.alpha = alpha
        
        assert learning_rate>0, "Choose the regularizer constant(alpha) > 0"
        self.l_r = learning_rate
       
        
        if self.use_bias == True:
            temp = np.ones((self.X.shape[0],self.X.shape[1] + 1))
            temp[:,:-1] = self.X
            self.X = temp
  #          print(temp)
        
              
    def fit(self):  
        
        self.w = np.zeros(self.X.shape[1])
        
        
        #Use Gradient Descent
        for i in range(self.no_iter):
            self._update_weights()
#         xt_x = self.X.T @ self.X
#         xt_x_ridge = xt_x + self.alpha*np.identity(self.X.shape[1])
        
#         xt_x_inv = np.linalg.inv(xt_x_ridge)
        
#         self.w = xt_x_inv @ self.X.T @ self.y
        self._check = True
    
    
    def weights(self):
        assert (self._check == True),"Model not fitted"
        
        return self.w
    
    def _update_weights(self):
        
        indices = self.w > 0
        inv_indices = indices != True
        adjustment_vector = np.zeros_like(self.w)
        adjustment_vector[indices] = 1
        adjustment_vector[inv_indices] = -1
        
        #Calculate Gradient
        self.dw = (-2/self.N)*self.X.T@(self.y - self.X@self.w) + self.alpha*(adjustment_vector)
        self.w = self.w - self.l_r*self.dw
        
        return self.w
              
    def predict(self,data,plot =False):
        
        if self.use_bias == True:
            temp = np.ones((data.shape[0],data.shape[1] + 1))
            temp[:,:-1] = data
            data = temp 
        p = data @ self.w
        return p