import numpy as np

class BinaryLogisticRegression():
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
#            print(temp)

    def _logistic_function(self,x):
        return(1/(1+np.exp(-x)))
        
        
              
    def fit(self):  
        
        self.w = np.zeros(self.X.shape[1])
        
        
        #Use Gradient Descent
        for i in range(self.no_iter):
            self._update_weights()
        self._check = True
    
    
    def weights(self):
        assert (self._check == True),"Model not fitted"
        
        return self.w
    
    def _update_weights(self):
                
        #Calculate Gradient
        output = self.X @ self.w
        
        self.dw = self.X.T @ (self._logistic_function(output)-self.y)/ self.X.shape[0]
        self.w = self.w - self.l_r*self.dw
        
        return self.w
              
    def predict(self,data,probs = False,plot =False):
        
        if self.use_bias == True:
            temp = np.ones((data.shape[0],data.shape[1] + 1))
            temp[:,:-1] = data
            data = temp 
        self.p = self._logistic_function(data @ self.w) 
        if probs == False:
            return (self.p>0.5).astype(float)
        else:
            return (self.p>0.5).astype(float) , self.p
        
        
        
        
        
        
        
        
        
        

class MultiLogisticRegression():
    '''
        Expects y to either be one hot encoded label or 1 dimensional label with label index
        
    '''
    def __init__(self,X,y,alpha = 0.5,learning_rate = 1,no_iter = 1000,use_bias = True,centralize = True):
        
        
        if centralize ==True:
            self.X = self._centralize(X)
        else:
            self.X = X
        self.y = y
        self._check = False
        self.use_bias = use_bias
        self.N = self.X.shape[0]
        self.no_iter = no_iter
        
        if(len(self.y.shape) == 1):
            self.max_index = max(self.y)
            self.one_hots = np.zeros((self.N,self.max_index+1))
            for i,_ in enumerate(y):
                self.one_hots[i,_] = 1
#             print(self.one_hots)
        
        
        assert alpha>0, "Choose the regularizer constant(alpha) > 0"
        self.alpha = alpha
        assert learning_rate>0, "Choose the regularizer constant(alpha) > 0"
        self.l_r = learning_rate
       
        
        if self.use_bias == True:
            temp = np.ones((self.X.shape[0],self.X.shape[1] + 1))
            temp[:,:-1] = self.X
            self.X = temp
#            print(temp)

    def _softmax(self,x):
        exponential = np.exp(x)
        sumer = exponential.sum(axis = 1,keepdims = True)
        return exponential/sumer
        #return (exponential.T/sumer).T
    
    def _centralize(self,x):
        x = x - np.mean(x, axis=1,keepdims =True)
        x = x/np.std(x, axis=1,keepdims = True)
        return x
              
    def fit(self):  
        
        self.w = np.zeros((self.X.shape[1],self.max_index+1))
        
        
        #Use Gradient Descent
        for i in range(self.no_iter):
            self._update_weights()
#             print(i, ':',self.w)
#             if i==100:
#                 break
        self._check = True
    
    
    def weights(self):
        assert (self._check == True),"Model not fitted"
        
        return self.w
    
    def _update_weights(self):
                
        #Calculate Gradient
        output = self.X @ self.w 
        #print(output)
        
        self.dw = self.X.T @ (self._softmax(output)-self.one_hots)/ self.X.shape[0]
#         print(self.dw)
        self.w = self.w - self.l_r*self.dw
        
        return self.w
              
    def predict(self,data,probs = False,plot =False):
        
        if self.use_bias == True:
            temp = np.ones((data.shape[0],data.shape[1] + 1))
            temp[:,:-1] = data
            data = temp 
        #print(self.w)
        self.p = self._softmax(data @ self.w)
        return self.p