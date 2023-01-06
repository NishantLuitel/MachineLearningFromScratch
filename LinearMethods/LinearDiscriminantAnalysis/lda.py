import numpy as np
import sys
sys.path.append('../LinearRegression')

from linear_models import LinearRegression


class LDA():
    def __init__(self,X,y,alpha = 0.5,learning_rate = 1,no_iter = 1000,use_bias = True):
        
        self.lr = LinearRegression(X,y,use_bias = True)
        
                 
    def fit(self):       
        self.lr.fit()
    
    
    def weights(self):
        return self.lr.weights()
    
              
    def predict(self,data,probs = False,plot =False):
        self.p = self.lr.predict(data,plot)
        
        if probs == False:
            return (self.p>0.5).astype(float)
        else:
            return (self.p>0.5).astype(float) , self.p