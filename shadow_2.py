'''
This is an two layered neural network
when initiating set the number of layers accordingly!!!!!
'''

import numpy as np
from scipy.stats import multivariate_normal

class NeuralNet():
    def __init__(self,layers=[15,15,1],lr=0.0003,iterations=1000):
        self.params={}
        self.lr=lr
        self.iterations=iterations
        self.loss=[]
        self.sample_size=None
        self.layers=layers

    def initial_weights(self):
        np.random.seed(1)
        self.params['w1']=np.random.randn(self.layers[0],self.layers[1])
        self.params['b1']=np.random.rand(self.layers[1],)
        self.params['w2']=np.random.rand(self.layers[1],self.layers[2])
        self.params['b2']=np.random.rand(self.layers[2],)

    def relu(self,z):
        return np.maximum(0,z)
        

    def tanh(self,z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    
    
    def back_propagation(self,loss):
        #gradients calculated by loss using backpropagation
        dw1=np.gradient(self.params['w1'],loss,axis=0)
        dw2=np.gradient(self.params['w2'],loss,axis=0)
        db1=np.gradient(self.params['b1'],loss,axis=0)
        db2=np.gradient(self.params['b2'],loss,axis=0)

        #update the weights and bias
        #stochastic gradient descent
        self.params['W1'] = self.params['W1'] - self.learning_rate * dw1
        self.params['W2'] = self.params['W2'] - self.learning_rate * dw2
        self.params['b1'] = self.params['b1'] - self.learning_rate * db1
        self.params['b2'] = self.params['b2'] - self.learning_rate * db2


    def train(self,loss):
        self.loss_store=[]
        self.initial_weights()
        for i in range(self.iterations):
            self.back_propagation(loss)
            self.loss_store.append(loss)

    def predict(self, X):
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        pred = self.tanh(Z2)
        return np.round(pred) 