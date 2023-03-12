'''
This is an two layered neural network
when initiating set the number of layers accordingly!!!!!
'''

import numpy as np

class NeuralNet_actor():
    def __init__(self,layers,lr,iterations=1000):
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
    

    
    def predict(self, X):
        self.X=X
        self.Z1 = X.dot(self.params['w1']) + self.params['b1']
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1.dot(self.params['w2']) + self.params['b2']
        self.pred = self.tanh(self.Z2)
        return np.round(self.pred) 
    

    def back_propagation(self,loss):
        #gradients calculated by loss using backpropagation
        # Compute the gradient of the loss with respect to the output of the second layer
        delta2 = loss

        # Compute the gradient of the loss with respect to the weights and biases of the second layer
        dw2 = np.dot(self.Z1[:,:,np.newaxis], delta2[np.newaxis,:,:])
        db2 = np.sum(delta2, axis=0)

        # Compute the gradient of the loss with respect to the output of the first layer
        delta1 = np.dot(delta2[:,:,np.newaxis], self.params['w2'][:,:,np.newaxis]) * (1 - np.power(self.Z1, 2))

        # Compute the gradient of the loss with respect to the weights and biases of the first layer
        dw1 = np.dot(self.X[:,:np.newaxis], delta1[np.newaxis,:,:])
        db1 = np.sum(delta1, axis=0)

        # Update the weights and biases using the gradients
        self.params['w1'] -= self.lr * dw1
        self.params['w2'] -= self.lr * dw2
        self.params['b1'] -= self.lr * db1
        self.params['b2'] -= self.lr * db2

    def train(self,loss):
        self.loss_store=[]
        self.initial_weights()
        for i in range(self.iterations):
            self.back_propagation(loss)
            self.loss_store.append(loss)
