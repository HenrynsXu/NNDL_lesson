import numpy as np

def Sigmoid(z):
    return 0.5*(np.tanh(0.5*z)+1)

def Sigmoid_drv1(z):
    return Sigmoid(z)*(1-Sigmoid(z))

def ReLU(z):
    return np.maximum(0,z)

def ReLU_drv1(z):
    return np.array([1 if zz>0 else 0 for zz in z]).reshape((len(z),1))

def Softmax(z):
    c = np.max(z)
    return np.exp(z-c) / np.sum(np.exp(z-c))

