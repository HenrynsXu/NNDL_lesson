import numpy as np
import activate_func
from activate_func import Softmax
import random,os
from tqdm import tqdm
import matplotlib.pyplot as plt

class MLP():
    def __init__(self,size=[784,80,10],lr=1e-2,L2 = 0.0,mini_batch:int=256,act_func = 'ReLU') -> None:
        self.lr = lr
        self.l2 = L2 # regular coef
        self.size = size
        self.layer_num = len(size)
        self.actv_fuc = getattr(activate_func,act_func)
        self.actv_fuc_drv = getattr(activate_func,f'{act_func}_drv1')
        self.W = [np.array([0])] + [np.random.randn(y, x)/np.sqrt(x) for y, x in zip(size[1:], size[:-1])]
        self.b = [np.array([0])] + [np.random.randn(y, 1) for y in size[1:]]
        self._zs = [np.zeros(bias.shape) for bias in self.b]
        self._activations = [np.zeros(bias.shape) for bias in self.b]
        self.mini_batch = mini_batch
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_acc_list = []
    
    def __forward(self,x):
        self._activations[0] = x
        for i in range(1,self.layer_num):
            self._zs[i] = np.dot(self.W[i],self._activations[i-1])+self.b[i]
            if i == self.layer_num - 1:
                self._activations[i] = Softmax(self._zs[i])
            else:
                self._activations[i] = self.actv_fuc(self._zs[i])
    
    def __backward(self,x,y):
        lmd_W = [np.zeros(weight.shape) for weight in self.W]
        lmd_b = [np.zeros(bias.shape) for bias in self.b]
        error = (self._activations[-1] - y)
        lmd_b[-1] = error
        lmd_W[-1] = np.dot(error,self._activations[-2].transpose())
        for i in range(self.layer_num-2,0,-1):
            error = np.multiply(np.dot(self.W[i + 1].transpose(),error),self.actv_fuc_drv(self._zs[i]))
            lmd_b[i] = error
            lmd_W[i] = np.dot(error,self._activations[i-1].transpose())
        return lmd_b,lmd_W
    
    def fit(self, training_data,epochs,val_data):
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_acc_list = []
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + self.mini_batch] for k in range(0, len(training_data), self.mini_batch)]
            loss = 0
            for mini_batch in tqdm(mini_batches):
                lmd_W = [np.zeros(weight.shape) for weight in self.W]
                lmd_b = [np.zeros(bias.shape) for bias in self.b]
                for x,y in mini_batch:
                    self.__forward(x)
                    loss+=self.cross_entropy(self._activations[-1].T,y)
                    d_lmd_b,d_lmd_W = self.__backward(x,y)
                    lmd_b = [nb + dnb for nb, dnb in zip(lmd_b, d_lmd_b)]
                    lmd_W = [nW + dnW for nW, dnW in zip(lmd_W, d_lmd_W)]
                self.W = [w - (0.99**epoch*self.lr / self.mini_batch) * dw + self.l2*w/self.mini_batch for w, dw in zip(self.W, lmd_W)]
                self.b = [b - (0.99**epoch*self.lr / self.mini_batch) * db for b, db in zip(self.b, lmd_b)]
            print(f'Epoch {epoch + 1}, trainning loss:{loss.item()/(len(mini_batch)*len(mini_batches))}')
            self.train_loss_list.append(loss.item()/(len(mini_batch)*len(mini_batches)))
            train_acc = self.validate_for_train(training_data) / len(training_data)
            print(f"Epoch {epoch + 1}, train accuracy {train_acc}.")
            self.train_acc_list.append(train_acc)
            if val_data:
                val_acc = self.validate(val_data) / len(val_data)
                print(f"Epoch {epoch + 1}, val accuracy {val_acc}.")
                self.val_acc_list.append(val_acc)
            else:
                print(f"Processed epoch {epoch}.")
    def validate(self,data):
        val_results = [(self.predict(x) == y) for x ,y in data]
        return sum(res for res in val_results)
    def predict(self,x):
        self.__forward(x)
        return np.argmax(self._activations[-1])
    
    def validate_for_train(self,data):
        val_results = [(self.predict(x) == np.argmax(y)) for x ,y in data]
        return sum(res for res in val_results)
    
    def load(self,filename = 'model'):
        npy_num = np.load(filename,allow_pickle=True)
        self.W = list(npy_num['W'])
        self.b = list(npy_num['b'])
        self.size = [b.shape[0] for b in self.b]
        self.layer_num = len(self.size)
        self._zs = [np.zeros(bias.shape) for bias in self.b]
        self._activations = [np.zeros(bias.shape) for bias in self.b]
        self.mini_batch = int(npy_num['mini_batch'])
        self.lr = float(npy_num['lr'])
        self.train_loss_list = list(npy_num['Loss'])
        self.train_acc_list = list(npy_num['train_acc'])
        self.val_acc_list = list(npy_num['val_acc'])
    
    def save(self,filename = 'model'):
        if not os.path.exists(os.path.join(os.curdir,'models')): os.mkdir(os.path.join(os.curdir,'models'))
        np.savez(file=os.path.join(os.curdir,'models',filename),W = self.W,b = self.b,mini_batch = self.mini_batch,lr = self.lr,Loss = self.train_loss_list,train_acc = self.train_acc_list,val_acc = self.val_acc_list)

    def cross_entropy(self,y_pre,y_true):
        return -np.dot(y_pre,np.log(y_true+0.00001))

    def plot_curve(self,loss_curve_name = 'ex1_loss',acc_curve_name = 'ex1_accs'):
        if not os.path.exists(os.path.join(os.curdir,'pictures')): os.mkdir(os.path.join(os.curdir,'pictures'))
        epochs = [i for i in range(len(self.val_acc_list))]
        plt.figure()
        plt.plot(epochs,self.train_loss_list)
        plt.xlabel('Epochs')
        plt.ylabel('Train Loss')
        plt.savefig(f'./pictures/{loss_curve_name}')
        plt.figure()
        plt.plot(epochs,self.train_acc_list,'r',label = 'Train Acc')
        plt.plot(epochs,self.val_acc_list,'g',label = 'Val Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'./pictures/{acc_curve_name}')
