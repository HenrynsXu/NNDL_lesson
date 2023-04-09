import gzip,os,pickle,wget
import numpy as np
from nn import MLP
import argparse

def onehot(x):
    re = np.zeros((10,1))
    re[x] = 1.0
    return re

def load_MNIST():
    if not os.path.exists(os.path.join(os.curdir,'data')):
        os.mkdir(os.path.join(os.curdir,'data'))
        wget.download('https://resources.oreilly.com/live-training/inside-unsupervised-learning/-/raw/master/data/mnist_data/mnist.pkl.gz',out='data')
    data_file = gzip.open(os.path.join(os.curdir,'data','mnist.pkl.gz'),'rb')
    train_data, val_data, test_data = pickle.load(data_file,encoding='latin1')
    data_file.close()

    train_input = [np.reshape(x,(784,1)) for x in train_data[0]]
    train_label = [onehot(y) for y in train_data[1]]
    train_dd = list(zip(train_input,train_label))
    val_input = [np.reshape(x, (784, 1)) for x in val_data[0]]
    val_dd = list(zip(val_input,val_data[1]))
    test_input = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_dd = list(zip(test_input,test_data[1]))
    return train_dd,val_dd,test_dd

if __name__ == '__main__':
    np.random.seed(79888)
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',default=35,type=int)
    parser.add_argument('--hidden_size',default=160,type=int)
    parser.add_argument('--lr',default=0.5,type=float)
    parser.add_argument('--batch_size',default=64,type=int)
    parser.add_argument('--L2',default=0.0001,type=float)
    args = parser.parse_args()
    arg_dic = args.__dict__
    epochs,hidden,lr,L2,mini_batch = arg_dic['epochs'],arg_dic['hidden_size'],arg_dic['lr'],arg_dic['L2'],arg_dic['batch_size']
    size = [784,10]
    size.insert(1,hidden)
    train_data,val_data,test_data = load_MNIST()
    nn = MLP(size,lr,L2,mini_batch)
    nn.fit(train_data,epochs,val_data)
    accuracy = nn.validate(test_data) / len(test_data)
    print(f"Test Accuracy: {accuracy}.")

    nn.save(filename='model6_l1e4')
    nn.plot_curve('ex6_l1e4_loss','ex6_l1e4_accs')
