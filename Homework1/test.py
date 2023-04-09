from nn import MLP
from main import load_MNIST
train_data,val_data,test_data = load_MNIST()
nn = MLP()
nn.load('models/model6_l1e4.npz') # model_name
print(nn.validate(test_data)/ len(test_data))
