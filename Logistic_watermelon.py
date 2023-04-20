import numpy as np
import math

print("this is a great beginning")
class LogisticModel():
    
    def __init__(self, dataset, feature_dim, output_dim = 1, epoch = 100, learning_rate = 0.1):
        self.w = np.random.normal(loc = 0, scale = 1, size = (feature_dim, output_dim))
        self.b = np.zeros((output_dim,1))
        self.beta = np.concatenate(self.w.transpose, self.b, axis = 1)
        self.lr = learning_rate
        self.dataset = dataset
        print(self.beta)

    def logistic(z):
        return 1/(1 + math.exp(-z))

    def forward(x):
        # work as prediction process, actually not used in training
        x_hat = np.concatenate(x, np.ones((1, 1)), axis = 0)
        z = np.matmul(self.beta,x_hat)
        return logistic(z)

    def loss(dset):#要求传入dset，因为有可能有train 与 validation两种loss
        total_loss = 0
        data_x, data_y = np.split(self.dset,[self.dset.shape[1]-1],axis = 1)
        for i in range(0, dataset.shape[0], 1):
            x = data_x[j]
            y = data_y[j]
            x = x.transpose()
            x_hat = np.concatenate(x, np.ones((1, 1)), axis = 0)
            (-y) * np.matmul(self.beta, x_hat) + math.log(1 + math.exp(np.matmul(self.beta, x_hat)))

    def Gradient_decent(batchsize):
        gradient = 0.0
        data_x, data_y = np.split(self.dataset,[self.dataset.shape[1]-1],axis = 1)
        for j in range(0, batchsize, 1):
            x = data_x[j]
            y = data_y[j]
            x = x.transpose()
            x_hat = np.concatenate(x, np.ones((1, 1)), axis = 0)
            gradient = gradient + (-1) * x_hat * (y - (math.exp(np.matmul(self.beta,x_hat)))/(1 + math.exp(np.matmul(self.beta,x_hat))))

    def train(batchsize):
        for i in range(0, epoch, 1):
            np.random.shuffle(self.dataset)# 检查这里是否改变了self.dataset,还是需要temp = np.random.shuffle(self.dataset)
            self.beta = self.beta - (Gradient_decent(batchsize) * self.lr).transpose()
            currentLoss = loss()
            print(currentLoss)
    
    def validation():
        pass
        

def main():
    """读入csv,并拆分为训练集与验证集 然后调用逻辑回归模型 进行训练"""
    pass

if __name__ == '__main__':
    main()