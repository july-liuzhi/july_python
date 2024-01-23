# 完整步骤线性回归
# 导入需要的模块
import torch 
from d2l import torch as d2l
import matplotlib.pyplot as plt
import random

# 1、根据带有噪声的线性模型构造一个人造数据集
def synthetic_data(w,b,num_exmple):
    X = torch.rand(num_exmple,len(w))
    X = X * 10                      
    y = torch.matmul(X,w) + b
    y += torch.normal(0,0.1,y.shape)
    return X,y.reshape((-1,1))

# 将w和b的真实值传入构造数据集，构造10000个数据集样本
true_w = torch.tensor([1.000, 9.280, -6.210, 3.300, -0.015, -7.050, 0.100, -0.170, 0.000, 12.310])
true_b = 9.15
n = 10000
features, labels=synthetic_data(true_w,true_b,n)

"""
# detach() 方法用于从计算图中分离张量，以避免梯度计算
# numpy() 方法将张量转换为 NumPy 数组 
d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1) 
plt.show()
"""

#2、生成大小为batch_size的小批量

def data_iter(batch_size, features, labels):
    num_examples = len(features)   # 10000
    indices = list(range(num_examples)) # 0 - 9999
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
       batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
    yield features[batch_indices], labels[batch_indices]

# 从数据集中随机拿出100个数据
# batch_size = 100

"""
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
"""
# 3、定义初始化参数

w = torch.normal(0,0.01,true_w.shape, requires_grad = True)
b = torch.zeros(1,requires_grad = True)

# 4、定义模型
def linrge(X,w,b):
    return torch.matmul(X,w) + b

# 5、定义损失函数
def quare_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape)) **2 / 2

# 6、定义优化器
def sgd(params, lr, batch_size): 
    with torch.no_grad():    #  在该语句块中不需要计算梯度。这样可以避免在更新模型参数时计算梯度
        for param in params:     #  遍历整个参数列表
            param -= lr * param.grad / batch_size   # 根据小批量随机梯度下降算法的公式，使用当前参数的梯度
            param.grad.zero_()   # 在更新完参数后，需要将参数的梯度清零，以便下一轮迭代计算新的梯度。

# 7、训练
# 定义超参数
lr = 0.0075
num_epoch = 10000
net = linrge
loss = quare_loss
batch_size = 10
# 训练
for epoch in range(num_epoch):   # 训练次数
    for X,y in data_iter(batch_size,features,labels):  # 小批量提出数据
        l = loss(net(X,w,b) ,y)  # 损失函数
        l.sum().backward()     # 对损失函数进行求导
        sgd([w,b],lr,batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}') #表示整个数据集的平均损失，:f表示使用浮点数格式输出
# 打印误差
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')






