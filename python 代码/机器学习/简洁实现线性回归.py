# 简介实现线性回归代码
# 生成数据集
import numpy as np
import torch
from torch.utils import data   #  用于数据加载和预处理的模块
from d2l import torch as d2l   #  是一个深度学习教材和代码库

true_w = torch.tensor([1.000, 9.280, -6.210, 3.300, -0.015, -7.050, 0.100, -0.170, 0.000, 12.310])
true_b = 9.15
n = 1000
features,labels = d2l.synthetic_data(true_w,true_b,n)

# 读取数据集
def load_array(data_arrays,batch_size,is_train = True):
     #   构造一个PyTorch数据迭代器
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 100
data_iter = load_array((features, labels), batch_size)

# 使用next从迭代器中获取第一项
next(iter(data_iter))

# 定义模型
# nn是神经网络的缩写
from torch import nn
net = nn.Sequential(nn.Linear(10, 1)) # 全连接线性层，它接受一个大小为10的输入，并输出一个大小为1的输出

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.0075)

# 训练
num_epochs = 1000
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差:', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差:', true_b - b)