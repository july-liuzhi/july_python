# 导入需要的模块  
import numpy as np  
import torch   
import torch.nn as nn  
import torch.optim as optim  
  
# 根据带有噪声的线性模型构造一个人造数据集  
def synthetic_data(w,b,num_exmple):  
    x = torch.rand(num_exmple,len(w))  
    x = x * 10                        
    y = torch.matmul(x,w) + b  
    y += torch.normal(0,0.1,y.shape)  
    return x,y.reshape(-1,1)  
  
# 将w和b的真实值传入构造数据集，构造10000个数据集样本  
true_w = torch.tensor([1.000, 9.280, -6.210, 3.300, -0.015, -7.050, 0.100, -0.170, 0.000, 12.310])  
true_b = 9.15  
n = 10000  
X, ture_Y = synthetic_data(true_w, true_b, n)  
ture_Y = ture_Y.to(torch.float32)  # 这里需要将y转为tensor  
  
# 回归模型  
def regression_model(x, w, b):  
    return torch.matmul(x, w) + b  
  
# 随机定义w,b的值  
w = torch.randn(10,1,requires_grad= True)  
b = torch.randn(1,requires_grad= True)  
  
criterion = nn.MSELoss()  # 定义损失函数  
  
# 多次训练  
for i in range(10000):  
    y_pre = regression_model(X,w,b)  
    loss = criterion(y_pre, ture_Y)  # 这里需要将ture_Y转为tensor  
    loss.backward()  # 对损失进行求导  
    with torch.no_grad():  
        w -= w.grad * 1e-3  
        b -= b.grad * 1e-3  
        w.grad.zero_()  
        b.grad.zero_()  # 将梯度重置为零，防止重置梯度  
  
y_pre = regression_model(X,w,b)  
loss = criterion(y_pre, ture_Y)  # 这里需要将ture_Y转为tensor  
print(loss)  # 打印最终损失值
print(w)     # 打印参数w的预测值
print(b)     # 打印参数b的预测值
