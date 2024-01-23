# 导入所需要的库
import numpy as np  
import matplotlib.pyplot as plt

# 构建凸函数的特征
Q = np.array([[10,-9],[-9,10],],dtype="float32")  
b = np.array([4,-15],dtype = "float32").reshape(-1,1)   
func = lambda x: 0.5 * np.dot(x.T, np.dot(Q, x)) + np.dot(b.T, x)  
gradient = lambda x: np.dot(Q, x) + b  

# 声明最优点
x_0 = np.array([5,6]).reshape(-1,1)

# 定义最速下降法  
def gradient_descent(start_point, func, gradient, epsilon, max_iter=1000):      
    global Q, b       
    x_k_1, iter_num , loss_error = start_point, 0 ,[]       # 初始点赋值
    xs = [x_k_1]  
  
    while iter_num < max_iter:  
        g_k = gradient(x_k_1).reshape([-1, 1])              # 计算梯度
        if np.sqrt(np.sum(g_k ** 2)) < epsilon:             # 检查梯度大小
            break  
        alpha_k = np.dot(g_k.T, g_k) / (np.dot(g_k.T, np.dot(Q, g_k)))   # 更新步长
        x_k_2 = x_k_1 - alpha_k * g_k                       # 更新点的位置
        iter_num += 1                                       # 更新迭代次数
        xs.append(x_k_2)                                    # 保存点的位置
        if np.fabs(func(x_k_2) - func(x_k_1)) < epsilon:    # 查看函数值的变化
            break
        loss_error.append(float(np.fabs(func(x_k_2) - func(x_0))))   # 与真实值大小比较
        x_k_1 = x_k_2                                       # 更新当前点位置为下一步的起始点 
    return xs, iter_num , loss_error 

# 输入参数 
x0 = np.array([5,6], dtype="float32").reshape([-1, 1])  
xs, iter_num ,loss_error = gradient_descent(start_point=x0, func=func,gradient=gradient,epsilon=1e-20)  

# 打印所需要的数据
a = xs[-1]
print(xs)
print('\n')
print(iter_num)
print('\n')  
print(a) 
print('\n') 
print(func(a))

# 数据可视化展示
plt.figure(figsize = [12,6])
plt.plot(loss_error)
plt.xlabel("# iteration", fontsize=12)
plt.ylabel("Loss: $|f(x_k) - f(x^*)|$", fontsize=12)
plt.yscale("log")
plt.show()