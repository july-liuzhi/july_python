# 导入所需要的库  
import numpy as np    
  
# 构建凸函数的特征  
Q = np.array([[10,-9],[-9,10]], dtype="float32")    
b = np.array([4,-15],dtype = "float32").reshape(-1,1)     
func = lambda x: 0.5 * np.dot(x.T, np.dot(Q, x)) + np.dot(b.T, x)    
gradient = lambda x: np.dot(Q, x) + b    

# 给定参数需要的值
c_1 = 0.5
c_2 = 0.9        # wolfe的参数设置
gamma = 0.5      # 线搜索回退法  

# 定义最速下降法    
def gradient_descent(start_point, func, gradient, epsilon, c_1,c_2 ,gamma, max_iter = 1000):        
    global Q, b         
    x_k_1 = start_point  # 初始点赋值  
    iter_num = 0  # 初始化迭代次数  
    xs = [x_k_1]    
    alpha_k = 1                            # 给定初始步长

    while iter_num < max_iter:    
        g_k = gradient(x_k_1).reshape(-1, 1)
        c_k = gradient(x_k_1 - alpha_k * g_k).reshape(-1,1)  # 计算梯度  
        if np.sqrt(np.sum(g_k ** 2)) < epsilon:  # 检查梯度大小  
            break      
        while func(x_k_1 - alpha_k * g_k) > func(x_k_1) + c_1 * alpha_k * np.dot(g_k.T,-g_k) and np.dot(c_k.T,g_k) < c_2 * np.dot(g_k.T,-g_k):  # wolfe准则  
            alpha_k = gamma * alpha_k  # 更新步长  
        x_k_2 = x_k_1 - alpha_k * g_k  # 更新点位置  
        iter_num += 1  # 更新迭代次数  
        xs.append(x_k_2)  # 保存点位置 
        if np.fabs(func(x_k_2) - func(x_k_1)) < epsilon:      # 查看函数值的变化
            break
        x_k_1 = x_k_2  # 更新当前点位置为下一步的起始点  
    return xs, iter_num  # 返回点序列和迭代次数  
  
# 输入参数   
x0 = np.array([0,0], dtype="float32").reshape([-1, 1])  # 使用numpy数组定义初始点  
xs, iter_num = gradient_descent(start_point=x0, func=func,gradient=gradient,c_1 = c_1,c_2 = c_2,gamma= gamma,epsilon=1e-20)  # 调用gradient descent函数并传入参数  
  
# 打印所需要的数据  
a = xs[-1]  # 取最后一步的点位置作为结束点  
print(xs)  # 打印所有点位置序列  
print('\n')  # 新行分隔符  
print(iter_num)  # 打印迭代次数  
print('\n')  # 新行分隔符  
print(a)  # 打印结束点位置  
print('\n')  # 新行分隔符  
print(func(a))  # 打印函数在结束点位置的值