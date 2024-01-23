import numpy as np    
from scipy.special import roots_legendre    
    
def gauss_legendre_quadrature(f, a, b, n):    
    """    
    f: 被积函数    
    a, b: 积分区间的下限和上限    
    n: 使用的点数，决定了积分的精度    
    """    
    # 计算n阶Legendre多项式的根    
    x_n, w_n = roots_legendre(n)       
    # 对根进行缩放以适应积分区间    
    x = (x_n * (b - a) / 2) + (a + b) / 4    
    # 计算被积函数在每个点上的值
    y = np.polyval(f(x), x)     
    # 计算求积公式的结果    
    result = np.sum(w_n * y)       
    return result    
    
# 测试函数为 f(x) = x^2, 积分区间为 [0, 1]    
f = lambda x: x**2    
a, b = 0, 1    
n =5   # 使用5个点进行求积    
    
result = gauss_legendre_quadrature(f, a, b, n)  
print(f"Gauss-Legendre求积结果: {result}")