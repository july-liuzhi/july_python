# SSOR迭代求解线性方程组
import numpy as np

# 定义SSOR函数
def SSor(A,b,x0,max_iter,tol,w):
    n = len(b)
    x = x0
    iter_num = 0
    
    while iter_num < max_iter:
        x_new1 = np.zeros(n)                     # 每次迭代的时候都需要提前清零
        x_new2 = np.zeros(n)
        for i in range(n):
            s1 = np.sum(A[i,:i]*x_new1[:i]) + np.sum(A[i,i+1:]*x[i+1:]) 
            x_new1[i] = (1-w) * x[i] + w * (b[i] - s1) / A[i,i]
        for k in range(n-1,-1,-1):
            s2 = np.sum(A[k,:k] * x_new1[:k]) + np.sum(A[k,k+1:]*x_new2[k+1:])
            x_new2[k] = (1-w) * x_new1[k] + w * (b[k] - s2) / A[k,k] 
        if np.linalg.norm(x_new2 - x) < tol:
            break
        x = x_new2
        iter_num += 1
        print(f"iter_num:{iter_num}\n{x}")
    return x_new2,iter_num

# 测试
A = np.array([[10, -1, -2],
              [-1, 10, -2],
              [-1, -1, 5]])
b = np.array([7.2,8.3,4.2])
x0 = np.array([0,0,0])
max_iter = 1000
tol = 1e-6
w = 1.8
x,num = SSor(A,b,x0,max_iter,tol,w)
print(x)
print("\n")
print(num)