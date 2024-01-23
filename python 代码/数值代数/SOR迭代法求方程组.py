# SOR迭代求解线性方程组
import numpy as np

# 定义SOR函数
def Sor(A,b,x0,max_iter,tol,w):
    n = len(b)
    x = x0
    iter_num = 0
    while iter_num < max_iter:
        x_new = np.zeros(n)
        for i in range(n):
            s = np.sum(A[i,:i]*x_new[:i]) + np.sum(A[i,i+1:]*x[i+1:]) 
            x_new[i] = (1-w) * x[i] + w * (b[i] - s) / A[i,i]
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
        iter_num += 1
    return x_new,iter_num

# 测试
if __name__ == "__main__":
    A = np.array([[10, -1, -2],
              [-1, 10, -2],
              [-1, -1, 5]])
    b = np.array([7.2,8.3,4.2])
    x0 = np.array([0,0,0])
    max_iter = 1000
    tol = 1e-6
    w = 1.05
    x,num = Sor(A,b,x0,max_iter,tol,w)
    print(x)
    print("\n")
    print(num)

