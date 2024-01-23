import numpy as np
import matplotlib.pyplot as plt

# 定义jacobi函数
def jacobi(A,b,max_iter):
    n = len(b)
    x = np.zeros(n)
    x_true = np.linalg.solve(A,b)
    error1 = np.zeros(max_iter)
    iter_num = 0
    while iter_num < max_iter:
        x_new = np.zeros(n)
        for i in range(n):
            s = 0
            for j in range(n):             
                if j != i:
                    s += A[i,j] * x[j]                                                                 
            x_new[i] = (b[i] - s) / A[i,i]
        x = x_new
        error1[iter_num] = np.linalg.norm(x_true - x)
        iter_num += 1

    return error1

# 定义G-S迭代法
def Gauss_Sedidal(A,b,max_iter):
    n = len(b)
    x = np.zeros(n)
    x_true = np.linalg.solve(A,b)
    error2 = np.zeros(max_iter)
    iter_num = 0
    while iter_num < max_iter:
        x_new = np.zeros(n)
        for i in range(n):
            s = np.sum(A[i,:i]*x_new[:i]) + np.sum(A[i,i+1:]*x[i+1:]) 
            x_new[i] = (b[i] - s) / A[i,i]
        x = x_new
        error2[iter_num] = np.linalg.norm(x_true - x)
        iter_num += 1
    return error2


# 定义SOR函数
def SOR_methed(A,b,w,max_iter):
    n = len(A)
    x = np.zeros(n)
    x_true = np.linalg.solve(A,b)
    error3 = np.zeros(max_iter)
    D = np.zeros_like(A)
    D[np.arange(n),np.arange(n)] = A[np.arange(n),np.arange(n)]
    LU = D - A
    L = np.tril(LU)
    U = np.triu(LU)
    D_wL = D - w * L
    D_wl_inv = np.linalg.inv(D_wL)
    iter_num = 0
    while iter_num < max_iter:
        x = D_wl_inv@((1-w)*D + w*U)@x + w * D_wl_inv@b
        error3[iter_num] = np.linalg.norm(x_true - x)
        iter_num += 1    
    return error3

A = np.array([[10, -1, -2],
              [-1, 10, -2],
              [-1, -1, 5]])
b = np.array([7.2,8.3,4.2])
max_iter = 15
w = 1.1
error1 = jacobi(A,b,max_iter)
error2 = Gauss_Sedidal(A,b,max_iter)
error3 = SOR_methed(A,b,w,max_iter)
x= np.arange(0,max_iter)
plt.plot(x,error1,marker='o',label ='Jacobi')
plt.plot(x,error2,marker = 'v',label = 'G-S')
plt.plot(x,error3,marker = 's',label = 'Sor')
plt.xlabel('Iterations')
plt.ylabel('Iteration Error')
plt.title('Jacobi,G-S And Sor Error Comparison')
plt.grid(True)
plt.legend()
plt.show()
