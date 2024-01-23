# 矩阵的LU分解，具体数学理论见书P44
import numpy as np

def LU(A):
    n = len(A)
    L = np.eye(n)
    U = np.zeros((n,n))
    for k in range(n):
        for i in range(k+1,n):
            L[i,k] = A[i,k] / A[k,k]
        for j in range(k,n):
            U[k,j] = A[k,j]
        for i in range(k+1,n):
            for j in range(k+1,n):
                A[i,j] = A[i,j] -L[i,k]*U[k,j]
    return L,U

# 测试
if __name__ =="__main__":
    A = np.array([[2, 4, -2, 2],
              [4, 9, -3, 8],
              [-2, -3, 7, -6],
              [2, 8, -6, 9]])
    L,U = LU(A)
    b = np.array([4,20,-6,26]).reshape(-1,1)

    # 求解Ly=b(向前回带求解)
    n1 = len(L)
    y = np.zeros(n1)      # 以行储存的方式
    y[0] = b[0] / L[0,0]
    for i in range(1,n1):
        for j in range(0,i):
            b[i] = b[i] - L[i,j] * y[j]
        y[i] = b[i] / L[i,i]    

    # 求解Ux=y(向后回带求解)   
    y = y.reshape(-1,1)          
    x = np.zeros(n1).reshape(-1,1)  # 以列储存的方式
    for k in range(n1-1,-1,-1):
        x[k] = y[k] / U[k,k]
        for i in range(k-1,-1,-1):
            y[i] = y[i] - x[k] * U[i,k]
    print(x)