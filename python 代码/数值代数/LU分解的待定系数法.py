# 矩阵的LU分解，Doolittle方法（待定系数法，A矩阵自身储存）具体数学理论书P43
import numpy as np

def Doolittle(A):
    n = len(A)
    L = np.eye(n)
    U = np.zeros((n,n))
    for k in range(n):
        for j in range(k,n):
            A[k,j] = A[k,j] - np.sum(A[k,:k] * A[:k,j])
        for i in range(k+1,n):
            A[i,k] = (A[i,k] - np.sum(A[i,:k] * A[:k,k])) / A[k,k]  # 这里可以直接返回A，
    for i in range(n):
        for j in range(n):
            if i <= j:
                U[i,j] = A[i,j]
            else:
                L[i,j] = A[i,j]
    return U,L
# 测试，也可以直接返回A
if __name__ =="__main__":
    
    A = np.array([[2,3,1],[4,7,1],[6,7,3]])
    U,L = Doolittle(A)
    print(f"L矩阵为:\n{L}")
    print('\n')
    print(f"U矩阵为:\n{U}")
