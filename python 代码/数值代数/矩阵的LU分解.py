# 矩阵的LU分解，具体数学理论见书P39
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
    A = np.array([[2,3,1],[4,7,1],[6,7,3]])
    L,U = LU(A)
    B = np.einsum('ij,jk -> ik',L,U)
    print(f"L矩阵为:\n{L}")
    print('\n')
    print(f"U矩阵为:\n{U}")
    print('\n')
    print(f"原矩阵为:\n{B}")

