# 矩阵的cholesky分解，具体数学理论见书P51
import numpy as np

def cholesky(A):
    n = len(A)
    L = np.eye(n)
    for j in range(n):
        L[j,j] =np.sqrt(A[j,j] - np.sum(L[j,:j]**2))      # 为什么这里是j，而不是j-1
        for i in range(j+1,n):                            # 原因:L[:j] 他的索引是0行到j-1行
            L[i,j] = (A[i,j] - np.sum(L[i,:j]*L[j,:j])) / L[j,j]
    return L

# 测试
if __name__ == "__main__":

    A = np.array([[4,2,8,0],[2,10,10,9],[8,10,21,6],[0,9,6,34]])
    L = cholesky(A)
    B = np.einsum('lk,kj -> lj',L,L.T)
    print(L)
    print('\n')
    print(B)
