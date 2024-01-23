# 矩阵的LDL.T分解，具体数学理论见书P53
import numpy as np

def LDL(A):
    n = len(A)
    L = np.eye(n)
    d = np.zeros(n)
    for j in range(n):
        d[j] = A[j,j] - np.sum(L[j,:j] ** 2 *d[:j])
        for i in range(j+1,n):
            L[i,j] = (A[i,j] - np.sum(L[i,:j]*d[:j]*L[j,:j])) / d[j]
    D = np.diag(d)      
    return D,L

# 测试
if __name__ == "__main__":

    A = np.array([[4,2,8,0],[2,10,10,9],[8,10,21,6],[0,9,6,34]])
    D,L = LDL(A)
    B = np.einsum('ij,jk,kl -> il',L,D,L.T)
    print(D)
    print('\n')
    print(L)
    print('\n')
    print(B)
