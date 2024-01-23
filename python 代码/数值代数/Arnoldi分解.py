import numpy as np

def arnoldi(A, v, k):     # 特别注意这里k≤rank(A),(思想是用两个矩阵进行储存每次的向量)
    """
    Arnoldi分解的实现
    :param A: 待分解的矩阵
    :param v: 初始向量
    :param k: Krylov子空间的维数
    :return: 上Hessenberg矩阵H和Q矩阵
    """
    n = A.shape[0]  # 获取矩阵A的行数
    Q = np.zeros((n, k+1))  # 初始化Q矩阵，Q的列数为k+1
    H = np.zeros((k+1, k))  # 初始化上Hessenberg矩阵H，H的行数为k+1，列数为k
    Q[:, 0] = v / np.linalg.norm(v)  # 将初始向量归一化后作为Q矩阵的第一列

    for j in range(k):
        w = A @ Q[:, j]  # 计算A与Q的第j列的乘积
        for i in range(j+1):
            H[i, j] = np.dot(Q[:, i], w)  # 计算H矩阵的第i行第j列的值
            w = w - H[i, j] * Q[:, i]  # 计算新的向量w
        H[j+1, j] = np.linalg.norm(w)  # 计算H矩阵的第j+1行第j列的值
        if H[j+1, j] == 0:
            break
        Q[:, j+1] = w / H[j+1, j]  # 将新的向量w归一化后作为Q矩阵的第j+1列

    return H, Q  # 返回上Hessenberg矩阵H和Q矩阵

# 测试Arnoldi分解的代码
if __name__ == "__main__":
    A = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])  # 定义待分解的矩阵A
    v = np.array([1, 1, 1])  # 定义初始向量v
    k = 2  # 定义Krylov子空间的维数k

    H, Q = arnoldi(A, v, k)  # 调用arnoldi函数进行Arnoldi分解
    print("上Hessenberg矩阵H:\n", H)  # 输出上Hessenberg矩阵H
    print("Q矩阵:\n", Q)  # 输出Q矩阵
