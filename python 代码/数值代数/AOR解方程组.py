# AOR迭代法
import numpy as np
def AOR(A,b,x0,max_iter,tol,w,r):
    n =len(b)
    x = x0
    iter_num = 0
    while iter_num < max_iter:
        x_new = np.zeros(n)
        for i in range(n):
            sum_1 = np.sum(A[i,:i] * x_new[:i])
            sum_h = np.sum(A[i,:i] * x[:i])
            sum_2 = np.sum(A[i, i+1:] * x[i+1:])
            x_new[i] = (1-w)*x[i] + (w*b[i]-r*sum_1 -(w-r)*sum_h -w*sum_2) / A[i,i]
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
        iter_num += 1
    return x_new,iter_num

# 测试
if __name__ == '__main__':
    A = np.array([[4, -1, 0],
              [-1, 4, -1],
              [0, -1, 3]])
    b = np.array([12,18,10])
    x0 = np.array([0,0,0])
    max_iter = 1000
    tol = 1e-6
    w = 1.2
    r = 0.5
    x,num = AOR(A,b,x0,max_iter,tol,w,r)
    print(x)
    print("\n")
    print(num)
