#导入库
from sympy import symbols
from scipy import integrate
def legendre_polynomial(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return ((2*n-1)*x*legendre_polynomial(n-1, x) - (n-1)*legendre_polynomial(n-2, x)) / n
def legendre_derivative(n, x):
    return n * (x * legendre_polynomial(n, x) - legendre_polynomial(n-1, x)) / (x**2 - 1)
def find_legendre_zeros(n):
    zeros = []
    epsilon = 1e-6  # 精度控制
    max_iterations = 100  # 最大迭代次数
    for i in range(1, n+1):
        x = -1 + (2*i-1) / n  # 初始猜测值
        for _ in range(max_iterations):
            f = legendre_polynomial(n, x)
            df = legendre_derivative(n, x)
            delta_x = -f / df
            x += delta_x
            if abs(delta_x) < epsilon:
                zeros.append(x)
                break
    return zeros
n = 2
y = find_legendre_zeros(n)
print(y)

# 求积分系数
x = symbols("x")
w = []
q = []
for i in range(n):
    for j in range(n):
        if j != i:
            a = 1
            a *= (x-y[j])/(y[i]-y[j])
    w.append(a)
print(w)

for i in range(n):
    b,l= integrate.quad(w[i],-1, 1)
    q.append(b)
print(q)









   


