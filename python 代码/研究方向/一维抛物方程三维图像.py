# 导入相应的的模块
import numpy as np
from fealpy.mesh import UniformMesh1d
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 导入数据
class Data:
    def domain(self):
        return [0,1]
    def duration(self):
        return [0,1]
    def source(self,p,t):
        return 0
    def init_solution(self,p):
        return np.sin(np.pi*p)
    def dirichlet(self,p,t):
        return 0

# 创建对象
pde = Data()
domain = pde.domain()
duration = pde.duration()

# 空间离散
nx = 10
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0,nx],h = hx,origin=domain[0])


# 时间离散
nt = 100
tau = (duration[1]-duration[0])/nt
NN = mesh.number_of_nodes()
r = tau/hx**2

uh0 = mesh.interpolate(pde.init_solution,intertype="node")

# 追赶法
def tridiagonal_solver(a, b, c, d):
    n = len(b)
    x = np.zeros(n)
    
    # 向前消元
    for i in range(1, n):
        m = a[i-1] / b[i-1]
        b = b.copy()  # 创建可写的 b 副本
        b[i] = b[i] - m * c[i-1]
        d = d.copy()  # 创建可写的 d 副本
        d[i] = d[i] - m * d[i-1]
    
    # 向后代入
    x[-1] = d[-1] / b[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    
    return x
# 向后欧拉
def forward(n):
    t = duration[0]+n*tau
    if n==0:
        return uh0
    else:
        i = 1
        for i in range(n+1):
            source = lambda p: pde.source(p, t + (i+1)*tau)
            f = mesh.interpolate(source, intertype='node')
            f *= tau
            f += uh0
            val = np.broadcast_to(1+2*r**2,(NN,))
            val1 = np.broadcast_to(-r,(NN-1,))
            uh0[:] = tridiagonal_solver(val1,val,val1,f)
            i += 1
        return uh0

H = np.zeros((100,11))
for i in range(10):
    H[i,:]=forward(i)
Z1 = H.flatten()
x1 = np.linspace(0,1,100)  # 时间
y1 = np.linspace(0,1,11)   # 空间
X1 = np.repeat(x1, 11)   # 时间
Y1 = np.tile(y1,100)     # 空间

# 创建三维图像
fig = plt.figure(figsize=(12, 6))

# 三维散点图
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X1, Y1, Z1, c=Z1, cmap='viridis')
ax1.set_xlabel('X')
ax1.set_ylabel('T')
ax1.set_zlabel('Z')
ax1.set_title('3D Scatter Plot')

# 三维插值图像
# 创建网格
xi = np.linspace(min(X1), max(X1), 50)
yi = np.linspace(min(Y1), max(Y1), 50)
X, Y = np.meshgrid(xi, yi)

# 插值
Z = griddata((X1, Y1), Z1, (X, Y), method='linear')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
ax2.set_xlabel('X')
ax2.set_ylabel('T')
ax2.set_zlabel('Z')
ax2.set_title('3D Interpolated Surface')
plt.show()

