# 一维有限体积法
# 导入相应的模块
import numpy as np 
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh1d
from scipy.sparse.linalg import spsolve
from scipy.sparse import*

# 创建模型
class pdeDATA:
    
    def domain(self):
        return [0,1]
    
    def solution(self,p):
        return np.exp(-10*p*(1-p))
    
    def source(self,p):
        return 0
    
    def dirichlet(self,p):
        return self.solution(p)
    
# 创建对象
pde = pdeDATA()
domain = pde.domain()

# 网格剖分 
nx = 10
hx = (domain[1] - domain[0]) / nx
mesh = UniformMesh1d([0,nx],h = hx,origin = domain[0])
NN = mesh.number_of_nodes()
node= mesh.node


# 矩阵组装
def operator():
    k = np.arange(NN)
    
    A = diags([2/hx+20*hx],[0],shape = (NN,NN))
    val1 = np.zeros(NN-1)
    val2 = np.zeros(NN-1)
    for i in range(NN-1):
        val1[i] = -1/hx + 5*(2*node[i]-1)   
        val2[i] = -1/hx - 5*(2*node[i+1]-1)

    I = k[1:]
    J = k[0:-1]

    A += csr_matrix((val1,(J,I)),shape = (NN,NN))
    A += csr_matrix((val2,(I,J)),shape = (NN,NN))
    return A

A = operator()

# 准备初值
uh = mesh.function()
f = mesh.interpolate(pde.source,"node")

# 边界处理
def dirichlet(gD, A, f, uh=None, threshold=None): 
    if uh is None:
        uh = mesh.function()
    
    node = mesh.node
    if threshold is None:
        isBdnode = mesh.ds.boundary_node_flag()
        index = isBdnode
    elif isinstance(threshold,int):
        index = threshold
    elif callable(threshold):
        index = threshold(node)
    else:
        raise ValueError(f"Invalid threshold: {threshold}")
    
    uh[index] = gD(node[index])

    f -= A@uh
    f[index] = uh[index]

    bdIdx = np.zeros(A.shape[0], dtype=np.int_)
    bdIdx[index] = 1
    D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
    D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
    A = D0@A@D0 + D1
    return A, f 

A,f = dirichlet(pde.dirichlet,A,f)

# 求解
uh[:] = spsolve(A,f)
# 画出真解与数值解的图像
x1 = np.linspace(domain[0],domain[1],100)
x2 = mesh.node
u = pde.solution(x1)
fig, ax = plt.subplots()
ax.plot(x1, u, label='u')
ax.plot(x2, uh, label='uh')
# 添加图例
ax.legend()
# 显示图像
plt.show()


# 误差计算
def error(u,uh,errortype = "all"):
    h = hx
    node = mesh.node
    uI = u(node)
    e = uI - uh

    if errortype == "all":
        emax = np.max(np.abs(e))
        e0 = np.sqrt(h * np.sum(e ** 2))
        de = e[1:] - e[0:-1]
        e1 = np.sqrt(np.sum(de ** 2) / h + e0 ** 2)
        return emax,e0,e1
    elif errortype == "max":
        emax = np.max(np.abs(e))
        return emax
    elif errortype == "L2":
        e0 = np.sqrt(h * np.sum(e ** 2))
        return e0
    elif errortype == "H1":
        de = e[1:] - e[0:-1]
        e1 = np.sqrt(np.sum(de ** 2) / h + e0 ** 2)
        return e1
    

max = error(pde.solution, uh,errortype = "max")
print(max)

