# 数值解
# 导入相关的模块
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.uniform_mesh_1d import UniformMesh1d
from scipy.sparse import*
from scipy.sparse.linalg import spsolve

# 创建模型
class PDEData:
    def domain(self):
        return [0,1]
    
    def solution(self,p):
        return np.sin(np.pi * p)
    
    def source(self,p):
        return (np.pi*np.pi + 1)*np.sin(np.pi * p)
    def dirichlet(self, p):
        return self.solution(p)

# 创建对象
pde = PDEData()
domain = pde.domain()

# 网格剖分
nx = 10
hx = (domain[1] - domain[0]) / nx
mesh = UniformMesh1d([0,nx],h = hx,origin = domain[0])
NN = mesh.number_of_nodes()

# 矩阵组装
def operator():
    cx = 1 / hx ** 2
    k = np.arange(NN)

    A = diags([2*cx + 1],[0],shape = (NN,NN))
    val = np.broadcast_to(-cx,(NN - 1))
    I = k[1:]
    J = k[0:-1]

    A += csr_matrix((val,(I,J)),shape = (NN,NN))
    A += csr_matrix((val,(J,I)),shape = (NN,NN))
    return A

A = operator()
uh = mesh.function()
f = mesh.interpolate(pde.source,"node")
# 边界处理（为了将系数矩阵变成对称矩阵）
def dirichlet(gD, A, f, uh=None, threshold=None): 
    if uh is None:
        uh = mesh.function('node')

    node = mesh.node
    if threshold is None:
        isBdNode = mesh.ds.boundary_node_flag()
        index = isBdNode
    elif isinstance(threshold, int):
        index = threshold
    elif callable(threshold):
        index = threshold(node)
    else:
        raise ValueError(f"Invalid threshold: {threshold}")
 
    uh[index]  = gD(node[index])

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
def error(u, uh, errortype='all'): # 可以点击这里查看 FEALPy 仓库中的代码。 
    h = hx
    node = mesh.node
    uI = u(node)
    e = uI - uh

    if errortype == 'all':
        emax = np.max(np.abs(e))
        e0 = np.sqrt(h * np.sum(e ** 2))
        de = e[1:] - e[0:-1]
        e1 = np.sqrt(np.sum(de ** 2) / h + e0 ** 2)
        return emax, e0, e1
    elif errortype == 'max':
        emax = np.max(np.abs(e))
        return emax
    elif errortype == 'L2':
        e0 = np.sqrt(h * np.sum(e ** 2))
        return e0
    elif errortype == 'H1':
        e0 = np.sqrt(h * np.sum(e ** 2))
        de = e[1:] - e[0:-1]
        e1 = np.sqrt(np.sum(de ** 2) / h + e0 ** 2)
        return e1
            
et = ['$|| u - u_h||_{\infty}$', '$|| u - u_h||_{0}$', '$|| u - u_h ||_{1}$']
eu = np.zeros(len(et), dtype=np.float64) 
eu[0], eu[1], eu[2] = error(pde.solution, uh)
et = np.array(et)
print(np.vstack((et, eu)))

# 测试收敛阶
maxit = 5 
em = np.zeros((len(et), maxit), dtype=np.float64)

for i in range(maxit):
    A = mesh.elliptic_operator(r = 1)
    uh = mesh.function() 
    f = mesh.interpolate(pde.source, 'node')
    A, f = mesh.apply_dirichlet_bc(gD=pde.dirichlet, A=A, f=f)
    uh[:] = spsolve(A, f) 
    em[0, i], em[1, i], em[2, i] = mesh.error(pde.solution, uh)

    if i < maxit:
        mesh.uniform_refine(1)

print("em:\n", em)
print("em_ratio:", em[:, 0:-1]/em[:, 1:])
