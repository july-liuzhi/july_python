# 导入所需要的模块
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh2d
from scipy.sparse import*
from scipy.sparse.linalg import spsolve

# 创建模型
class PDEData2D:
    def domain(self):
        return [0,2,0,1]
    
    def solution(self,p):
        x = p[...,0]
        y = p[...,1]
        return np.exp(x)*np.sin(2*np.pi*y)
    
    def source(self,p):
        x = p[...,0]
        y = p[...,1]
        return 4 * (np.pi**2) * np.exp(x) * np.sin(2*np.pi*y)
    
    def dirichlet(self,p):
        return self.solution(p)
    
# 创建对象
pde = PDEData2D()
domain = pde.domain()

# 网格剖分
nx = 20
ny = 20
hx = (domain[1] - domain[0]) / nx
hy = (domain[3] - domain[2]) / ny
mesh = UniformMesh2d((0,nx,0,ny),h = (hx,hy),origin = (domain[0],domain[2]))

NN = mesh.number_of_nodes()

# 矩阵组装
def operator(NN,nx,ny,hx,hy):
    n0 = nx + 1
    n1 = ny + 1
    cx = 1 / (hx**2)
    cy = 1 / (hy**2)
    NN = mesh.number_of_nodes()
    k = np.arange(NN).reshape(n0,n1)

    A = diags([2*(cx+cy) + 1],[0],shape = (NN,NN))
    
    val = np.broadcast_to(-cx, (NN-n1, ))
    I = k[1:, :].flat
    J = k[0:-1, :].flat
    A += csr_matrix((val, (I, J)), shape=(NN, NN))
    A += csr_matrix((val, (J, I)), shape=(NN, NN))

    val = np.broadcast_to(-cy, (NN-n0, ))
    I = k[:, 1:].flat
    J = k[:, 0:-1].flat
    A += csr_matrix((val, (I, J)), shape=(NN, NN))
    A += csr_matrix((val, (J, I)), shape=(NN, NN))
    return A
k = 0
maxit = 5
em = np.zeros((3, maxit), dtype=np.float64)

for i in range(maxit):
    NN = mesh.number_of_nodes()
    if k == 0:
        nx = 20
        ny = 20
        hx = (domain[1] - domain[0]) / nx
        hy = (domain[3] - domain[2]) / ny
    else:
        nx = nx
        ny = ny
        hx = hx
        hy = hy
    A = operator(NN,nx,ny,hx,hy)
    f = mesh.interpolate(pde.source, 'node')

    A, f = mesh.apply_dirichlet_bc(gD=pde.dirichlet, A=A, f=f)
    uh = mesh.function()
    uh.flat[:] = spsolve(A, f)
    em[0, i], em[1, i], em[2, i] = mesh.error(pde.solution, uh)

    if i < maxit:
        mesh.uniform_refine()
    k += 1
    nx = nx*2
    ny = ny*2
    hx = (domain[1] - domain[0]) / nx
    hy = (domain[3] - domain[2]) / ny

print("em_ratio:", em[:, 0:-1]/em[:, 1:])