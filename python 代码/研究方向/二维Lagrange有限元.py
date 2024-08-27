# 导入相应模块
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh
from scipy.sparse import*
from scipy.sparse.linalg import spsolve
from sympy import*

# 创建类
class pdeData:
    def __init__(self,u:str,x:str,y:str,domain = [0,1,0,1]):
        u = sympify(u)
        self.u = lambdify((x,y),u,'numpy')
        self.du_dx = lambdify((x, y), diff(u, x))
        self.du_dy = lambdify((x, y), diff(u, y))
        self.f = lambdify((x, y), -diff(u, x, 2) - diff(u, y, 2))
        self.d = domain
    
    def domain(self): 
        return self.d
    
    def solution(self,p):
        x = p[...,0]
        y = p[...,1]
        return self.u(x,y)
    
    def source(self,p):
        x = p[...,0]
        y = p[...,1]
        return self.f(x,y)
    
    def gradient(self, p):
        x = p[...,0]
        y = p[...,1]
        return self.du_dx(x),self.du_dy(y)
    
    def dirichlet(self,p):
        return self.solution(p)

# 创建对象
pde = pdeData('sin(pi*x)*sin(pi*y)', 'x','y', domain=[0,1,0,1])
domain = pde.domain()

#创建网格
N = 5
mesh = TriangleMesh.from_box(box=[0,1,0,1],nx=N,ny=N)

"""
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes,showindex=True,fontsize=10)
mesh.find_cell(axes,showindex=True,fontsize=10)
mesh.find_edge(axes,showindex=True,fontsize=10)
plt.show()
"""

# 获取网格信息
NN = mesh.number_of_nodes()
node = mesh.entity("node")
NC = mesh.number_of_cells()
cell = mesh.entity("cell")
GD = mesh.geo_dimension()
cm = mesh.entity_measure("cell")

# 组装单元刚度矩阵
# 插值点的个数为3
K = np.array([
            [0.6666666666666670,	0.1666666666666670,     0.1666666666666670,	0.3333333333333330],
            [0.1666666666666670,	0.6666666666666670,     0.1666666666666670,	0.3333333333333330],
            [0.1666666666666670,	0.1666666666666670,     0.6666666666666670,	0.3333333333333330]],dtype=np.float64)
bcs = K[:,0:3]
w = np.array([0.3333333333333330,0.3333333333333330,0.3333333333333330])
v0 = node[cell[:,2],:] - node[cell[:,1],:]
v1 = node[cell[:,0],:] - node[cell[:,2],:]
v2 = node[cell[:,1],:] - node[cell[:,0],:]
nv = np.cross(v2,v0)
Dlambda = np.zeros((NC,3,GD))
W = np.array([[0,1],[-1,0]])
Dlambda[:,0,:] = v0@W/nv.reshape(-1,1)
Dlambda[:,1,:] = v1@W/nv.reshape(-1,1)
Dlambda[:,2,:] = v2@W/nv.reshape(-1,1)
Dlambda=Dlambda[np.newaxis,:]

gphi = np.repeat(Dlambda,3,axis=0)
A = np.einsum('q,qcid,qcjd,c-> cij', w,gphi,gphi,cm)

# 全局映射
I = np.broadcast_to(cell[:,:,None],shape=A.shape)
J = np.broadcast_to(cell[:,None,:],shape=A.shape)

# 组装成全局刚度矩阵
A = csr_matrix((A.flat,(I.flat,J.flat)),shape=(NN,NN))

# 组装全局荷载向量
phi = bcs[:,np.newaxis,:]
ps = mesh.bc_to_point(bcs)
fval = pde.source(ps)
bb = np.einsum("q,qc,qci,c->ci",w,fval,phi,cm)
b = np.zeros(NN, dtype=np.float64)
np.add.at(b, cell, bb)

# dirichlet边界处理
isBdNode = mesh.ds.boundary_node_flag()
uh = np.zeros(NN)
uh[isBdNode] =pde.dirichlet((node[isBdNode])).reshape(-1)
b -= A@uh
b[isBdNode] = uh[isBdNode]
bdIdx = np.zeros(A.shape[0])
bdIdx[isBdNode] = 1
D0 = spdiags(1-bdIdx,0,A.shape[0],A.shape[0])
D1 = spdiags(bdIdx,0,A.shape[0],A.shape[0])
A = D0@A@D0 + D1

# 求解

uh[:] = spsolve(A,b)
uval = pde.solution(node)
error = np.max(np.abs(uh-uval))
print("error:",error)


