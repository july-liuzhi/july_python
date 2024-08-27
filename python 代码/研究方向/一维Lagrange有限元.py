# 导入所需要的库
import numpy as np

import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse  import csr_matrix
from scipy.sparse import spdiags
from fealpy.mesh import IntervalMesh

# 创建类
class SinData:
    def domain(self):
        return [0,1]
    def solution(self,p):
        return np.sin(4*np.pi*p)
    def source(self,p):
        return 16*np.pi*np.pi*np.sin(4*np.pi*p)
    def dirichlet(self,p):
        return self.solution(p)

# 创建对象   
Pde = SinData()
domain = Pde.domain()


# 创建网格
mesh = IntervalMesh.from_interval_domain(domain,nx = 5)

# 获取网格的信息
node = mesh.entity("node")
NC = mesh.number_of_cells()
NN = mesh.number_of_nodes()
GD = mesh.geo_dimension()
cell = mesh.entity("cell")
cm = mesh.entity_measure("cell")
# 组装矩阵
qf = mesh.integrator(3)
bcs,ws = qf.get_quadrature_points_and_weights()
v = node[cell[:,1]] - node[cell[:,0]]
h2 = np.sum(v**2,axis=-1)
v /= h2.reshape(-1,1)
Dlambda = np.zeros((NC,2,GD))
Dlambda[:,0,:] =  v
Dlambda[:,1,:] = -v
Dlambda = Dlambda[np.newaxis,:]
gphi = np.repeat(Dlambda,3,axis=0)
A = np.einsum('q,qcid,qcjd,c-> cij', ws,gphi,gphi,cm)

I = np.broadcast_to(cell[:,:,None],shape=A.shape)
J = np.broadcast_to(cell[:,None,:],shape=A.shape)
A = csr_matrix((A.flat,(I.flat,J.flat)),shape=(NN,NN))

# 组装荷载向量
phi = bcs[:, np.newaxis, :]
ps = mesh.bc_to_point(bcs)
fval = np.squeeze(Pde.source(ps), axis=2)
bb = np.einsum('q,qc,qci,c-> ci',ws,fval,phi,cm)

b = np.zeros(NN, dtype=np.float64)
np.add.at(b, cell, bb) 


# dirichlet边界处理
isBdNode = mesh.ds.boundary_node_flag()
uh = np.zeros(NN)
uh[isBdNode] = Pde.dirichlet((node[isBdNode])).reshape(-1)
b -= A@uh
b[isBdNode] = uh[isBdNode]
bdIdx = np.zeros(A.shape[0])
bdIdx[isBdNode] = 1
D0 = spdiags(1-bdIdx,0,A.shape[0],A.shape[0])
D1 = spdiags(bdIdx,0,A.shape[0],A.shape[0])
A = D0@A@D0 + D1

# 求解
uh[:] = spsolve(A,b)
x1 = np.linspace(domain[0],domain[1],100)
x2 = mesh.node
u = Pde.solution(x1)
fig, ax = plt.subplots()
ax.plot(x1, u, label='u')
ax.plot(x2, uh, label='uh')
ax.legend()
plt.show()
