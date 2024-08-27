import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from fealpy.pde.poisson_2d import CosCosData
from fealpy.mesh import TriangleMesh
from fealpy.functionspace  import LagrangeFESpace
from scipy.sparse import*
from scipy.sparse.linalg import spsolve
from sympy import*

pde = CosCosData()
domain = pde.domain()
mesh = TriangleMesh.from_box(domain, nx=10, ny=10)
node = mesh.entity("node")

p = 1                                              # 空间基函数的次数
q = 3                                              # 高斯插值点的个数
space = LagrangeFESpace(mesh, p=p)                 # 选择Lagrange有限元空间
GDOF = space.number_of_global_dofs()               # 全局自由度
LDOF = space.number_of_local_dofs()                # 局部自由度

cell2dof = space.cell_to_dof() # (NC, LDOF, 1)     # cell = mesh.entity("cell")是一样的

ipoints = space.interpolation_points() # (GDOF, GD)

qf = mesh.integrator(q)                            # 插值点
bcs, ws = qf.get_quadrature_points_and_weights()   # 获取插值点的重心坐标和权重
ps = mesh.bc_to_point(bcs) # (NQ, NC, GD)          # 将重心坐标转换为笛卡尔坐标

cm = mesh.entity_measure('cell') #(NC, )           # 获取单元的测度
gphi = space.grad_basis(bcs) #(NQ, NC, LDOF, GD)   # 获取基函数的导数

A = np.einsum('q, qcid, qcjd, c->cij', ws, gphi, gphi, cm) # (NC, LDOF, LDOF)  # 单元刚度矩阵的组装

fval = pde.source(ps) # (NQ,NC)                    # 获取源项的值
phi = space.basis(bcs) # (NQ, 1, LDOF)             # 获取基函数
b = np.einsum('q, qc, qci, c->ci', ws, fval, phi, cm) # (NC, LDOF)  # 单元载荷向量的组装

F = np.zeros((GDOF, ), dtype=np.float64)
np.add.at(F, cell2dof, b)                          # 全局荷载向量的组装

I = np.broadcast_to(cell2dof[:, :, None], shape=A.shape)
J = np.broadcast_to(cell2dof[:, None, :], shape=A.shape)

A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(GDOF, GDOF))   # 全局刚度矩阵的组装

# dirichlet边界处理
isBdNode = mesh.ds.boundary_node_flag()
uh = np.zeros(GDOF)
uh[isBdNode] =pde.dirichlet((node[isBdNode])).reshape(-1)
F -= A@uh
F[isBdNode] = uh[isBdNode]
bdIdx = np.zeros(A.shape[0])
bdIdx[isBdNode] = 1
D0 = spdiags(1-bdIdx,0,A.shape[0],A.shape[0])
D1 = spdiags(bdIdx,0,A.shape[0],A.shape[0])
A = D0@A@D0 + D1

uh[:] = spsolve(A,F)
uval = pde.solution(node)

error = np.max(np.abs(uh-uval))
print("error:",error)

