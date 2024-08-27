# 导入数据模型
import numpy as np
import matplotlib.pyplot as plt
from fealpy.pde.elliptic_1d import SinPDEData
from scipy.sparse.linalg import spsolve

pde = SinPDEData()

# 网格对象
from fealpy.mesh import IntervalMesh

nx  = 10
domain = pde.domain() 
mesh = IntervalMesh.from_interval(domain, nx=nx)

# 导入拉格朗日有限元空间 LagrangeFEMSpace 类，并建立分片线性的连续空间对象 space
from fealpy.functionspace import LagrangeFESpace
space = LagrangeFESpace(mesh, p=1)   # 用户可以指定更高次数

# 导入扩散算子的积分子 DiffusionIntegrator和双线性型 BilinearForm。
# 前者负责计算 (u', v') 的单元矩阵，后者负责组装总刚度矩阵
from fealpy.fem import DiffusionIntegrator
from fealpy.fem import BilinearForm

bform = BilinearForm(space)
bform.add_domain_integrator(DiffusionIntegrator(q=3))
A = bform.assembly()

# 导入标量函数形式的源项积分子 ScalarSourceIntegrator 和线性型 LinearForm
# 前者负责计算右端单元向量，后者负责组装总向量的组装。
from fealpy.fem import ScalarSourceIntegrator
from fealpy.fem import LinearForm

lform = LinearForm(space)
lform.add_domain_integrator(ScalarSourceIntegrator(pde.source, q=3))
F = lform.assembly()

# 导入 Dirichlet 边界处理模块 DirichletBC
from fealpy.fem import DirichletBC

bc = DirichletBC(space, pde.dirichlet) 
uh = space.function() 
A, F = bc.apply(A, F, uh)
uh[:] = spsolve(A, F)

# 计算 $$L^2$$ 和 $$H^1$$误差。
L2Error = mesh.error(pde.solution, uh, q=3)
H1Error = mesh.error(pde.gradient, uh.grad_value, q=3)