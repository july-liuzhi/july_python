# 向后欧拉公式
# 导入相应的模块
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from fealpy.pde.parabolic_1d import SinExpPDEData
from fealpy.mesh import UniformMesh1d

# 创造对象
pde = SinExpPDEData()

#空间离散
domain = pde.domain()
nx = 100
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 时间离散
duration = pde.duration()
nt = 320
tau = (duration[1] - duration[0])/nt

# 初值准备
uh0 = mesh.interpolate(pde.init_solution, intertype='node')

#离散系统求解（SOR）方法
def SOR_methed(A,b,w,error):
    n = len(A)
    x = np.zeros(n)
    D = np.zeros_like(A)
    D[np.arange(n),np.arange(n)] = A[np.arange(n),np.arange(n)]
    LU = D - A
    L = np.tril(LU)
    U = np.triu(LU)
    D_wL = D - w * L
    D_wl_inv = np.linalg.inv(D_wL)
    i = 0
    while np.linalg.norm(A@x-b)>error:
        x = D_wl_inv@((1-w)*D + w*U)@x + w * D_wl_inv@b
    return x

# 向后欧拉公式
def advance_backward(n, *fargs):
    w = 1.1
    error = 1e-6
    t = duration[0] + n*tau
    if n== 0:
        return uh0, t
    else:
        A = mesh.parabolic_operator_backward(tau)
        source: Callable[[np.ndarray], np.ndarray] = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype = 'node')
        f*= tau
        f+= uh0

        gD: Callable[[np.ndarray], np.ndarray] = lambda p: pde.dirichlet(p, t + tau)
        A ,f = mesh.apply_dirichlet_bc(gD, A, f)
        A = A.toarray()
        print(A.shape)
        uh0[:] = SOR_methed(A,f,w,error)

        solution: Callable[[np.ndarray], np.ndarray] = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype = 'max')
        print(f"the max error is {e}")

        return uh0, t
# 制作动画化
fig, axes = plt.subplots()
box = [0, 1, -1.5, 1.5]
mesh.show_animation(fig, axes, box, advance_backward, frames=nt + 1)
plt.show()
    