# 导入相应的模块
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh1d
from scipy.sparse.linalg import spsolve
from scipy.sparse import*
from typing import Callable, Tuple, Any

# 创建模型
class PdeDATA:
    def domain(self):
        return [0,1,0,1]
    
    def solution(self,p,t):
        return np.sin(4*np.pi*p)*np.exp(-10*t)
    
    def init_solution(self,p):
        return np.sin(4*np.pi*p)
    
    def source(self,p,t):
        pi = np.pi
        return -10*np.exp(-10*t)*np.sin(4*pi*p) + 16*pi**2*np.exp(-10*t)*np.sin(4*pi*p)
    def dirichlet(self,p,t):
        return self.solution(p,t)
    
# 创建对象
pde = PdeDATA()

#空间离散
domain = pde.domain()
nx = 20
hx = (domain[1] - domain[0]) / nx
mesh = UniformMesh1d([0,nx],h = hx,origin=domain[0])

#时间离散
nt = 1000
tau = (domain[3] - domain[2]) / nt

# 向前欧拉矩阵组装
def operator(tau): 
    r = tau/mesh.h**2 
    if r > 0.5:
        raise ValueError(f"The r: {r} should be smaller than 0.5")

    NN = mesh.number_of_nodes()
    k = np.arange(NN)

    A = diags([1 - 2 * r], 0, shape=(NN, NN), format='csr')

    val = np.broadcast_to(r, (NN-1, ))
    I = k[1:]
    J = k[0:-1]
    A += csr_matrix((val, (I, J)), shape=(NN, NN))
    A += csr_matrix((val, (J, I)), shape=(NN, NN))
    return A


# 准备初值
uh0 = mesh.interpolate(pde.init_solution,"node")

# 向前欧拉
def advance(n):
    t = domain[2]+n*tau
    if n == 0 :
        return  uh0,t
    else:
        A = operator(tau)
        source:Callable[[np.ndarray],np.ndarray] = lambda p:pde.source(p,t+tau)
        f = mesh.interpolate(source,"node")
        uh0[:] = A@uh0 + tau*f
        gD:Callable[[np.ndarray],np.ndarray] = lambda p:pde.dirichlet(p,t+tau)
        mesh.update_dirichlet_bc(gD,uh0)

        solution:Callable[[np.ndarray],np.ndarray] = lambda p:pde.solution(p,t+tau)
        e = mesh.error(solution,uh0,errortype="max")
        print(f"the max error is {e}")
        return uh0,t
fig, axes = plt.subplots()
box = [0, 1, -1.5, 1.5] # 图像显示的范围 0 <= x <= 1, -1.5 <= y <= 1.5

mesh.show_animation(fig, axes, box, advance, fname='advance_forward.mp4', 
                    frames=nt+1, lw=2, interval=50, linestyle='--', color='red')
plt.show()





