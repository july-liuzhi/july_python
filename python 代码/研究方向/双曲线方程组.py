# 导入相应的模块
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from fealpy.mesh import UniformMesh1d
import matplotlib.pyplot as plt
from sympy import*
from scipy.sparse import*


class SinData:
    def __init__(self,D=[0,1],T=[0,2]):
        self._domain = D
        self._duration = T
    
    def domain(self):
        return self._domain
    
    def duration(self):
        return self._duration
    
    def solution(self,p,t):
        pi = np.pi   
        term1 = np.sin(pi * (p - t)) + np.sin(pi * (p + t))  
        term2 = np.sin(pi * (p - t)) - np.sin(pi * (p + t))  
        return (term1 / 2) - (term2 / (2 * pi))  
    
    def init_solution(self,p):
        return np.sin(np.pi*p)
    
    def init_solution_diff_t(self, p):
        return np.cos(np.pi*p)
    
    def source(self,p,t):
        return np.zeros_like(p)
    
    def dirichlet(self,p,t):
        return self.solution(p,t)

pde = SinData()

domain = pde.domain()
duration = pde.duration()
nx = 80
nt = 220

hx = (domain[1] - domain[0])/nx
tau = (duration[1] - duration[0])/nt

mesh = UniformMesh1d([0,nx],h = hx,origin=domain[0])
NN = mesh.number_of_nodes()


uh0 = mesh.interpolate(pde.init_solution,intertype="node")
vh0 = mesh.interpolate(pde.init_solution_diff_t,intertype="node")
uh1 = mesh.function("node")
# 计算第一层
t = duration[0]+2*tau
rx = tau/hx 
uh1[1:-1] = rx**2*(uh0[0:-2] + uh0[2:])/2.0 + (1-rx**2)*uh0[1:-1] + tau*vh0[1:-1]
gD = lambda p: pde.dirichlet(p, t)
mesh.update_dirichlet_bc(gD, uh1)
uh1 = uh1

def advance_exp(n):
    a = 1
    t = duration[0] + tau
    if n == 0:
        return uh0, t
    elif n == 1:
        t = duration[0]+2*tau
        return uh1, t
    else:
        i = 3
        while i<n+1:
            t = duration[0]+i*tau
            A = mesh.wave_operator_explicit(tau, a)
            source = lambda p: pde.source(p, t+tau)
            f = mesh.interpolate(source, intertype='node')
            f *= tau**2
            uh2 = A@uh1 - uh0 + f  # 根据差分方程进行更新
            gD = lambda p: pde.dirichlet(p, t+tau)
            mesh.update_dirichlet_bc(gD, uh2)
            uh0[:] = uh1
            uh1[:] = uh2
            solution = lambda p:pde.solution(p,t+tau)
            e = mesh.error(solution,uh2,errortype="L2")
            print(f"the L2 error is {e}")
            i = i+1
        return uh2

n = 15
a = advance_exp(n)

x1 = np.linspace(domain[0],domain[1],30)
t = np.broadcast_to(duration[0]+n*tau,(30,))
u = pde.solution(x1,t)
x2 = mesh.node
fig, ax = plt.subplots()
ax.plot(x1, u, label='u')
ax.plot(x2, a, label='uh')
plt.show()

