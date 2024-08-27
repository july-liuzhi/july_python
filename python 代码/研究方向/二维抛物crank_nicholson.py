import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from sympy import*
from fealpy.mesh import UniformMesh2d
from scipy.sparse import*


# 定义数据模型
class sinData:
    def __init__(self,u:str,x:str,y:str,t:str,D=[0,1,0,1],T=[0,1]):
        u = sympify(u)
        self.u = lambdify([x,y,t],u,'numpy')
        self.f = lambdify([x,y,t],diff(u,t,1)-diff(u,x,2)-diff(u,y,2))
        self.domian = D
        self._duration = T

    def domain(self):
        return self.domian
    def duration(self):
        return self._duration
    def solution(self,p,t):
        x = p[...,0]
        y = p[...,1]
        return self.u(x,y,t)
    def init_solution(self,p):
        x = p[...,0]
        y = p[...,1]
        return self.u(x,y,t=self._duration[0])
    def source(self,p,t):
        x = p[...,0]
        y = p[...,1]
        return self.f(x,y,t)
    def dirichlet(self,p,t):
        return self.solution(p,t)
    
pde = sinData('exp(-t)*sin(pi*x)*sin(pi*y)','x','y','t',D=[0,1,0,1],T=[0,1])

domain = pde.domain()
duration = pde.duration()


nx =10
ny =10
nt = 200 
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny 
tau = (duration[1] - duration[0])/nt

mesh = UniformMesh2d((0,nx,0,ny),h = (hx,hy),origin=(domain[0],domain[2]))
NN = mesh.number_of_nodes()

def advance_crank_nicholson(n):
    uh0 = mesh.interpolate(pde.init_solution,intertype="node")
    if n == 0:
        t = duration[0]
        return uh0, t
    else:
        uh = uh0.copy()
        i = 1
        for i in range(n+1):
            t = duration[0] + i*tau
            A, B = mesh.parabolic_operator_crank_nicholson(tau)
            source = lambda p: pde.source(p, t +tau)
            f = mesh.interpolate(source, intertype='node') 
            f *= tau
            f.flat[:] += B@uh0.flat[:]
            
            gD = lambda p: pde.dirichlet(p, t+tau)
            A, f = mesh.apply_dirichlet_bc(gD, A, f)
            uh.flat = spsolve(A, f)
            uh0 = uh

            #solution = lambda p: pde.solution(p, t + tau)
            #e = mesh.error(solution, uh0, errortype='max')
            #print(f"the max error is {e}")
            i += 1

        return uh0

uh1 = advance_crank_nicholson(200)

# 画出真解的图像
x = np.linspace(0, 1, 11)
y = np.linspace(0, 1, 11)
X, Y = np.meshgrid(x, y)

p = np.array([X, Y]).T
t = 1
Z = pde.solution(p,t)
K = np.abs(uh1 - Z)
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

ax1.plot_surface(X, Y, K, cmap='jet')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
plt.show()
