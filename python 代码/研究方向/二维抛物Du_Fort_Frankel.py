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
    def uh1_solution(self,p):
        x = p[...,0]
        y = p[...,1]
        return self.u(x,y,t=0.005)
    def source(self,p,t):
        x = p[...,0]
        y = p[...,1]
        return self.f(x,y,t)
    def dirichlet(self,p,t):
        return self.solution(p,t)
    
pde = sinData('exp(-t)*sin(pi*x)*sin(pi*y)','x','y','t',D=[0,1,0,1],T=[0,1])

domain = pde.domain()
duration = pde.duration()


nx = 10
ny = 10
nt = 200 
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny 
tau = (duration[1] - duration[0])/nt

mesh = UniformMesh2d((0,nx,0,ny),h = (hx,hy),origin=(domain[0],domain[2]))
NN = mesh.number_of_nodes()
n0 = nx + 1
n1 = ny + 1
cx = 2/(hx**2)
cy = 2/(hy**2)

uh0 = mesh.interpolate(pde.init_solution,'node')    
uh1 = mesh.function('node')
uh2 = mesh.function('node')
def Du_Fort_Frankel(mesh,tau):
    rx = tau / mesh.h[0]**2
    ry = tau / mesh.h[1]**2
    NN = mesh.number_of_nodes()
    n0 = mesh.nx + 1
    n1 = mesh.ny + 1
    K = np.arange(NN).reshape(n0, n1)
    A = diags([0], [0], shape=(NN, NN), format='csr')
    val = np.broadcast_to(2 * rx, (NN - n1,))
    I = K[1:, :].flat
    J = K[0:-1, :].flat
    A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=mesh.ftype)
    A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=mesh.ftype)
    val = np.broadcast_to(2 * ry, (NN - n0,))
    I = K[:, 1:].flat
    J = K[:, 0:-1].flat
    A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=mesh.ftype)
    A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=mesh.ftype)
    B = diags([1 - 2*rx - 2*ry], [0], shape=(NN, NN), format='csr')
    C = diags([1 + 2*rx + 2*ry], [0], shape=(NN, NN), format='csr')
    return A , B ,C

def advance_DFF(n):
    if n == 0:
        return uh0
    
    elif n == 1:
        A1, B1 = mesh.parabolic_operator_crank_nicholson(tau)
        source = lambda p: pde.source(p, t )
        f = mesh.interpolate(source, intertype='node') # f.shape = (nx+1,ny+1)
        f *= tau
        f.flat[:] += B1@uh0.flat[:]
        gD = lambda p: pde.dirichlet(p, t )
        A1, f = mesh.apply_dirichlet_bc(gD, A1, f)
        uh1.flat = spsolve(A1, f)
        return uh1
    
    else:
        i = 2
        for i in range(n+1):
            t = duration[0] + i*tau
            A,B,C = Du_Fort_Frankel(mesh,tau)
            source0 = lambda p: pde.source(p, t-tau )
            source1 = lambda p: pde.source(p, t )
            source2 = lambda p: pde.source(p, t+tau )
            f0 = mesh.interpolate(source0, intertype='node') # f.shape = (nx+1,ny+1)
            f1 = mesh.interpolate(source1, intertype='node')
            f2 = mesh.interpolate(source2, intertype='node')
            f = (tau/2)*(f0 + 2*f1 +f2)
            f.flat[:] += (A@uh1.flat[:] + B@uh0.flat[:])
            gD = lambda p: pde.dirichlet(p, t+tau )
            C, f = mesh.apply_dirichlet_bc(gD, C, f)
            uh2.flat = spsolve(C, f)
            uh0[:] = uh1
            uh1[:] = uh2
            solution = lambda p: pde.solution(p, t+tau )
            e = mesh.error(solution, uh1, errortype='max')
            print(f"the max error is {e}")
            i += 1
        return uh1
    
uh = advance_DFF(200)


# 画出误差解的图像
x = np.linspace(0, 1, 11)
y = np.linspace(0, 1, 11)
X, Y = np.meshgrid(x, y)

p = np.array([X, Y]).T
t = 1
Z = pde.solution(p,t)
K = np.abs(uh - Z)
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

ax1.plot_surface(X, Y, K, cmap='jet')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
plt.show()
