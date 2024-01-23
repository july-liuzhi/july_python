import numpy as np
import matplotlib.pyplot as plt
from fealpy.decorator import cartesian
from fealpy.mesh import UniformMesh1d

class SinPDEData: 
    def domain(self) :
        return [0, 1]

    @cartesian    
    def solution(self, p: np.ndarray) -> np.ndarray:
        return np.sin(4*np.pi*p)
    
    @cartesian    
    def source(self, p: np.ndarray) -> np.ndarray:
        return 16*np.pi**2*np.sin(4*np.pi*p)
    
    @cartesian    
    def gradient(self, p: np.ndarray) -> np.ndarray:
        return 4*np.pi*np.cos(4*np.pi*p)

    @cartesian    
    def dirichlet(self, p: np.ndarray) -> np.ndarray:
        return self.solution(p)

pde = SinPDEData()
domain = pde.domain()
nx = 100
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()
#矩阵组装
A = mesh.laplace_operator()
#边界条件处理
uh = mesh.function()
f = mesh.interpolate(pde.source,'node')
A , f = mesh.apply_dirichlet_bc(pde.dirichlet,A,f)
A = A.toarray()
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
        print("第",i+1,"次迭代结果为：",x)
        i += 1
        
    return x

w = 1.4
error = 10
uh[:] = SOR_methed(A,f,w,error)
fig = plt.figure()
axes = fig.gca()
mesh.show_function(axes,uh)
plt.show()
