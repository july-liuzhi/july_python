# 导入所需要的模块
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.uniform_mesh_2d import UniformMesh2d
from scipy.sparse import*
from scipy.sparse.linalg import spsolve
from sympy import*

# 创建类
class pdeData:
    def __init__(self,u:str,x:str,y:str,domain = [0,1,0,1]):
        self.u = lambdify((x,y),sympify(u))
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
pde = pdeData('sin(pi*x)*sin(pi*y)-cos(pi*x)*cos(pi*y)+1', 'x','y', domain=[0,1,0,1])
domain = pde.domain()

# 网格划分
nx = 10
ny = 10
hx = (domain[1] - domain[0]) / nx
hy = (domain[3] - domain[2]) / ny
mesh = UniformMesh2d((0,nx,0,ny),h=(hx,hy),origin=(domain[0],domain[2]))
NN = mesh.number_of_nodes()

# 矩阵组装
def operator():
    n0 = nx + 1
    n1 = ny + 1
    cx = 1 / hx**2
    cy = 1 / hy**2
    k = np.arange(NN).reshape(n0,n1)
    A = diags([2*(cx+cy)],[0],shape=(NN,NN))
    
    val = np.broadcast_to(-cx,(NN-n1,))
    I = k[1:,:].flat
    J = k[0:-1,:].flat
    A += csr_matrix((val,(I,J)),shape=(NN,NN))
    A += csr_matrix((val,(J,I)),shape=(NN,NN))

    val = np.broadcast_to(-cy,(NN-n0,))
    I = k[:,1:].flat
    J = k[:,0:-1].flat
    A += csr_matrix((val,(I,J)),shape=(NN,NN))
    A += csr_matrix((val,(J,I)),shape=(NN,NN))
    
    return A

A = operator()
f = mesh.interpolate(pde.source,"node")

# 处理边界条件
def dirichlet(gD,A,f,uh=None,threshold=None):
    if uh is None:
        uh = mesh.function("node").reshape(-1)
    else:
        uh = uh.reshape(-1)
    f = f.reshape(-1)

    node = mesh.entity('node')
    if threshold is None:
        isBdNode = mesh.ds.boundary_node_flag()
    elif isinstance(threshold, int):
        isBdNode = (np.arange(node.shape[0]) == threshold)
    elif callable(threshold):
        isBdNode = threshold(node)
    else:
        raise ValueError(f"Invalid threshold: {threshold}")    
    
    uh[isBdNode] = gD(node[isBdNode])
    f -= A@uh
    f[isBdNode] = uh[isBdNode]

    bdIdx = np.zeros(A.shape[0],dtype=mesh.itype)
    bdIdx[isBdNode] = 1
    D0 = spdiags(1-bdIdx,0,A.shape[0],A.shape[0])
    D1 = spdiags(bdIdx,0,A.shape[0],A.shape[0])
    A = D0@A@D0 + D1
    return A,f

A, f = dirichlet(gD=pde.dirichlet, A=A, f=f)

# 求解及图像
uh = mesh.function()
uh.flat[:] = spsolve(A,f)
# 画出真解的图像
x = np.linspace(0, 1, 1000)
y = np.linspace(0, 1, 1000)
X, Y = np.meshgrid(x, y)

p = np.array([X, Y]).T
Z = pde.solution(p)
fig = plt.figure(1)
ax1 = fig.add_subplot(111, projection='3d')

ax1.plot_surface(X, Y, Z, cmap='jet')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
print(uh.shape)
# 画出数值解图像
fig = plt.figure(2)
ax2 = fig.add_subplot(111, projection='3d')
mesh.show_function(ax2, uh.reshape(nx+1, ny + 1))
plt.title("Numerical solution after processing the boundary conditions")
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
plt.show()

# 定义误差
def error(u,uh,errortype = all):
    assert (uh.shape[0] == nx+1) and (uh.shape[1] == ny+1)
    hx = hx
    hy = hy
    nx = nx
    ny = ny
    node = mesh.node
    uI = u(node)
    e = uI - uh

    if errortype == 'all':
        emax = np.max(np.abs(e))
        e0 = np.sqrt(hx * hy * 2 * np.sum(e ** 2))
        el2 = np.sqrt(1 / ((nx - 1) * (ny - 1)) * np.sum(e ** 2))

        return emax, e0, el2
    elif errortype == 'max':
        emax = np.max(np.abs(e))
        return emax
    elif errortype == 'L2':
        h = np.zeros(2)
        h[0] = hx
        h[1] = hy
        e0 = np.sqrt(h ** 2 * np.sum(e ** 2))
        return e0
    elif errortype == 'l2':
        el2 = np.sqrt(1 / ((nx - 1) * (ny - 1)) * np.sum(e ** 2))
        return el2

et = ['$|| u - u_h||_{\infty}$', '$|| u - u_h||_{0}$', '$|| u - u_h ||_{l2}$']
eu = np.zeros(len(et), dtype=np.float64) 
eu[0], eu[1], eu[2] = mesh.error(pde.solution, uh)
et = np.array(et)
print(np.vstack((et, eu)))
print("----------------------------------------------------------------------")
maxit = 5
em = np.zeros((3, maxit), dtype=np.float64)

# 测试收敛阶
for i in range(maxit):
    A = mesh.laplace_operator() 
    uh = mesh.function()
    f = mesh.interpolate(pde.source, 'node')
    A, f = mesh.apply_dirichlet_bc(pde.dirichlet, A, f)
    uh.flat[:] = spsolve(A, f)  
    em[0, i], em[1, i], em[2, i] = mesh.error(pde.solution, uh)

    if i < maxit:
        mesh.uniform_refine()

print("em_ratio:", em[:, 0:-1]/em[:, 1:])


