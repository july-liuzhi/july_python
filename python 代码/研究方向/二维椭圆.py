# 导入相应模块
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh
from fealpy.functionspace  import LagrangeFESpace
from scipy.sparse import*
from scipy.sparse.linalg import spsolve
from sympy import*
import time

# 创建类
class pdeData:
    def __init__(self,u:str,x:str,y:str,domain = [0,1,0,1]):
        u = sympify(u)
        self.u = lambdify((x,y),u,'numpy')
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
pde = pdeData('cos(2*pi*x)*cos(2*pi*y)', 'x','y', domain=[0,1,0,1])
domain = pde.domain()

start_time = time.time()
#创建网格
mesh = TriangleMesh.from_box(domain, nx=160, ny=160)
node = mesh.entity("node")

p = 1                                              
q = 3                                              
space = LagrangeFESpace(mesh, p=p)                 
GDOF = space.number_of_global_dofs()               
LDOF = space.number_of_local_dofs()                

cell2dof = space.cell_to_dof()                     

ipoints = space.interpolation_points()             

qf = mesh.integrator(q)                            
bcs, ws = qf.get_quadrature_points_and_weights()   
ps = mesh.bc_to_point(bcs) # (NQ, NC, GD)          

cm = mesh.entity_measure('cell') #(NC, )           
gphi = space.grad_basis(bcs) #(NQ, NC, LDOF, GD)   

A = np.einsum('q, qcid, qcjd, c->cij', ws, gphi, gphi, cm) 

fval = pde.source(ps) # (NQ,NC)
phi = space.basis(bcs) # (NQ, 1, LDOF)           
b = np.einsum('q, qc, qci, c->ci', ws, fval, phi, cm) 

F = np.zeros((GDOF, ), dtype=np.float64)
np.add.at(F, cell2dof, b)                          
I = np.broadcast_to(cell2dof[:, :, None], shape=A.shape)
J = np.broadcast_to(cell2dof[:, None, :], shape=A.shape)

A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(GDOF, GDOF))   

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
end_time = time.time()
execution_time = end_time - start_time
print("代码执行时间：", execution_time, "秒")
uval = pde.solution(node)
error = np.max(np.abs(uh-uval))
print("error:",error)


# 画出图像
x1 = np.linspace(0, 1, 1000)
y1 = np.linspace(0, 1, 1000)
X1, Y1 = np.meshgrid(x1, y1)
p = np.array([X1, Y1]).T
Z = pde.solution(p)
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, Y1, Z, cmap='jet')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

x2 = np.linspace(0,1,161)
y2 = np.linspace(0,1,161)
X2,Y2 = np.meshgrid(x2,y2)
uh = uh.reshape(161,161)
ax.plot_surface(X2, Y2, uh, cmap='jet')
plt.show()