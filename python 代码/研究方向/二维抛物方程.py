# 导入相应的模块
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from sympy import*
from fealpy.mesh import UniformMesh2d
from scipy.sparse import*


# 定义数据模型
class sinData:
    def __init__(self,u:str,x:str,y:str,t:str,D=[0,1,0,1],T=[0,0.1]):
        u = sympify(u)
        self.u = lambdify([x,y,t],u,'numpy')
        self.f = lambdify([x,y,t],diff(u,t,1)-diff(u,x,2)-diff(u,y,2)-u)
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
    
pde = sinData('exp(-2*pi*pi*t)*sin(pi*x)*sin(pi*y)','x','y','t',D=[0,1,0,1],T=[0,0.1])

domain = pde.domain()
duration = pde.duration()
nx =20
ny =20
nt = 400 
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny 
tau = (duration[1] - duration[0])/nt

print(hx)
mesh = UniformMesh2d((0,nx,0,ny),h = (hx,hy),origin=(domain[0],domain[2]))
NN = mesh.number_of_nodes()
uh0 = mesh.interpolate(pde.init_solution,intertype="node")
"""
# 向前欧拉
# 矩阵组装

def operator():
  rx = tau/hx**2
  ry = tau/hy**2
  if rx+ry>0.5:
      raise ValueError(f"The rx+ry:{rx+ry} should be smaller than 0.5")
  else:
      n0 = nx + 1
      n1 = ny + 1
      K = np.arange(NN).reshape(n0,n1)
      A = diags([1-2*rx-2*ry],[0],shape=(NN,NN))

      val = np.broadcast_to(rx,(NN-n1,))
      I = K[1:,:].flat
      J = K[0:-1,:].flat
      A += csr_matrix((val,(I,J)),shape=(NN,NN))
      A += csr_matrix((val,(J,I)),shape=(NN,NN))

      val = np.broadcast_to(ry,(NN-n0,))
      I = K[:,1:].flat
      J = K[:,0:-1].flat
      A += csr_matrix((val,(I,J)),shape=(NN,NN))
      A += csr_matrix((val,(J,I)),shape=(NN,NN))
      A -= diags([1],[0],shape=(NN,NN))
      return A
# 向前欧拉

def advance_forward(n):
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = operator()
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        uh0.flat = A@uh0.flat + (tau*f).flat
        gD = lambda p: pde.dirichlet(p, t + tau)
        mesh.update_dirichlet_bc(gD, uh0)
        
        solution = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t
"""

# 向后欧拉
def advance_backward(n):
    """
    @brief 时间步进格式为向后欧拉方法
    
    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.parabolic_operator_backward(tau) + diags([1],[0],shape=(NN,NN))
        
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        f *= tau
        f += uh0

        gD = lambda p: pde.dirichlet(p, t + tau)
        A, f = mesh.apply_dirichlet_bc(gD, A, f)
        uh0.flat = spsolve(A, f)
        
        solution = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t

# 画出动态图像
fig, axes = plt.subplots()
box = [0, 1, 0, 1, -1, 1] # 图像显示的范围 0 <= x <= 1, 0 <= y <= 1, -1 <= uh <= 1
mesh.show_animation(fig, axes, box, advance_backward, 
                    fname='parabolic_cn.mp4', plot_type='contourf', frames=nt + 1)
plt.show()




   
