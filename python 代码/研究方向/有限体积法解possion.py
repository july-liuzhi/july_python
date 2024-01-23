# 使用有限体积法解决二维possion问题
# 导入所需要的模块
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh2d
from fealpy.mesh import QuadrangleMesh
from scipy.sparse import coo_matrix
from fealpy.pde.poisson_2d import CosCosData
from scipy.sparse.linalg import spsolve
import ipdb # 强大的调试工具

class Poisson:
    # 定义域
    def domain(self):
        return np.array([0,1,0,1])
    # 真解
    def solution(self,p):
        x = p[...,0]
        y = p[...,1]
        pi = np.pi
        return np.cos(pi*x)*np.cos(pi*y)
    # 右端项
    def source(self,p):
        x = p[...,0]
        y = p[...,1]
        pi = np.pi
        return 2*pi**2*np.cos(pi*x)*np.cos(pi*y)
    # 定义边界
    def dirichlet(self,p):
        x = p[...,0]
        y = p[...,1]
        pi = np.pi
        return np.cos(pi*x)*np.cos(pi*y)
    def is_dirichlet_boundary(self,p):
        eps = 1e-12
        x = p[..., 0]
        y = p[..., 1]
        return (np.abs(y-1)<eps)|(np.abs(x-1)<eps)|(np.abs(x)<eps)|(np.abs(y)<eps)
    
pde = Poisson()
domain = pde.domain()
# 网格划分
nx = 40
ny = 40
mesh = QuadrangleMesh.from_box(box = domain,nx = nx,ny = ny) # 导入网格mesh

NC = mesh.number_of_cells()       # 单元的数量
node = mesh.entity("node")        # 节点的编号
edge = mesh.entity("edge")        # 边的编号
cell2edge = mesh.ds.cell_to_edge()   # 单元到边的映射关系,一个单元由哪些边构成
print(cell2edge)
'''
# 绘制网格图像
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, color='r', marker='o', markersize=8, fontsize=16, fontcolor='r')
mesh.find_cell(axes, showindex=True, color='b', marker='o', markersize=8, fontsize=16, fontcolor='b')
mesh.find_edge(axes, showindex=True, color='g', marker='o', markersize=8, fontsize=16, fontcolor='g')
plt.show()
'''
h = 1/nx

# 先组装对角线元素
I = np.arange(NC)
J = np.arange(NC)
val = 4*h/h*np.ones(NC)
A0 = coo_matrix((val,(I, J)), shape=(NC, NC))

# 在处理边界条件
e2e = mesh.ds.cell_to_cell()     # 单元周围的单元有哪些
flag = e2e[np.arange(NC)] == np.arange(NC)[:, None] # 判断是边界的,后面是NC×1数组
I = np.where(flag)[0]            # 含义是从布尔数组 flag 中获取为 True 的元素的索引，
                                 # 并将这些索引存储在名为 I 的变量中。非常重要
J = e2e[flag]
val = (h/(h/2)-h/h)*np.ones(I.shape)
A0 += coo_matrix((val,(I, J)), shape=(NC, NC))

# 处理非边界的条件(质量矩阵组装完成)
I = np.where(~flag)[0] 
J = e2e[~flag]
val = -h/h*np.ones(I.shape)
A0 += coo_matrix((val,(I, J)), shape=(NC, NC))

# 组装荷载向量
# 先将左边的已知项移动到右边
b = np.zeros(NC)
index = np.where(flag)[0]
index1 = cell2edge[flag]                        # 找对应的边界的边的编号
point = node[edge[index1]]                      # 找到该边的两个节点
point = point[:,0,:]*(1/2)+point[:,1,:]*(1/2)   # 计算点的中心位置，减少了一个维度
flag = pde.is_dirichlet_boundary(point)
bu = pde.dirichlet(point)[flag]
data = bu*h/(h/2)
np.add.at(b, index, data)                       # 将data的值通过索引index加到b上
print(b)
# 最后组装右边的向量(荷载向量组装完成)
bb = mesh.integral(pde.source, celltype=True)   # 计算在网格的单元上对源进行积分

# 解方程组
uh = spsolve(A0, bb+b)

# 误差以及图像
ipoint = mesh.entity_barycenter('cell')          # 获得网格的重心坐标        
u = pde.solution(ipoint)                         # 重心坐标的真解
e = u - uh
print('emax', np.max(np.abs(e)))
print('eL2', np.sqrt(np.sum(e**2)))
fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')
xx = ipoint[..., 0]
yy = ipoint[..., 1]
X = xx.reshape(nx, ny)
Y = yy.reshape(ny, ny)
Z = uh.reshape(nx, ny)
ax1.plot_surface(X, Y, Z, cmap='cool')   #'rainbow'、'viridis'、'plasma'、'inferno'、'magma'、'cool' 、'hot'
plt.show()
