# 创建网格的三种方法

# 1、直接调用TriangleMesh.from_box,这种方法可以直接调用，不需要创建对象
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh

mesh1 = TriangleMesh.from_box(box=[0,10,0,10],nx=10,ny=10)

fig = plt.figure()
axes = plt.gca()
mesh1.add_plot(axes)
mesh1.find_node(axes,showindex=True,fontsize=7)
mesh1.find_edge(axes,showindex = True,fontsize=7)
mesh1.find_cell(axes,showindex = True,fontsize=7)

# 2、通过给定网格的节点以及单元
node = np.array([
    [0,0],
    [0,1],
    [0,2],
    [1,0],
    [1,1],
    [1,2],
    [2,0],
    [2,1],
    [2,2]
])
cell = np.array([
    [0,3,1],
    [3,6,4],
    [1,4,2],
    [4,7,5],
    [4,3,1],
    [7,6,4],
    [5,4,2],
    [8,7,5]
])
mesh2 = TriangleMesh(node,cell)
fig = plt.figure()
axes = plt.gca()
mesh2.add_plot(axes)
mesh2.find_node(axes,showindex=True,fontsize=20)
mesh2.find_edge(axes, showindex=True,fontsize=20)
mesh2.find_cell(axes, showindex=True,fontsize=20)

# 3、通过导入UniformMesh1d
from fealpy.mesh import UniformMesh2d
domain = [0,1,0,1]
nx = 10
ny = 10
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
mesh3 = UniformMesh2d([0, nx, 0, ny], h=(hx, hy), origin=(domain[0], domain[2]))
fig = plt.figure()
axes = plt.gca()
mesh3.add_plot(axes)
mesh3.find_node(axes,showindex=True,fontsize=7)
mesh3.find_edge(axes, showindex=True,fontsize=7)
mesh3.find_cell(axes, showindex=True,fontsize=7)
plt.show()


