#一维均匀网格
"""
import numpy as np 
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh1d

domain = [0.0,1.0]
nx = 10
hx = (domain[1] - domain[0]) / nx
mesh = UniformMesh1d([0,nx],h=hx,origin=domain[0])
fig,axes = plt.subplots()
mesh.add_plot(axes)
mesh.find_node(axes,showindex=True,fontsize=10,fontcolor="g")
plt.show()
"""

# 二维均匀网格
'''
import numpy as np
import matplotlib.pyplot as plt 
from fealpy.mesh import UniformMesh2d

domain = [0,1,0,1]
nx = 10
ny = 10
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny

mesh = UniformMesh2d([0,nx,0,ny],h=(hx,hy),origin=(domain[0],domain[2]))
fig,axes = plt.subplots()
mesh.add_plot(axes)
mesh.find_node(axes,showindex=True,fontcolor="b",fontsize=12)
plt.show()
'''

# 三维均匀网格
'''
import numpy as np
import matplotlib.pyplot as plt 
from fealpy.mesh import UniformMesh3d

domain = [0,1,0,1,0,1]
nx = 5
ny = 5
nz = 5
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
hz = (domain[5] - domain[4])/nz

mesh = UniformMesh3d([0,nx,0,ny,0,nz],h=(hx,hy,hz),origin=(domain[0],domain[2],domain[4]))
fig = plt.figure()
axes = fig.add_subplot(111,projection="3d")
mesh.add_plot(axes)
mesh.find_node(axes,showindex=True,fontcolor="y",fontsize=12)
plt.show()
'''

# 一维区间网格
'''
import numpy as np 
import matplotlib.pyplot as plt
from fealpy.mesh import IntervalMesh

node = np.array([[0], [0.5], [1]], dtype=np.float64) # (NN, 1) 
cell = np.array([[0, 1], [1, 2]], dtype=np.int_) # (NN, 2) 
mesh = IntervalMesh(node, cell)
mesh.uniform_refine(n=2) # 加密

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()
'''

# Intervalmesh 嵌入到高一维的空间
'''
import numpy as np 
import matplotlib.pyplot as plt 
from fealpy.mesh import IntervalMesh

mesh = IntervalMesh.from_circle_boundary(n=100)
fig = plt.figure()
axes = plt.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True,fontsize=8)
plt.show()
'''

# 非结构三角形网格

import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh

i = 0

if i ==0:
    mesh = TriangleMesh.from_one_triangle(meshtype="iso")
if i ==1:
    mesh = TriangleMesh.from_unit_square(nx=5,ny=5)
if i ==2:    
    def threshold(p):
        x = p[..., 0]
        y = p[..., 1]
        return (x > 0.0) & (y < 0.0)
    mesh = TriangleMesh.from_box(box=[-1, 1, -1, 1], threshold=threshold)
if i ==3:
    mesh = TriangleMesh.from_unit_circle_gmsh(0.2)
if i == 4:
    vertices = np.array([(0, 0), (1, 0), (0.5, 1)])
    h = 0.1
    mesh = TriangleMesh.from_polygon_gmsh(vertices, h) 


fig = plt.figure()
axes = plt.gca()
mesh.add_plot(axes)
c = mesh.circumcenter()
mesh.find_node(axes, showindex=True,fontsize=7)
mesh.find_edge(axes, showindex=True,color='b', marker='v', fontsize=20, fontcolor='b')
mesh.find_cell(axes, showindex=True,color='k', marker='s', fontsize=20, fontcolor='y')
plt.show()


# 基于distmesh算法生成三角形网格示例
'''
import numpy as np
import matplotlib.pyplot as plt
from fealpy.geometry import LShapeDomain, CircleDomain
from fealpy.mesh import TriangleMesh

hmin = 0.05
hmax = .2
def sizing_function(p, *args):
    fd = args[0]
    x = p[:, 0]
    y = p[:, 1]
    h = hmin + np.abs(fd(p))*0.1
    h[h>hmax] = hmax 
    return h
domain = CircleDomain(fh=sizing_function)
domain.hmin = hmin
 
mesh = TriangleMesh.from_domain_distmesh(domain, maxit=1000)
fig, axes = plt.subplots()
mesh.add_plot(axes)
plt.show()
'''

# 嵌入到三维空间中的三角形网格
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fealpy.mesh import TriangleMesh

i = 1
if i == 0:
    mesh = TriangleMesh.from_torus_surface(5, 1, 30, 30)
elif i == 1:
    mesh = TriangleMesh.from_unit_sphere_surface(refine=3)

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
mesh.add_plot(axes)
plt.show()
'''

# 三角形网格上的形函数
'''
from fealpy.mesh import TriangleMesh    
TriangleMesh.show_lattice(1)
TriangleMesh.show_shape_function(2, funtype='L')
TriangleMesh.show_grad_shape_function(2, funtype='L')
'''

# 四边形网格
'''
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import QuadrangleMesh
from fealpy.mesh import TriangleMesh

i = 2
if i == 0:
    mesh = QuadrangleMesh.from_one_quadrangle(meshtype='square')  
    
if i == 1:
    mesh = QuadrangleMesh.from_unit_square(
    nx=10, ny=10, threshold=None)
    
if i == 2:
    mesh = TriangleMesh.from_unit_circle_gmsh(0.1)
    mesh = QuadrangleMesh.from_triangle_mesh(mesh)


fig, axes = plt.subplots()
mesh.add_plot(axes)
plt.show()
'''

# 多边形网格
'''
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import PolygonMesh, TriangleMesh

i = 4
if i == 0:
    mesh = PolygonMesh.from_one_triangle(meshtype='iso')
    
elif i == 1:
    mesh = PolygonMesh.from_one_square()
    
elif i == 2:
    mesh = PolygonMesh.from_one_pentagon()
    
elif i == 3:
    mesh = PolygonMesh.from_one_hexagon()
    
elif i == 4:
    vertices = np.array([(0, 0), (1, 0), (0.5, 1)])
    h = 0.1
    tmesh = TriangleMesh.from_polygon_gmsh(vertices, h)
    mesh = PolygonMesh.from_triangle_mesh_by_dual(tmesh, bc=True)

fig, axes = plt.subplots()
mesh.add_plot(axes, cellcolor=[0.5, 0.9, 0.45], edgecolor='k')
plt.show()
'''

# 四面体网格
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fealpy.mesh import TetrahedronMesh

i = 2
if i == 0:
    mesh = TetrahedronMesh.from_one_tetrahedron(meshtype='iso')

if i == 1:
    mesh = TetrahedronMesh.from_cylinder_gmsh(1, 5, 0.3)

if i == 2:
    points = np.array([
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        ])

    facets = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
        ])   
    mesh = TetrahedronMesh.from_meshpy(points, facets, 0.5)

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
mesh.add_plot(axes)
plt.show()
'''


# 六面体网格
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fealpy.mesh import TetrahedronMesh
from fealpy.mesh import HexahedronMesh

i = 1

if i==0:
    def threshold(p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        return (x > 0.0) & (y < 0.0) & (z > 0.0)
    box = [-1, 1, -1, 1, -1, 1]
    mesh = HexahedronMesh.from_box(
        box=box, nx=10, ny=10, nz=10, threshold=threshold)
    
if i==1:
    tmesh = TetrahedronMesh.from_one_tetrahedron(meshtype='equ')
    mesh = HexahedronMesh.from_tetrahedron_mesh(tmesh)
    
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
mesh.add_plot(axes)
plt.show()
'''

# 一维EdgeMesh
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fealpy.mesh import EdgeMesh

mesh = EdgeMesh.from_tower()

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
mesh.add_plot(axes)
plt.show()
'''

