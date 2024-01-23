import numpy as np
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.mesh.polygon_mesh import PolygonMesh

from fealpy.functionspace.lagrange_fe_space import LagrangeFESpace

from fealpy.fem.scalar_mass_integrator import ScalarMassIntegrator
from fealpy.fem.bilinear_form import BilinearForm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

node = np.array([[0.0, 0.0], [0.0, 0.5], [0.0, 1.0],
                 [0.5, 0.0], [0.5, 0.5], [0.5, 1.0],
                 [1.0, 0.0], [1.0, 0.5], [1.0, 1.0]], dtype=np.float64)
cell = np.array([0, 3, 4, 1, 3, 6, 7, 4, 1, 4, 5, 2, 4, 7, 8, 4, 8, 5], dtype=np.int_)
cellLocation = np.array([0, 4, 8, 12, 15, 18], dtype=np.int_)
mesh = PolygonMesh(node=node, cell=cell, cellLocation=cellLocation)
#mesh = PolygonMesh.from_box([0,1,0,1],nx=1,ny=1)

normal = mesh.edge_unit_normal()
flag = mesh.ds.cell_to_edge_sign()
cell2edge = mesh.ds.cell_to_edge()
outnormal = [[normal[edge] for edge in cell] for cell in cell2edge]

print(cell2edge)
"""
print(normal)
print(flag.toarray())
print(len(outnormal))
print(outnormal) 


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, color='r', marker='o', markersize=8, fontsize=16, fontcolor='r')
mesh.find_cell(axes, showindex=True, color='b', marker='o', markersize=8, fontsize=16, fontcolor='b')
mesh.find_edge(axes, showindex=True, color='g', marker='o', markersize=8, fontsize=16, fontcolor='g')
plt.show()


print("\n")
my_list = list()
for i in range(5):
    if i < 3:
        for j in range(4):
            k = cell2edge[i][j]
            if flag.toarray()[i,k] == True:
                my_list.append(normal[k,:])
            else:
                my_list.append(-1 * normal[k,:])
    else:
        for j in range(3):
            k = cell2edge[i][j]
            if flag.toarray()[i,k] == True:
                my_list.append(normal[k,:])
            else:
                my_list.append(-1 * normal[k,:])
print(my_list)
"""
u1 = np.zeros((3,4,2))
u2 = np.zeros((2,3,2))
for i in range(5):
    if i < 3:
        for j in range(4):
            k = cell2edge[i][j]
            if flag.toarray()[i,k] == True:
                u1[i,j,:] = normal[k,:]
            else:
                u1[i,j,:] = -1 * normal[k,:]        
    else:
        for j in range(3):
            k = cell2edge[i][j]
            if flag.toarray()[i,k] == True:
                u2[i-3,j,:] = normal[k,:]
            else:
                u2[i-3,j,:] = -1 * normal[k,:]
print(u1)
print("\n")
print(u2)
