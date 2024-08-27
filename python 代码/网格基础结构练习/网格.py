import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh

mesh = TriangleMesh.from_box([0,1,0,1],nx=2,ny=2)

'''
fig,axes = plt.subplots()
mesh.add_plot(axes, cellcolor='g')
mesh.find_node(axes, showindex=True,color='r', marker='o', fontsize=10, fontcolor='r')
mesh.find_edge(axes, showindex=True,color='b', marker='v', fontsize=10, fontcolor='b')
mesh.find_cell(axes, showindex=True,color='k', marker='s', fontsize=10, fontcolor='y')
plt.show()
'''

node = mesh.entity("node")
edge = mesh.entity("edge")
cell = mesh.entity("cell")
face = mesh.entity("face")


NN = mesh.number_of_nodes()
NE = mesh.number_of_edges()
NF = mesh.number_of_faces()
NC = mesh.number_of_cells()

t = mesh.edge_tangent()
ut = mesh.edge_unit_tangent()

n = mesh.edge_normal()
un = mesh.edge_unit_normal()

h = mesh.entity_measure("edge")
area = mesh.entity_measure("cell")
qf = mesh.integrator(3)
bcs,ws = qf.get_quadrature_points_and_weights()

cell2cell = mesh.ds.cell_to_cell()
node2node = mesh.ds.node_to_node()
node2cell = mesh.ds.node_to_cell()
cell2edge = mesh.ds.cell_to_edge()

isBdNode = mesh.ds.boundary_node_flag()
isBdEdge = mesh.ds.boundary_edge_flag()
isBdFace = mesh.ds.boundary_face_flag()
isBdCell = mesh.ds.boundary_cell_flag()

nidx = mesh.ds.boundary_node_index()
eidx = mesh.ds.boundary_edge_index()
fidx = mesh.ds.boundary_face_index()
cidx = mesh.ds.boundary_cell_index()

p = 5
ipoint = mesh.interpolation_points(p)   # p次插值点的坐标
cell2ipoint = mesh.cell_to_ipoint(p)    # 单元到p次插值点的映射数组
edge2ipoint = mesh.edge_to_ipoint(p)

np1 = mesh.number_of_local_ipoints(p)   # 单元插值点的个数
NP = mesh.number_of_global_ipoints(p)   # 全局插值点的个数
fig, axes= plt.subplots()
'''
mesh.add_plot(axes)
mesh.find_node(axes, node=ipoint, showindex=True)
plt.show()
'''





