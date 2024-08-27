import numpy as np
from fealpy.mesh import TriangleMesh
node = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]], dtype=np.float64) # 节点信息，给出点的坐标，形状为(NN, 2)

cell = np.array([
    [1, 2, 0], 
    [3, 0, 2]], dtype=np.int_) 
    #单元信息，给出构成每个单元的三个点的编号，形状为(NC, 3)，
mesh = TriangleMesh(node, cell) #建立三角形网格
NC = mesh.number_of_cells()
node = mesh.entity('node')
cell = mesh.entity('cell')

v0 = node[cell[:, 2], :] - node[cell[:, 1], :] # $x_2 - x_1$
v1 = node[cell[:, 0], :] - node[cell[:, 2], :] # $x_0 - x_2$
v2 = node[cell[:, 1], :] - node[cell[:, 0], :] # $x_1 - x_0$
nv = np.cross(v2, v0)
Dlambda = np.zeros((NC, 3, 2), dtype=np.float64)
length = nv 
W = np.array([[0, 1], [-1, 0]], dtype=np.int_)
Dlambda[:,0,:] = v0@W/length.reshape(-1, 1)
Dlambda[:,1,:] = v1@W/length.reshape(-1, 1)
Dlambda[:,2,:] = v2@W/length.reshape(-1, 1)
print('Dlambda:\n',Dlambda)