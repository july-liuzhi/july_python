import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh

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
mesh = TriangleMesh(node,cell)
# 对网格进行一致加密
# mesh.uniform_refine(2)

fig = plt.figure()
axes = plt.gca()
mesh.add_plot(axes)
mesh.find_node(axes,showindex=True,fontsize=20)
mesh.find_edge(axes, showindex=True,fontsize=20)
mesh.find_cell(axes, showindex=True,fontsize=20)
plt.show()

# 获取网格的节点个数
NN = mesh.number_of_nodes()
# 9

#获取网格的单元个数
NC = mesh.number_of_cells()
# 8

# 获取网格的边个数
NE = mesh.number_of_edges()
# 16

# 获取网格面的个数
NF = mesh.number_of_faces()
# 16

# 获取网格的几何维数
GD = mesh.geo_dimension()
# 2

# 获取网格的拓扑维数
TD = mesh.top_dimension()
# 2

# 获取网格的实体信息
# 获取网格的各个节点的坐标
node = mesh.entity("node")

# 获取网格各个边
edge = mesh.entity("edge")

# 获取网格的各个面
face = mesh.entity("face")

# 获取网格的各个单元
cell = mesh.entity("cell")

# 获取一些其他信息
# 获取每条边的重心坐标
ebc = mesh.entity_barycenter('edge')

# 获取每个单元的重心坐标
cbc = mesh.entity_barycenter('cell')

# 获取每条边的长度
eh = mesh.entity_measure('edge')

# 获取每个单元的面积
area = mesh.entity_measure('cell')

# 1、单元与其他的关系
# 获取单元与单元的邻接关系
cell2cell = mesh.ds.cell_to_cell()

# 获取单元与面的邻接关系
cell2face = mesh.ds.cell_to_face()

# 获取单元与边的邻接关系
cell2edge = mesh.ds.cell_to_edge()

# 获取单元与节点的邻接关系
cell2node = mesh.ds.cell_to_node()

# 2、获取面与其他的关系
# 获取面与单元之间的关系
face2cell = mesh.ds.face_to_cell()

# 获取面与面的邻接关系(二维三角形网格无)
# face2face = mesh.ds.face_to_face()

# 获取面与边的邻接关系（二维三角形网格无）
# face2edge = mesh.ds.face_to_edge()

# 获取面与节点的邻接关系
face2node = mesh.ds.face_to_node()

# 3、边与其他的邻接关系
# 获取边与单元的邻接关系
edge2cell = mesh.ds.edge_to_cell()


# 获取边与面的邻接关系（二维三角形网格无）
# edge2face = mesh.ds.edge_to_face()

# 获取边与边的邻接关系
edge2edge = mesh.ds.edge_to_edge()

# 获取边与点的邻接关系
edge2node = mesh.ds.edge_to_node()

# 4、节点与其他的关系
# 获取节点与单元的邻接关系
node2cell = mesh.ds.node_to_cell()

# 获取节点与面的邻接关系
node2face = mesh.ds.node_to_face()

# 获取节点与边的邻接关系
node2edge = mesh.ds.node_to_edge()

# 获取节点与节点的邻接关系
node2node = mesh.ds.node_to_node()

# 4、获取边界元素的方法
# 一维逻辑数组，标记边界节点
isBdnode = mesh.ds.boundary_node_flag()

# 一维逻辑数组，标记边界边
isBdedge = mesh.ds.boundary_edge_flag()

# 一维逻辑数组，标记边界面
isBdface = mesh.ds.boundary_face_flag()

# 一维逻辑数组，标记边界单元
isBdcell = mesh.ds.boundary_cell_flag()

# 一维整数数组，边界节点全局编号
bdNodeIdx = mesh.ds.boundary_node_index()

# 一维整数数组，边界边全局编号
bdEdgeIdx = mesh.ds.boundary_node_index()

# 一维整数数组，边界面全局编号
bdFaceIdx = mesh.ds.boundary_face_index()

# 一维整数数组，边界单元全局编号
bdCellIdx = mesh.ds.boundary_cell_index()

# 二维数组，边界边的坐标
bd_edge = mesh.ds.boundary_edge()

# 二维数组，边界面的坐标
bd_face = mesh.ds.boundary_face()

# 二维数组，边界单元的坐标
bd_cell = mesh.ds.boundary_cell()

# 求边重（中）心坐标返回二维数组(后面可以指定哪些范围)
Pj = mesh.entity_barycenter('edge')[isBdedge]  
print(Pj)