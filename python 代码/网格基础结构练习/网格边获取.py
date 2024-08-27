import  numpy as np
from fealpy.mesh import QuadrangleMesh
import matplotlib.pyplot as plt
node = np.array([
    [0,0],
    [0,1],
    [0,2],
    [0,3],
    [1,0],
    [1,1],
    [1,2],
    [1,3],
    [2,0],
    [2,1],
    [2,2],
    [2,3],
    [3,0],
    [3,1],
    [3,2],
    [3,3]
])
cell = np.array([
    [0,4,5,1],
    [4,8,9,5,],
    [8,12,13,9],
    [1,5,6,2],
    [5,9,10,6],
    [9,13,14,10],
    [2,6,7,3],
    [6,10,11,7],
    [10,14,15,11]
])
mesh = QuadrangleMesh(node,cell)
# 画出图形
'''
fig,axes = plt.subplots()
mesh.add_plot(axes,cellcolor = 'g')
mesh.find_node(axes,showindex=True,color="r",marker="o",fontsize=15,fontcolor="r")
mesh.find_edge(axes,showindex=True,color="b",marker="v",fontsize=20,fontcolor="b")
mesh.find_cell(axes,showindex=True,color="k",marker="s",fontsize=25,fontcolor="k")
plt.show()
'''
localedge = np.array([
    [0,1],
    [1,2],
    [2,3],
    [3,0]
])
totaledge = cell[:,localedge].reshape(-1,2)
stotaledge = np.sort(totaledge,axis=-1)

_,i0,j = np.unique(stotaledge,return_index=True,return_inverse=True,axis=0)
print(i0)
print("\n")

NE = mesh.number_of_edges()
i1 = np.zeros(NE)
i1[j] = range(36)
edge2cell = np.zeros((NE,4))
edge2cell[:,0]= i0//4 # 左边对应的单元的全局编号
edge2cell[:,1]= i1//4 # 右边对应的单元的全局编号
edge2cell[:,2]= i0% 4 # 在左边单元的局部编号
edge2cell[:,3]= i1% 4 # 在右边单元的局部编号
print(edge2cell)
edge2cell1 = mesh.ds.edge_to_cell()
# 检验是否和fealpy中一样
print(edge2cell1==edge2cell)