# 导入所需要的模块
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh

node = np.array([
                [0,0],
                [5,0],
                [5,5],
                [0,5]
])

cell = np.array([
                [1,2,0],
                [3,0,2]
])

mesh = TriangleMesh(node,cell)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes,showindex = True,fontsize=40)
mesh.find_edge(axes,showindex = True,fontsize=40)
mesh.find_cell(axes,showindex = True,fontsize=40)
plt.show()
