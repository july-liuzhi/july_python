import numpy as np                                                          
import matplotlib.pyplot as plt
from fealpy.mesh import QuadrangleMesh  # 从FEALPy导入QuadrangleMesh                                    
node = np.array([  #节点信息，给出点的坐标，形状为(NN, 2)
    (0.0, 0.0), # 0 号点                                                    
    (0.0, 1.0), # 1 号点                                                    
    (0.0, 2.0), # 2 号点                                                    
    (1.0, 0.0), # 3 号点                                                    
    (1.0, 1.0), # 4 号点                                                    
    (1.0, 2.0), # 5 号点                                                    
    (2.0, 0.0), # 6 号点                                                    
    (2.0, 1.0), # 7 号点                                                    
    (2.0, 2.0), # 8 号点                                                    
    ], dtype=np.float64)                                                    
cell = np.array([  #单元信息，给出构成每个单元的四个点的编号，形状为(NC, 4)
    (0, 3, 4, 1), # 0 号单元                                                
    (1, 4, 5, 2), # 1 号单元                                                
    (3, 6, 7, 4), # 2 号单元                                                
    (4, 7, 8, 5), # 3 号单元                                                
    ], dtype=np.int_)                                                       
                                                                            
mesh = QuadrangleMesh(node, cell)  #建立四边形网格                                         
#画图                                                              
fig = plt.figure()                                                          
axes = fig.gca()                                                            
mesh.add_plot(axes)                                                         
mesh.find_node(axes, showindex=True)#展示节点的编号                         
mesh.find_edge(axes, showindex=True)#展示边的编号                         
mesh.find_cell(axes, showindex=True)#展示单元的编号                         
plt.show()    


# 四叉树网格结构
from fealpy.mesh import Quadtree
import numpy as np
import matplotlib.pyplot as plt 
node = np.array([  #节点信息，给出点的坐标，形状为(NN, 2)
    (0.0, 0.0), # 0 号点                                                    
    (0.0, 1.0), # 1 号点                                                    
    (0.0, 2.0), # 2 号点                                                    
    (1.0, 0.0), # 3 号点                                                    
    (1.0, 1.0), # 4 号点                                                    
    (1.0, 2.0), # 5 号点                                                    
    (2.0, 0.0), # 6 号点                                                    
    (2.0, 1.0), # 7 号点                                                    
    (2.0, 2.0), # 8 号点                                                    
    ], dtype=np.float64)                                                    
cell = np.array([  #单元信息，给出构成每个单元的四个点的编号，形状为(NC, 4)
    (0, 3, 4, 1), # 0 号单元                                                
    (1, 4, 5, 2), # 1 号单元                                                
    (3, 6, 7, 4), # 2 号单元                                                
    (4, 7, 8, 5), # 3 号单元                                                
    ], dtype=np.int_)  
Quadtree = Quadtree(node,cell)
isMarkedCell = [True,False,False,True]
Quadtree.refine_1(isMarkedCell=isMarkedCell)
fig = plt.figure()                                                          
axes = fig.gca()                                                            
Quadtree.add_plot(axes)                                                      
Quadtree.find_node(axes, showindex=True)#展示节点的编号                                       
Quadtree.find_cell(axes, showindex=True)#展示单元的编号                     
plt.show()