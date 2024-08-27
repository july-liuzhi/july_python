from fealpy.mesh import TetrahedronMesh
mesh = TetrahedronMesh.from_one_tetrahedron('iso')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
mesh.add_plot(axes)

import numpy as np

node = mesh.entity('node') 
cell = mesh.entity('cell') 


NC = mesh.number_of_cells() 
GD = mesh.geo_dimension()

localFace = np.array([[1, 2, 3],  
                      [0, 3, 2], 
                      [0, 1, 3], 
                      [0, 2, 1]]) 

Lambda = np.zeros((NC,4,GD))


v01 = node[cell[:,1]] - node[cell[:,0]]
v02 = node[cell[:,2]] - node[cell[:,0]]
v03 = node[cell[:,3]] - node[cell[:,0]]
volune = np.sum(v03*np.cross(v01,v02),axis=1)/6
for i in range(4):
    k,j,m = localFace[i]
    vmj = node[cell[:,j]] - node[cell[:,k]]
    vmk = node[cell[:,m]] - node[cell[:,k]]
    Lambda[:,i] = np.cross(vmj,vmk)/(6*volune.reshape(1,-1))
print(Lambda)
