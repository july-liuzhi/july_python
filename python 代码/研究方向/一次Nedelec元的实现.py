import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh
import numpy as np
from sympy import*
from scipy.sparse import*

mesh = TriangleMesh.from_box(box=[0,1,0,1],nx=2,ny=2)

'''
fig = plt.figure()
axes = plt.gca()
mesh.add_plot(axes)
mesh.find_node(axes,showindex=True,fontsize=7)
mesh.find_edge(axes,showindex = True,fontsize=10)
# mesh.find_cell(axes,showindex = True,fontsize=7)
plt.show()
'''

# 获取网格数据结构
NN = mesh.number_of_nodes()
NE = mesh.number_of_edges()
NC = mesh.number_of_cells()
node = mesh.entity("node")
cell = mesh.entity("cell")
edge = mesh.entity("edge")
cm = mesh.entity_measure("cell")


# 计算重心坐标梯度
v0 = node[cell[:,2],:] - node[cell[:,1],:]
v1 = node[cell[:,0],:] - node[cell[:,2],:]
v2 = node[cell[:,1],:] - node[cell[:,0],:]

nv = np.cross(v2,v0)

Dlambda = np.zeros((NC, 3, 2), dtype=np.float64)
length = nv 
W = np.array([[0, 1], [-1, 0]], dtype=np.int_)
Dlambda[:,0,:] = v0@W/length.reshape(-1, 1)
Dlambda[:,1,:] = v1@W/length.reshape(-1, 1)
Dlambda[:,2,:] = v2@W/length.reshape(-1, 1)

Dlambda1 = np.zeros((NC,3,1),dtype=np.float64)

x0 =2*np.cross(Dlambda[:,0,:],Dlambda[:,1,:])
x1 =2*np.cross(Dlambda[:,1,:],Dlambda[:,2,:])
x2 =2*np.cross(Dlambda[:,2,:],Dlambda[:,0,:])

'''
# 分量形式实现
x0 =2*(Dlambda[:,0,0]*Dlambda[:,1,1]-Dlambda[:,0,1]*Dlambda[:,1,0])
x1 =2*(Dlambda[:,1,0]*Dlambda[:,2,1]-Dlambda[:,1,1]*Dlambda[:,2,0])
x2 =2*(Dlambda[:,2,0]*Dlambda[:,0,1]-Dlambda[:,2,1]*Dlambda[:,0,0])
'''

Dlambda1[:,0,:]=x0.reshape(-1,1)
Dlambda1[:,1,:]=x1.reshape(-1,1)
Dlambda1[:,2,:]=x2.reshape(-1,1)

K = np.array([
            [0.6666666666666670,	0.1666666666666670,     0.1666666666666670,	0.3333333333333330],
            [0.1666666666666670,	0.6666666666666670,     0.1666666666666670,	0.3333333333333330],
            [0.1666666666666670,	0.1666666666666670,     0.6666666666666670,	0.3333333333333330]],dtype=np.float64)
bcs = K[:,0:3]
w = np.array([0.3333333333333330,0.3333333333333330,0.3333333333333330])
Dlambda1=Dlambda1[np.newaxis,:]
gphi = np.repeat(Dlambda1,3,axis=0)

A = np.einsum('q,qcid,qcjd,c-> cij',w,gphi,gphi,cm)

cell2edge = mesh.ds.cell_to_edge()

# 全局映射
I = np.broadcast_to(cell2edge[:,:,None],shape=A.shape)
J = np.broadcast_to(cell2edge[:,None,:],shape=A.shape)

# 组装成全局刚度矩阵
A = csr_matrix((A.flat,(I.flat,J.flat)),shape=(NE,NE))

print(A.toarray())
# 组装单元质量矩阵 
bcs = bcs[:,np.newaxis,:]
phi = np.zeros((3,NC,3,2))
y0 = np.einsum('qc,qcd ->qcd',bcs[...,0],Dlambda[None,...][...,1,:])-np.einsum('qc,qcd ->qcd',bcs[...,1],Dlambda[None,...][...,0,:])
y1 = np.einsum('qc,qcd ->qcd',bcs[...,1],Dlambda[None,...][...,2,:])-np.einsum('qc,qcd ->qcd',bcs[...,2],Dlambda[None,...][...,1,:])
y2 = np.einsum('qc,qcd ->qcd',bcs[...,2],Dlambda[None,...][...,0,:])-np.einsum('qc,qcd ->qcd',bcs[...,0],Dlambda[None,...][...,2,:])
phi[...,0,:]=y0
phi[...,1,:]=y1
phi[...,2,:]=y2
B = np.einsum('q,qcid,qcjd,c -> cij',w,phi,phi,cm)

# 组装全局质量矩阵
B = csr_matrix((B.flat,(I.flat,J.flat)),shape=(NE,NE))
cell2node = mesh.ds.cell_to_node()


