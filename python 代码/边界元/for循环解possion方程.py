# 边界元法解poisson方程
# 导入所需要的库
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import*

# 创建函数的信息
class PoissonModelConstantDirichletBC2d:
    """
    二维常单元纯 Dirichlet 边界 Poisson 方程模型
    """
    # 函数定义域
    def domain(self):
        return np.array([0, 1, 0, 1])

    # 函数真解
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.sin(pi * x) * np.sin(pi * y)
        return val
    # 函数右端项
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = -2 * pi * pi * np.sin(pi * x) * np.sin(pi * y)
        return val

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = - pi * np.cos(pi * x) * np.sin(pi * y)
        val[..., 1] = - pi * np.sin(pi * x) * np.cos(pi * y)
        return val

    def dirichlet(self, p):
        return self.solution(p)

# 创建对象
pde = PoissonModelConstantDirichletBC2d()

# 创建网格
box = pde.domain()
nx = 5
ny = 5
mesh = TriangleMesh.from_box(box,nx,ny)


maxite = 4


for k in range(maxite):
    
    # 获取网格的信息
    NN = mesh.number_of_nodes()  # 获取节点个数
    NE = mesh.number_of_edges()  # 获取变的条数
    NC = mesh.number_of_cells()  # 获取单元个数

    node = mesh.entity("node")   # 获取节点的坐标
    edge = mesh.entity("edge")   # 获取边的坐标（按点的全局编号）
    cell = mesh.entity("cell")   # 获取单元的坐标（按点的全局编号）

    bd_node_idx = mesh.ds.boundary_node_index() # 获得边界点的全局编号
    bd_cell_idx = mesh.ds.boundary_cell_index() # 获得边界单元的全局编号
    bd_edge_idx = mesh.ds.boundary_edge_index() # 获得边界边的全局编号
    bd_edge = mesh.ds.boundary_edge()           # 获得边界边的坐标
    bd_edge_flag = mesh.ds.boundary_edge_flag() # 判断是否为边界边
    bd_edge_measure = mesh.entity_measure('face', index=bd_edge_idx) # 计算边界边的边长
    cell_measure = mesh.entity_measure('cell')  # 计算每个单元的面积

    bd_val = pde.dirichlet(node)  # 求精确解
    uh = np.zeros_like(bd_val)  # 构造近视解形式

    # 计算边界中心点的坐标
    Pj = mesh.entity_barycenter('edge')[bd_edge_flag]

    # 计算离散边界中点法向导数值（这里给定矩阵大小）
    G = np.zeros((bd_edge.shape[0], bd_edge.shape[0]), dtype=float)
    H = np.zeros_like(G)
    B = np.zeros(bd_edge.shape[0])

    # 构建边界 Gauss 积分子，获取积分点重心坐标及其权重
    edge_qf = mesh.integrator(q=2, etype='edge')  # 定义积分对象
    edge_bcs,edge_ws = edge_qf.get_quadrature_points_and_weights()

    # 计算积分点笛卡尔坐标，q 为每个边界上积分点，j 为边界端点，f 为边界，d 为空间维数
    edge_ps = np.einsum('qj, fjd->fqd', edge_bcs, node[bd_edge])

    # 单元 Gauss 积分子
    cell_qf = mesh.integrator(q=3, etype='cell')
    cell_bcs, cell_ws = cell_qf.get_quadrature_points_and_weights()
    cell_ps = np.einsum('qj, ejd->eqd', cell_bcs, node[cell])  # 求高斯积分点

    x1 = node[bd_edge[:,0]]
    x2 = node[bd_edge[:,1]]

    # hij的计算
    for i in range(bd_edge.shape[0]):
        xi = Pj[i]
        c = np.sign((x1[:,0]-xi[0])*(x2[:,1]-xi[1]) - (x2[:,0]-xi[0])*(x1[:,1]-xi[1]))
        hij = c * np.abs((xi[0] - Pj[:, 0]) * (x2[:, 1] - x1[:, 1]) - (xi[1] - Pj[:, 1]) * (x2[:, 0] - x1[:, 0])) / bd_edge_measure
        # rij的计算
        rij = np.sqrt(np.sum((edge_ps - xi) ** 2, axis=-1))
        H[..., i, :] = np.einsum('f,f,q,fq->f', -bd_edge_measure, hij, edge_ws, 1 / rij ** 2) / 2 / np.pi
        G[..., i, :] = bd_edge_measure / 2 / np.pi * np.einsum('q, fq ->f', edge_ws, np.log(1 / rij))
        # 计算源积分值
        cell_rij = np.sqrt(np.sum((cell_ps - xi) ** 2, axis=-1))  
        b = pde.source(cell_ps)
        B[i] = np.einsum('e,b,eb,eb->', cell_measure, cell_ws, b, np.log(1 / cell_rij)) / np.pi / 2

        # 填充对角线元素
        np.fill_diagonal(H,0.5)
        np.fill_diagonal(G,(bd_edge_measure* (np.log(2 / bd_edge_measure) + 1))/(np.pi*2))

    # 求解方程组Gq=Hu+B，获得节点的导数值
    bd_u_val = pde.dirichlet(Pj)
    bd_un_val = np.linalg.solve(G,H @ bd_u_val + B)

    # 计算内部点的函数值
    internal_node = node[~mesh.ds.boundary_node_flag()]
    uh[bd_node_idx] = bd_val[bd_node_idx]
    internal_idx = np.arange(NN)[~mesh.ds.boundary_node_flag()]

    # 计算内部节点相关矩阵元素值
    for i in range(internal_node.shape[0]):
        Hi = 0
        Mi = 0
        xi = internal_node[i]
        c = np.sign((x1[:, 0] - xi[0]) * (x2[:, 1] - xi[1]) - (x2[:, 0] - xi[0]) * (x1[:, 1] - xi[1]))
        hij = c * np.abs(
            (xi[0] - Pj[:, 0]) * (x2[:, 1] - x1[:, 1]) - (xi[1] - Pj[:, 1]) * (x2[:, 0] - x1[:, 0])) / bd_edge_measure

        rij = np.sqrt(np.sum((edge_ps - xi) ** 2, axis=-1))
        Hi = np.einsum('f...,f,f,q,fq->...', bd_u_val, -bd_edge_measure, hij, edge_ws, 1 / rij ** 2) / 2 / np.pi
        Mi = np.einsum('f,f...,q,fq->...', bd_un_val, bd_edge_measure, edge_ws, np.log(1 / rij)) / 2 / np.pi
        cell_rij = np.sqrt(np.sum((cell_ps - xi) ** 2, axis=-1))
        b = -2 * np.pi ** 2 * np.sin(np.pi * cell_ps[..., 0]) * np.sin(np.pi * cell_ps[..., 1])
        Bi = np.einsum('f,q,fq,fq->', cell_measure, cell_ws, b, np.log(1 / cell_rij)) / np.pi / 2
        uh[internal_idx[i]] = Mi - Hi - Bi

    # 误差计算
    real_solution = pde.solution(node)  
    h = np.max(mesh.entity_measure('cell'))
    errorMatrix = np.sqrt(np.sum((uh - real_solution) ** 2) * h)
    
    if errorMatrix > 2e-5:
        mesh.uniform_refine(1)
    else:
        break

    print(f'迭代{k}次，结果如下：')
    print("误差：\n", errorMatrix)
    errorMatrix = 0

