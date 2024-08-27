import numpy as np
import itertools

from fealpy.functionspace import BernsteinFESpace 
from fealpy.functionspace.Function import Function
from fealpy.quadrature import FEMeshIntegralAlg

from fealpy.functionspace.femdof import multi_index_matrix2d, multi_index_matrix3d
from fealpy.mesh import TetrahedronMesh
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

from fealpy.decorator import cartesian, barycentric

class FirstNedelecDof3d():
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.multiindex = multi_index_matrix3d(p)

    def number_of_local_dofs(self, doftype='all'):
        p = self.p
        if doftype == 'all': # number of all dofs on a cell 
            return 6*(p+1)+4*(p+1)*p+(p+1)*(p-1)*p//2
        elif doftype in {'cell', 3}: # number of dofs on each edge 
            return (p+1)*(p-1)*p//2
        elif doftype in {'face', 2}: # number of dofs inside the cell 
            return (p+1)*p
        elif doftype in {'edge', 1}: # number of dofs on each edge 
            return p+1
        elif doftype in {'node', 0}: # number of dofs on each node
            return 0

    def number_of_global_dofs(self):
        NC = self.mesh.number_of_cells()
        NF = self.mesh.number_of_faces()
        NE = self.mesh.number_of_edges()
        edof = self.number_of_local_dofs(doftype='edge')
        fdof = self.number_of_local_dofs(doftype='face')
        cdof = self.number_of_local_dofs(doftype='cell')
        return NE*edof + NC*cdof + NF*fdof

    def edge_to_dof(self, index=np.s_[:]):
        NE = self.mesh.number_of_edges()
        edof = self.number_of_local_dofs(doftype='edge')
        return np.arange(NE*edof).reshape(NE, edof)[index]

    def face_to_internal_dof(self, index=np.s_[:]):
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        edof = self.number_of_local_dofs(doftype='edge')
        fdof = self.number_of_local_dofs(doftype='face')
        return np.arange(NE*edof, NE*edof+NF*fdof).reshape(NF, fdof)[index]

    def face_to_dof(self):
        p = self.p
        fldof = self.number_of_local_dofs('face')
        eldof = self.number_of_local_dofs('edge')
        fdof = fldof + eldof*3
        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()
        e2dof = self.edge_to_dof()

        NF = self.mesh.number_of_faces()
        NE = self.mesh.number_of_edges()
        f2e = self.mesh.ds.face_to_edge()
        edge = self.mesh.entity('edge')
        face = self.mesh.entity('face')

        f2d = np.zeros((NF, fdof), dtype=np.int_)
        f2d[:, :eldof*3] = e2dof[f2e].reshape(NF, eldof*3)
        s = [1, 0, 0]
        for i in range(3):
            flag = face[:, s[i]] == edge[f2e[:, i], 0]
            f2d[flag, eldof*i:eldof*(i+1)] = f2d[flag, eldof*i:eldof*(i+1)][:, ::-1]
        f2d[:, eldof*3:] = self.face_to_internal_dof() 
        return f2d

    def cell_to_dof(self):
        p = self.p
        cldof = self.number_of_local_dofs('cell')
        fldof = self.number_of_local_dofs('face')
        fldof_2 = fldof//2
        eldof = self.number_of_local_dofs('edge')
        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()
        e2dof = self.edge_to_dof()
        f2dof = self.face_to_internal_dof()

        NC = self.mesh.number_of_cells()
        NF = self.mesh.number_of_faces()
        NE = self.mesh.number_of_edges()
        c2e = self.mesh.ds.cell_to_edge()
        c2f = self.mesh.ds.cell_to_face()
        edge = self.mesh.entity('edge')
        face = self.mesh.entity('face')
        cell = self.mesh.entity('cell')

        c2d = np.zeros((NC, ldof), dtype=np.int_)
        c2d[:, :eldof*6] = e2dof[c2e].reshape(NC, eldof*6)
        s = [0, 0, 0, 1, 1, 2]
        for i in range(6):
            flag = cell[:, s[i]] == edge[c2e[:, i], 0]
            c2d[flag, eldof*i:eldof*(i+1)] = c2d[flag, eldof*i:eldof*(i+1)][:, ::-1]

        if fldof > 0:
            locFace = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int_)
            midx2num = lambda a : (a[:, 1]+a[:, 2])*(1+a[:, 1]+a[:, 2])//2 + a[:, 2]

            midx = multi_index_matrix2d(p-1)
            perms = np.array(list(itertools.permutations([0, 1, 2])))
            indices = np.zeros((6, len(midx)), dtype=np.int_)
            for i in range(6):
                indices[i] = midx2num(midx[:, perms[i]])

            c2fp = self.mesh.ds.cell_to_face_permutation(locFace=locFace)

            perm2num = lambda a : a[:, 0]*2 + (a[:, 1]>a[:, 2]) 
            for i in range(4):
                #perm = np.argsort(c2fp[:, i], axis=1)
                perm =c2fp[:, i]
                pnum = perm2num(perm)
                N = eldof*6+fldof*i
                c2d[:, N:N+fldof_2] = f2dof[c2f[:, i, None], indices[None, pnum]]
                c2d[:, N+fldof_2:N+fldof] = f2dof[c2f[:, i, None], fldof_2 +
                        indices[None, pnum]]

        if cldof > 0:
            c2d[:, eldof*6+fldof*4:] = np.arange(NE*eldof+NF*fldof, gdof).reshape(NC, cldof) 
        return c2d

    @property
    def cell2dof(self):
        return self.cell_to_dof()

    def boundary_dof(self):
        eidx = self.mesh.ds.boundary_edge_index()
        e2d = self.edge_to_dof(index=eidx)
        return e2d.reshape(-1)

    def is_boundary_dof(self):
        bddof = self.boundary_dof()

        gdof = self.number_of_global_dofs()
        flag = np.zeros(gdof, dtype=np.bool_)

        flag[bddof] = True
        return flag

class FirstNedelecFiniteElementSpace3d():
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.dof = FirstNedelecDof3d(mesh, p)

        self.bspace = BernsteinFESpace(mesh, p)
        self.cellmeasure = mesh.entity_measure('cell')
        self.integralalg = FEMeshIntegralAlg(mesh, p+3, cellmeasure=self.cellmeasure)
        self.integrator = self.integralalg.integrator

        self.ftype=np.float_

    @barycentric
    def basis(self, bc):

        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        ldof = self.dof.number_of_local_dofs()
        cldof = self.dof.number_of_local_dofs("cell")
        fldof = self.dof.number_of_local_dofs("face")
        eldof = self.dof.number_of_local_dofs("edge")
        gdof = self.dof.number_of_global_dofs()
        glambda = mesh.grad_lambda()
        ledge = mesh.ds.localEdge

        c2esign = mesh.ds.cell_to_edge_sign() #(NC, 3, 2)

        l = np.zeros((4, )+bc[..., 0, None, None, None].shape, dtype=np.float_)
        l[0] = bc[..., 0, None, None, None]
        l[1] = bc[..., 1, None, None, None]
        l[2] = bc[..., 2, None, None, None]
        l[3] = bc[..., 3, None, None, None] #(NQ, NC, ldof, 2)

        #l = np.tile(l, (1, NC, 1, 1))

        # edge basis
        phi = self.bspace.basis(bc, p=p)
        multiIndex = self.mesh.multi_index_matrix(p, 3)
        val = np.zeros(bc.shape[:-1]+(NC, ldof, 3), dtype=np.float_)
        locEdgeDual = np.array([[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]])
        for i in range(6):
            flag = np.all(multiIndex[:, locEdgeDual[i]]==0, axis=1)
            phie = phi[:, :, flag] #(NQ, NC, eldof)
            c2esi = c2esign[:, i] #(NC, )
            v = l[ledge[i, 0]]*glambda[:, ledge[i, 1], None] - l[ledge[i, 1]]*glambda[:, ledge[i, 0], None]
            v[..., ~c2esi, :, :] *= -1 #(NQ, NC, eldof, 2)
            val[..., eldof*i:eldof*(i+1), :] = phie[..., None]*v

        # face basis
        if(p > 0):
            phi = self.bspace.basis(bc, p=p-1) #(NQ, NC, cldof)
            multiIndex = self.mesh.multi_index_matrix(p-1, 3)
            permcf = self.mesh.ds.cell_to_face_permutation()
            localFace = self.mesh.ds.localFace
            for i in range(4):

                flag = multiIndex[:, i]==0
                phif = phi[:, :, flag] #(NQ, NC, fldof//2)

                permci = localFace[i, permcf[:, i]] #(NC, 3)
                #l0 = l[permci[:, 0], ..., np.arange(NC), :, :].swapaxes(0, 1)
                #l1 = l[permci[:, 1], ..., np.arange(NC), :, :].swapaxes(0, 1)
                #l2 = l[permci[:, 2], ..., np.arange(NC), :, :].swapaxes(0, 1)
                l0 = l[permci[:, 0], ..., 0, :, :].swapaxes(0, 1)
                l1 = l[permci[:, 1], ..., 0, :, :].swapaxes(0, 1)
                l2 = l[permci[:, 2], ..., 0, :, :].swapaxes(0, 1)

                g0 = glambda[np.arange(NC), permci[:, 0], None]
                g1 = glambda[np.arange(NC), permci[:, 1], None]
                g2 = glambda[np.arange(NC), permci[:, 2], None]

                v0 = l2*(l0*g1 - l1*g0) #(NQ, NC, fldof//2, 2)
                v1 = l0*(l1*g2 - l2*g1)

                N = eldof*6+fldof*i
                val[..., N:N+fldof//2, :] = v0*phif[..., None] 
                val[..., N+fldof//2:N+fldof, :] = v1*phif[..., None] 

        if(p > 1):
            phi = self.bspace.basis(bc, p=p-2) #(NQ, NC, cldof)
            v0 = l[2]*l[3]*(l[0]*glambda[:, 1, None] - l[1]*glambda[:, 0, None]) #(NQ, NC, ldof, 2)
            v1 = l[0]*l[3]*(l[1]*glambda[:, 2, None] - l[2]*glambda[:, 1, None]) #(NQ, NC, ldof, 2)
            v2 = l[0]*l[1]*(l[2]*glambda[:, 3, None] - l[3]*glambda[:, 2, None]) #(NQ, NC, ldof, 2)

            N = eldof*6+fldof*4
            val[..., N:N+cldof//3, :] = v0*phi[..., None] 
            val[..., N+cldof//3:N+2*cldof//3, :] = v1*phi[..., None] 
            val[..., N+2*cldof//3:N+cldof, :] = v2*phi[..., None] 

        return val

    @barycentric
    def curl_basis(self, bc):

        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        ldof = self.dof.number_of_local_dofs()
        cldof = self.dof.number_of_local_dofs("cell")
        fldof = self.dof.number_of_local_dofs("face")
        eldof = self.dof.number_of_local_dofs("edge")
        gdof = self.dof.number_of_global_dofs()
        glambda = mesh.grad_lambda()
        ledge = mesh.ds.localEdge

        c2esign = mesh.ds.cell_to_edge_sign() #(NC, 6, 2)

        l = np.zeros((4, )+bc[..., 0, None, None, None].shape, dtype=np.float_)
        l[0] = bc[..., 0, None, None, None]
        l[1] = bc[..., 1, None, None, None]
        l[2] = bc[..., 2, None, None, None]
        l[3] = bc[..., 3, None, None, None] #(NQ, NC, ldof, 2)

        #l = np.tile(l, (1, NC, 1, 1))

        phi = self.bspace.basis(bc, p=p)
        gphi = self.bspace.grad_basis(bc, p=p)
        multiIndex = self.mesh.multi_index_matrix(p, 3)
        val = np.zeros(bc.shape[:-1]+(NC, ldof, 3), dtype=np.float_)

        # edge basis
        locEdgeDual = np.array([[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]])
        for i in range(6):
            flag = np.all(multiIndex[:, locEdgeDual[i]]==0, axis=1)
            phie = phi[..., flag] #(NQ, NC, eldof)
            gphie = gphi[..., flag, :] #(NQ, NC, eldof, 2)
            c2esi = c2esign[:, i] #(NC, )
            v = l[ledge[i, 0]]*glambda[:, ledge[i, 1], None] - l[ledge[i, 1]]*glambda[:, ledge[i, 0], None]
            v[..., ~c2esi, :, :] *= -1 #(NQ, NC, eldof, 2)
            cv = 2*np.cross(glambda[:, ledge[i, 0]], glambda[:, ledge[i, 1]]) #(NC, )
            cv[~c2esi] *= -1
            val[..., eldof*i:eldof*(i+1), :] = phie[..., None]*cv[:, None] + np.cross(gphie, v)

        # face basis
        if(p > 0):
            phi = self.bspace.basis(bc, p=p-1) #(NQ, NC, cldof)
            gphi = self.bspace.grad_basis(bc, p=p-1)
            multiIndex = self.mesh.multi_index_matrix(p-1, 3)
            permcf = self.mesh.ds.cell_to_face_permutation()
            localFace = self.mesh.ds.localFace
            for i in range(4):

                flag = multiIndex[:, i]==0
                phif = phi[..., flag] #(NQ, NC, fldof//2)
                gphif = gphi[..., flag, :] #(NQ, NC, fldof//2, 2)

                permci = localFace[i, permcf[:, i]] #(NC, 3)
                #l0 = l[permci[:, 0], ..., np.arange(NC), :, :].swapaxes(0, 1)
                #l1 = l[permci[:, 1], ..., np.arange(NC), :, :].swapaxes(0, 1)
                #l2 = l[permci[:, 2], ..., np.arange(NC), :, :].swapaxes(0, 1)

                l0 = l[permci[:, 0], ..., 0, :, :].swapaxes(0, 1)
                l1 = l[permci[:, 1], ..., 0, :, :].swapaxes(0, 1)
                l2 = l[permci[:, 2], ..., 0, :, :].swapaxes(0, 1)

                g0 = glambda[np.arange(NC), permci[:, 0], None]
                g1 = glambda[np.arange(NC), permci[:, 1], None]
                g2 = glambda[np.arange(NC), permci[:, 2], None]

                v0 = l2*(l0*g1 - l1*g0) #(NQ, NC, fldof//2, 2)
                v1 = l0*(l1*g2 - l2*g1)

                cv0 = np.cross(g2, (l0*g1 - l1*g0)) + 2*l2*np.cross(g0, g1)
                cv1 = np.cross(g0, (l1*g2 - l2*g1)) + 2*l0*np.cross(g1, g2)

                N = eldof*6+fldof*i
                val[..., N:N+fldof//2, :] = -np.cross(v0, gphif)+phif[..., None]*cv0
                val[..., N+fldof//2:N+fldof, :] = -np.cross(v1, gphif)+phif[..., None]*cv1 

        # cell basis
        if(p > 1):
            phi = self.bspace.basis(bc, p=p-2) #(NQ, NC, cldof)
            gphi = self.bspace.grad_basis(bc, p=p-2) #(NQ, NC, cldof)

            g0 = glambda[:, 0, None]
            g1 = glambda[:, 1, None]
            g2 = glambda[:, 2, None]
            g3 = glambda[:, 3, None]

            v0 = l[2]*l[3]*(l[0]*g1 - l[1]*g0) #(NQ, NC, ldof, 2)
            v1 = l[0]*l[3]*(l[1]*g2 - l[2]*g1) #(NQ, NC, ldof, 2)
            v2 = l[0]*l[1]*(l[2]*g3 - l[3]*g2) #(NQ, NC, ldof, 2)
            cv0 = np.cross(l[2]*g3+l[3]*g2, l[0]*g1-l[1]*g0) + 2*l[2]*l[3]*np.cross(g0, g1)
            cv1 = np.cross(l[0]*g3+l[3]*g0, l[1]*g2-l[2]*g1) + 2*l[0]*l[3]*np.cross(g1, g2)
            cv2 = np.cross(l[0]*g1+l[1]*g0, l[2]*g3-l[3]*g2) + 2*l[0]*l[1]*np.cross(g2, g3)

            N = eldof*6+fldof*4
            val[..., N:N+cldof//3, :] = np.cross(gphi, v0) + phi[..., None]*cv0 
            val[..., N+cldof//3:N+2*cldof//3, :] = np.cross(gphi, v1) + phi[...,
                    None]*cv1  
            val[..., N+2*cldof//3:N+cldof, :] = np.cross(gphi, v2) + phi[...,
                    None]*cv2  
        return val

    def cell_to_dof(self):
        return self.dof.cell2dof

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype)

    @barycentric
    def value(self, uh, bc, index=np.s_[:]):
        '''@
        @brief 计算一个有限元函数在每个单元的 bc 处的值
        @param bc : (..., GD+1)
        @return val : (..., NC, GD)
        '''
        phi = self.basis(bc)
        c2d = self.dof.cell_to_dof()
        # uh[c2d].shape = (NC, ldof); phi.shape = (..., NC, ldof, GD)
        val = np.einsum("cl, ...clk->...ck", uh[c2d], phi)
        return val

    @barycentric
    def curl_value(self, uh, bc, index=np.s_[:]):
        '''@
        @brief 计算一个有限元函数在每个单元的 bc 处的值
        @param bc : (..., GD+1)
        @return val : (..., NC, GD)
        '''
        cphi = self.curl_basis(bc)
        c2d = self.dof.cell_to_dof()
        # uh[c2d].shape = (NC, ldof); phi.shape = (..., NC, ldof, GD)
        val = np.einsum("cl, ...cli->...ci", uh[c2d], cphi)
        return val

    @barycentric
    def div_value(self, uh, bc, index=np.s_[:]):
        pass

    @barycentric
    def grad_value(self, uh, bc, index=np.s_[:]):
        pass

    @barycentric
    def edge_value(self, uh, bc, index=np.s_[:]):
        pass

    @barycentric
    def face_value(self, uh, bc, index=np.s_[:]):
        pass

    def mass_matrix(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()
        cm = self.cellmeasure
        c2d = self.dof.cell_to_dof() #(NC, ldof)

        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        phi = self.basis(bcs) #(NQ, NC, ldof, GD)
        mass = np.einsum("qclg, qcdg, c, q->cld", phi, phi, cm, ws)

        I = np.broadcast_to(c2d[:, :, None], shape=mass.shape)
        J = np.broadcast_to(c2d[:, None, :], shape=mass.shape)
        M = csr_matrix((mass.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return M 

    def curl_matrix(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()
        cm = self.cellmeasure

        c2d = self.dof.cell_to_dof() #(NC, ldof)
        bcs, ws = self.integrator.get_quadrature_points_and_weights()

        cphi = self.curl_basis(bcs) #(NQ, NC, ldof)
        print(cphi.shape)
        A = np.einsum("qclg, qcdg, c, q->cld", cphi, cphi, cm, ws) #(NC, ldof, ldof)

        I = np.broadcast_to(c2d[:, :, None], shape=A.shape)
        J = np.broadcast_to(c2d[:, None, :], shape=A.shape)
        B = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return B

    def source_vector(self, f):
        mesh = self.mesh
        cm = self.cellmeasure
        ldof = self.dof.number_of_local_dofs()
        gdof = self.dof.number_of_global_dofs()
        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        c2d = self.dof.cell_to_dof() #(NC, ldof)

        p = mesh.bc_to_point(bcs) #(NQ, NC, GD)
        fval = f(p) #(NQ, NC, GD)

        phi = self.basis(bcs) #(NQ, NC, ldof, GD)
        val = np.einsum("qcg, qclg, q, c->cl", fval, phi, ws, cm)# (NC, ldof)
        vec = np.zeros(gdof, dtype=np.float_)
        np.add.at(vec, c2d, val)
        return vec

mesh = TetrahedronMesh.from_box(nx=1,ny=1,nz=1)
p = 1
a = FirstNedelecFiniteElementSpace3d(mesh=mesh,p=p)
bcs = np.array([[0.1,0.2,0.1,0.6],
                [0.5,0.3,0.1,0.1]])
uh = np.arange(74)
def func(k):
    x = k[...,0]
    y = k[...,1]
    z = k[...,2]
    E1= 4*x+3*z
    E2= y +12
    E3= x*y +2*z
    array = np.zeros_like(k)
    array[...,0] = E1
    array[...,1] = E2
    array[...,2] = E3
    return array


b = a.source_vector(f=func)

print((b,))


