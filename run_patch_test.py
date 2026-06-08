#!/usr/bin/env python3
import numpy as np
from Domain import Domain
from element.ElementGroup import ElementTypes


def unpack_packed_upper(packed, n):
    K = np.zeros((n,n), dtype=np.double)
    pos = 0
    for j in range(n):
        for i in range(j, -1, -1):
            K[i,j] = packed[pos]
            K[j,i] = packed[pos]
            pos += 1
    return K


def integrate_internal_force(element, u_e):
    # Recompute K via integration and multiply by u_e to get internal force
    ND = element._ND
    mat = element._ElementMaterial
    if mat is None:
        E = 1.0; nu = 0.3
    else:
        E = float(mat.E); nu = float(getattr(mat, 'nu', 0.3))
    factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = np.zeros((6,6), dtype=np.double)
    D[0,0]=D[1,1]=D[2,2]=(1-nu)*factor
    D[0,1]=D[0,2]=D[1,0]=D[1,2]=D[2,0]=D[2,1]=nu*factor
    D[3,3]=D[4,4]=D[5,5]=0.5*(1-2*nu)*factor

    gp = [-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)]
    gw = [1.0, 1.0]

    Kloc = np.zeros((ND,ND), dtype=np.double)

    xi_coords = np.array([ -1,  1,  1, -1, -1,  1,  1, -1 ], dtype=np.double)
    eta_coords= np.array([ -1, -1,  1,  1, -1, -1,  1,  1 ], dtype=np.double)
    zeta_coords= np.array([ -1, -1, -1, -1,  1,  1,  1,  1 ], dtype=np.double)

    for a in range(2):
        for b in range(2):
            for c in range(2):
                xi = gp[a]; eta = gp[b]; zeta = gp[c]
                w = gw[a]*gw[b]*gw[c]
                dN_dxi = np.zeros(8); dN_deta = np.zeros(8); dN_dzeta = np.zeros(8)
                for i in range(8):
                    n1 = xi_coords[i]; n2 = eta_coords[i]; n3 = zeta_coords[i]
                    dN_dxi[i] = 0.125 * n1 * (1.0 + n2*eta) * (1.0 + n3*zeta)
                    dN_deta[i]= 0.125 * (1.0 + n1*xi) * n2 * (1.0 + n3*zeta)
                    dN_dzeta[i]=0.125 * (1.0 + n1*xi) * (1.0 + n2*eta) * n3
                J = np.zeros((3,3), dtype=np.double)
                for i in range(8):
                    x = element._nodes[i].XYZ[0]; y = element._nodes[i].XYZ[1]; z = element._nodes[i].XYZ[2]
                    J[0,0] += dN_dxi[i]*x; J[0,1] += dN_deta[i]*x; J[0,2] += dN_dzeta[i]*x
                    J[1,0] += dN_dxi[i]*y; J[1,1] += dN_deta[i]*y; J[1,2] += dN_dzeta[i]*y
                    J[2,0] += dN_dxi[i]*z; J[2,1] += dN_deta[i]*z; J[2,2] += dN_dzeta[i]*z
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                dN_dx = np.zeros((8,3), dtype=np.double)
                for i in range(8):
                    dN_nat = np.array([dN_dxi[i], dN_deta[i], dN_dzeta[i]])
                    dN_phys = invJ.dot(dN_nat)
                    dN_dx[i,0]=dN_phys[0]; dN_dx[i,1]=dN_phys[1]; dN_dx[i,2]=dN_phys[2]
                B = np.zeros((6,ND), dtype=np.double)
                for i in range(8):
                    i3 = 3*i
                    dNxi = dN_dx[i,0]; dNyi = dN_dx[i,1]; dNzi = dN_dx[i,2]
                    B[0,i3    ] = dNxi
                    B[1,i3 + 1] = dNyi
                    B[2,i3 + 2] = dNzi
                    B[3,i3    ] = dNyi
                    B[3,i3 + 1] = dNxi
                    B[4,i3 + 1] = dNzi
                    B[4,i3 + 2] = dNyi
                    B[5,i3    ] = dNzi
                    B[5,i3 + 2] = dNxi
                Kloc += B.T.dot(D.dot(B)) * detJ * w
    fint = Kloc.dot(u_e)
    return fint, Kloc


if __name__ == '__main__':
    FEM = Domain()
    # we assume h8_test.dat exists in workspace root; read it
    FEM.ReadData('h8_test.dat', 'h8_test.out')
    # find H8 element groups
    for grp in FEM.GetEleGrpList():
        etype = grp.GetElementType()
        if ElementTypes.get(etype) == 'H8':
            NUME = grp.GetNUME()
            print('Found H8 group with', NUME, 'elements')
            for Ele in range(NUME):
                elem = grp[Ele]
                # build nodal displacement vector from linear field u = [ax+by+cz, 0, 0]
                ND = elem._ND
                u_e = np.zeros(ND, dtype=np.double)
                # choose u_x = x, u_y = 2*y, u_z = 3*z
                for i in range(8):
                    x = elem._nodes[i].XYZ[0]; y = elem._nodes[i].XYZ[1]; z = elem._nodes[i].XYZ[2]
                    u_e[3*i + 0] = x
                    u_e[3*i + 1] = 2.0 * y
                    u_e[3*i + 2] = 3.0 * z
                # get analytical stiffness
                packed_size = elem.SizeOfStiffnessMatrix()
                packed = np.zeros(packed_size, dtype=np.double)
                elem.ElementStiffness(packed)
                K_ana = unpack_packed_upper(packed, ND)
                # compute internal force via numerical integration routine
                f0, Kloc = integrate_internal_force(elem, u_e)
                # numeric differentiation: columns of Knum
                eps = 1e-6
                Knum = np.zeros((ND,ND), dtype=np.double)
                for j in range(ND):
                    du = np.zeros(ND); du[j] = eps
                    f1, _ = integrate_internal_force(elem, u_e + du)
                    f2, _ = integrate_internal_force(elem, u_e - du)
                    Knum[:, j] = (f1 - f2) / (2*eps)
                # compare
                diff = K_ana - Knum
                rel_err = np.linalg.norm(diff) / (np.linalg.norm(Knum) + 1e-20)
                print('Element', Ele+1, 'relative error between analytical K and numeric K:', rel_err)

