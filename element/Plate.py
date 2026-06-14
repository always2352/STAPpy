#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/*****************************************************************************/
/*  STAPpy : A python FEM code sharing the same input data file with STAP90  */
/*     Computational Dynamics Laboratory                                     */
/*     School of Aerospace Engineering, Tsinghua University                  */
/*                                                                           */
/*     Created on Mon Jun 22, 2020                                           */
/*                                                                           */
/*     @author: thurcni@163.com, xzhang@tsinghua.edu.cn                      */
/*     http://www.comdyn.cn/                                                 */
/*****************************************************************************/
"""
import sys
sys.path.append('../')
import numpy as np
from element.Element import CElement


class CPlate(CElement):
    """
    Flat shell element (4 nodes): Kirchhoff bending + Q4 plane-stress membrane
    + a fictitious drilling stiffness, so each node carries the full 6 DOF
    (u, v, w, theta_x, theta_y, theta_z). Bending and membrane are decoupled
    in linear flat-shell theory, so pure-bending results are unchanged.
    """
    def __init__(self):
        super().__init__()
        self._NEN = 4 # Each element has 4 nodes
        self._nodes = [None for _ in range(self._NEN)]

        self._ND = 24            # 4 nodes x 6 DOF (membrane + bending + drilling)
        self._LocationMatrix = np.zeros(self._ND, dtype=int)

    def Read(self, input_file, Ele, MaterialSets, NodeList):
        line = input_file.readline().split()

        N = int(line[0])
        if N != Ele + 1:
            error_info = "\n*** Error *** Elements must be inputted in order !" \
                         "\n   Expected element : {}" \
                         "\n   Provided element : {}".format(Ele + 1, N)
            raise ValueError(error_info)

        # anti-clockwise
        N1, N2, N3, N4 = int(line[1]), int(line[2]), int(line[3]), int(line[4])

        # Material
        MSet = int(line[5])
        self._ElementMaterial = MaterialSets[MSet - 1]
        self._nodes[0] = NodeList[N1 - 1]
        self._nodes[1] = NodeList[N2 - 1]
        self._nodes[2] = NodeList[N3 - 1]
        self._nodes[3] = NodeList[N4 - 1]

    def Write(self, output_file, Ele):
        element_info = "%5d%11d%9d%9d%9d%12d\n" % (
            Ele + 1,
            self._nodes[0].NodeNumber,
            self._nodes[1].NodeNumber,
            self._nodes[2].NodeNumber,
            self._nodes[3].NodeNumber,
            self._ElementMaterial.nset,
        )

        # print the element info on the screen
        print(element_info, end='')
        # write the element info to output file
        output_file.write(element_info)

    def _DofSlots(self):
        """
        Global 6-DOF node slots for the local DOF order (u, v, w, tx, ty, tz):
        the two in-plane translations along axes p, q; the transverse
        translation along the (snapped) normal axis k; the two bending
        rotations about p, q; and the drilling rotation about k.
        """
        e1, e2, e3, area = self._ExtractGeometry()
        k = int(np.argmax(np.abs(e3)))
        p, q = [i for i in range(3) if i != k]
        return [p, q, k, 3 + p, 3 + q, 3 + k]

    def GenerateLocationMatrix(self):
        """ Map the six shell DOFs per node to the 6-DOF node slots. """
        slots = self._DofSlots()
        i = 0
        for N in range(self._NEN):
            for d in range(6):
                self._LocationMatrix[i] = self._nodes[N].bcode[slots[d]]
                i += 1

    def MarkActiveDofs(self):
        """ A flat shell stiffens all six DOFs (membrane + bending + drilling). """
        slots = self._DofSlots()
        for node in self._nodes:
            for s in slots:
                node.active[s] = True

    def SizeOfStiffnessMatrix(self):
        """ Upper-triangular size of the 24x24 shell stiffness matrix. """
        return int(self._ND * (self._ND + 1) // 2)
    
    def _ExtractLocalSize(self):
        x0, y0 = self._nodes[0].XYZ[0], self._nodes[0].XYZ[1]
        x1, y1 = self._nodes[1].XYZ[0], self._nodes[1].XYZ[1]
        x2, y2 = self._nodes[2].XYZ[0], self._nodes[2].XYZ[1]
        x3, y3 = self._nodes[3].XYZ[0], self._nodes[3].XYZ[1]
        
        width1 = np.abs(x1 - x0)
        width2 = np.abs(x2 - x3)
        a = (width1 + width2) / 4.0
        
        height1 = np.abs(y3 - y0)
        height2 = np.abs(y2 - y1)
        b = (height1 + height2) / 4.0
        
        return a, b

    def _ExtractGeometry(self):
        v1 = self._nodes[1].XYZ - self._nodes[0].XYZ
        v2 = self._nodes[3].XYZ - self._nodes[0].XYZ

        normal = np.cross(v1, v2)
        area = np.linalg.norm(normal)
        e3 = normal / area

        e1 = v1 / np.linalg.norm(v1)
        e2 = np.cross(e3, e1)

        return e1, e2, e3, area

    def _GetTransformationMatrix(self):
        e1, e2, e3, area = self._ExtractGeometry()

        T_node = np.zeros((3, 3))
        T_node[0, 0] = e3[2]
        T_node[1, 1] = e1[0]
        T_node[1, 2] = e2[0]
        T_node[2, 1] = e1[1]
        T_node[2, 2] = e2[1]

        T = np.zeros((12, 12))
        for I in range(4):
            T[I*3:I*3+3, I*3:I*3+3] = T_node

        return T, e1, e2, e3, area

    def _AcmCurvatureB(self, xi, eta):
        """
        Kirchhoff (ACM / MZC non-conforming rectangle) curvature-displacement
        matrix Bb (3x12) at natural point (xi, eta), in the local (w, tx, ty)
        DOFs with tx = -w,y and ty = w,x (same convention as the shell DOF
        slots).  Returns also the area Jacobian a*b.  The thin-plate kinematics
        impose theta = grad(w) exactly, so there is NO transverse shear.

        Built from the 12-term polynomial
          w = a1 + a2 xi + a3 eta + ... + a11 xi^3 eta + a12 xi eta^3
        in natural coordinates (well conditioned); valid for a rectangular
        element (the deck mesh), with a, b the element half-sides.
        """
        e1, e2, e3, area = self._ExtractGeometry()
        k = int(np.argmax(np.abs(e3)))
        p, q = [i for i in range(3) if i != k]
        c = np.array([[nd.XYZ[p], nd.XYZ[q]] for nd in self._nodes])
        # Map each node to its (+/-1, +/-1) corner from its actual position, so
        # the element is robust to node ordering / orientation (the deck quads
        # are reconnected with a cyclic shift).  Assumes axis-aligned rectangle.
        cen = c.mean(axis=0)
        dp = c[:, 0] - cen[0]; dq = c[:, 1] - cen[1]
        a = float(np.mean(np.abs(dp))); b = float(np.mean(np.abs(dq)))
        xi_I = np.sign(dp); eta_I = np.sign(dq)

        def P(s, t):
            return np.array([1, s, t, s*s, s*t, t*t, s**3, s*s*t, s*t*t, t**3, s**3*t, s*t**3], float)

        def Ps(s, t):    # d/d(xi)
            return np.array([0, 1, 0, 2*s, t, 0, 3*s*s, 2*s*t, t*t, 0, 3*s*s*t, t**3], float)

        def Pt(s, t):    # d/d(eta)
            return np.array([0, 0, 1, 0, s, 2*t, 0, s*s, 2*s*t, 3*t*t, s**3, 3*s*t*t], float)

        # nodal-DOF -> polynomial-coefficient map; rows per node: (w, tx=-w,y, ty=w,x)
        C = np.zeros((12, 12))
        for i in range(4):
            s, t = xi_I[i], eta_I[i]
            C[3*i + 0] = P(s, t)
            C[3*i + 1] = -(1.0 / b) * Pt(s, t)        # tx = -w,y
            C[3*i + 2] = (1.0 / a) * Ps(s, t)         # ty =  w,x
        Cinv = np.linalg.inv(C)

        Pss = np.array([0, 0, 0, 2, 0, 0, 6*xi, 2*eta, 0, 0, 6*xi*eta, 0], float)
        Ptt = np.array([0, 0, 0, 0, 0, 2, 0, 0, 2*xi, 6*eta, 0, 6*xi*eta], float)
        Pst = np.array([0, 0, 0, 0, 1, 0, 0, 2*xi, 2*eta, 0, 3*xi*xi, 3*eta*eta], float)
        # physical curvatures [w,xx ; w,yy ; 2 w,xy]
        Bnat = np.vstack([Pss / (a*a), Ptt / (b*b), 2.0 * Pst / (a*b)])
        return Bnat.dot(Cinv), a * b

    def _BendingStiffness(self):
        """
        Kirchhoff thin-plate bending in the (w, tx, ty) DOFs using the ACM/MZC
        non-conforming rectangle (theta = grad w, no transverse shear).  3x3
        Gauss integration is exact for the quartic Bb^T Db Bb integrand.
        Valid for the rectangular deck elements.
        """
        mat = self._ElementMaterial
        E, nu, t = mat.E, mat.nu, mat.thick
        Db = (E * t**3 / (12.0 * (1.0 - nu**2))) * np.array([[1.0, nu, 0.0],
                                                             [nu, 1.0, 0.0],
                                                             [0.0, 0.0, (1.0 - nu) / 2.0]])
        g = np.sqrt(0.6)
        gp = [-g, 0.0, g]; gw = [5.0/9.0, 8.0/9.0, 5.0/9.0]
        Kb = np.zeros((12, 12))
        for xi, wi in zip(gp, gw):
            for eta, wj in zip(gp, gw):
                Bb, detJ = self._AcmCurvatureB(xi, eta)
                Kb += (wi * wj) * Bb.T.dot(Db).dot(Bb) * detJ
        return Kb

    def _MembraneStiffness(self):
        """ 8x8 Q4 plane-stress membrane stiffness in the (u, v) DOFs. """
        material = self._ElementMaterial
        E, nu, t = material.E, material.nu, material.thick
        e1, e2, e3, area = self._ExtractGeometry()
        k = int(np.argmax(np.abs(e3)))
        p, q = [i for i in range(3) if i != k]
        coords = np.array([[nd.XYZ[p], nd.XYZ[q]] for nd in self._nodes])

        Dm = (E * t / (1.0 - nu**2)) * np.array([[1.0, nu, 0.0],
                                                 [nu, 1.0, 0.0],
                                                 [0.0, 0.0, (1.0 - nu) / 2.0]])
        xi_I = [-1.0, 1.0, 1.0, -1.0]
        eta_I = [-1.0, -1.0, 1.0, 1.0]
        gp = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        Km = np.zeros((8, 8))
        for xi in gp:
            for eta in gp:
                dN = np.zeros((2, 4))
                for I in range(4):
                    dN[0, I] = 0.25 * xi_I[I] * (1.0 + eta_I[I] * eta)
                    dN[1, I] = 0.25 * eta_I[I] * (1.0 + xi_I[I] * xi)
                J = dN.dot(coords)
                detJ = np.linalg.det(J)
                dNxy = np.linalg.inv(J).dot(dN)
                B = np.zeros((3, 8))
                for I in range(4):
                    B[0, 2 * I] = dNxy[0, I]
                    B[1, 2 * I + 1] = dNxy[1, I]
                    B[2, 2 * I] = dNxy[1, I]
                    B[2, 2 * I + 1] = dNxy[0, I]
                Km += B.T.dot(Dm).dot(B) * detJ
        return Km

    def ElementStiffness(self, stiffness):
        """
        Assemble the 24x24 flat-shell stiffness (membrane + bending + drilling)
        and pack the upper triangle column by column.
        """
        for i in range(self.SizeOfStiffnessMatrix()):
            stiffness[i] = 0.0

        Km = self._MembraneStiffness()
        Kb = self._BendingStiffness()

        K = np.zeros((24, 24))
        m_idx = [6 * I + d for I in range(4) for d in (0, 1)]      # u, v
        b_idx = [6 * I + d for I in range(4) for d in (2, 3, 4)]   # w, tx, ty
        K[np.ix_(m_idx, m_idx)] += Km
        K[np.ix_(b_idx, b_idx)] += Kb

        # small fictitious drilling stiffness so the tz DOF is not singular
        k_drill = 1.0e-3 * np.mean(np.diag(Kb))
        for I in range(4):
            K[6 * I + 5, 6 * I + 5] += k_drill

        count = 0
        for col in range(24):
            for row in range(col, -1, -1):
                stiffness[count] = K[row, col]
                count += 1

    def ElementStress(self, stress, displacement):
        """ Bending moments (Mx, My, Mxy) at the element centre. """
        mat = self._ElementMaterial
        E, nu, t = mat.E, mat.nu, mat.thick
        Db = (E * t**3 / (12.0 * (1.0 - nu**2))) * np.array([[1.0, nu, 0.0],
                                                             [nu, 1.0, 0.0],
                                                             [0.0, 0.0, (1.0 - nu) / 2.0]])
        Bb, _ = self._AcmCurvatureB(0.0, 0.0)
        de = self._GatherBendingDof(displacement)
        moment = Db.dot(Bb.dot(de))
        stress[0] = moment[0]
        stress[1] = moment[1]
        stress[2] = moment[2]

    def _GatherBendingDof(self, displacement):
        """ Pull the (w, tx, ty) DOFs out of the 24-DOF location matrix. """
        b_idx = [6 * I + d for I in range(4) for d in (2, 3, 4)]
        de = np.zeros(12)
        for j, i in enumerate(b_idx):
            eq = self._LocationMatrix[i]
            if eq > 0:
                de[j] = displacement[eq - 1]
        return de

    def CalculateWAtPoint(self, xi, eta, displacement):
        de = self._GatherBendingDof(displacement)
        xi_I = [-1.0, 1.0, 1.0, -1.0]
        eta_I = [-1.0, -1.0, 1.0, 1.0]
        w = 0.0
        for I in range(4):
            N = 0.25 * (1.0 + xi_I[I] * xi) * (1.0 + eta_I[I] * eta)
            w += N * de[3 * I]
        return w

    def GetShapeFunctions(self, xi, eta, zeta=0.0):
        """
        Bilinear shape functions; N[I,0] = N_I drives the transverse (w)
        self-weight load (a pressure does work only on w in a Mindlin shell).
        """
        xi_I = [-1.0, 1.0, 1.0, -1.0]
        eta_I = [-1.0, -1.0, 1.0, 1.0]
        N = np.zeros((4, 3))
        for I in range(4):
            N[I, 0] = 0.25 * (1.0 + xi_I[I] * xi) * (1.0 + eta_I[I] * eta)
        return N

    def GetIntegrationPoints(self):
        """
        Get integration points for plate element (2*2 Gauss)
        """
        gp = [-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)]
        points = [(xi, eta, 0.0) for xi in gp for eta in gp]
        weights = [1.0, 1.0, 1.0, 1.0]
        return points, weights

    def GetDetJ(self, xi=0.0, eta=0.0, zeta=0.0):
        """
        Calculate determinant of Jacobian for plate element
        """
        a, b = self._ExtractLocalSize()
        return a * b