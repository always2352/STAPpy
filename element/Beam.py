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


class CBeam(CElement):
    """
    3D Euler-Bernoulli space-frame element: 2 nodes, 6 DOF/node
    (u, v, w, theta_x, theta_y, theta_z). Local stiffness includes axial,
    St-Venant torsion and bending about both principal axes (no shear, i.e.
    Bernoulli-Euler -- adequate for slender members).
    """
    def __init__(self):
        super().__init__()
        self._NEN = 2
        self._nodes = [None for _ in range(self._NEN)]

        self._ND = 12               # 2 nodes x 6 DOF
        self._LocationMatrix = np.zeros(self._ND, dtype=int)

    def Read(self, input_file, Ele, MaterialSets, NodeList):
        line = input_file.readline().split()
        N = int(line[0])
        if N != Ele + 1:
            raise ValueError("\n*** Error *** Elements must be inputted in order !"
                             "\n   Expected element : {}"
                             "\n   Provided element : {}".format(Ele + 1, N))
        N1, N2, MSet = int(line[1]), int(line[2]), int(line[3])
        self._ElementMaterial = MaterialSets[MSet - 1]
        self._nodes[0] = NodeList[N1 - 1]
        self._nodes[1] = NodeList[N2 - 1]

    def Write(self, output_file, Ele):
        element_info = "%5d%11d%9d%12d\n" % (
            Ele + 1, self._nodes[0].NodeNumber, self._nodes[1].NodeNumber,
            self._ElementMaterial.nset)
        print(element_info, end='')
        output_file.write(element_info)

    def GenerateLocationMatrix(self):
        """ Map the 12 element DOFs to the 6-DOF node slots (all 6 per node). """
        i = 0
        for N in range(self._NEN):
            for d in range(6):
                self._LocationMatrix[i] = self._nodes[N].bcode[d]
                i += 1

    def MarkActiveDofs(self):
        """ A space frame stiffens all six DOFs of each node. """
        for node in self._nodes:
            for d in range(6):
                node.active[d] = True

    def SizeOfStiffnessMatrix(self):
        """ Upper-triangular size of the 12x12 frame stiffness. """
        return int(self._ND * (self._ND + 1) // 2)

    def _LocalFrame(self):
        """
        Orthonormal local axes: e1 = element axis; e2, e3 = principal section
        axes (chosen automatically -- the box section is symmetric so the
        in-plane orientation is immaterial).
        """
        d = self._nodes[1].XYZ - self._nodes[0].XYZ
        length = np.sqrt(d.dot(d))
        if length <= 0.0:
            raise ValueError("Beam element has zero length.")
        e1 = d / length
        ref = np.array([0.0, 0.0, 1.0]) if abs(e1[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
        e2 = np.cross(ref, e1)
        e2 = e2 / np.sqrt(e2.dot(e2))
        e3 = np.cross(e1, e2)
        return length, e1, e2, e3

    def _GetTransformationMatrix(self):
        length, e1, e2, e3 = self._LocalFrame()
        Lam = np.array([e1, e2, e3])
        T = np.zeros((12, 12))
        for b in range(4):
            T[3*b:3*b+3, 3*b:3*b+3] = Lam
        return T, length

    def _GetLocalStiffness(self, L):
        mat = self._ElementMaterial
        E, A, I = mat.E, mat.Area, mat.Inertia
        J = getattr(mat, 'J', I)
        nu = getattr(mat, 'nu', 0.3)
        As = getattr(mat, 'As', 0.0)            # shear area; 0 -> Euler-Bernoulli
        G = E / (2.0 * (1.0 + nu))
        Iy = Iz = I
        L2, L3 = L*L, L*L*L

        # Timoshenko shear parameter Phi = 12 EI / (G As L^2); Phi=0 recovers
        # Euler-Bernoulli.  The bridge's support-beam members are very stocky
        # (L/h ~ 2), so shear deformation is large and must not be neglected.
        Phi = (12.0 * E * I / (G * As * L2)) if As > 0.0 else 0.0
        opi = 1.0 + Phi

        K = np.zeros((12, 12))
        EA, GJ = E*A/L, G*J/L
        K[0, 0] = EA; K[0, 6] = -EA; K[6, 0] = -EA; K[6, 6] = EA
        K[3, 3] = GJ; K[3, 9] = -GJ; K[9, 3] = -GJ; K[9, 9] = GJ

        az = 12*E*Iz/L3/opi; bz = 6*E*Iz/L2/opi
        cz = (4.0+Phi)*E*Iz/L/opi; dz = (2.0-Phi)*E*Iz/L/opi
        K[1, 1] = az; K[1, 5] = bz; K[1, 7] = -az; K[1, 11] = bz
        K[5, 1] = bz; K[5, 5] = cz; K[5, 7] = -bz; K[5, 11] = dz
        K[7, 1] = -az; K[7, 5] = -bz; K[7, 7] = az; K[7, 11] = -bz
        K[11, 1] = bz; K[11, 5] = dz; K[11, 7] = -bz; K[11, 11] = cz

        ay = 12*E*Iy/L3/opi; by = 6*E*Iy/L2/opi
        cy = (4.0+Phi)*E*Iy/L/opi; dy = (2.0-Phi)*E*Iy/L/opi
        K[2, 2] = ay; K[2, 4] = -by; K[2, 8] = -ay; K[2, 10] = -by
        K[4, 2] = -by; K[4, 4] = cy; K[4, 8] = by; K[4, 10] = dy
        K[8, 2] = -ay; K[8, 4] = by; K[8, 8] = ay; K[8, 10] = by
        K[10, 2] = -by; K[10, 4] = dy; K[10, 8] = by; K[10, 10] = cy
        return K

    def ElementStiffness(self, stiffness):
        for i in range(self.SizeOfStiffnessMatrix()):
            stiffness[i] = 0.0
        T, L = self._GetTransformationMatrix()
        K_global = np.dot(T.T, np.dot(self._GetLocalStiffness(L), T))
        count = 0
        for col in range(12):
            for row in range(col, -1, -1):
                stiffness[count] = K_global[row, col]
                count += 1

    def ElementStress(self, stress, displacement):
        """ stress[0] axial force; stress[1], stress[2] bending-moment
            resultant at the two ends. """
        T, L = self._GetTransformationMatrix()
        K_local = self._GetLocalStiffness(L)
        d_global = np.zeros(12)
        for i in range(12):
            eq = self._LocationMatrix[i]
            if eq > 0:
                d_global[i] = displacement[eq - 1]
        f = np.dot(K_local, np.dot(T, d_global))
        stress[0] = f[0]
        stress[1] = np.sqrt(f[4]**2 + f[5]**2)
        stress[2] = np.sqrt(f[10]**2 + f[11]**2)

    def GetShapeFunctions(self, xi, eta=0.0, zeta=0.0):
        return np.array([0.5 * (1.0 - xi), 0.5 * (1.0 + xi)])

    def GetIntegrationPoints(self):
        gp = 1.0 / np.sqrt(3.0)
        return [(-gp, 0.0, 0.0), (gp, 0.0, 0.0)], [1.0, 1.0]

    def GetDetJ(self, xi=0.0, eta=0.0, zeta=0.0):
        return self._LocalFrame()[0] / 2.0
