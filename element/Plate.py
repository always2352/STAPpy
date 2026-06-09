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
    """ Plate Element class """
    def __init__(self):
        super().__init__()
        self._NEN = 4 # Each element has 4 nodes
        self._nodes = [None for _ in range(self._NEN)]

        self._ND = 12
        self._LocationMatrix = np.zeros(self._ND, dtype=int)

    def Read(self, input_file, Ele, MaterialSets, NodeList):
        """
        Read element data from stream Input

        :param input_file: (_io.TextIOWrapper) the object of input file
        :param Ele: (int) check index
        :param MaterialSets: (list(CMaterial)) the material list in Domain
        :param NodeList: (list(CNode)) the node list in Domain
        :return: None
        """
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
        """
        Write element data to stream

        :param output_file: (_io.TextIOWrapper) the object of output file
        :param Ele: the element number
        :return: None
        """
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
        Map the plate's (w, theta_x, theta_y) to the 6-DOF node slots:
        w -> translation along the (snapped) normal axis k;
        theta_x, theta_y -> rotations about the two in-plane axes.
        """
        e1, e2, e3, area = self._ExtractGeometry()
        k = int(np.argmax(np.abs(e3)))
        p, q = (i for i in range(3) if i != k)
        return [k, 3 + p, 3 + q]

    def GenerateLocationMatrix(self):
        """
        Generate location matrix: map the three plate DOFs per node to the
        corresponding 6-DOF node slots.
        """
        slots = self._DofSlots()
        i = 0
        for N in range(self._NEN):
            for D in range(3):
                self._LocationMatrix[i] = self._nodes[N].bcode[slots[D]]
                i += 1

    def MarkActiveDofs(self):
        """ Out-of-plane translation + the two bending rotations. """
        slots = self._DofSlots()
        for node in self._nodes:
            for s in slots:
                node.active[s] = True

    def SizeOfStiffnessMatrix(self):
        """
        Return the size of the element stiffness matrix
        (stored as an array column by column)
        For 4 node Plate element, element stiffness is a 12x12 matrix,
        whose upper triangular part has 78 elements
        """
        return 78
    
    def _CalculateBMatrix(self, xi, eta, a, b):
        B = np.zeros((3, 12))

        xi_I  = [-1.0,  1.0, 1.0, -1.0]
        eta_I = [-1.0, -1.0, 1.0,  1.0]

        for I in range(4):
            xI = xi_I[I]
            eI = eta_I[I]
            BI = np.zeros((3, 3))

            BI[0, 0] = -3.0 * b / a * xI * xi * (1.0 + eI * eta)
            BI[0, 1] = 0.0
            BI[0, 2] = -b * xI * (1.0 + 3.0 * xI * xi) * (1.0 + eI * eta)
            
            BI[1, 0] = -3.0 * a / b * eI * eta * (1.0 + xI * xi)
            BI[1, 1] = a * eI * (1.0 + 3.0 * eI * eta) * (1.0 + xI * xi)
            BI[1, 2] = 0.0
            
            BI[2, 0] = xI * eI * (4.0 - 3.0 * xi**2 - 3.0 * eta**2)
            BI[2, 1] = b * xI * (3.0 * eta**2 + 2.0 * eI * eta - 1.0)
            BI[2, 2] = a * eI * (1.0 - 2.0 * xI * xi - 3.0 * xi**2)

            BI = BI / (4.0 * a * b)

            col_start = I * 3
            B[:, col_start : col_start + 3] = BI

        return B

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

    def ElementStiffness(self, stiffness):
        """
        Calculate element stiffness matrix
        Upper triangular matrix, stored as an array column by colum
        starting from the diagonal element
        """
        for i in range(self.SizeOfStiffnessMatrix()):
            stiffness[i] = 0.0

        material = self._ElementMaterial
        E = material.E
        nu = material.nu
        t = material.thick

        D0 = (E * t**3) / (12.0 * (1.0 - nu**2))
        D = D0 * np.array([
            [1.0,  nu, 0.0],
            [ nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0]
        ])

        a, b = self._ExtractLocalSize()
        detJ = a * b

        gauss_points = [-np.sqrt(0.6), 0.0, np.sqrt(0.6)]
        gauss_weights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
    
        Ke_full = np.zeros((12, 12))
        for xi, w_xi in zip(gauss_points, gauss_weights):
            for eta, w_eta in zip(gauss_points, gauss_weights):
                B = self._CalculateBMatrix(xi, eta, a, b)
                Ke_point = np.dot(B.T, np.dot(D, B)) * detJ * w_xi * w_eta
                Ke_full += Ke_point
        
        count = 0
        for col in range(12):
            for row in range(col, -1, -1):
                stiffness[count] = Ke_full[row, col]
                count += 1

    def ElementStress(self, stress, displacement):
        """
        Calculate element stress
        """
        material = self._ElementMaterial
        E = material.E
        nu = material.nu
        t = material.thick

        D0 = (E * t**3) / (12.0 * (1.0 - nu**2))
        D = D0 * np.array([
            [1.0,  nu, 0.0],
            [ nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0]
        ])

        a, b = self._ExtractLocalSize()

        B = self._CalculateBMatrix(0.0, 0.0, a, b)

        de = np.zeros(12)
        for i in range(12):
            global_eq = self._LocationMatrix[i]
            if global_eq > 0:
                de[i] = displacement[global_eq - 1]
            else:
                de[i] = 0.0

        kappa = np.dot(B, de)
        moment = - np.dot(D, kappa)
    
        stress[0] = moment[0]
        stress[1] = moment[1]
        stress[2] = moment[2]
    
    def CalculateWAtPoint(self, xi, eta, displacement):
        de = np.zeros(12)
        for i in range(12):
            global_eq = self._LocationMatrix[i]
            if global_eq > 0:
                de[i] = displacement[global_eq - 1]
            else:
                de[i] = 0.0

        xi_I  = [-1.0,  1.0, 1.0, -1.0]
        eta_I = [-1.0, -1.0, 1.0,  1.0]

        a, b = self._ExtractLocalSize()
        w_interpolated = 0.0

        for I in range(4):
            xI = xi_I[I]
            eI = eta_I[I]

            factor = 0.125 * (1.0 + xI * xi) * (1.0 + eI * eta)

            N_w      = factor * (2.0 + xI * xi + eI * eta - xi**2 - eta**2)
            N_thetax = factor * (-b * eI * (1.0 - eta**2))
            N_thetay = factor * (a * xI * (1.0 - xi**2))

            idx = I * 3
            
            w_interpolated += N_w * de[idx] + N_thetax * de[idx + 1] + N_thetay * de[idx + 2]

        return w_interpolated

    def GetShapeFunctions(self, xi, eta, zeta=0.0):
        """
        Get shape function values for 4-node plate element
        Returns shape functions for (w, theta_x, theta_y) at each node
        """
        xi_I  = [-1.0,  1.0, 1.0, -1.0]
        eta_I = [-1.0, -1.0, 1.0,  1.0]
        a, b = self._ExtractLocalSize()
        
        N = np.zeros((4, 3))  # [node][dof: w, theta_x, theta_y]
        
        for I in range(4):
            xI = xi_I[I]
            eI = eta_I[I]
            
            factor = 0.125 * (1.0 + xI * xi) * (1.0 + eI * eta)
            
            N[I, 0] = factor * (2.0 + xI * xi + eI * eta - xi**2 - eta**2)  # N_w
            N[I, 1] = factor * (-b * eI * (1.0 - eta**2))  # N_theta_x
            N[I, 2] = factor * (a * xI * (1.0 - xi**2))  # N_theta_y
        
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