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
import numpy as np


class CNode(object):
    # Degrees of freedom per node: (ux, uy, uz, theta_x, theta_y, theta_z).
    # All element types share this 6-DOF space; each element activates only
    # the DOFs it stiffens (see Domain.MarkActiveDofs / auto-suppression).
    NDF = 6

    def __init__(self, x=0.0, y=0.0, z=0.0):
        super().__init__()
        # x, y and z coordinates of the node
        self.XYZ = np.zeros(3)
        self.XYZ[0] = x; self.XYZ[1] = y; self.XYZ[2] = z

        # Boundary code of each degree of freedom of the node
        #     0: The corresponding degree of freedom is active
        #     1: The corresponding degree of freedom is constrained
        # After Domain.CalculateEquationNumber(), bcode stores the global
        # equation number corresponding to each degree of freedom.
        self.bcode = np.zeros(CNode.NDF, dtype=int)

        # True once some element contributes stiffness to the DOF; DOFs that
        # stay inactive are suppressed so the global matrix stays non-singular.
        self.active = np.zeros(CNode.NDF, dtype=bool)

        self.is_constrained = np.zeros(CNode.NDF, dtype=int)
        self.prescribed_values = np.zeros(CNode.NDF, dtype=np.double)

        # Node number
        self.NodeNumber = 0

    def Read(self, input_file, check_np):
        """
        Read nodal point data from stream Input
        Format: N  b0 b1 b2 b3 b4 b5  X Y Z  [prescribed values for fixed DOFs]
        """
        line = input_file.readline().split()

        N = int(line[0])
        if N != check_np + 1:
            error_info = "\n*** Error *** Nodes must be inputted in order !" \
                         "\n   Expected node number : {}" \
                         "\n   Provided node number : {}".format(check_np+1, N)
            raise ValueError(error_info)

        self.NodeNumber = N

        for d in range(CNode.NDF):
            self.bcode[d] = int(line[1 + d])
        self.is_constrained = np.array([int(line[1 + d]) for d in range(CNode.NDF)])

        self.XYZ[0] = np.double(line[1 + CNode.NDF])
        self.XYZ[1] = np.double(line[2 + CNode.NDF])
        self.XYZ[2] = np.double(line[3 + CNode.NDF])

        current_idx = 1 + CNode.NDF + 3
        for dof in range(CNode.NDF):
            if self.is_constrained[dof] == 1:
                if current_idx < len(line):
                    self.prescribed_values[dof] = np.double(line[current_idx])
                    current_idx += 1
                else:
                    self.prescribed_values[dof] = 0.0

    def Write(self, output_file):
        """
        Output nodal point data to stream
        """
        codes = ''.join("%5d" % self.bcode[d] for d in range(CNode.NDF))
        node_info = "%9d%s%15.6e%15.6e%15.6e\n" % (
            self.NodeNumber, codes, self.XYZ[0], self.XYZ[1], self.XYZ[2])
        print(node_info, end='')
        output_file.write(node_info)

    def WriteEquationNo(self, output_file):
        """
        Output equation numbers of nodal point to stream
        """
        equation_info = "%9d       " % self.NodeNumber

        for dof in range(CNode.NDF):
            equation_info += "%5d" % self.bcode[dof]

        equation_info += '\n'
        print(equation_info, end='')
        output_file.write(equation_info)

    def WriteNodalDisplacement(self, output_file, displacement):
        """
        Write nodal displacement
        """
        displacement_info = "%5d        " % self.NodeNumber

        for dof in range(CNode.NDF):
            if self.is_constrained[dof] == 1:
                val = self.prescribed_values[dof]
                displacement_info += "%18.6e" % val
            elif self.bcode[dof] > 0:
                displacement_info += "%18.6e" % displacement[self.bcode[dof] - 1]
            else:
                displacement_info += "%18.6e" % 0.0

        displacement_info += '\n'
        print(displacement_info, end='')
        output_file.write(displacement_info)
