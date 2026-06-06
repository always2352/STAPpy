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
from utils.Singleton import Singleton
from utils.Outputter import COutputter
from element.Node import CNode
from LoadCaseData import CLoadCaseData
from element.ElementGroup import CElementGroup
from utils.SkylineMatrix import CSkylineMatrix
import numpy as np
import sys


@Singleton
class Domain(object):
    """
    Domain class : Define the problem domain
    Only a single instance of Domain class can be created
    """
    def __init__(self):
        super().__init__()

        # Input file stream for reading data from input data file
        self.input_file = None

        # Heading information for use in labeling the output
        self.Title = '0'

        # Solution MODEX
        # 		0 : Data check only
        # 		1 : Execution
        self.MODEX = 0

        # Total number of nodal points
        self.NUMNP = 0

        # List of all nodes in the domain
        self.NodeList = []

        # Total number of element groups
        self.NUMEG = 0

        # Element group list
        self.EleGrpList = []

        # Number of load cases
        self.NLCASE = 0

        # Number of concentrated loads applied in each load case
        self.NLOAD = []

        # List of all load cases
        self.LoadCases = []

        # Total number of equations in the system
        self.NEQ = 0

        # Global nodal force/displacement vector
        self.Force = None

        # Banded stiffness matrix
        # A one-dimensional array storing only the elements below the
        # skyline of the global stiffness matrix.
        self.StiffnessMatrix = None
    
    def GetGRAVITY(self):
        return self.GRAVITY
    
    def GetMODEX(self):
        return self.MODEX

    def GetTitle(self):
        return self.Title

    def GetNEQ(self):
        return self.NEQ

    def GetNUMNP(self):
        return self.NUMNP

    def GetNodeList(self):
        return self.NodeList

    def GetNUMEG(self):
        return self.NUMEG

    def GetEleGrpList(self):
        return self.EleGrpList

    def GetForce(self):
        return self.Force

    def GetDisplacement(self):
        return self.Force

    def GetNLCASE(self):
        return self.NLCASE

    def GetNLOAD(self):
        return self.NLOAD

    def GetLoadCases(self):
        return self.LoadCases

    def GetStiffnessMatrix(self):
        return self.StiffnessMatrix

    def ReadData(self, input_filename, output_filename):
        """ Read domain data from the input data file """
        try:
            self.input_file = open(input_filename)
        except FileNotFoundError as e:
            print(e)
            sys.exit(3)

        Output = COutputter(output_filename)

        # Read the heading line
        self.Title = self.input_file.readline()
        Output.OutputHeading()

		# Read the control line
        line = self.input_file.readline().split()
        self.NUMNP = int(line[0])
        self.NUMEG = int(line[1])
        self.NLCASE = int(line[2])
        self.MODEX = int(line[3])
        self.GRAVITY = np.double(line[4])

        # Read nodal point data
        if self.ReadNodalPoints():
            Output.OutputNodeInfo()
        else:
            return False

        # Update equation number
        self.CalculateEquationNumber()
        Output.OutputEquationNumber()

        # Read load data
        if self.ReadLoadCases():
            Output.OutputLoadInfo()
        else:
            return False

        # Read element data
        if self.ReadElements():
            Output.OutputElementInfo()
        else:
            return False

        self.AssembleEquivalentLoads()

        return True

    def ReadNodalPoints(self):
        """ Read nodal point data """
        self.NodeList = [CNode() for _ in range(self.NUMNP)]

        for np in range(self.NUMNP):
            try:
                self.NodeList[np].Read(self.input_file, np)
            except ValueError as e:
                print(e)
                return False

        return True

    def CalculateEquationNumber(self):
        """
        Calculate global equation numbers corresponding to every
        degree of freedom of each node
        """
        self.NEQ = 0

        for np in range(self.NUMNP):
            for dof in range(CNode.NDF):
                if self.NodeList[np].bcode[dof]:
                    self.NodeList[np].bcode[dof] = 0
                else:
                    self.NEQ += 1
                    self.NodeList[np].bcode[dof] = self.NEQ

    def ReadLoadCases(self):
        """ Read load case data """
        self.LoadCases = [CLoadCaseData() for _ in range(self.NLCASE)]

        for lcase in range(self.NLCASE):
            try:
                self.LoadCases[lcase].Read(self.input_file, lcase)
            except ValueError as e:
                print(e)
                return False

        return True

    def ReadElements(self):
        """ Read element data """
        self.EleGrpList = [CElementGroup() for _ in range(self.NUMEG)]

        for EleGrp in range(self.NUMEG):
            if not self.EleGrpList[EleGrp].Read(self.input_file):
                return False

        return True

    def CalculateColumnHeights(self):
        """ Calculate column heights """
        for EleGrp in range(self.NUMEG):
            ElementGrp = self.EleGrpList[EleGrp]
            NUME = ElementGrp.GetNUME()

            for Ele in range(NUME):
                Element = ElementGrp[Ele]

                Element.GenerateLocationMatrix()

                self.StiffnessMatrix.CalculateColumnHeight(
                    Element.GetLocationMatrix(), Element.GetND())

        self.StiffnessMatrix.CalculateMaximumHalfBandwidth()

    def AssembleStiffnessMatrix(self):
        """ Assemble the banded gloabl stiffness matrix """
        # Loop over for all element groups
        for EleGrp in range(self.NUMEG):
            ElementGrp = self.EleGrpList[EleGrp]
            NUME = ElementGrp.GetNUME()
            size = ElementGrp[0].SizeOfStiffnessMatrix()
            Matrix = np.zeros(size, dtype=np.double)

            # Loop over for all elements in group EleGrp
            for Ele in range(NUME):
                Element = ElementGrp[Ele]
                Element.ElementStiffness(Matrix)
                self.StiffnessMatrix.Assembly(Matrix,
                    Element.GetLocationMatrix(), Element.GetND())

            del Matrix

    def AssembleForce(self, LoadCase):
        """ Assemble the global nodal force vector for load case LoadCase """
        if LoadCase > self.NLCASE:
            return False

        LoadData = self.LoadCases[LoadCase - 1]
        
        if LoadCase == 1:
			# Loop over for all concentrated loads in load case LoadCase
			# 节点集中力可直接组装至全局力向量
            for lnum in range(LoadData.nloads):
                dof = self.NodeList[LoadData.node[lnum]-1].bcode[LoadData.dof[lnum]-1]
                if dof:
                    self.Force[dof - 1] += LoadData.load[lnum]
        elif LoadCase == 2:
            self.AssembleGravityForce()
        elif LoadCase == 3:
            self.AssembleSurfaceForce()
        elif LoadCase == 4:
            self.AssembleBodyForce()

        
        for EleGrp in range(self.NUMEG):
            ElementGrp = self.EleGrpList[EleGrp]
            if ElementGrp.GetElementType() != 6: 
                continue
            
            NUME = ElementGrp.GetNUME()
            size = ElementGrp[0].SizeOfStiffnessMatrix()
            stiffness_array = np.zeros(size, dtype=np.double)

            for Ele in range(NUME):
                element = ElementGrp[Ele]
                element.ElementStiffness(stiffness_array)
                
                Ke_full = np.zeros((12, 12))
                count = 0
                for col in range(12):
                    for row in range(col, -1, -1):
                        Ke_full[row, col] = stiffness_array[count]
                        Ke_full[col, row] = stiffness_array[count]
                        count += 1
                
                U_boundary = np.zeros(12)
                for I in range(4):
                    node = element._nodes[I]
                    for d in range(3):
                        if node.is_constrained[d] == 1:
                            U_boundary[I*3 + d] = node.prescribed_values[d]

                element_dof_global = np.zeros(12, dtype=int)
                idx = 0
                for I in range(4):
                    node = element._nodes[I]
                    for d in range(3):
                        element_dof_global[idx] = node.bcode[d]
                        idx += 1

                for r in range(12):
                    global_eq_r = element_dof_global[r]
                    if global_eq_r > 0: 
                        
                        for c in range(12):
                            node_c = element._nodes[c // 3]
                            dof_c = c % 3
                            
                            if node_c.is_constrained[dof_c] == 1 and np.abs(U_boundary[c]) > 1e-15:
                                # F = F - K_rc * U_boundary_c
                                self.Force[global_eq_r - 1] -= Ke_full[r, c] * U_boundary[c]
        return True
    
    def AssembleGravityForce(self):
        """ Assemble gravity forces for all elements """
        for EleGrp in range(self.NUMEG):
            ElementGrp = self.EleGrpList[EleGrp]
            NUME = ElementGrp.GetNUME() # 该单元组的单元个数
            
            if ElementGrp.GetElementType() == 1:  # bar element
                for Ele in range(NUME):
                    Element = ElementGrp[Ele]
                    nodes = Element.GetNodes()
                    material = Element.GetElementMaterial()
                    
                    dx = nodes[1].XYZ[0] - nodes[0].XYZ[0]
                    dy = nodes[1].XYZ[1] - nodes[0].XYZ[1]
                    dz = nodes[1].XYZ[2] - nodes[0].XYZ[2]
                    length = np.sqrt(dx**2 + dy**2 + dz**2)
                    weight = material.rho * material.Area * length * self.GRAVITY
                    
                    loc = Element.GetLocationMatrix() # 单元局部自由度对应的全局自由度编号
                    if loc[2] != 0:
                        self.Force[loc[2] - 1] -= weight / 2.0
                    if loc[5] != 0:
                        self.Force[loc[5] - 1] -= weight / 2.0
            elif ElementGrp.GetElementType() == 4:  # H8 element
                pass
            elif ElementGrp.GetElementType() == 5:  # beam element
                pass
            elif ElementGrp.GetElementType() == 6:  # plate element
                pass
            
    def AssembleSurfaceForce(self):
        """ Assemble surface forces """
        pass
    
    def AssembleBodyForce(self):
        """ Assemble body forces """
        pass
    
    def AllocateMatrices(self):
        """
        Allocate storage for matrices Force, ColumnHeights, DiagonalAddress
        and StiffnessMatrix and calculate the column heights and address
        of diagonal elements
        """
        # Allocate for global force/displacement vector
        self.Force = np.zeros(self.NEQ, dtype=np.double)

        # Create the banded stiffness matrix
        self.StiffnessMatrix = CSkylineMatrix(self.NEQ)

        # Calculate column heights
        self.CalculateColumnHeights()

        # Calculate address of diagonal elements in banded matrix
        self.StiffnessMatrix.CalculateDiagnoalAddress()

        # Allocate for banded global stiffness matrix
        self.StiffnessMatrix.Allocate()

        Output = COutputter()
        Output.OutputTotalSystemData()


    def AssembleEquivalentLoads(self):
        NUMNP = self.GetNUMNP()

        for lcase_data in self.LoadCases: 
            if lcase_data.LL == 4:
                q_magnitude = lcase_data.q_magnitude
                
                global_nodal_forces = np.zeros((NUMNP + 1, 3))
                
                xi_I  = [-1.0,  1.0,  1.0, -1.0]
                eta_I = [-1.0, -1.0,  1.0,  1.0]

                for EleGrp in range(self.NUMEG):
                    ElementGrp = self.EleGrpList[EleGrp]
                    NUME = ElementGrp.GetNUME()

                    for Ele in range(NUME):
                        element = ElementGrp[Ele]

                        node1 = element._nodes[0]
                        node2 = element._nodes[1]
                        node4 = element._nodes[3]
                    
                        a = (node2.XYZ[0] - node1.XYZ[0]) / 2.0
                        b = (node4.XYZ[1] - node1.XYZ[1]) / 2.0
                        
                        C = (q_magnitude * a * b) / 3.0

                        for I in range(4):
                            node_obj = element._nodes[I]
                            global_node_num = node_obj.NodeNumber
                            
                            xI, eI = xi_I[I], eta_I[I]
                            
                            global_nodal_forces[global_node_num, 0] += C * 3.0
                            global_nodal_forces[global_node_num, 1] += C * b * eI
                            global_nodal_forces[global_node_num, 2] += -C * a * xI

                valid_loads = []
                for node_num in range(1, NUMNP + 1):
                    for dof_idx in range(3):
                        val = global_nodal_forces[node_num, dof_idx]
                        if np.abs(val) > 1e-11:
                            valid_loads.append((node_num, dof_idx + 1, val))

                NL_equivalent = len(valid_loads)
                lcase_data.Allocate(NL_equivalent)
                
                for i, load_item in enumerate(valid_loads):
                    lcase_data.node[i] = load_item[0]
                    lcase_data.dof[i] = load_item[1]
                    lcase_data.load[i] = load_item[2]