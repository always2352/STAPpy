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


class CLoadCaseData(object):
    """ Class LoadData is used to store load data """
    def __init__(self):
        self.nloads = 0    #!< Number of concentrated loads in this load case
        self.node = None   #!< Node number to which this load is applied
        self.dof = None    #!< Degree of freedom number for this load component
        self.load = None   #!< Magnitude of load

    def Allocate(self, num):
        self.nloads = num
        self.node = np.zeros(num, dtype=int)
        self.dof = np.zeros(num, dtype=int)
        self.load = np.zeros(num, dtype=np.double)

    def Read(self, input_file, lcase):
        """
        Read load case data from stream Input

        :param input_file: (_io.TextIOWrapper) the object of input file
        :param lcase: check index
        :return: None
        """
        line = input_file.readline().split()

        LL = int(line[0])
        NL = int(line[1])

        if LL != lcase + 1:
            error_info = "\n*** Error *** Load case must be inputted in order !" \
                         "\n   Expected load case : {}" \
                         "\n   Provided load case : {}".format(lcase + 1, LL)
            raise ValueError(error_info)

        self.load_type = 'Concentrated' if NL >= 0 else 'Uniform'
        if self.load_type == 'Concentrated':
            self.Allocate(NL)

            for i in range(NL):
                line = input_file.readline().split()
                self.node[i] = int(line[0])
                self.dof[i] = int(line[1])
                self.load[i] = np.double(line[2])
        else:
            line = input_file.readline().split()
            q_magnitude = np.double(line[0])  
            
            import Domain as Domain
            FEMData = Domain()
            NodeList = FEMData.GetNodeList() 
            NUMNP = FEMData.GetNUMNP()

            global_nodal_forces = np.zeros((NUMNP + 1, 3))
            for group in FEMData.EleGrpList:
                for element in group._ElementList:
                    nodes = element.nodes 
                    
                    x1 = NodeList[nodes[0] - 1].XYZ[0]
                    x2 = NodeList[nodes[1] - 1].XYZ[0]
                    y1 = NodeList[nodes[0] - 1].XYZ[1]
                    y4 = NodeList[nodes[3] - 1].XYZ[1]
                    
                    ele_width = np.abs(x2 - x1)
                    ele_height = np.abs(y4 - y1)

                    a = ele_width / 2.0
                    b = ele_height / 2.0
                    
                    C = (q_magnitude * a * b) / 3.0
                    
                    xi_I  = [-1.0,  1.0, 1.0, -1.0]
                    eta_I = [-1.0, -1.0, 1.0,  1.0]
                    
                    for I in range(4):
                        global_node_num = nodes[I] 
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
            self.Allocate(NL_equivalent)
            self.nloads = NL_equivalent

            for i, load_item in enumerate(valid_loads):
                self.node[i] = load_item[0]
                self.dof[i] = load_item[1]
                self.load[i] = load_item[2]

    def Write(self, output_file, lcase):
        """
        Write load case data to stream

        :param output_file: (_io.TextIOWrapper) the object of output file
        :param lcase: the index of load case
        :return: None
        """
        for i in range(self.nloads):
            load_info = "%7d%13d%19.6e\n"%(self.node[i], self.dof[i],
                                           self.load[i])
            print(load_info, end="")
            output_file.write(load_info)