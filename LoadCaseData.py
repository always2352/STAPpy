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
        from Domain import Domain
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

            FEMData = Domain()
            NodeList = FEMData.GetNodeList()
            NUMNP = FEMData.GetNUMNP()

            x_coords = [node.XYZ[0] for node in NodeList]
            y_coords = [node.XYZ[1] for node in NodeList]
            max_x, min_x = max(x_coords), min(x_coords)
            max_y, min_y = max(y_coords), min(y_coords)
            
            L_x = max_x - min_x  
            L_y = max_y - min_y 

            unique_x = np.sort(np.unique(np.round(x_coords, 6)))
            unique_y = np.sort(np.unique(np.round(y_coords, 6)))
            
            ele_width = unique_x[1] - unique_x[0] if len(unique_x) > 1 else L_x
            ele_height = unique_y[1] - unique_y[0] if len(unique_y) > 1 else L_y
            a = ele_width / 2.0
            b = ele_height / 2.0

            global_nodal_forces = np.zeros((NUMNP + 1, 3))
            N_div_x = int(round(L_x / ele_width))  
            N_div_y = int(round(L_y / ele_height))

            for row in range(N_div_y):
                for col in range(N_div_x):
                    n1 = row * (N_div_x + 1) + col + 1
                    n2 = n1 + 1
                    n3 = n2 + (N_div_x + 1)
                    n4 = n3 - 1
                    
                    virtual_connectivity = [n1, n2, n3, n4]
                    
                    C = (q_magnitude * a * b) / 3.0
                    
                    xi_I  = [-1.0,  1.0, 1.0, -1.0]
                    eta_I = [-1.0, -1.0, 1.0,  1.0]
                    
                    for I in range(4):
                        global_node_num = virtual_connectivity[I]
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