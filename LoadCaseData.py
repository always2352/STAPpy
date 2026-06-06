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

        self.surface_pressure = 0.0 #！< Magnitude of surface pressure
        self.body_density = 0.0 #！< Magnitude of body force

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
        LL = int(line[0]) #载荷工况编号
        # Store load case number
        self.LL = LL
        
        if LL == 1:
            NL = int(line[1]) #载荷数量
            self.Allocate(NL)
            for i in range(NL):
                line = input_file.readline().split()
                self.node[i] = int(line[0])
                self.dof[i] = int(line[1])
                self.load[i] = np.double(line[2])
        elif LL == 2:
            self.Allocate(0)
        elif LL == 3:
            self.surface_pressure = np.double(line[1])
        elif LL == 4:
            self.surface_pressure = np.double(line[1])
    
    def Write(self, output_file, lcase):
        """
		Write load case data to stream

		:param output_file: (_io.TextIOWrapper) the object of output file
		:param lcase: the index of load case
		:return: None
		"""
        
        LL = lcase + 1	

        if LL == 1:
            for i in range(self.nloads):
                load_info = "%7d%13d%19.6e\n"%(self.node[i], self.dof[i],self.load[i])
                print(load_info, end="")
                output_file.write(load_info)
        elif LL == 3:
            pass
        elif LL == 4:
            pass