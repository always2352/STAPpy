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
	""" Beam element class based on Bernoulli-Euler beam theory """
	def __init__(self):
		super().__init__()
		self._NEN = 2
		self._nodes = [None for _ in range(self._NEN)]

		self._ND = 6
		self._LocationMatrix = np.zeros(self._ND, dtype=int)

	def Read(self, input_file, Ele, MaterialSets, NodeList):
		"""
		Read element data from stream Input
		"""
		line = input_file.readline().split()

		N = int(line[0])
		if N != Ele + 1:
			error_info = "\n*** Error *** Elements must be inputted in order !" \
						 "\n   Expected element : {}" \
						 "\n   Provided element : {}".format(Ele + 1, N)
			raise ValueError(error_info)

		N1 = int(line[1])
		N2 = int(line[2])
		MSet = int(line[3])

		self._ElementMaterial = MaterialSets[MSet - 1]
		self._nodes[0] = NodeList[N1 - 1]
		self._nodes[1] = NodeList[N2 - 1]

	def Write(self, output_file, Ele):
		"""
		Write element data to stream
		"""
		element_info = "%5d%11d%9d%12d\n" % (
			Ele + 1,
			self._nodes[0].NodeNumber,
			self._nodes[1].NodeNumber,
			self._ElementMaterial.nset,
		)

		print(element_info, end='')
		output_file.write(element_info)

	def GenerateLocationMatrix(self):
		"""
		Generate location matrix: the global equation number that
		corresponding to each DOF of the element
		"""
		i = 0
		for N in range(self._NEN):
			for D in range(3):
				self._LocationMatrix[i] = self._nodes[N].bcode[D]
				i += 1

	def SizeOfStiffnessMatrix(self):
		"""
		Return the size of the element stiffness matrix
		(stored as an array column by column)
		For 2 node beam element, element stiffness is a 6x6 matrix,
		whose upper triangular part has 21 elements
		"""
		return 21

	def _ExtractGeometry(self):
		"""
		Build the local frame from the 3D node coordinates.
		e1: element axis; e3: bending-plane normal (material normal snapped to
		the nearest global axis); e2 = e3 x e1: in-plane transverse direction.
		The beam bends in the e1-e2 plane and rotates about axis k (= e3).
		"""
		d = self._nodes[1].XYZ - self._nodes[0].XYZ
		length = np.sqrt(d.dot(d))
		if length <= 0.0:
			raise ValueError("Beam element has zero length.")
		e1 = d / length

		nrm = self._ElementMaterial.normal
		k = int(np.argmax(np.abs(nrm)))
		sgn = 1.0 if nrm[k] >= 0.0 else -1.0
		e3 = np.zeros(3)
		e3[k] = sgn

		e2 = np.cross(e3, e1)
		n2 = np.sqrt(e2.dot(e2))
		if n2 <= 1e-12:
			raise ValueError("Beam axis is parallel to its bending-plane normal.")
		e2 = e2 / n2
		return length, e1, e2, e3, k, sgn

	def _GetTransformationMatrix(self):
		"""
		Transformation from the 3 nodal DOFs (two in-plane translations + one
		rotation about the normal axis) to the local (u, v, theta) DOFs.
		"""
		length, e1, e2, e3, k, sgn = self._ExtractGeometry()
		p, q = (i for i in range(3) if i != k)

		block = np.zeros((3, 3))
		block[0, p] = e1[p]; block[0, q] = e1[q]   # u : axial
		block[1, p] = e2[p]; block[1, q] = e2[q]   # v : in-plane transverse
		block[2, k] = sgn                          # theta : rotation about normal

		T = np.zeros((6, 6))
		T[0:3, 0:3] = block
		T[3:6, 3:6] = block
		return T, length

	def _GetLocalStiffness(self, length):
		material = self._ElementMaterial
		E = material.E
		A = material.Area
		I = material.Inertia

		EA_L = E * A / length
		EI = E * I
		L2 = length * length
		L3 = L2 * length

		K = np.zeros((6, 6))
		K[0, 0] = EA_L
		K[0, 3] = -EA_L
		K[3, 0] = -EA_L
		K[3, 3] = EA_L

		K[1, 1] = 12.0 * EI / L3
		K[1, 2] = 6.0 * EI / L2
		K[1, 4] = -12.0 * EI / L3
		K[1, 5] = 6.0 * EI / L2

		K[2, 1] = 6.0 * EI / L2
		K[2, 2] = 4.0 * EI / length
		K[2, 4] = -6.0 * EI / L2
		K[2, 5] = 2.0 * EI / length

		K[4, 1] = -12.0 * EI / L3
		K[4, 2] = -6.0 * EI / L2
		K[4, 4] = 12.0 * EI / L3
		K[4, 5] = -6.0 * EI / L2

		K[5, 1] = 6.0 * EI / L2
		K[5, 2] = 2.0 * EI / length
		K[5, 4] = -6.0 * EI / L2
		K[5, 5] = 4.0 * EI / length

		return K

	def ElementStiffness(self, stiffness):
		"""
		Calculate element stiffness matrix
		Upper triangular matrix, stored as an array column by column
		starting from the diagonal element
		"""
		for i in range(self.SizeOfStiffnessMatrix()):
			stiffness[i] = 0.0

		T, length = self._GetTransformationMatrix()
		K_local = self._GetLocalStiffness(length)
		K_global = np.dot(T.T, np.dot(K_local, T))

		count = 0
		for col in range(6):
			for row in range(col, -1, -1):
				stiffness[count] = K_global[row, col]
				count += 1

	def ElementStress(self, stress, displacement):
		"""
		Calculate beam internal force resultants
		stress[0]: axial force
		stress[1]: end moment at node I
		stress[2]: end moment at node J
		"""
		T, length = self._GetTransformationMatrix()
		K_local = self._GetLocalStiffness(length)

		d_global = np.zeros(6)
		for i in range(6):
			global_eq = self._LocationMatrix[i]
			if global_eq > 0:
				d_global[i] = displacement[global_eq - 1]

		d_local = np.dot(T, d_global)
		local_force = np.dot(K_local, d_local)

		stress[0] = local_force[0]
		stress[1] = local_force[2]
		stress[2] = local_force[5]

	def GetShapeFunctions(self, xi, eta=0.0, zeta=0.0):
		"""
		Get shape function values for 2-node Bernoulli-Euler beam element
		Returns shape functions for (u, v, theta) at each node
		"""
		length = self._ExtractGeometry()[0]
		N = np.zeros((2, 3))

		N1 = 0.5 * (1.0 - xi)
		N2 = 0.5 * (1.0 + xi)
		H1 = 0.25 * (1.0 - xi) ** 2 * (2.0 + xi)
		H2 = 0.125 * length * (1.0 - xi) ** 2 * (1.0 + xi)
		H3 = 0.25 * (1.0 + xi) ** 2 * (2.0 - xi)
		H4 = 0.125 * length * (1.0 + xi) ** 2 * (xi - 1.0)

		N[0, 0] = N1
		N[0, 1] = H1
		N[0, 2] = H2
		N[1, 0] = N2
		N[1, 1] = H3
		N[1, 2] = H4

		return N

	def GetIntegrationPoints(self):
		"""
		Get integration points for beam element (2-point Gauss)
		"""
		points = [(-1.0 / np.sqrt(3.0), 0.0, 0.0), (1.0 / np.sqrt(3.0), 0.0, 0.0)]
		weights = [1.0, 1.0]
		return points, weights

	def GetDetJ(self, xi=0.0, eta=0.0, zeta=0.0):
		"""
		Calculate determinant of Jacobian for beam element
		"""
		length = self._ExtractGeometry()[0]
		return length / 2.0
