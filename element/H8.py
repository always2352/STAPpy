#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H8 (8-node hexahedral) element - minimal implementation
"""
import sys
sys.path.append('../')
import numpy as np
from element.Element import CElement


class CH8(CElement):
	HG_COEF = 0.05

	def __init__(self):
		super().__init__()
		self._NEN = 8
		self._nodes = [None for _ in range(self._NEN)]

		# 8 nodes * 3 DOF per node
		self._ND = 24
		self._LocationMatrix = np.zeros(self._ND, dtype=int)

	def Read(self, input_file, Ele, MaterialSets, NodeList):
		"""Read H8 element connectivity
		Format: EleID N1 N2 N3 N4 N5 N6 N7 N8 MSet
		"""
		line = input_file.readline().split()

		N = int(line[0])
		if N != Ele + 1:
			error_info = "\n*** Error *** Elements must be inputted in order !" \
				 "\n   Expected element : {}" \
				 "\n   Provided element : {}".format(Ele + 1, N)
			raise ValueError(error_info)

		# read node numbers
		nodes = [int(x) for x in line[1:9]]
		MSet = int(line[9])
		self._ElementMaterial = MaterialSets[MSet - 1]

		for i in range(self._NEN):
			self._nodes[i] = NodeList[nodes[i] - 1]

	def Write(self, output_file, Ele):
		# ELEMENT     NODE ... MATERIAL
		nodes_str = ''.join(["%10d" % (n.NodeNumber,) for n in self._nodes])
		element_info = "%5d%s%12d\n" % (Ele+1, nodes_str, self._ElementMaterial.nset)
		print(element_info, end='')
		output_file.write(element_info)

	def GenerateLocationMatrix(self):
		i = 0
		for N in range(self._NEN):
			for D in range(3):
				self._LocationMatrix[i] = self._nodes[N].bcode[D]
				i += 1

	def MarkActiveDofs(self):
		""" A solid element stiffens only the three translations. """
		for node in self._nodes:
			for D in range(3):
				node.active[D] = True

	def SizeOfStiffnessMatrix(self):
		# upper triangular size of 24x24 matrix
		return int(self._ND * (self._ND + 1) / 2)

	def ElementStiffness(self, stiffness):
		"""
		Reduced (1-point) integration at the element centroid plus
		Flanagan-Belytschko hourglass control -- the technology of Abaqus
		C3D8R. One-point quadrature is exact for constant strain (the patch
		test passes), and the hourglass stiffness only resists the spurious
		zero-energy modes it leaves behind.
		"""
		ND = self._ND
		K = np.zeros((ND, ND), dtype=np.double)

		mat = self._ElementMaterial
		if mat is None:
			E = 1.0; nu = 0.3
		else:
			E = float(mat.E); nu = float(getattr(mat, 'nu', 0.3))

		factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
		D = np.zeros((6, 6), dtype=np.double)
		D[0, 0] = D[1, 1] = D[2, 2] = (1.0 - nu) * factor
		D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = nu * factor
		D[3, 3] = D[4, 4] = D[5, 5] = 0.5 * (1.0 - 2.0 * nu) * factor

		xi_c = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=np.double)
		eta_c = np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=np.double)
		zeta_c = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=np.double)
		X = np.array([nd.XYZ[0] for nd in self._nodes])
		Y = np.array([nd.XYZ[1] for nd in self._nodes])
		Z = np.array([nd.XYZ[2] for nd in self._nodes])

		# shape-function derivatives at the centroid (xi = eta = zeta = 0)
		dN_nat = 0.125 * np.vstack([xi_c, eta_c, zeta_c])      # 3 x 8
		J = dN_nat.dot(np.vstack([X, Y, Z]).T)                 # 3 x 3
		detJ = np.linalg.det(J)
		if detJ <= 0:
			raise ValueError("Jacobian determinant non-positive: {}".format(detJ))
		dN_dx = np.linalg.inv(J).dot(dN_nat)                   # 3 x 8 (d/dx, d/dy, d/dz)
		bx, by, bz = dN_dx[0], dN_dx[1], dN_dx[2]

		B = np.zeros((6, ND), dtype=np.double)
		for i in range(8):
			i3 = 3 * i
			B[0, i3] = bx[i]; B[1, i3 + 1] = by[i]; B[2, i3 + 2] = bz[i]
			B[3, i3] = by[i]; B[3, i3 + 1] = bx[i]
			B[4, i3 + 1] = bz[i]; B[4, i3 + 2] = by[i]
			B[5, i3] = bz[i]; B[5, i3 + 2] = bx[i]

		# reduced integration: single Gauss point, weight 2^3 = 8
		K += B.T.dot(D.dot(B)) * detJ * 8.0

		# Flanagan-Belytschko hourglass stabilization on the four hourglass modes
		mu = 0.5 * E / (1.0 + nu)
		vol = detJ * 8.0
		c_hg = self.HG_COEF * mu * vol * (bx.dot(bx) + by.dot(by) + bz.dot(bz))
		for h in (xi_c * eta_c, eta_c * zeta_c, zeta_c * xi_c, xi_c * eta_c * zeta_c):
			g = h - h.dot(X) * bx - h.dot(Y) * by - h.dot(Z) * bz   # orthogonal to linear field
			gg = c_hg * np.outer(g, g)
			for d in range(3):
				idx = np.arange(d, ND, 3)
				K[np.ix_(idx, idx)] += gg

		# small diagonal regularization for under-constrained rigid modes
		k_eps = 1e-9 * E
		for ii in range(ND):
			K[ii, ii] += k_eps

		pos = 0
		for j in range(ND):
			for i in range(j, -1, -1):
				stiffness[pos] = K[i, j]
				pos += 1

	def ElementStress(self, stress, displacement):
		"""Compute elemental stress at element center (ξ,η,ζ = 0)
		stress: array-like, will receive [vonMises, s_xx, s_yy]
		displacement: global displacement vector
		"""
		ND = self._ND
		# gather elemental displacement
		u = np.zeros(ND, dtype=np.double)
		for i in range(ND):
			lm = self._LocationMatrix[i]
			if lm:
				u[i] = displacement[lm - 1]
			else:
				u[i] = 0.0

		# evaluate B at centroid (xi=0,eta=0,zeta=0)
		xi = eta = zeta = 0.0

		# derivatives of N at centroid
		xi_coords = np.array([ -1,  1,  1, -1, -1,  1,  1, -1 ], dtype=np.double)
		eta_coords= np.array([ -1, -1,  1,  1, -1, -1,  1,  1 ], dtype=np.double)
		zeta_coords= np.array([ -1, -1, -1, -1,  1,  1,  1,  1 ], dtype=np.double)

		dN_dxi = np.zeros(8); dN_deta = np.zeros(8); dN_dzeta = np.zeros(8)
		for i in range(8):
			n1 = xi_coords[i]; n2 = eta_coords[i]; n3 = zeta_coords[i]
			dN_dxi[i] = 0.125 * n1 * (1.0 + n2*eta) * (1.0 + n3*zeta)
			dN_deta[i]= 0.125 * (1.0 + n1*xi) * n2 * (1.0 + n3*zeta)
			dN_dzeta[i]= 0.125 * (1.0 + n1*xi) * (1.0 + n2*eta) * n3

		J = np.zeros((3,3), dtype=np.double)
		for i in range(8):
			x = self._nodes[i].XYZ[0]
			y = self._nodes[i].XYZ[1]
			z = self._nodes[i].XYZ[2]
			J[0,0] += dN_dxi[i] * x; J[0,1] += dN_deta[i] * x; J[0,2] += dN_dzeta[i] * x
			J[1,0] += dN_dxi[i] * y; J[1,1] += dN_deta[i] * y; J[1,2] += dN_dzeta[i] * y
			J[2,0] += dN_dxi[i] * z; J[2,1] += dN_deta[i] * z; J[2,2] += dN_dzeta[i] * z

		detJ = np.linalg.det(J)
		invJ = np.linalg.inv(J)

		dN_dx = np.zeros((8,3), dtype=np.double)
		for i in range(8):
			dN_nat = np.array([dN_dxi[i], dN_deta[i], dN_dzeta[i]])
			dN_phys = invJ.dot(dN_nat)
			dN_dx[i,0] = dN_phys[0]; dN_dx[i,1] = dN_phys[1]; dN_dx[i,2] = dN_phys[2]

		B = np.zeros((6, ND), dtype=np.double)
		for i in range(8):
			i3 = 3 * i
			dNxi = dN_dx[i,0]; dNyi = dN_dx[i,1]; dNzi = dN_dx[i,2]
			B[0, i3    ] = dNxi
			B[1, i3 + 1] = dNyi
			B[2, i3 + 2] = dNzi
			B[3, i3    ] = dNyi
			B[3, i3 + 1] = dNxi
			B[4, i3 + 1] = dNzi
			B[4, i3 + 2] = dNyi
			B[5, i3    ] = dNzi
			B[5, i3 + 2] = dNxi

		strain = B.dot(u)

		# material D
		mat = self._ElementMaterial
		if mat is None:
			E = 1.0; nu = 0.3
		else:
			E = float(mat.E); nu = float(getattr(mat, 'nu', 0.3))
		factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
		D = np.zeros((6, 6), dtype=np.double)
		D[0, 0] = D[1, 1] = D[2, 2] = (1.0 - nu) * factor
		D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = nu * factor
		D[3, 3] = D[4, 4] = D[5, 5] = 0.5 * (1.0 - 2.0 * nu) * factor

		stress_vec = D.dot(strain)

		sxx = stress_vec[0]; syy = stress_vec[1]; szz = stress_vec[2]
		sxy = stress_vec[3]; syz = stress_vec[4]; sxz = stress_vec[5]

		von = np.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2)
		               + 3.0 * (sxy**2 + syz**2 + sxz**2))

		# fill output: von Mises stress and principal stresses
		stress[0] = von
		stress[1] = sxx
		stress[2] = syy
		if len(stress) > 3:
			stress[3] = szz
		if len(stress) > 4:
			stress[4] = sxy
		if len(stress) > 5:
			stress[5] = syz
		if len(stress) > 6:
			stress[6] = sxz

	def GetShapeFunctions(self, xi, eta, zeta=0.0):
		N = np.zeros(8)
		xi_coords = np.array([-1,  1,  1, -1, -1,  1,  1, -1], dtype=np.double)
		eta_coords = np.array([-1, -1,  1,  1, -1, -1,  1,  1], dtype=np.double)
		zeta_coords = np.array([-1, -1, -1, -1,  1,  1,  1,  1], dtype=np.double)
		
		for i in range(8):
			n1 = xi_coords[i]; n2 = eta_coords[i]; n3 = zeta_coords[i]
			N[i] = 0.125 * (1.0 + n1*xi) * (1.0 + n2*eta) * (1.0 + n3*zeta)
		
		return N

	def GetIntegrationPoints(self):
		# 2-point Gauss quadrature in each direction
		gp = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
		gw = [1.0, 1.0]
		
		points = []
		weights = []
		
		for a in range(2):
			for b in range(2):
				for c in range(2):
					points.append((gp[a], gp[b], gp[c]))
					weights.append(gw[a] * gw[b] * gw[c])
		
		return (points, weights)

	def GetDetJ(self, xi, eta, zeta=0.0):
		xi_coords = np.array([-1,  1,  1, -1, -1,  1,  1, -1], dtype=np.double)
		eta_coords = np.array([-1, -1,  1,  1, -1, -1,  1,  1], dtype=np.double)
		zeta_coords = np.array([-1, -1, -1, -1,  1,  1,  1,  1], dtype=np.double)

		dN_dxi = np.zeros(8); dN_deta = np.zeros(8); dN_dzeta = np.zeros(8)
		for i in range(8):
			n1 = xi_coords[i]; n2 = eta_coords[i]; n3 = zeta_coords[i]
			dN_dxi[i] = 0.125 * n1 * (1.0 + n2*eta) * (1.0 + n3*zeta)
			dN_deta[i] = 0.125 * (1.0 + n1*xi) * n2 * (1.0 + n3*zeta)
			dN_dzeta[i] = 0.125 * (1.0 + n1*xi) * (1.0 + n2*eta) * n3

		J = np.zeros((3,3), dtype=np.double)
		for i in range(8):
			x = self._nodes[i].XYZ[0]
			y = self._nodes[i].XYZ[1]
			z = self._nodes[i].XYZ[2]
			J[0,0] += dN_dxi[i] * x; J[0,1] += dN_deta[i] * x; J[0,2] += dN_dzeta[i] * x
			J[1,0] += dN_dxi[i] * y; J[1,1] += dN_deta[i] * y; J[1,2] += dN_dzeta[i] * y
			J[2,0] += dN_dxi[i] * z; J[2,1] += dN_deta[i] * z; J[2,2] += dN_dzeta[i] * z

		return np.linalg.det(J)
