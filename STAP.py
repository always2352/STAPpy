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

Usage:
	$ python STAP.py file_name
or
	>>> STAP file_name

Command line arguments:
	file_name: Input file name with the postfix of .dat or without postfix
"""
from Domain import Domain
from utils.Outputter import COutputter
from utils.Clock import Clock
#from utils.PostProcessor import GenerateVisualization
from solver.LDLTSolver import CLDLTSolver
from sys import argv, exit


if __name__ == "__main__":
	nargs = len(argv)
	if nargs != 2:
		print("Usage: \n\t$ python STAP.py InputFileName")
		exit(1)

	filename = argv[1]
	found = filename.rfind('.')

	# If the input file name is provided with an extension
	if found != -1:
		if filename[found:] == ".dat":
			filename = filename[:found]
		else:
			print("*** Error *** Invalid file extension: {}".format(
				filename[found+1:]))
			exit(1)

	input_filename = filename + ".dat"
	output_filename = filename + ".out"

	FEMData = Domain()

	timer = Clock()
	timer.Start()

	# Read data and define the problem domain
	if not FEMData.ReadData(input_filename, output_filename):
		print("*** Error *** Data input failed!")
		exit(1)

	time_input = timer.ElapsedTime()

	# Allocate global vectors and matrices, such as the Force, ColumnHeights,
	# DiagonalAddress and StiffnessMatrix, and calculate the column heights
	# and address of diagonal elements
	FEMData.AllocateMatrices()

	# Assemble the banded gloabl stiffness matrix
	FEMData.AssembleStiffnessMatrix()

	time_assemble = timer.ElapsedTime()

	# Solve the linear equilibrium equations for displacements
	Solver = CLDLTSolver(FEMData.GetStiffnessMatrix())

	# Perform L*D*L(T) factorization of stiffness matrix
	Solver.LDLT()

	Output = COutputter()

	# Loop over for all load cases (using actual load case numbers from dictionary)
	# pyrefly: ignore [missing-attribute]
	for LL in FEMData.GetLoadCases().keys():
		# Assemble righ-hand-side vector (force vector)
		FEMData.AssembleForce(LL)

	# Reduce right-hand-side force vector and back substitute
	Solver.BackSubstitution(FEMData.GetForce())
	Output.OutputNodalDisplacement(0)

	time_solution = timer.ElapsedTime()

	# Calculate and output stresses of all elements
	Output.OutputElementStress()

	time_stress = timer.ElapsedTime()

	timer.Stop()

	time_info = "\n S O L U T I O N   T I M E   L O G   I N   S E C \n\n" \
				"     TIME FOR INPUT PHASE = {}\n" \
				"     TIME FOR CALCULATION OF STIFFNESS MATRIX = {}\n" \
				"     TIME FOR FACTORIZATION AND LOAD CASE SOLUTIONS = {}\n" \
				"     T O T A L   S O L U T I O N   T I M E = {}\n".format(
		time_input, time_assemble - time_input,
		time_solution - time_assemble, time_stress
	)
	Output.OutputSolutionTime(time_info)

	# Generate ParaView/VTK visualization files
	#print("\n Generating visualization files for ParaView...")
	#vis_filename = filename + "_results"
	#if GenerateVisualization(FEMData, vis_filename, scale_factor=1.0):
	#	print(f" ✓ Visualization files created:")
	#	print(f"   - {vis_filename}.vtk (for ParaView)")
	#	print(f"   - {vis_filename}.dat (for Tecplot)")
	#	print(f"\n To view results in ParaView:")
	#	print(f"   1. Open {vis_filename}.vtk")
	#	print(f"   2. Apply filters -> Warp by Vector (for displacement)")
	#	print(f"   3. Color by von_mises_stress or other scalars")
	#else:
	#	print(" ✗ Failed to generate visualization files")
