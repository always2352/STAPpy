#!/usr/bin/env python3
import os
import numpy as np
from Domain import Domain
from solver.LDLTSolver import CLDLTSolver
from utils.Outputter import COutputter

input_file = 'data/h8_test.dat'
output_file = 'data/h8_test.out'

FEM = Domain()
if not FEM.ReadData(input_file, output_file):
    raise SystemExit('ReadData failed')

FEM.AllocateMatrices()
FEM.AssembleStiffnessMatrix()
Solver = CLDLTSolver(FEM.GetStiffnessMatrix())
Solver.LDLT()

# Loop over load cases and solve; store displacement for first load case
for lcase in range(FEM.GetNLCASE()):
    FEM.AssembleForce(lcase + 1)
    Solver.BackSubstitution(FEM.GetForce())
    # write post files for this load case
    disp = FEM.GetDisplacement()
    # extract nodal displacements in array (NUMNP x 3)
    NodeList = FEM.GetNodeList()
    NUMNP = FEM.GetNUMNP()
    U = np.zeros((NUMNP, 3))
    for i in range(NUMNP):
        for d in range(3):
            val = NodeList[i].bcode[d]
            if val:
                U[i, d] = disp[val - 1]
            else:
                U[i, d] = 0.0

    # write VTK (legacy unstructured grid)
    vtk_path = f'data/post_lcase{lcase+1}.vtk'
    with open(vtk_path, 'w') as f:
        f.write('# vtk DataFile Version 2.0\n')
        f.write('H8 result\n')
        f.write('ASCII\n')
        f.write('DATASET UNSTRUCTURED_GRID\n')
        f.write(f'POINTS {NUMNP} float\n')
        for n in range(NUMNP):
            x,y,z = NodeList[n].XYZ
            f.write(f'{x} {y} {z}\n')
        # build cell list
        cells = []
        for EleGrp in FEM.GetEleGrpList():
            if EleGrp.GetElementType() == 4 or EleGrp.GetElementType() == 6:
                for e in range(EleGrp.GetNUME()):
                    elem = EleGrp[e]
                    node_ids = [nd.NodeNumber - 1 for nd in elem.GetNodes()]
                    cells.append(node_ids)
        total_idx = sum([1 + len(c) for c in cells])
        f.write(f'\nCELLS {len(cells)} {total_idx}\n')
        for c in cells:
            f.write(str(len(c)) + ' ' + ' '.join(map(str, c)) + '\n')
        f.write(f'\nCELL_TYPES {len(cells)}\n')
        for c in cells:
            # VTK_HEXAHEDRON = 12
            if len(c) == 8:
                f.write('12\n')
            else:
                f.write('0\n')
        # point data: displacement
        f.write(f'\nPOINT_DATA {NUMNP}\n')
        f.write('VECTORS displacement float\n')
        for n in range(NUMNP):
            f.write(f'{U[n,0]} {U[n,1]} {U[n,2]}\n')
    print('Wrote', vtk_path)

    # write Tecplot ASCII (FE Zone)
    tec_path = f'data/post_lcase{lcase+1}.dat'
    with open(tec_path, 'w') as f:
        f.write('TITLE = "H8 result"\n')
        f.write('VARIABLES = "X","Y","Z","Ux","Uy","Uz"\n')
        # unstructured zone
        num_cells = len(cells)
        f.write(f'ZONE T="Loadcase{lcase+1}", N={NUMNP}, E={num_cells}, F=FEPOINT, ET=BRICK\n')
        for n in range(NUMNP):
            x,y,z = NodeList[n].XYZ
            ux,uy,uz = U[n]
            f.write(f'{x} {y} {z} {ux} {uy} {uz}\n')
        for c in cells:
            # Tecplot node indices are 1-based
            f.write(' '.join(str(i+1) for i in c) + '\n')
    print('Wrote', tec_path)

print('Postprocessing export complete')
