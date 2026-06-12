#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-processing: export the solved model to a legacy VTK unstructured grid
(.vtk) for ParaView. Point data carries the nodal displacement/rotation
vectors; cell data carries a per-element stress measure.

The file is plain ASCII VTK, so it needs no extra Python packages and can be
opened directly in ParaView (File > Open) or driven with paraview.simple.
"""
import numpy as np

# STAP element type -> (VTK cell type, nodes per cell)
#   VTK_LINE = 3, VTK_QUAD = 9, VTK_HEXAHEDRON = 12
_VTK_TYPE = {1: (3, 2), 5: (3, 2), 6: (9, 4), 4: (12, 8)}


def _nodal_vectors(nodes, disp):
    """Return (translations Nx3, rotations Nx3) for every node."""
    trans = np.zeros((len(nodes), 3))
    rot = np.zeros((len(nodes), 3))
    for i, nd in enumerate(nodes):
        for d in range(6):
            if nd.is_constrained[d] == 1:
                val = nd.prescribed_values[d]
            elif nd.bcode[d] > 0:
                val = disp[nd.bcode[d] - 1]
            else:
                val = 0.0
            (trans if d < 3 else rot)[i, d % 3] = val
    return trans, rot


def _element_stress(element, etype, disp):
    """A single representative stress measure per element."""
    if etype == 1:                       # bar: axial stress
        s = np.zeros(1)
        element.ElementStress(s, disp)
        return abs(s[0])
    if etype == 5:                       # beam: largest end moment
        s = np.zeros(3)
        element.ElementStress(s, disp)
        return max(abs(s[1]), abs(s[2]))
    if etype == 6:                       # plate: von Mises of the moments
        s = np.zeros(3)
        element.ElementStress(s, disp)
        mx, my, mxy = s[0], s[1], s[2]
        return float(np.sqrt(mx * mx - mx * my + my * my + 3.0 * mxy * mxy))
    if etype == 4:                       # solid: von Mises stress
        s = np.zeros(7)
        element.ElementStress(s, disp)
        return s[0]
    return 0.0


def WriteVTK(filename, write_stress=True):
    """ Write the current (solved) Domain to a legacy .vtk file.

    write_stress=False skips the per-element stress computation and the
    CELL_DATA section (faster for large meshes).
    """
    from Domain import Domain
    FEMData = Domain()
    nodes = FEMData.GetNodeList()
    disp = FEMData.GetDisplacement()

    trans, rot = _nodal_vectors(nodes, disp)

    cell_conn, cell_types, cell_stress = [], [], []
    for g in range(FEMData.GetNUMEG()):
        grp = FEMData.GetEleGrpList()[g]
        etype = grp.GetElementType()
        vtk_type, _ = _VTK_TYPE[etype]
        for e in range(grp.GetNUME()):
            ele = grp[e]
            conn = [nd.NodeNumber - 1 for nd in ele.GetNodes()]
            cell_conn.append(conn)
            cell_types.append(vtk_type)
            if write_stress:
                cell_stress.append(_element_stress(ele, etype, disp))

    ncell = len(cell_conn)
    cell_size = sum(len(c) + 1 for c in cell_conn)

    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("STAPpy results\nASCII\nDATASET UNSTRUCTURED_GRID\n")

        f.write("POINTS %d double\n" % len(nodes))
        for nd in nodes:
            f.write("%g %g %g\n" % (nd.XYZ[0], nd.XYZ[1], nd.XYZ[2]))

        f.write("CELLS %d %d\n" % (ncell, cell_size))
        for c in cell_conn:
            f.write("%d %s\n" % (len(c), " ".join(str(i) for i in c)))
        f.write("CELL_TYPES %d\n" % ncell)
        for t in cell_types:
            f.write("%d\n" % t)

        f.write("POINT_DATA %d\n" % len(nodes))
        f.write("VECTORS Displacement double\n")
        for t in trans:
            f.write("%g %g %g\n" % (t[0], t[1], t[2]))
        f.write("SCALARS Displacement_Magnitude double 1\nLOOKUP_TABLE default\n")
        for t in trans:
            f.write("%g\n" % np.linalg.norm(t))
        f.write("VECTORS Rotation double\n")
        for r in rot:
            f.write("%g %g %g\n" % (r[0], r[1], r[2]))

        if write_stress:
            f.write("CELL_DATA %d\n" % ncell)
            f.write("SCALARS Stress_Measure double 1\nLOOKUP_TABLE default\n")
            for s in cell_stress:
                f.write("%g\n" % s)

    print(" Visualization written: %s (%d points, %d cells)"
          % (filename, len(nodes), ncell))
