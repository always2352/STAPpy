#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve the full bridge under self-weight and export a ParaView .vtk file.

The model is read and assembled by the STAP CDomain exactly as STAP.py does;
the (large, banded) linear system is then factored with SciPy because the
in-house pure-Python skyline LDLT is impractically slow at this size. The
displacement is written back into CDomain so the standard stress output and
VTK exporter can be reused.

Usage:  python bridge_solve_export.py  [data/Bridge-1.dat]  [Bridge-1.vtk]
"""
import os
import sys
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from Domain import Domain
from utils.Outputter import COutputter
from utils.PostProcessor import WriteVTK


def skyline_to_csc(K):
    NEQ = K.dim()
    DA, CH, data = K._DiagonalAddress, K._ColumnHeights, K._data
    rows, cols, vals = [], [], []
    for j in range(1, NEQ + 1):
        o = np.arange(int(CH[j - 1]) + 1)
        rows.append(j - o)
        cols.append(np.full(o.size, j))
        vals.append(data[DA[j - 1] - 1 + o])
    U = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows) - 1, np.concatenate(cols) - 1)),
                   shape=(NEQ, NEQ)).tocsr()
    return (U + U.T - diags(U.diagonal())).tocsc()


def apply_ties(A, mpc_path, nodes):
    """
    Apply node-to-node translation ties (Abaqus *Tie, no rotation) by the
    penalty method: for each tied pair the three translations are forced equal.
    A no-op when the .mpc file is missing/empty (all interfaces already merged).
    """
    if not os.path.exists(mpc_path):
        return A
    pairs = [tuple(int(x) for x in ln.split()) for ln in open(mpc_path) if ln.split()]
    if not pairs:
        return A
    alpha = 1.0e6 * np.median(A.diagonal())
    r, c, v = [], [], []
    for s, m in pairs:
        for d in range(3):                       # ux, uy, uz
            es, em = nodes[s - 1].bcode[d], nodes[m - 1].bcode[d]
            if es > 0 and em > 0 and es != em:
                i, j = es - 1, em - 1
                r += [i, j, i, j]
                c += [i, j, j, i]
                v += [alpha, alpha, -alpha, -alpha]
    if not v:
        return A
    P = coo_matrix((v, (r, c)), shape=A.shape).tocsc()
    print(" applied %d ties (penalty)" % len(pairs))
    return (A + P).tocsc()


def self_weight(FEMData):
    """ Total self-weight = sum over elements of rho * g * volume. """
    g = FEMData.GetGRAVITY()
    W = 0.0
    for grp_i in range(FEMData.GetNUMEG()):
        grp = FEMData.GetEleGrpList()[grp_i]
        et = grp.GetElementType()
        for e in range(grp.GetNUME()):
            ele = grp[e]
            mat = ele.GetElementMaterial()
            nodes = ele.GetNodes()
            if et in (1, 5):                       # bar / beam
                L = np.linalg.norm(nodes[1].XYZ - nodes[0].XYZ)
                vol = mat.Area * L
            elif et == 6:                          # plate
                _, _, _, area = ele._ExtractGeometry()
                vol = mat.thick * area
            else:                                  # H8 solid
                pts, wts = ele.GetIntegrationPoints()
                vol = sum(ele.GetDetJ(*p) * w for p, w in zip(pts, wts))
            W += mat.rho * g * vol
    return W


def main(dat, vtk):
    FEMData = Domain()
    COutputter(os.path.splitext(dat)[0] + ".out")
    FEMData.ReadData(dat, os.path.splitext(dat)[0] + ".out")
    FEMData.AllocateMatrices()
    FEMData.AssembleStiffnessMatrix()
    FEMData.AssembleForce(2)                        # self-weight load case

    K = FEMData.GetStiffnessMatrix()
    A = skyline_to_csc(K)
    F = np.array(FEMData.GetForce(), dtype=np.double)
    A = apply_ties(A, dat[:-4] + ".mpc", FEMData.GetNodeList())
    u = spsolve(A, F)

    # write the solution back into CDomain (GetDisplacement returns Force)
    FEMData.GetForce()[:] = u

    NEQ = K.dim()
    W = self_weight(FEMData)
    uz = np.array([u[nd.bcode[2] - 1] if nd.bcode[2] > 0 else 0.0
                   for nd in FEMData.GetNodeList()])
    print("=== bridge self-weight solution ===")
    print(" NEQ (6-DOF)            :", NEQ)
    print(" total self-weight  W   : %.6e N" % W)
    print(" solve residual         : %.2e" % (np.linalg.norm(A @ u - F)
                                              / np.linalg.norm(F)))
    print(" vertical deflection uz : min %.4e (down)  max %.4e (up)"
          % (uz.min(), uz.max()))
    WriteVTK(vtk)


if __name__ == "__main__":
    dat = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "Bridge-1.dat")
    vtk = sys.argv[2] if len(sys.argv) > 2 else os.path.join("data", "Bridge-1.vtk")
    main(dat, vtk)
