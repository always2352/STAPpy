#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General self-weight solver + post-processor for the bridge models.

Handles every mesh size in one path -- the small Bridge-1 (~1.5e4 DOF) up to the
fine Bridge-3 (~8e5 DOF) -- with low memory and an automatically chosen linear
solver:

  * assembly  : the global stiffness goes straight into a SciPy sparse (CSR)
                matrix, flushed in small element batches, so peak memory stays a
                few hundred MB even for ~10^6 DOF (the dense skyline would need
                hundreds of GB and is never built);
  * ties      : the Abaqus *Tie constraints (companion .mpc) are removed by EXACT
                master-slave elimination (K_r = T^T K T) -- well conditioned, no
                penalty, no spurious stiffness;
  * solve     : direct sparse LU (SuperLU) when the reduced system is small,
                otherwise Jacobi-preconditioned conjugate gradient (memory-lean);
  * output    : a ParaView .vtk with the nodal displacement field (and, unless
                --fast, a per-element stress measure), plus a one-line summary.

Usage:
    python solve_bridge.py  data/Bridge-1.dat  [out.vtk]  [--fast] [--direct|--cg]
"""
import os
import sys
import gc
import time
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, cg, LinearOperator

from element.H8 import CH8
from Domain import Domain
from utils.Outputter import COutputter
from utils.PostProcessor import WriteVTK

# The slender towers' lateral bending is carried by the H8 hourglass term;
# this matches the Abaqus C3D8R towers (see notes in bridge_solve_export.py).
CH8.HG_COEF = 0.006

DIRECT_MAX = 150000     # reduced equations at/below which direct LU is used
CHUNK = 5000            # elements per assembly flush (memory bound)


def log(msg, t0):
    print("[%7.1fs] %s" % (time.time() - t0, msg))
    sys.stdout.flush()


def assemble_K(FEMData, NEQ, t0):
    """ Assemble the global stiffness into CSR, flushing in element batches. """
    A = csr_matrix((NEQ, NEQ))
    buf = {"RI": [], "CI": [], "VV": []}

    def flush():
        if not buf["VV"]:
            return A
        Kg = coo_matrix((np.concatenate(buf["VV"]),
                         (np.concatenate(buf["RI"]), np.concatenate(buf["CI"]))),
                        shape=(NEQ, NEQ)).tocsr()
        buf["RI"].clear(); buf["CI"].clear(); buf["VV"].clear()
        gc.collect()
        return Kg

    for g in range(FEMData.GetNUMEG()):
        grp = FEMData.GetEleGrpList()[g]
        n = grp.GetNUME()
        if n == 0:
            continue
        ND = grp[0].GetND()
        Psz = ND * (ND + 1) // 2
        prow = np.empty(Psz, np.int32)
        pcol = np.empty(Psz, np.int32)
        c = 0
        for col in range(ND):                       # unpack order of ElementStiffness
            for row in range(col, -1, -1):
                prow[c] = row; pcol[c] = col; c += 1
        offdiag = prow != pcol
        arr = np.zeros(Psz)
        for e in range(n):
            ele = grp[e]
            ele.ElementStiffness(arr)
            lm = np.asarray(ele.GetLocationMatrix())
            gi, gj = lm[prow], lm[pcol]
            m = (gi > 0) & (gj > 0)                  # drop constrained DOFs
            buf["RI"].append(gi[m] - 1); buf["CI"].append(gj[m] - 1)
            buf["VV"].append(arr[m].copy())
            mo = m & offdiag                         # mirror to the lower triangle
            buf["RI"].append(lm[pcol[mo]] - 1); buf["CI"].append(lm[prow[mo]] - 1)
            buf["VV"].append(arr[mo].copy())
            if (e + 1) % CHUNK == 0:
                A = A + flush()
        A = A + flush()
        log("group %d (type %d, %d elems) assembled" % (g, grp.GetElementType(), n), t0)
    return A.tocsr()


def build_tie_transform(NEQ, nodes, mpc_path, diag):
    """
    Exact master-slave transform T (NEQ x n_reduced) for the translation ties:
    every tied slave DOF follows its master (resolving chains).  Returns (T, nr);
    (None, NEQ) when there is no .mpc.

    The .mpc lists each tie as an (unordered) node pair, so the slave/master
    orientation is chosen here from the assembled diagonal: the DOF with the
    SMALLER diagonal stiffness becomes the slave.  This is essential -- an
    in-plane cable's transverse translation carries no element stiffness (it is
    held only by the tie), so it MUST be eliminated as the slave; keeping it as a
    master would leave a stiffness-less free DOF and a singular system.
    """
    if not os.path.exists(mpc_path):
        return None, NEQ, np.arange(NEQ)
    pairs = [tuple(int(x) for x in ln.split()) for ln in open(mpc_path) if ln.split()]
    master = {}
    for a, b in pairs:
        for d in range(3):
            ea, eb = nodes[a - 1].bcode[d], nodes[b - 1].bcode[d]
            if ea > 0 and eb > 0 and ea != eb:
                s, m = (ea, eb) if diag[ea - 1] <= diag[eb - 1] else (eb, ea)
                master[s - 1] = m - 1
    if not master:
        return None, NEQ, np.arange(NEQ)

    cache = {}

    def root(i):
        seen = []
        while i in master and i not in cache:
            seen.append(i); i = master[i]
        r = cache.get(i, i)
        for s in seen:
            cache[s] = r
        return r

    is_slave = np.zeros(NEQ, bool)
    is_slave[list(master)] = True
    red = np.full(NEQ, -1, np.int64)
    free = np.where(~is_slave)[0]
    nr = free.size
    red[free] = np.arange(nr)
    cols = red.copy()
    for s in master:                                # only the (few) slave DOFs
        cols[s] = red[root(s)]
    T = csr_matrix((np.ones(NEQ), (np.arange(NEQ), cols)), shape=(NEQ, nr))
    return T, nr, free


def rigid_body_modes(nodes, NEQ):
    """
    The 6 rigid-body near-null-space modes (3 translations + 3 rotations) in
    global-DOF equation space -- the near-null-space AMG needs to coarsen 3-D
    elasticity well.  DOF order per node is ux,uy,uz,rx,ry,rz; coordinates are
    centred to keep the rotation modes well scaled.
    """
    dof = np.full(NEQ, -1, np.int16)
    xyz = np.zeros((NEQ, 3))
    for nd in nodes:
        for d in range(6):
            e = nd.bcode[d]
            if e > 0:
                dof[e - 1] = d
                xyz[e - 1] = nd.XYZ
    xyz -= xyz.mean(axis=0)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    B = np.zeros((NEQ, 6))
    B[dof == 0, 0] = 1.0                                          # translate x
    B[dof == 1, 1] = 1.0                                          # translate y
    B[dof == 2, 2] = 1.0                                          # translate z
    B[dof == 1, 3] = -z[dof == 1]; B[dof == 2, 3] = y[dof == 2]   # rotate about x
    B[dof == 3, 3] = 1.0
    B[dof == 0, 4] = z[dof == 0]; B[dof == 2, 4] = -x[dof == 2]   # rotate about y
    B[dof == 4, 4] = 1.0
    B[dof == 0, 5] = -y[dof == 0]; B[dof == 1, 5] = x[dof == 1]   # rotate about z
    B[dof == 5, 5] = 1.0
    return B


def solve_linear(K, F, nr, force_mode, t0, B=None):
    """
    Direct SuperLU for small systems; otherwise an iterative solve.  The
    iterative path prefers algebraic multigrid (pyamg smoothed aggregation),
    which is built for 3-D elasticity and converges in tens of cycles; it falls
    back to Jacobi-CG if pyamg is missing (Jacobi stalls on the global bending
    modes of a fine mesh, so AMG is strongly preferred for the large meshes).
    """
    use_direct = (force_mode == "direct") or (force_mode is None and nr <= DIRECT_MAX)
    if use_direct:
        log("direct sparse LU (SuperLU) on %d equations ..." % nr, t0)
        return spsolve(K.tocsc(), F)

    K = K.tocsr()
    nrm = np.linalg.norm(F)
    try:
        import pyamg
        log("AMG (smoothed aggregation) on %d equations ..." % nr, t0)
        ml = pyamg.smoothed_aggregation_solver(K, B=B, max_coarse=2000)
        rh = []
        u = ml.solve(F, tol=1e-9, accel="cg", maxiter=500, residuals=rh)
        log("AMG-CG: %d cycles, final rel.res %.2e (grid: %s)"
            % (len(rh) - 1, rh[-1] / nrm, [lvl.A.shape[0] for lvl in ml.levels]), t0)
        return u
    except ImportError:
        log("pyamg unavailable -> Jacobi-CG on %d equations ..." % nr, t0)

    diag = K.diagonal().copy()
    diag[diag == 0.0] = 1.0
    M = LinearOperator((nr, nr), matvec=lambda x: x / diag)
    it = [0]

    def cb(xk):
        it[0] += 1
        if it[0] % 500 == 0:
            log("  cg iter %d  rel.res %.2e" % (it[0], np.linalg.norm(K @ xk - F) / nrm), t0)

    u, info = cg(K, F, rtol=1e-8, atol=0.0, maxiter=50000, M=M, callback=cb)
    log("CG stopped: info=%d (0=converged), iters=%d" % (info, it[0]), t0)
    return u


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    dat = args[0] if args else os.path.join("data", "Bridge-1.dat")
    vtk = args[1] if len(args) > 1 else os.path.splitext(dat)[0] + ".vtk"
    force_mode = "direct" if "--direct" in flags else "cg" if "--cg" in flags else None
    write_stress = "--fast" not in flags

    t0 = time.time()
    FEMData = Domain()
    COutputter(os.path.splitext(dat)[0] + ".out")
    FEMData.ReadData(dat, os.path.splitext(dat)[0] + ".out")
    NEQ = FEMData.NEQ
    nodes = FEMData.GetNodeList()
    log("model read: NEQ=%d, nodes=%d" % (NEQ, len(nodes)), t0)

    # equivalent nodal self-weight (needs the location matrices)
    FEMData.Force = np.zeros(NEQ)
    for g in range(FEMData.GetNUMEG()):
        grp = FEMData.GetEleGrpList()[g]
        for e in range(grp.GetNUME()):
            grp[e].GenerateLocationMatrix()
    FEMData.AssembleGravityForce()
    F = FEMData.Force.copy()
    log("self-weight assembled: total |Fz|=%.6e N" % np.abs(F).sum(), t0)

    A = assemble_K(FEMData, NEQ, t0)
    log("global K: nnz=%d (%.0f MB)" % (A.nnz, A.data.nbytes / 1e6), t0)

    T, nr, free = build_tie_transform(NEQ, nodes, os.path.splitext(dat)[0] + ".mpc",
                                      A.diagonal())
    if T is not None:
        K = (T.T @ A @ T).tocsr()
        Fr = T.T @ F
        log("ties eliminated (master-slave): %d -> %d equations" % (NEQ, nr), t0)
    else:
        K, Fr = A, F

    B = rigid_body_modes(nodes, NEQ)[free]              # AMG near-null-space
    ur = solve_linear(K, Fr, nr, force_mode, t0, B=B)
    u = (T @ ur) if T is not None else ur
    # residual of the REDUCED system (A u - F on the full system is, by design,
    # the non-zero tie reaction, so it is not the solve error)
    res = np.linalg.norm(K @ ur - Fr) / np.linalg.norm(Fr)
    log("solved: reduced relative residual %.2e" % res, t0)

    FEMData.GetForce()[:] = u                        # so the post-processor sees it
    uz = np.array([u[nd.bcode[2] - 1] if nd.bcode[2] > 0 else 0.0 for nd in nodes])
    print("=== %s ===" % os.path.basename(dat))
    print(" equations (after ties) : %d" % nr)
    print(" residual               : %.2e" % res)
    print(" vertical uz  min / max : %.4e / %.4e" % (uz.min(), uz.max()))

    WriteVTK(vtk, write_stress=write_stress)
    log("done", t0)


if __name__ == "__main__":
    main()
