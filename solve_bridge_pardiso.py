"""
Optimized self-weight solver + post-processor for the bridge models.
Streamlined for maximum performance using Intel MKL PARDISO.

Usage:
    $env:MKL_NUM_THREADS=8
    python solve_bridge_pardiso.py  data/Bridge-1.dat  data/Bridge-1.vtk  [--fast]
"""
import os
import sys
import gc
import time
import psutil
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import pypardiso  # Intel MKL PARDISO 高性能求解器

from element.H8 import CH8
from Domain import Domain
from utils.Outputter import COutputter
from utils.PostProcessor import WriteVTK
from assembler import assemble_full as _assemble_full

# Hourglass stabilization coefficient
CH8.HG_COEF = 0.006


def log(msg, t0):
    print("[%7.1fs] %s" % (time.time() - t0, msg))
    sys.stdout.flush()

def get_windows_peak_memory_mb():
    return psutil.Process().memory_info().peak_wset / (1024 * 1024)

def assemble_K(FEMData, NEQ, t0, chunk=20000):
    # Full symmetric global stiffness (CSR).
    A = _assemble_full(FEMData, NEQ, chunk=chunk)
    log("global K assembled: nnz=%d (%.0f MB)" % (A.nnz, A.data.nbytes / 1e6), t0)
    return A


def build_tie_transform(NEQ, nodes, mpc_path, diag):
    """ Exact master-slave transform T (NEQ x n_reduced) for the translation ties. """
    if not os.path.exists(mpc_path):
        return None, NEQ
    pairs = [tuple(int(x) for x in ln.split()) for ln in open(mpc_path) if ln.split()]
    master = {}
    for a, b in pairs:
        for d in range(3):
            ea, eb = nodes[a - 1].bcode[d], nodes[b - 1].bcode[d]
            if ea > 0 and eb > 0 and ea != eb:
                s, m = (ea, eb) if diag[ea - 1] <= diag[eb - 1] else (eb, ea)
                master[s - 1] = m - 1
    if not master:
        return None, NEQ

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
    for s in master:                                
        cols[s] = red[root(s)]
    T = csr_matrix((np.ones(NEQ), (np.arange(NEQ), cols)), shape=(NEQ, nr))
    return T, nr


def solve_linear(K, F, nr, t0):
    # Intel MKL PARDISO 直接求解. 先尝试对称正定（mtype=2），失败后尝试对称不定（mtype=-2），最后退回非对称（mtype=11）。
    from scipy.sparse import triu
    Kc = K.tocsr()
    try:
        from pypardiso import PyPardisoSolver
        Ku = triu(Kc, format="csr"); Ku.sort_indices()
        for mt in (2, -2):                       # SPD first, then symmetric-indefinite
            solver = PyPardisoSolver(mtype=mt)
            try:
                log("Calling PARDISO (symmetric mtype=%d) on %d equations ..." % (mt, nr), t0)
                x = solver.solve(Ku, F)
                try:
                    solver.free_memory(everything=True)
                except Exception:
                    pass
                return x
            except Exception as ex:
                log("  mtype=%d failed (%s)" % (mt, ex), t0)
                try:
                    solver.free_memory(everything=True)
                except Exception:
                    pass
    except ImportError:
        pass
    log("symmetric PARDISO unavailable; using unsymmetric solver (mtype=11)", t0)
    return pypardiso.spsolve(Kc, F)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    dat = args[0] if args else os.path.join("data", "Bridge-1.dat")
    vtk = args[1] if len(args) > 1 else os.path.splitext(dat)[0] + ".vtk"
    write_stress = "--fast" not in flags

    t0 = time.time()
    FEMData = Domain()
    COutputter(os.path.splitext(dat)[0] + ".out")
    # verbose=False: skip the per-node / per-element echo to the terminal and
    # the .out file (the bulk model dump that floods the console on big meshes).
    FEMData.ReadData(dat, os.path.splitext(dat)[0] + ".out", verbose=False)
    NEQ = FEMData.NEQ
    nodes = FEMData.GetNodeList()
    log("model read: NEQ=%d, nodes=%d" % (NEQ, len(nodes)), t0)

    # 组装自重载荷
    FEMData.Force = np.zeros(NEQ)
    for g in range(FEMData.GetNUMEG()):
        grp = FEMData.GetEleGrpList()[g]
        for e in range(grp.GetNUME()):
            grp[e].GenerateLocationMatrix()
    FEMData.AssembleGravityForce()
    F = FEMData.Force.copy()
    log("self-weight assembled: total |Fz|=%.6e N" % np.abs(F).sum(), t0)

    # 组装全局刚度矩阵
    A = assemble_K(FEMData, NEQ, t0)
    log("global K completed: nnz=%d (%.0f MB)" % (A.nnz, A.data.nbytes / 1e6), t0)

    # 消除多点约束（Tie Constraints）
    T, nr = build_tie_transform(NEQ, nodes, os.path.splitext(dat)[0] + ".mpc", A.diagonal())
    if T is not None:
        K = (T.T @ A @ T).tocsr()
        Fr = T.T @ F
        log("ties eliminated: %d -> %d equations" % (NEQ, nr), t0)
    else:
        K, Fr = A, F

    # 核心求解阶段
    ur = solve_linear(K, Fr, nr, t0)
    u = (T @ ur) if T is not None else ur
    t1 = time.time()
    
    # 计算相对残差
    res = np.linalg.norm(K @ ur - Fr) / np.linalg.norm(Fr)
    log("solved: reduced relative residual %.2e" % res, t0)

    # 后处理与导出
    FEMData.GetForce()[:] = u                        
    uz = np.array([u[nd.bcode[2] - 1] if nd.bcode[2] > 0 else 0.0 for nd in nodes])
    print("=== %s ===" % os.path.basename(dat))
    print(" equations (after ties) : %d" % nr)
    print(" residual               : %.2e" % res)
    print(" vertical uz  min / max : %.4e / %.4e" % (uz.min(), uz.max()))

    WriteVTK(vtk, write_stress=write_stress)
    print(" Peak Memory Usage (OS) : %.2f MB" % get_windows_peak_memory_mb())
    print(" Total Time             : %.2f seconds" % (t1 - t0))
    log("done", t0)


if __name__ == "__main__":
    main()