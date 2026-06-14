# -*- coding: utf-8 -*-
"""
Assembly v2: upper-triangular, preallocated-CSR (symbolic + numeric), with every
element group VECTORISED (bar/beam/plate/H8).  Produces the SAME matrix as the
per-element assembler (upper triangle), feeds the symmetric PARDISO directly.

  * #1 upper-triangle only: half the entries, half the memory/work, no triu copy.
  * #2 two-phase: pass-1 builds the CSR pattern from connectivity (indices only,
        no values); pass-2 scatters batched element stiffness straight into the
        preallocated data[] via searchsorted -- no COO triplet pile, no dedup.
  * #3 batched element stiffness for bar/beam/plate (H8 already batched).

Validated against the per-element ElementStiffness (K_e to ~1e-12) and against
the full assembler's upper triangle.
"""
import os
import sys
import gc
import numpy as np
from scipy.sparse import csr_matrix

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from element.H8 import CH8                       # noqa: E402

# ---------- H8 (8-node solid), reduced integ + FB hourglass ----------
_XI = np.array([-1, 1, 1, -1, -1, 1, 1, -1], float)
_ETA = np.array([-1, -1, 1, 1, -1, -1, 1, 1], float)
_ZE = np.array([-1, -1, -1, -1, 1, 1, 1, 1], float)
_dN = 0.125 * np.vstack([_XI, _ETA, _ZE])
_HG = np.array([_XI*_ETA, _ETA*_ZE, _ZE*_XI, _XI*_ETA*_ZE])
_N8 = np.arange(8)


def _h8_Ke(XYZe, E, nu, hg):
    Ne = XYZe.shape[0]
    J = np.einsum('ij,ejk->eik', _dN, XYZe)
    detJ = np.linalg.det(J)
    dNx = np.einsum('eij,jk->eik', np.linalg.inv(J), _dN)
    bx, by, bz = dNx[:, 0, :], dNx[:, 1, :], dNx[:, 2, :]
    c0, c1, c2 = 3*_N8, 3*_N8+1, 3*_N8+2
    B = np.zeros((Ne, 6, 24))
    B[:, 0, c0] = bx; B[:, 1, c1] = by; B[:, 2, c2] = bz
    B[:, 3, c0] = by; B[:, 3, c1] = bx
    B[:, 4, c1] = bz; B[:, 4, c2] = by
    B[:, 5, c0] = bz; B[:, 5, c2] = bx
    f = E/((1+nu)*(1-2*nu)); dv = (1-nu)*f; ov = nu*f; G = 0.5*(1-2*nu)*f
    D = np.zeros((Ne, 6, 6))
    for a in range(3):
        D[:, a, a] = dv
        for b in range(3):
            if a != b:
                D[:, a, b] = ov
    D[:, 3, 3] = D[:, 4, 4] = D[:, 5, 5] = G
    K = np.einsum('eai,eaj->eij', B, np.einsum('eab,ebj->eaj', D, B)) * (detJ*8.0)[:, None, None]
    mu = 0.5*E/(1+nu)
    chg = hg*mu*(detJ*8.0)*(np.einsum('ei,ei->e', bx, bx)+np.einsum('ei,ei->e', by, by)+np.einsum('ei,ei->e', bz, bz))
    X, Y, Z = XYZe[:, :, 0], XYZe[:, :, 1], XYZe[:, :, 2]
    er = np.arange(Ne)
    for h in _HG:
        g = h[None, :] - (X@h)[:, None]*bx - (Y@h)[:, None]*by - (Z@h)[:, None]*bz
        gg = chg[:, None, None]*np.einsum('ei,ej->eij', g, g)
        for d in range(3):
            idx = np.arange(d, 24, 3)
            K[np.ix_(er, idx, idx)] += gg
    di = np.arange(24)
    K[:, di, di] += 1e-9*E[:, None]
    return K


# ---------- Bar (3-D truss) ----------
def _bar_Ke(XYZe, E, A):
    d = XYZe[:, 1, :] - XYZe[:, 0, :]
    L = np.linalg.norm(d, axis=1)
    c = d / L[:, None]
    cc = np.einsum('ei,ej->eij', c, c)
    k = (E*A/L)[:, None, None]
    K = np.zeros((XYZe.shape[0], 6, 6))
    K[:, 0:3, 0:3] = k*cc; K[:, 3:6, 3:6] = k*cc
    K[:, 0:3, 3:6] = -k*cc; K[:, 3:6, 0:3] = -k*cc
    return K


# ---------- Beam (3-D Euler-Bernoulli frame) ----------
def _beam_Ke(XYZe, E, A, I, Jt, nu):
    Ne = XYZe.shape[0]
    d = XYZe[:, 1, :] - XYZe[:, 0, :]
    L = np.linalg.norm(d, axis=1)
    e1 = d / L[:, None]
    ref = np.where((np.abs(e1[:, 2]) < 0.99)[:, None],
                   np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))
    e2 = np.cross(ref, e1); e2 /= np.linalg.norm(e2, axis=1)[:, None]
    e3 = np.cross(e1, e2)
    Lam = np.stack([e1, e2, e3], axis=1)               # (Ne,3,3)
    T = np.zeros((Ne, 12, 12))
    for b in range(4):
        T[:, 3*b:3*b+3, 3*b:3*b+3] = Lam
    G = E/(2*(1+nu))
    L2 = L*L; L3 = L2*L
    Kl = np.zeros((Ne, 12, 12))
    EA = E*A/L; GJ = G*Jt/L
    Kl[:, 0, 0] = EA; Kl[:, 0, 6] = -EA; Kl[:, 6, 0] = -EA; Kl[:, 6, 6] = EA
    Kl[:, 3, 3] = GJ; Kl[:, 3, 9] = -GJ; Kl[:, 9, 3] = -GJ; Kl[:, 9, 9] = GJ
    az = 12*E*I/L3; bz = 6*E*I/L2; cz = 4*E*I/L; dz = 2*E*I/L          # Phi=0 (Euler)
    Kl[:, 1, 1] = az; Kl[:, 1, 5] = bz; Kl[:, 1, 7] = -az; Kl[:, 1, 11] = bz
    Kl[:, 5, 1] = bz; Kl[:, 5, 5] = cz; Kl[:, 5, 7] = -bz; Kl[:, 5, 11] = dz
    Kl[:, 7, 1] = -az; Kl[:, 7, 5] = -bz; Kl[:, 7, 7] = az; Kl[:, 7, 11] = -bz
    Kl[:, 11, 1] = bz; Kl[:, 11, 5] = dz; Kl[:, 11, 7] = -bz; Kl[:, 11, 11] = cz
    ay, by_, cy, dy = az, bz, cz, dz
    Kl[:, 2, 2] = ay; Kl[:, 2, 4] = -by_; Kl[:, 2, 8] = -ay; Kl[:, 2, 10] = -by_
    Kl[:, 4, 2] = -by_; Kl[:, 4, 4] = cy; Kl[:, 4, 8] = by_; Kl[:, 4, 10] = dy
    Kl[:, 8, 2] = -ay; Kl[:, 8, 4] = by_; Kl[:, 8, 8] = ay; Kl[:, 8, 10] = by_
    Kl[:, 10, 2] = -by_; Kl[:, 10, 4] = dy; Kl[:, 10, 8] = by_; Kl[:, 10, 10] = cy
    return np.einsum('eki,ekl,elj->eij', T, Kl, T)


# ---------- Plate (flat shell: Q4 membrane + ACM Kirchhoff bending + drilling) ----------
_gp2 = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
_g3 = np.sqrt(0.6); _gp3 = np.array([-_g3, 0.0, _g3]); _gw3 = np.array([5/9., 8/9., 5/9.])
_xiI = np.array([-1.0, 1.0, 1.0, -1.0]); _etI = np.array([-1.0, -1.0, 1.0, 1.0])


def _plate_Ke(coords, E, nu, t):
    """coords (n,4,2) in local (p,q); returns K (n,24,24) in node-major
    (u,v,w,tx,ty,tz) order, matching CPlate."""
    n = coords.shape[0]
    # ---- membrane Q4 (u,v) ----
    Dm = np.zeros((n, 3, 3))
    cm = E*t/(1-nu**2)
    Dm[:, 0, 0] = cm; Dm[:, 1, 1] = cm; Dm[:, 0, 1] = cm*nu; Dm[:, 1, 0] = cm*nu
    Dm[:, 2, 2] = cm*(1-nu)/2
    Km = np.zeros((n, 8, 8))
    for xi in _gp2:
        for eta in _gp2:
            dN = np.zeros((n, 2, 4))
            for Ii in range(4):
                dN[:, 0, Ii] = 0.25*_xiI[Ii]*(1+_etI[Ii]*eta)
                dN[:, 1, Ii] = 0.25*_etI[Ii]*(1+_xiI[Ii]*xi)
            Jm = np.einsum('eij,ejk->eik', dN, coords)
            detJ = np.linalg.det(Jm)
            dNx = np.einsum('eij,ejk->eik', np.linalg.inv(Jm), dN)
            B = np.zeros((n, 3, 8))
            for Ii in range(4):
                B[:, 0, 2*Ii] = dNx[:, 0, Ii]
                B[:, 1, 2*Ii+1] = dNx[:, 1, Ii]
                B[:, 2, 2*Ii] = dNx[:, 1, Ii]; B[:, 2, 2*Ii+1] = dNx[:, 0, Ii]
            Km += np.einsum('eai,eab,ebj->eij', B, Dm, B) * detJ[:, None, None]
    # ---- ACM Kirchhoff bending (w,tx,ty) ----
    Db = np.zeros((n, 3, 3))
    cb = E*t**3/(12*(1-nu**2))
    Db[:, 0, 0] = cb; Db[:, 1, 1] = cb; Db[:, 0, 1] = cb*nu; Db[:, 1, 0] = cb*nu
    Db[:, 2, 2] = cb*(1-nu)/2
    cen = coords.mean(axis=1)
    dp = coords[:, :, 0] - cen[:, 0:1]; dq = coords[:, :, 1] - cen[:, 1:2]
    a = np.mean(np.abs(dp), axis=1); b = np.mean(np.abs(dq), axis=1)
    sx = np.sign(dp); sy = np.sign(dq)                      # (n,4) corner signs

    def P(s, t_):
        return np.stack([np.ones_like(s), s, t_, s*s, s*t_, t_*t_, s**3, s*s*t_, s*t_*t_, t_**3, s**3*t_, s*t_**3], -1)

    def Ps(s, t_):
        return np.stack([np.zeros_like(s), np.ones_like(s), np.zeros_like(s), 2*s, t_, np.zeros_like(s), 3*s*s, 2*s*t_, t_*t_, np.zeros_like(s), 3*s*s*t_, t_**3], -1)

    def Pt(s, t_):
        return np.stack([np.zeros_like(s), np.zeros_like(s), np.ones_like(s), np.zeros_like(s), s, 2*t_, np.zeros_like(s), s*s, 2*s*t_, 3*t_*t_, s**3, 3*s*t_*t_], -1)

    C = np.zeros((n, 12, 12))
    for i in range(4):
        s, t_ = sx[:, i], sy[:, i]
        C[:, 3*i+0, :] = P(s, t_)
        C[:, 3*i+1, :] = -(1.0/b)[:, None]*Pt(s, t_)
        C[:, 3*i+2, :] = (1.0/a)[:, None]*Ps(s, t_)
    Cinv = np.linalg.inv(C)
    Kb = np.zeros((n, 12, 12))
    for xi, wi in zip(_gp3, _gw3):
        for eta, wj in zip(_gp3, _gw3):
            Pss = np.array([0, 0, 0, 2, 0, 0, 6*xi, 2*eta, 0, 0, 6*xi*eta, 0], float)
            Ptt = np.array([0, 0, 0, 0, 0, 2, 0, 0, 2*xi, 6*eta, 0, 6*xi*eta], float)
            Pst = np.array([0, 0, 0, 0, 1, 0, 0, 2*xi, 2*eta, 0, 3*xi*xi, 3*eta*eta], float)
            Bnat = np.zeros((n, 3, 12))
            Bnat[:, 0, :] = Pss[None, :]/(a*a)[:, None]
            Bnat[:, 1, :] = Ptt[None, :]/(b*b)[:, None]
            Bnat[:, 2, :] = 2.0*Pst[None, :]/(a*b)[:, None]
            Bb = np.einsum('eij,ejk->eik', Bnat, Cinv)
            Kb += (wi*wj)*np.einsum('eai,eab,ebj->eij', Bb, Db, Bb)*(a*b)[:, None, None]
    # ---- assemble 24x24 ----
    K = np.zeros((n, 24, 24))
    m_idx = [6*Ii+d for Ii in range(4) for d in (0, 1)]
    b_idx = [6*Ii+d for Ii in range(4) for d in (2, 3, 4)]
    K[np.ix_(np.arange(n), m_idx, m_idx)] += Km
    K[np.ix_(np.arange(n), b_idx, b_idx)] += Kb
    kdr = 1e-3*np.mean(np.diagonal(Kb, axis1=1, axis2=2), axis=1)
    for Ii in range(4):
        K[:, 6*Ii+5, 6*Ii+5] += kdr
    return K


# ---------- per-group gather + batched Ke ----------
def _group_gather(grp, node_xyz, bcode):
    """Collect connectivity, location matrix and per-element material/geometry
    arrays for one group -- cheap, no element stiffness computed here."""
    n = grp.GetNUME(); NEN = grp[0]._NEN
    conn = np.fromiter((nd.NodeNumber - 1 for e in range(n) for nd in grp[e].GetNodes()),
                       np.int64, n*NEN).reshape(n, NEN)
    et = grp.GetElementType()
    mats = [grp[e].GetElementMaterial() for e in range(n)]
    g = {'et': et, 'n': n}
    if et == 4:
        XYZe = node_xyz[conn]
        E = np.fromiter((m.E for m in mats), float, n)
        nu = np.fromiter((m.nu for m in mats), float, n)
        g['XYZe'] = XYZe; g['E'] = E; g['nu'] = nu
        g['LM'] = bcode[conn, :3].reshape(n, 24)
        # identical-element cache: geometrically congruent hexes (the structured
        # pier/riverbank blocks) have bit-identical relative geometry, so their
        # stiffness is computed ONCE per distinct (rel-coords, E, nu).  Exact.
        rel = (XYZe - XYZe.mean(axis=1, keepdims=True)).reshape(n, 24)
        sig = np.ascontiguousarray(np.column_stack([rel, E, nu]))
        view = sig.view(np.dtype((np.void, sig.dtype.itemsize * sig.shape[1])))
        _, rep, inv = np.unique(view, return_index=True, return_inverse=True)
        g['Kcache'] = _h8_Ke(XYZe[rep], E[rep], nu[rep], CH8.HG_COEF)
        g['inv'] = inv.ravel()
    elif et == 1:
        g['XYZe'] = node_xyz[conn]
        g['E'] = np.fromiter((m.E for m in mats), float, n)
        g['A'] = np.fromiter((m.Area for m in mats), float, n)
        g['LM'] = bcode[conn, :3].reshape(n, 6)
    elif et == 5:
        g['XYZe'] = node_xyz[conn]
        g['E'] = np.fromiter((m.E for m in mats), float, n)
        g['A'] = np.fromiter((m.Area for m in mats), float, n)
        g['I'] = np.fromiter((m.Inertia for m in mats), float, n)
        g['J'] = np.fromiter((getattr(m, 'J', m.Inertia) for m in mats), float, n)
        g['nu'] = np.fromiter((getattr(m, 'nu', 0.3) for m in mats), float, n)
        g['LM'] = bcode[conn, :6].reshape(n, 12)
    elif et == 6:
        v1 = node_xyz[conn[:, 1]] - node_xyz[conn[:, 0]]
        v2 = node_xyz[conn[:, 3]] - node_xyz[conn[:, 0]]
        e3 = np.cross(v1, v2); e3 /= np.linalg.norm(e3, axis=1)[:, None]
        kk = np.argmax(np.abs(e3), axis=1)
        if not np.all(kk == kk[0]):
            raise ValueError("plate normals not all aligned; per-element p,q needed")
        k = int(kk[0]); p, q = [i for i in range(3) if i != k]
        g['coords'] = node_xyz[conn][:, :, [p, q]]
        g['E'] = np.fromiter((m.E for m in mats), float, n)
        g['nu'] = np.fromiter((m.nu for m in mats), float, n)
        g['t'] = np.fromiter((m.thick for m in mats), float, n)
        g['LM'] = bcode[conn][:, :, [p, q, k, 3+p, 3+q, 3+k]].reshape(n, 24)
    else:
        raise ValueError("unknown element type %d" % et)
    return g


def _ke(g, sl=slice(None)):
    """Batched element stiffness (m, ND, ND) for a slice of gathered group g."""
    et = g['et']
    if et == 4:
        return g['Kcache'][g['inv'][sl]]          # gather from the congruent-hex cache
    if et == 1:
        return _bar_Ke(g['XYZe'][sl], g['E'][sl], g['A'][sl])
    if et == 5:
        return _beam_Ke(g['XYZe'][sl], g['E'][sl], g['A'][sl], g['I'][sl], g['J'][sl], g['nu'][sl])
    return _plate_Ke(g['coords'][sl], g['E'][sl], g['nu'][sl], g['t'][sl])


def _upper_pairs(LM):
    """upper-triangular (row,col) global index pairs (0-based) for one group."""
    n, ND = LM.shape
    gi = np.broadcast_to(LM[:, :, None], (n, ND, ND))
    gj = np.broadcast_to(LM[:, None, :], (n, ND, ND))
    m = (gi > 0) & (gj > 0) & (gi <= gj)
    return (gi[m] - 1).astype(np.int32), (gj[m] - 1).astype(np.int32), m


def assemble_upper(FEMData, NEQ, chunk=20000):
    """Upper-triangular CSR of K via symbolic pattern + numeric scatter."""
    nodes = FEMData.GetNodeList()
    node_xyz = np.array([nd.XYZ for nd in nodes], float)
    bcode = np.array([nd.bcode for nd in nodes], np.int32)
    grps = [FEMData.GetEleGrpList()[g] for g in range(FEMData.GetNUMEG())
            if FEMData.GetEleGrpList()[g].GetNUME() > 0]

    # ---- pass 1: gather groups (cheap), build CSR pattern from (row,col) ----
    gd = [_group_gather(grp, node_xyz, bcode) for grp in grps]
    R, Cc = [], []
    for g in gd:
        r, c, _ = _upper_pairs(g['LM'])
        R.append(r); Cc.append(c)
    rows = np.concatenate(R); cols = np.concatenate(Cc); del R, Cc; gc.collect()
    pat = csr_matrix((np.ones(rows.size, np.int8), (rows, cols)), shape=(NEQ, NEQ))
    pat.sum_duplicates(); pat.sort_indices()
    indptr, indices = pat.indptr, pat.indices
    nnz = indices.size
    key = np.repeat(np.arange(NEQ, dtype=np.int64), np.diff(indptr)) * NEQ + indices
    del rows, cols, pat; gc.collect()

    # ---- pass 2: scatter batched element stiffness straight into data[] ----
    data = np.zeros(nnz)
    for g in gd:
        LM = g['LM']; n = g['n']
        step = chunk if g['et'] == 4 else n          # chunk only the big H8 group
        for s in range(0, n, step):
            sl = slice(s, s + step)
            Ke = _ke(g, sl)
            r, c, m = _upper_pairs(LM[sl])
            pos = np.searchsorted(key, r.astype(np.int64) * NEQ + c)
            data += np.bincount(pos, weights=Ke[m], minlength=nnz)   # fast scatter-add
            gc.collect()
    return csr_matrix((data, indices, indptr), shape=(NEQ, NEQ))




def assemble_full(FEMData, NEQ, chunk=20000):
    """Full symmetric CSR of K (what the tie elimination T^T K T needs),
    built via the upper-triangular path then mirrored: K = U + triu(U,1)^T.
    Same matrix as the per-element assembler, lower peak memory + faster."""
    from scipy.sparse import triu
    U = assemble_upper(FEMData, NEQ, chunk=chunk)
    A = (U + triu(U, 1).transpose()).tocsr()
    A.sum_duplicates()
    return A
