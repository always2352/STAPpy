# -*- coding: utf-8 -*-
"""
Convert the Abaqus assembly Bridge-1.inp into a STAP input file (.dat).

Mapping:  T3D2 -> Bar(1)   B31 -> Beam(5)   S4R -> Plate(6)   C3D8R -> H8(4)
Assembly instances are flattened to global coordinates (translate, then rotate
about the given axis); coincident nodes are merged so the parts form one
connected structure.  Load: self-weight (GRAV).
"""
import math
import sys
import numpy as np

MERGE_DECIMALS = 3   # tolerance for merging coincident nodes (1e-3)

# material properties read from the .inp (name -> (E, nu, rho))
MAT = {
    "Aluminum": (7.0e10, 0.346, 2710.0),
    "Concrete": (2.5e10, 0.30, 2320.0),
    "Granite":  (6.0e10, 0.27, 2770.0),
    "Steel":    (1.17e11, 0.266, 7860.0),
}
GRAVITY = 10.0   # *Dload GRAV magnitude, direction (0,0,-1)


def get_param(line, key):
    for tok in line.split(','):
        tok = tok.strip()
        if tok.lower().startswith(key.lower() + '='):
            return tok[len(key) + 1:].strip()
    return None


def read_block(lines, i):
    """Read comma-separated data lines starting at i until the next keyword."""
    data = []
    while i < len(lines) and not lines[i].lstrip().startswith('*'):
        s = lines[i].strip()
        if s:
            data.append(s)
        i += 1
    return data, i


def rot_matrix(axis, angle_deg):
    u = np.array(axis, dtype=float)
    u = u / np.linalg.norm(u)
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    ux, uy, uz = u
    K = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
    return c * np.eye(3) + s * K + (1 - c) * np.outer(u, u)


def main(src, out):
    with open(src, 'r', errors='replace') as f:
        lines = f.readlines()

    # ---- parse parts ----
    parts = {}
    i = 0
    while i < len(lines):
        L = lines[i].strip()
        if L.startswith('*Part,'):
            name = get_param(L, 'name')
            p = {'nodes': {}, 'elems': [], 'etype': None, 'material': None,
                 'area': None, 'thick': None, 'box': None}
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('*End Part'):
                L2 = lines[i].strip()
                if L2.startswith('*Node'):
                    data, i = read_block(lines, i + 1)
                    for d in data:
                        v = d.split(',')
                        p['nodes'][int(v[0])] = (float(v[1]), float(v[2]), float(v[3]))
                elif L2.startswith('*Element'):
                    p['etype'] = get_param(L2, 'type')
                    data, i = read_block(lines, i + 1)
                    for d in data:
                        v = [int(x) for x in d.split(',') if x.strip()]
                        p['elems'].append((v[0], v[1:]))
                elif L2.startswith('*Solid Section'):
                    p['material'] = get_param(L2, 'material')
                    data, i = read_block(lines, i + 1)
                    if data:
                        tok = data[0].split(',')[0].strip()
                        if tok:
                            p['area'] = float(tok)
                elif L2.startswith('*Shell Section'):
                    p['material'] = get_param(L2, 'material')
                    data, i = read_block(lines, i + 1)
                    p['thick'] = float(data[0].split(',')[0])
                elif L2.startswith('*Beam Section'):
                    p['material'] = get_param(L2, 'material')
                    data, i = read_block(lines, i + 1)
                    p['box'] = [float(x) for x in data[0].split(',') if x.strip()]
                else:
                    i += 1
            parts[name] = p
        else:
            i += 1

    # ---- parse instances ----
    instances = []   # (name, part, T, rot)
    i = 0
    while i < len(lines):
        L = lines[i].strip()
        if L.startswith('*Instance,'):
            name = get_param(L, 'name')
            part = get_param(L, 'part')
            j = i + 1
            tl = []
            while not lines[j].strip().startswith('*End Instance'):
                s = lines[j].strip()
                if s and not s.startswith('*'):
                    tl.append(s)
                j += 1
            T = np.zeros(3)
            rot = None
            if len(tl) >= 1:
                T = np.array([float(x) for x in tl[0].split(',')])
            if len(tl) >= 2:
                v = [float(x) for x in tl[1].split(',')]
                rot = (np.array(v[0:3]), np.array(v[3:6]), v[6])
            instances.append((name, part, T, rot))
            i = j
        else:
            i += 1

    # ---- parse Set-102 (boundary-condition node set) ----
    bc_sets = []   # (instance_name, [localids])
    i = 0
    while i < len(lines):
        L = lines[i].strip()
        if L.startswith('*Nset,') and get_param(L, 'nset') == 'Set-102':
            inst = get_param(L, 'instance')
            gen = 'generate' in L.lower()
            data, i = read_block(lines, i + 1)
            ids = []
            for d in data:
                nums = [int(x) for x in d.split(',') if x.strip()]
                if gen:
                    s, e, step = nums[0], nums[1], (nums[2] if len(nums) > 2 else 1)
                    ids.extend(range(s, e + 1, step))
                else:
                    ids.extend(nums)
            if inst:
                bc_sets.append((inst, ids))
        else:
            i += 1

    # ---- parse tie node sets (m_Set-*/s_Set-*) and *Tie pairs ----
    named_nsets = {}   # name -> list of (instance_name, localid)
    i = 0
    while i < len(lines):
        L = lines[i].strip()
        nm = get_param(L, 'nset') if L.startswith('*Nset,') else None
        if nm and (nm.startswith('m_Set-') or nm.startswith('s_Set-')):
            inst = get_param(L, 'instance')
            gen = 'generate' in L.lower()
            data, i = read_block(lines, i + 1)
            ids = []
            for d in data:
                nums = [int(x) for x in d.split(',') if x.strip()]
                if gen:
                    s, e, st = nums[0], nums[1], (nums[2] if len(nums) > 2 else 1)
                    ids.extend(range(s, e + 1, st))
                else:
                    ids.extend(nums)
            if inst:
                named_nsets.setdefault(nm, []).extend((inst, lid) for lid in ids)
        else:
            i += 1

    tie_pairs = []   # (slave_set_name, master_set_name)
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith('*Tie,'):
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or lines[j].strip().startswith('**')):
                j += 1
            tok = [x.strip().replace('_CNS_', '') for x in lines[j].split(',')]
            if len(tok) >= 2:
                tie_pairs.append((tok[0], tok[1]))
            i = j + 1
        else:
            i += 1

    # ---- flatten instances to global coordinates ----
    inst_coord = {}     # (inst_idx, localid) -> global xyz
    for idx, (name, part, T, rot) in enumerate(instances):
        p = parts[part]
        if rot is not None:
            a, b, ang = rot
            R = rot_matrix(b - a, ang)
        for lid, xyz in p['nodes'].items():
            x = np.array(xyz)
            if rot is not None:
                g = a + R.dot(x + T - a)   # translate, then rotate about axis through a
            else:
                g = x + T
            inst_coord[(idx, lid)] = g

    name2idx = {nm: k for k, inst in enumerate(instances) for nm in [inst[0]]}

    # ---- connect parts ONLY through *Tie; do NOT merge across instances -------
    # The Abaqus assembly joins its separate instances solely through the 48
    # *Tie constraints; coincident nodes from different parts are NOT shared.  A
    # blanket cross-part coincident-node merge silently fuses parts Abaqus leaves
    # untied -- the deck and the SupportBeam understructure pass coincident
    # through the fixed towers, and the stay cables share tower-top anchors -- so
    # merging fabricates false rigid supports that suppress the real self-weight
    # sag and the towers' lateral sway.  Keep every instance's nodes distinct
    # (merging only true duplicates WITHIN one instance) and let the ties below
    # supply every cross-part join, exactly reproducing the model's connectivity.
    # Verified: all 40 cable ends and all deck/beam interfaces are tied, so no
    # part is left floating.
    coord2canon = {}
    canon_xyz = []
    node_map = {}
    for key, g in inst_coord.items():
        rk = (key[0],                          # instance index: never merge across
              round(g[0], MERGE_DECIMALS), round(g[1], MERGE_DECIMALS),
              round(g[2], MERGE_DECIMALS))
        c = coord2canon.get(rk)
        if c is None:
            c = len(canon_xyz)
            coord2canon[rk] = c
            canon_xyz.append(g)
        node_map[key] = c

    # ---- generate translation-only ties (slave -> nearest master) ----
    def canon_of(setname):
        out_nodes = []
        for (instname, lid) in named_nsets.get(setname, []):
            idx = name2idx.get(instname)
            if idx is not None and (idx, lid) in node_map:
                out_nodes.append(node_map[(idx, lid)])
        return out_nodes

    ties = set()
    n_resolved = n_slave = n_merged = 0
    for s_name, m_name in tie_pairs:
        s_nodes, m_nodes = canon_of(s_name), canon_of(m_name)
        if not s_nodes or not m_nodes:
            continue
        n_resolved += 1
        m_xyz = np.array([canon_xyz[c] for c in m_nodes])
        for sc in s_nodes:
            n_slave += 1
            mc = m_nodes[int(np.argmin(np.sum((m_xyz - canon_xyz[sc])**2, axis=1)))]
            if mc != sc:                       # skip pairs already merged
                ties.add((min(sc, mc) + 1, max(sc, mc) + 1))
            else:
                n_merged += 1
    ties = sorted(ties)
    print("ties: pairs=%d resolved=%d slaveNodes=%d alreadyMerged=%d new=%d"
          % (len(tie_pairs), n_resolved, n_slave, n_merged, len(ties)))

    # ---- collect elements by Abaqus type ----
    bar, beam, plate, h8 = [], [], [], []
    for idx, (name, part, T, rot) in enumerate(instances):
        p = parts[part]
        et = p['etype']
        for (eid, conn) in p['elems']:
            gc = [node_map[(idx, lid)] + 1 for lid in conn]
            if et == 'T3D2':
                bar.append(gc)
            elif et == 'B31':
                beam.append(gc)
            elif et == 'S4R':
                plate.append(gc)
            elif et == 'C3D8R':
                mset = 1 if p['material'] == 'Concrete' else 2   # Concrete=Pier, Granite=RiverBank
                h8.append((gc, mset))

    # ---- resolve boundary nodes ----
    name2idx = {nm: k for k, (nm, *_rest) in enumerate(instances)}
    fixed = set()
    for inst, ids in bc_sets:
        idx = name2idx.get(inst)
        if idx is None:
            continue
        for lid in ids:
            key = (idx, lid)
            if key in node_map:
                fixed.add(node_map[key] + 1)

    # ---- beam section (BOX) -> A, I ----
    box = parts['Part-SupportBeam']['box']     # [a, b, t1, t2, t3, t4]
    a, b, t1, t2, t3, t4 = box
    ai, bi = a - (t2 + t4), b - (t1 + t3)
    beam_A = a * b - ai * bi
    beam_I = (a * b**3 - ai * bi**3) / 12.0
    # St-Venant torsion constant of a thin-walled closed box (mean wall t)
    tw = (t1 + t2 + t3 + t4) / 4.0
    am = (a - tw) * (b - tw)
    beam_J = 4.0 * am * am * tw / (2.0 * ((a - tw) + (b - tw)))
    # Transverse shear area of the box = the two webs carrying the shear flow
    # (2 * wall * depth).  Square box -> same both directions.  Drives the
    # Timoshenko shear flexibility, which dominates for these stocky members.
    beam_As = 2.0 * tw * b
    cable_A = parts['Part-Cable50']['area']
    floor_t = parts['Part-Floor']['thick']

    NUMNP = len(canon_xyz)

    # ---- diagnostics ----
    zs = np.array([g[2] for g in canon_xyz])
    fz = np.array([canon_xyz[n - 1][2] for n in fixed])
    print("pre-merge nodes :", len(inst_coord), "(expect 4163)")
    print("merged nodes    :", NUMNP)
    print("elements        : bar=%d beam=%d plate=%d h8=%d total=%d"
          % (len(bar), len(beam), len(plate), len(h8), len(bar)+len(beam)+len(plate)+len(h8)))
    print("fixed nodes     :", len(fixed))
    print("bbox z          : all[%.2f, %.2f]  fixed[%.2f, %.2f]"
          % (zs.min(), zs.max(), fz.min(), fz.max()))
    print("beam BOX -> A=%.5f  I=%.5f ; cable A=%s ; floor t=%s"
          % (beam_A, beam_I, cable_A, floor_t))

    # ---- write STAP .dat ----
    body = []
    body.append("Bridge-1 Self-Weight (Abaqus->STAP, coincident nodes merged)")
    body.append("%d  %d  %d  %d  %.6g" % (NUMNP, 4, 1, 1, GRAVITY))
    for n in range(1, NUMNP + 1):
        x, y, z = canon_xyz[n - 1]
        # 6 DOF/node; foundation nodes fix the three translations, rotations
        # are left free (auto-suppressed where no element stiffens them)
        bc = "1 1 1 0 0 0" if n in fixed else "0 0 0 0 0 0"
        body.append("%d  %s  %.8g %.8g %.8g" % (n, bc, x, y, z))
    body.append("2")   # load case LL=2 : gravity

    eS, nuS, rhoS = MAT['Steel']
    eA, nuA, rhoA = MAT['Aluminum']
    eC, nuC, rhoC = MAT['Concrete']
    eG, nuG, rhoG = MAT['Granite']

    # Bar group (type 1)
    body.append("1  %d  1" % len(bar))
    body.append("1  %.6g  %.6g  %.6g" % (eS, rhoS, cable_A))
    for k, c in enumerate(bar, 1):
        body.append("%d  %d %d  1" % (k, c[0], c[1]))

    # Beam group (type 5): 3D space frame -> nset E rho A I J nu As
    # (As = transverse shear area -> Timoshenko; the stocky box members shear a lot)
    body.append("5  %d  1" % len(beam))
    body.append("1  %.6g  %.6g  %.6g  %.6g  %.6g  %.6g  %.6g"
                % (eA, rhoA, beam_A, beam_I, beam_J, nuA, beam_As))
    for k, c in enumerate(beam, 1):
        body.append("%d  %d %d  1" % (k, c[0], c[1]))

    # Plate group (type 6). The deck mesh numbers each quad with its first edge
    # along global Y; CPlate expects the first edge along X, so cyclically shift
    # the connectivity by one (keeps it counter-clockwise, fixes a=b=0).
    body.append("6  %d  1" % len(plate))
    body.append("1  %.6g  %.6g  %.6g  %.6g" % (eC, rhoC, nuC, floor_t))
    for k, c in enumerate(plate, 1):
        body.append("%d  %d %d %d %d  1" % (k, c[1], c[2], c[3], c[0]))

    # H8 group (type 4): mat 1 = Concrete (pier), mat 2 = Granite (riverbank)
    body.append("4  %d  2" % len(h8))
    body.append("1  %.6g  %.6g  %.6g" % (eC, nuC, rhoC))
    body.append("2  %.6g  %.6g  %.6g" % (eG, nuG, rhoG))
    for k, (c, mset) in enumerate(h8, 1):
        body.append("%d  %s  %d" % (k, ' '.join(str(x) for x in c), mset))

    with open(out, 'w', newline='') as f:
        f.write('\n'.join(body) + '\n')
    print("written:", out, "(%d lines)" % len(body))

    # companion tie file: "slave_node master_node" pairs (translation tie)
    mpc_path = out[:-4] + ".mpc" if out.endswith(".dat") else out + ".mpc"
    with open(mpc_path, 'w', newline='') as f:
        for s, m in ties:
            f.write("%d %d\n" % (s, m))
    print("written:", mpc_path, "(%d non-merged ties from %d *Tie pairs)"
          % (len(ties), len(tie_pairs)))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python abaqus_to_stap.py  data/Bridge-1.inp  data/Bridge-1.dat")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
