#!/usr/bin/env python3
import numpy as np
import math
import os

# convergence test: interpolation error of trilinear H8 element
# domain: unit cube [0,1]^3
# exact displacement field: u = [sin(pi x) sin(pi y) sin(pi z), 0, 0]

def u_exact(x,y,z):
    val = math.sin(math.pi*x)*math.sin(math.pi*y)*math.sin(math.pi*z)
    return np.array([val, 0.0, 0.0])

# shape functions and derivatives at natural coords
def shape_functions(xi, eta, zeta):
    xi_coords = np.array([ -1,  1,  1, -1, -1,  1,  1, -1 ], dtype=np.double)
    eta_coords= np.array([ -1, -1,  1,  1, -1, -1,  1,  1 ], dtype=np.double)
    zeta_coords= np.array([ -1, -1, -1, -1,  1,  1,  1,  1 ], dtype=np.double)
    N = np.zeros(8)
    dN_dxi = np.zeros(8); dN_deta = np.zeros(8); dN_dzeta = np.zeros(8)
    for i in range(8):
        n1 = xi_coords[i]; n2 = eta_coords[i]; n3 = zeta_coords[i]
        N[i] = 0.125 * (1.0 + n1*xi) * (1.0 + n2*eta) * (1.0 + n3*zeta)
        dN_dxi[i] = 0.125 * n1 * (1.0 + n2*eta) * (1.0 + n3*zeta)
        dN_deta[i]= 0.125 * (1.0 + n1*xi) * n2 * (1.0 + n3*zeta)
        dN_dzeta[i]=0.125 * (1.0 + n1*xi) * (1.0 + n2*eta) * n3
    return N, dN_dxi, dN_deta, dN_dzeta

# quadrature points and weights for 2-point Gauss
gp = [-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)]
gw = [1.0, 1.0]

results = []
for nx in [1,2,4,8]:
    ny = nz = nx
    # generate nodes
    nxn = nx + 1
    nyn = ny + 1
    nzn = nz + 1
    coords = []
    index = {}
    cnt = 1
    for k in range(nzn):
        z = k / nz
        for j in range(nyn):
            y = j / ny
            for i in range(nxn):
                x = i / nx
                coords.append((x,y,z))
                index[(i,j,k)] = cnt
                cnt += 1
    NUMNP = len(coords)
    # elements connectivity
    elems = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # nodes ordering consistent with H8: N1..N4 bottom face ccw, N5..N8 top face ccw
                n1 = index[(i, j, k)]
                n2 = index[(i+1, j, k)]
                n3 = index[(i+1, j+1, k)]
                n4 = index[(i, j+1, k)]
                n5 = index[(i, j, k+1)]
                n6 = index[(i+1, j, k+1)]
                n7 = index[(i+1, j+1, k+1)]
                n8 = index[(i, j+1, k+1)]
                elems.append((n1,n2,n3,n4,n5,n6,n7,n8))
    NUME = len(elems)

    # compute interpolation L2 error: integrate |u_exact - u_h|^2 over domain
    err2 = 0.0
    vol = 0.0
    for e in elems:
        # get nodal coords for element
        nodes = [coords[node-1] for node in e]
        # nodal u values (interpolated from exact)
        u_nodes = [u_exact(*nodes[i]) for i in range(8)]
        # integrate over gauss points
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    xi = gp[a]; eta = gp[b]; zeta = gp[c]
                    N, dN_dxi, dN_deta, dN_dzeta = shape_functions(xi, eta, zeta)
                    # jacobian
                    J = np.zeros((3,3))
                    xg = yg = zg = 0.0
                    for i_node in range(8):
                        x,y,z = nodes[i_node]
                        J[0,0] += dN_dxi[i_node] * x; J[0,1] += dN_deta[i_node] * x; J[0,2] += dN_dzeta[i_node] * x
                        J[1,0] += dN_dxi[i_node] * y; J[1,1] += dN_deta[i_node] * y; J[1,2] += dN_dzeta[i_node] * y
                        J[2,0] += dN_dxi[i_node] * z; J[2,1] += dN_deta[i_node] * z; J[2,2] += dN_dzeta[i_node] * z
                        xg += N[i_node] * x; yg += N[i_node] * y; zg += N[i_node] * z
                    detJ = np.linalg.det(J)
                    if detJ <= 0:
                        raise ValueError('Nonpositive detJ')
                    u_ex = u_exact(xg, yg, zg)
                    # u_h at gauss point
                    u_h = np.zeros(3)
                    for i_node in range(8):
                        u_h += N[i_node] * u_nodes[i_node]
                    diff = u_ex - u_h
                    err2 += (diff[0]**2 + diff[1]**2 + diff[2]**2) * detJ * gw[a]*gw[b]*gw[c]
                    vol += detJ * gw[a]*gw[b]*gw[c]
    L2 = math.sqrt(err2)
    h = 1.0 / nx
    results.append((nx, NUME, h, L2))

# compute convergence rates
print('nx, NUME, h, L2_error, rate')
prev = None
for item in results:
    nx, NUME, h, L2 = item
    rate = ''
    if prev is not None:
        L2prev = prev[3]
        rate = math.log(L2prev / L2) / math.log(prev[2] / h)
        rate = '{:.3f}'.format(rate)
    print(f'{nx:2d}, {NUME:4d}, {h:.4f}, {L2:.6e}, {rate}')
    prev = item

# also write results to data/convergence_h8.csv
os.makedirs('data', exist_ok=True)
with open('data/convergence_h8.csv', 'w') as f:
    f.write('nx,NUME,h,L2_error,rate\n')
    prev = None
    for item in results:
        nx, NUME, h, L2 = item
        rate = ''
        if prev is not None:
            L2prev = prev[3]
            rate = math.log(L2prev / L2) / math.log(prev[2] / h)
            rate = '{:.6f}'.format(rate)
        f.write(f'{nx},{NUME},{h},{L2},{rate}\n')

print('\nWrote data/convergence_h8.csv')

# plot convergence
try:
    import matplotlib.pyplot as plt

    hs = [r[2] for r in results]
    errs = [r[3] for r in results]

    plt.figure()
    plt.loglog(hs, errs, '-o')
    plt.gca().invert_xaxis()
    plt.xlabel('h')
    plt.ylabel('L2 error')
    plt.title('H8 interpolation convergence')
    plt.grid(True, which='both', ls='--')
    plt.savefig('data/convergence_h8.png', dpi=200)
    print('Wrote data/convergence_h8.png')
except Exception as e:
    print('Could not plot convergence (matplotlib may be missing):', e)
