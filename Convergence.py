import numpy as np
import matplotlib.pyplot as plt
import math
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix' 
plt.rcParams['axes.unicode_minus'] = False 

def parse_out_file(file_path):
    x_w, w_vals = [], []
    x_mx, mx_vals = [], []
    
    if not os.path.exists(file_path):
        print(f" WARNING: File {file_path} not found. Skipping this curve.")
        return None, None, None, None

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_block = None
    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue
            
        if "W-DISPLACEMENT" in clean_line:
            current_block = "W"
            continue
        elif "MX-MOMENT" in clean_line:
            current_block = "MX"
            continue
        
        if "X-COORDINATE" in clean_line or "CENTERLINE" in clean_line:
            continue
            
        try:
            parts = clean_line.split()
            if len(parts) == 2:
                coord = float(parts[0])
                value = float(parts[1])
                
                if current_block == "W":
                    x_w.append(coord)
                    w_vals.append(value)
                elif current_block == "MX":
                    x_mx.append(coord)
                    mx_vals.append(value)
        except ValueError:
            continue

    return np.array(x_w), np.array(w_vals), np.array(x_mx), np.array(mx_vals)

def calculate_exact_solution(L, q, D_val, nu, x_samples):
    w_exact = np.zeros_like(x_samples)
    mx_exact = np.zeros_like(x_samples)
    
    pi = math.pi
    a = L
    b = L
    K = -4.0 * q * a**2 / (pi**3)
    m_all = np.array([1, 3, 5, 7])
    
    E = np.zeros(8, float)
    E[1] = 0.3722 * K
    E[3] = -0.0380 * K
    E[5] = -0.0178 * K
    E[7] = -0.0085 * K

    x_exact_sys = x_samples - a / 2.0
    y = 0.0

    for index, xi in enumerate(x_exact_sys):
        w1, w2, w3 = 0.0, 0.0, 0.0
        w1_xx, w2_xx, w3_xx = 0.0, 0.0, 0.0
        w1_yy, w2_yy, w3_yy = 0.0, 0.0, 0.0
        
        for m in m_all:
            a_m = m * pi * b / (2 * a)  
            b_m = m * pi * a / (2 * b)
            
            A1 = 4 * q * a**4 / (pi**5 * D_val) * (-1)**((m-1)/2.0) / m**5
            B1 = (a_m * math.tanh(a_m) + 2) / (2 * math.cosh(a_m))
            C1 = 1 / (2 * math.cosh(a_m))
            D1 = m * pi / a
            
            A2 = -a**2 / (2 * pi**2 * D_val) * E[m] * (-1)**((m-1)/2.0) / (m**2 * math.cosh(a_m))
            B2 = a_m * math.tanh(a_m)
            D2 = m * pi / a
            
            A3 = -b**2 / (2 * pi**2 * D_val) * E[m] * (-1)**((m-1)/2.0) / (m**2 * math.cosh(b_m))
            B3 = b_m * math.tanh(b_m)
            D3 = m * pi / b
            
            w1 += A1 * math.cos(D1*xi) * (1 - B1 * math.cosh(D1*y) + C1 * D1 * y * math.sinh(D1*y))
            w2 += A2 * math.cos(D2*xi) * (D2 * y * math.sinh(D2*y) - B2 * math.cosh(D2*y))
            w3 += A3 * math.cos(D3*y) * (D3 * xi * math.sinh(D3*xi) - B3 * math.cosh(D3*xi))
            
            w1_xx += A1 * (-D1**2 * math.cos(D1*xi)) * (1 - B1 * math.cosh(D1*y) + C1 * D1 * y * math.sinh(D1*y))
            w2_xx += A2 * (-D2**2 * math.cos(D2*xi)) * (D2 * y * math.sinh(D2*y) - B2 * math.cosh(D2*y))
            w3_xx += A3 * math.cos(D3*y) * ((2 - B3) * D3**2 * math.cosh(D3*xi) + D3**3 * xi * math.sinh(D3*xi))
            
            w1_yy += A1 * math.cos(D1*xi) * ((2 * C1 - B1) * D1**2 * math.cosh(D1*y) + C1 * D1 * y**3 * math.sinh(D1*y))
            w2_yy += A2 * math.cos(D2*xi) * ((2 - B2) * D2**2 * math.cosh(D2*y) + D2**3 * y * math.sinh(D2*y))
            w3_yy += A3 * (-D3**2 * math.cos(D3*y)) * (D3 * xi * math.sinh(D3*xi) - B3 * math.cosh(D3*xi))
            
        w_exact[index] = w1 + w2 + w3
        
        w_xx = w1_xx + w2_xx + w3_xx
        w_yy = w1_yy + w2_yy + w3_yy
        mx_exact[index] = -D_val * (w_xx + nu * w_yy)
        
    return w_exact, mx_exact

def plot_file_convergence(file_configs):
    fig1, ax1 = plt.subplots(figsize=(6.5, 4.5))  
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5)) 
    
    x_axis = np.linspace(0.0, 8.0, 200)
    
    L, q, E_mod, nu, t = 8.0, -1.0, 200.0e9, 0.3, 0.01
    D_val = (E_mod * t**3) / (12.0 * (1.0 - nu**2))
    
    w_exact, mx_exact = calculate_exact_solution(L, q, D_val, nu, x_samples=x_axis)

    ax1.plot(x_axis, w_exact, 'k-', linewidth=2.0, label='Analytical Solution')
    ax2.plot(x_axis, mx_exact, 'k-', linewidth=2.0, label='Analytical Solution')

    academic_colors = ['#1f77b4', '#d62728', '#2ca02c']
    markers = ['^', 's', 'o']
    
    mesh_sizes = []
    relative_errors = []

    for idx, cfg in enumerate(file_configs):
        xw, ww, xmx, mx = parse_out_file(cfg['path'])
        
        color = academic_colors[idx]
        marker = markers[idx]
        
        if xw is not None:
            ax1.plot(xw, ww, color=color, marker=marker, linestyle='--', 
                     linewidth=1.2, markersize=4, alpha=0.9, label=cfg['name'])
            center_idx = np.abs(xw - 4.0).argmin()
            w_fem_center = ww[center_idx]
            w_exact_center = -0.0002818

            h_size = L / cfg['mesh_num']
            error = np.abs(w_fem_center - w_exact_center) / np.abs(w_exact_center)

            mesh_sizes.append(h_size)
            relative_errors.append(error)

        if xmx is not None:
            ax2.plot(xmx, mx, color=color, marker=marker, linestyle='--', 
                     linewidth=1.2, markersize=4, alpha=0.9, label=cfg['name'])

    ax1.set_title('Case Validation: Deflection $w$ along Centerline ($y = 4.0$ m)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Coordinate $x$ (m)', fontsize=10)
    ax1.set_ylabel('Deflection $w$ (m)', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='best', frameon=True, edgecolor='none', facecolor='#f5f5f5')

    ax2.set_title('Case Validation: Bending Moment $M_x$ along Centerline ($y = 4.0$ m)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Coordinate $x$ (m)', fontsize=10)
    ax2.set_ylabel('Bending Moment $M_x$ ($\mathrm{N}\cdot\mathrm{m/m}$)', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='best', frameon=True, edgecolor='none', facecolor='#f5f5f5')

    
    if len(mesh_sizes) > 1:
        fig3, ax3 = plt.subplots(figsize=(6.0, 4.5))
        ax3.loglog(mesh_sizes, relative_errors, color='#2c3e50', marker='D', 
               linestyle='-', linewidth=1.5, markersize=6, label='Proposed Plate Element')
        if len(mesh_sizes) > 1:
            slope, _ = np.polyfit(np.log(mesh_sizes), np.log(relative_errors), 1)
            ax3.text(0.05, 0.05, f'Convergence Rate: {slope:.4f}', 
             fontsize=10, color='#d62728', fontfamily='serif', 
             transform=ax3.transAxes, fontweight='bold')
            
    ax3.set_title('Error Convergence Spectrum (Centerline Deflection)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Mesh Characteristic Size $h$ (m)', fontsize=10)
    ax3.set_ylabel('Relative Error (Log Scale)', fontsize=10)
    ax3.grid(True, which="both", linestyle=':', alpha=0.6)
    ax3.legend(loc='best', frameon=True, facecolor='#f5f5f5', edgecolor='none')

    os.makedirs('results', exist_ok=True)
    fig1.savefig('results/Convergence_Deflection.png', dpi=300, bbox_inches='tight')
    fig2.savefig('results/Convergence_Mx.png', dpi=300, bbox_inches='tight')
    fig3.savefig('results/Error_LogLog_Plot.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    file_settings = [
        {'name': '2$\\times$2 Mesh', 'path': 'data/Plate_Validation_2_plotting.out', 'mesh_num': 2},
        {'name': '4$\\times$4 Mesh', 'path': 'data/Plate_Validation_4_plotting.out', 'mesh_num': 4},
        {'name': '8$\\times$8 Mesh', 'path': 'data/Plate_Validation_8_plotting.out', 'mesh_num': 8}
    ]
    plot_file_convergence(file_settings)