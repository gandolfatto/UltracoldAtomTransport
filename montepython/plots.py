import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from typing import Dict, Any
from matplotlib.axes import Axes
import pickle

from constants import *
from lattice_funcs import *



###
suffix = 'transport_sim'
def load(file_name, suffix=suffix):
    with open(f'{file_name+suffix}.pkl', 'rb') as inp:
        data = pickle.load(inp)
    return data


def info_axis(ax: Axes, params: Dict[str, Any]):

    ax.axis("off")

    texts = [
        (0.01, 0.90, 'Beam Parameters:'),
        ############################
        (0.1, 0.85, f'Wavelength = {1e9*params["wavelength"]:.4} nm'),
        (0.1, 0.80, f'(w_0(1), w_0(2)) = ({1e6*params["w0_1"]:.3} \u03BCm, {1e6*params["w0_1"]:.3} \u03BCm)'),
        (0.1, 0.75, f'(P(1), P(2)) = ({1e0*params["P1"]:.3} W, {1e0*params["P2"]:.3} W)'),

        (0.01, 0.65, 'Simulation Parameters:'),
        ############################
        (0.1, 0.60, f'N = {params["N atoms"]}'),
        (0.1, 0.55, f'T = {1e6*params["T"]:.3} \u03BCK'),
        (0.1, 0.50, f'dt = {1e6*params["dt"]:.3} \u03BCs'),
        (0.1, 0.45, f'Sim time = {1e3*params["t sim"]:.3} ms'),
    
        (0.01, 0.35, 'Transport Parameters:'),
        ############################
        (0.1, 0.30, f'Transport distance = {1e3*params["d max"]:.3} mm'),
        (0.1, 0.25, f'Transport time = {1e3*params["d max"]:.3} ms'),        
    ]
    
    for x, y, text in texts:
        ax.text(x, y, text)
    return ax
###





#===============================================#
#                 TRAJECTORIES                  #
#===============================================#
# 
###
def trajectory_plot(r_max: float, z_max: float):

    results = load('results')
    params = load('parameters')
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(2, 2, width_ratios=[1, 2], figsize=(12, 6))

    i_show = 1

    xs = np.linspace(-r_max, r_max, 200)
    ys = np.linspace(-r_max, r_max, 200)
    zs = np.linspace(0, z_max, 20)



    depth_xy = np.array([[-(U_lattice(x, y, 0, 0*ms, params["lens elements"], 
                                      params["r0_1"], params["r0_2"], 
                                      params["prop dir1"], params["prop dir2"],
                                      params["wavelength"], 
                                      params["w0_1"], params["w0_2"], 
                                      params["P1"], params["P2"],
                                      params["ramp type"], params["d max"], params["t max"],
                                      params["transitions"]))/kb * 1e6 
                            for x in xs] 
                        for y in ys])

    depth_xz = np.array([[-(U_lattice(x, 0, z, 0*ms, params["lens elements"], 
                                      params["r0_1"], params["r0_2"], 
                                      params["prop dir1"], params["prop dir2"],
                                      params["wavelength"], 
                                      params["w0_1"], params["w0_2"], 
                                      params["P1"], params["P2"],
                                      params["ramp type"], params["d max"], params["t max"],
                                      params["transitions"]))/kb * 1e6 
                            for x in xs] 
                        for z in zs])

    depth_yz = np.array([[-(U_lattice(0, y, z, 0*ms, params["lens elements"], 
                                      params["r0_1"], params["r0_2"], 
                                      params["prop dir1"], params["prop dir2"],
                                      params["wavelength"], 
                                      params["w0_1"], params["w0_2"], 
                                      params["P1"], params["P2"],
                                      params["ramp type"], params["d max"], params["t max"],
                                      params["transitions"]))/kb * 1e6 
                            for y in ys] 
                        for z in zs])
    
    X, Y = np.meshgrid(xs, ys)
    c_xy = ax[0, 0].contourf(X/1e-3, Y/1e-3, depth_xy, 
                             levels=200, 
                             vmin=0, vmax=np.max(depth_xy), 
                             cmap='Reds')
    
    X, Z = np.meshgrid(xs, zs)
    c_xz = ax[0, 1].contourf(Z/1e-3, X/1e-3, depth_xz, 
                             levels=200, 
                             vmin=0, vmax=np.max(depth_xz), 
                             cmap='Reds')
    
    Y, Z = np.meshgrid(ys, zs)
    c_yz = ax[1, 1].contourf(Z/1e-3, Y/1e-3, depth_yz, 
                             levels=200, 
                             vmin=0, vmax=np.max(depth_yz), 
                             cmap='Reds')
    
    
    fig.colorbar(c_xy, 
                 ax=ax[0, 0], 
                 orientation = "vertical", 
                 label = "Lattice Depth (\u03BCK)")
    
    fig.colorbar(c_xz, 
                 ax=ax[0, 1], 
                 orientation = "vertical", 
                 label = "Lattice Depth (\u03BCK)")
    
    fig.colorbar(c_yz, 
                 ax=ax[1, 1], 
                 orientation = "vertical", 
                 label = "Lattice Depth (\u03BCK)")
    

    beam_widths0 = np.array([beam_width(z, params["lens elements"], 
                                        params["r0_1"], 
                                        params["prop dir1"], params["wavelength"],
                                        params["w0_1"],) for z in zs])
    beam_widths1 = np.array([beam_width(z, params["lens elements"], 
                                        params["r0_2"], 
                                        params["prop dir2"], params["wavelength"],
                                        params["w0_2"],) for z in zs])


    ax[1, 1].plot(1e3*zs, +1e3*beam_widths0, label='Beam Width w(z)', color='purple', linewidth=2)
    ax[1, 1].plot(1e3*zs, -1e3*beam_widths0, color='purple', linewidth=2)
    ax[1, 1].plot(1e3*zs, +1e3*beam_widths1, label='Beam Width w(z)', color='purple', linewidth=2)
    ax[1, 1].plot(1e3*zs, -1e3*beam_widths1, color='purple', linewidth=2)

    ax[1, 1].axvline(1e3*params["pos load"][2], color='black', linestyle='--', label='Load Position')
    ax[1, 1].axvline(1e3*(params["pos load"][2] + params["d max"]), color='black', linestyle='--', label='Drop Position')
    
    for ele in params["lens elements"]: 
        ax[1, 1].axvline(1e3*ele[0], color='black', linestyle='-', label='Lens Position')


    rs = results['rs']

    for i in range(params['N atoms']):
        if (i != i_show):
            ax[0, 0].plot(1e3*rs[i, :, 0], 1e3*rs[i, :, 1], color='skyblue', linewidth=1)
            ax[0, 1].plot(1e3*rs[i, :, 2], 1e3*rs[i, :, 0], color='skyblue', linewidth=1)
            ax[1, 1].plot(1e3*rs[i, :, 2], 1e3*rs[i, :, 1], color='skyblue', linewidth=1)

    ax[0, 0].plot(1e3*rs[i_show, :, 0], 1e3*rs[i_show, :, 1], color='blue', linewidth=2)
    ax[0, 1].plot(1e3*rs[i_show, :, 2], 1e3*rs[i_show, :, 0], color='blue', linewidth=2)
    ax[1, 1].plot(1e3*rs[i_show, :, 2], 1e3*rs[i_show, :, 1], color='blue', linewidth=2)


    ax[0, 0].set_xlabel('$x$ (mm)', fontsize=12)
    ax[0, 0].set_ylabel('$y$ (mm)', fontsize=12)
    ax[0, 1].set_xlabel('$z$ (mm)', fontsize=12)
    ax[0, 1].set_ylabel('$y$ (mm)', fontsize=12)
    ax[1, 1].set_xlabel('$z$ (mm)', fontsize=12)
    ax[1, 1].set_ylabel('$y$ (mm)', fontsize=12)
    

    ax[0, 0].set_xlim([-1e3*r_max/2, 1e3*r_max/2])
    ax[0, 0].set_ylim([-1e3*r_max/2, 1e3*r_max/2])
    ax[0, 1].set_xlim([0, 1e3*z_max])
    ax[0, 1].set_ylim([-1e3*r_max, 1e3*r_max])
    ax[1, 1].set_xlim([0, 1e3*z_max])
    ax[1, 1].set_ylim([-1e3*r_max, 1e3*r_max])


    info_axis(ax[1, 0], params)

    ax[1, 1].axvline(1e3*params["pos load"][2], color='black', linestyle='--', label='Load Position')
    ax[1, 1].axvline(1e3*(params["pos load"][2] + params["d max"]), color='black', linestyle='--', label='Drop Position')

    fig.tight_layout()
    
    return fig, ax
###





#===============================================#
#                   ENERGIES                    #
#===============================================#
# 
###
def energies(results: Dict[str, Any], params: Dict[str, Any]):

    step = 100

    m = params["mass"]
    N_atoms = params["N atoms"]
    ts = results["ts"][::step]
    rs = results["rs"][:, ::step, :]
    vs = results["vs"][:, ::step, :]

    K_abs = np.zeros((N_atoms, len(ts)))
    K_rel = np.zeros((N_atoms, len(ts)))
    Ug    = np.zeros((N_atoms, len(ts)))
    Ul    = np.zeros((N_atoms, len(ts)))

    for i in range(N_atoms):
        for j in range( int( params["t sim"] / (step * params["dt"]) )) :
            
            K_abs[i, j] = 0.5*m*(
                                (vs[i, j, 0])**2 + 
                                (vs[i, j, 1])**2 + 
                                (vs[i, j, 2])**2
                                )
            
            K_rel[i, j] = 0.5*m*(
                                (vs[i, j, 0])**2 + 
                                (vs[i, j, 1])**2 + 
                                (vs[i, j, 2] - v_latt(params["ramp type"], ts[j], params["d max"], params["t max"]))**2
                                )
            
            Ug[i, j] = m*g0*rs[i, j, 1]

            Ul[i, j] = U_lattice(rs[i, j, 0], rs[i, j, 1], rs[i, j, 2], ts[j],
                                 params["lens elements"], 
                                 params["r0_1"], params["r0_2"], 
                                 params["prop dir1"], params["prop dir2"],
                                 params["wavelength"], 
                                 params["w0_1"], params["w0_2"], 
                                 params["P1"], params["P2"],
                                 params["ramp type"], params["d max"], params["t max"],
                                 params["transitions"])
            
    return ts, K_abs, K_rel, Ug, Ul



def energy_plot(title: str = 'Energy in Moving Lattice'):

    results = load('results')
    params  = load('parameters')
    # ts = results['ts']

    ts, K_abs, K_rel, Ug, Ul = energies(results, params)
    
    i_show = 1
    fig, (ax_plot, ax_info) = plt.subplots(1, 2, width_ratios=[4, 1])

    # ax_plot.plot(ts[:]/1e-3, 1e6*(K_abs[i_show])/kb, color='orange', linewidth=1, label="KE_{abs.}")
    ax_plot.plot(ts[:]/1e-3, 1e6*(K_rel[i_show])/kb, color='red', linewidth=1, label="KE_{rel.}")
    ax_plot.plot(ts[:]/1e-3, 1e6*(Ug[i_show])/kb, color='blue', linewidth=1, label="$U_g$")
    ax_plot.plot(ts[:]/1e-3, 1e6*(Ul[i_show])/kb, color='green', linewidth=1, label='U_{lat.}')
    ax_plot.plot(ts[:]/1e-3, 1e6*(K_rel[i_show] + Ug[i_show] + Ul[i_show])/kb, color='purple', linewidth=1, label='Total Energy')
    
    ax_plot.set_xlabel('t (ms)', fontsize=15)
    ax_plot.set_ylabel('$E$ (\u03BCK)', fontsize=15)
    ax_plot.set_title(title)

    info_axis(ax_info, params)
    return fig, ax_info
###





#===============================================#
#               SPEED DISTRIBUTION              #
#===============================================#
# 
###
def boltzmann(m, T, v):

    A = 4*np.pi*(m/(2*np.pi*kb*T))**(3/2)
    return A * v**2 * np.exp(-m*v**2 / (2*kb*T))


def speed_dist():

    results = load('results')
    params  = load('parameters')

    time_step = 0

    m = params["mass"]
    T = params["T"]
    N_atoms = params["N atoms"]
    vs = results["vs"]
    speeds = np.sqrt(vs[:, time_step, 0]**2 + vs[:, time_step, 1]**2 + vs[:, time_step, 2]**2)

    fig, (ax_plot, ax_info) = plt.subplots(1, 2, width_ratios=[4, 1])

    v_grid = np.linspace(0, np.max(speeds)*1.2, 200)
    v_exp = (2/np.sqrt(np.pi)) * np.sqrt(2*kb*T/m)

    ax_plot.hist(speeds, bins=N_atoms//5, density=True, edgecolor='black')
    ax_plot.plot(v_grid, boltzmann(m, T, v_grid), color='red', label='Maxwell-Boltzmann distribution')
    ax_plot.axvline(v_exp, color='black', linestyle='--', label='Avg. Speed')

    ax_plot.set_xlabel('Speed (m/s)', fontsize=15)
    ax_plot.set_ylabel('Frequency', fontsize=15)
    ax_plot.set_title("Distribution of speeds", fontsize=15)
    ax_plot.legend()

    info_axis(ax_info, params)
    return fig, ax_info





trajectory_plot(r_max=1e-3, 
                z_max=load("parameters")["r0_2"])
plt.show()


# energy_plot()
# plt.show()


speed_dist()
plt.show()
