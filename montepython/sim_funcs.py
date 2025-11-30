import numpy as np
from numba import jit, prange

from constants import *
from lattice_funcs import *



#=======================================================#
#                   SIMULATION FUNCTIONS                #   
#=======================================================# 
#
###



@jit(nopython=True, fastmath=True, cache=True)
def MC_step(pos, vel, t, 
            lens_elements, 
            r0_1, r0_2, prop_dir1, prop_dir2, wavelength, w0_1, w0_2, P1, P2, 
            ramp_type, d_max, t_max, 
            transitions,
            dt,
            m): 
    
    """
    Perform one step of the Monte Carlo (MC) simulation. 

    Parameters
    ----------
    x (float)
    
    y (float)

    x (float)

    t (float)

    lens_elements (np.ndarray)

    r0_1(2) (float)

    prop_dir1(2) (float)

    wavelength (float)

    w0_1(2) (float)

    P1(2) (float)

    d_max (float)

    t_max (float)

    transitions (np.ndarray)

    dt (float):
        Time step.

    Returns
    -------
    r1 (np.ndarray): 
        Position of atom after one MC step. 
        Shape: (3,)

    v1 (np.ndarray): 
        Velocity of atom after one MC step. 
        Shape: (3,)
    """ 
        
    x, y, z = pos[0], pos[1], pos[2]
    x1, y1, z1 = x, y, z

    vx, vy, vz = vel[0], vel[1], vel[2]
    vx1, vy1, vz1 = vx, vy, vz

    t1 = t
    Fx, Fy, Fz = F_lattice(x1, y1, z1, t1, 
                           lens_elements, 
                           r0_1, r0_2, prop_dir1, prop_dir2, wavelength, w0_1, w0_2, P1, P2, 
                           ramp_type, d_max, t_max, 
                           transitions)

    dvx1, dvy1, dvz1 = (Fx/m)*dt, (g0 + Fy/m)*dt, (Fz/m)*dt
    dx1, dy1, dz1 = (vx1 + dvx1/2)*dt, (vy1 + dvy1/2)*dt, (vz1 + dvz1/2)*dt

    Fx, Fy, Fz = F_lattice(x1 + dx1, y1 + dy1, z1 + dz1, t1 + dt, 
                           lens_elements, 
                           r0_1, r0_2, prop_dir1, prop_dir2, wavelength, w0_1, w0_2, P1, P2, 
                           ramp_type, d_max, t_max, 
                           transitions)

    dvx2, dvy2, dvz2 = (Fx/m)*dt, (g0 + Fy/m)*dt, (Fz/m)*dt

    x1 += dx1
    y1 += dy1
    z1 += dz1

    vx1 += (dvx1 + dvx2)/2
    vy1 += (dvy1 + dvy2)/2
    vz1 += (dvz1 + dvz2)/2
    
    t1 += dt
            
    r1 = np.array([x1, y1, z1])
    v1 = np.array([vx1, vy1, vz1])

    return r1, v1





@jit(nopython=True, fastmath=True, cache=True)
def one_atom_sim(r0, v0, 
                 lens_elements, 
                 r0_1, r0_2, prop_dir1, prop_dir2, wavelength, w0_1, w0_2, P1, P2, 
                 ramp_type, d_max, t_max, 
                 transitions, 
                 dt, t_sim,
                 m):
    
    """
    Perform MC simulation for one atom.  

    Parameters
    ----------
    x (float)
    
    y (float)

    x (float)

    t (float)

    lens_elements (np.ndarray)

    r0_1(2) (float)

    prop_dir1(2) (float)

    wavelength (float)

    w0_1(2) (float)

    P1(2) (float)

    d_max (float)

    t_max (float)

    transitions (np.ndarray)

    dt (float)

    t_sim (float): 
        Simulation time. 

    Returns
    -------
    rs (np.ndarray): 
        Positions of atom at each time step of the simulation. 
        Shape: (Number of time steps, 3)

    vs (np.ndarray): 
        Velocities of atom at each time step of the simulation. 
        Shape: (Number of time steps, 3)
    """ 
    
    times = np.arange(0, t_sim, dt)
    Nt = int(t_sim/dt)

    rs = np.zeros((Nt, 3))
    vs = np.zeros((Nt, 3))

    rs[0], vs[0] = r0, v0
    r, v = r0, v0

    for i in range(Nt):
        r, v = MC_step(r, v, times[i], 
                       lens_elements, 
                       r0_1, r0_2, prop_dir1, prop_dir2, wavelength, w0_1, w0_2, P1, P2, 
                       ramp_type, d_max, t_max, 
                       transitions, 
                       dt,
                       m)
        rs[i], vs[i] = r, v

    return rs, vs





@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def run_MC_sim(pos_load, vel_load, 
               lens_elements, 
               r0_1, r0_2, prop_dir1, prop_dir2, wavelength, w0_1, w0_2, P1, P2, 
               ramp_type, d_max, t_max, 
               transitions, 
               dt, t_sim, N_atoms, sigma_x, sigma_v,
               m):
    
    """
    Perform MC simulation for N_atoms in parallel. 

    Parameters
    ----------
    x (float)
    
    y (float)

    x (float)

    t (float)

    lens_elements (np.ndarray)

    r0_1(2) (float)

    prop_dir1(2) (float)

    wavelength (float)

    w0_1(2) (float)

    P1(2) (float)

    d_max (float)

    t_max (float)

    transitions (np.ndarray)

    dt (float)

    t_sim (float)

    N_atoms (int):
        Number of atoms to simulate.
    
    sigma_x (np.ndarray):
        Width of intitial position distribution. 
    
    sigma_v (np.ndarray):
        Width of intitial velocity distribution. 

    Returns
    -------
    times (np.ndarray): 
        Simulation times. 
        Shape: (, Number of time steps)

    rs (np.ndarray): 
        Positions of each atom at each time step of the simulation. 
        Shape: (Number of atoms, Number of time steps, 3)

    vs (np.ndarray): 
        Velocities of each atom at each time step of the simulation. 
        Shape: (Number of atoms, Number of time steps, 3)
    """ 
    
    times = np.arange(0, t_sim, dt)
    Nt = int(t_sim/dt)

    rs = np.zeros((N_atoms, Nt, 3))
    vs = np.zeros((N_atoms, Nt, 3))

    r0_arr = pos_load + sigma_x*np.random.randn(N_atoms, 3)
    v0_arr = vel_load + sigma_v*np.random.randn(N_atoms, 3)

    for i in prange(N_atoms):

        rs[i], vs[i] = one_atom_sim(r0_arr[i], v0_arr[i], 
                                    lens_elements, 
                                    r0_1, r0_2, prop_dir1, prop_dir2, wavelength, w0_1, w0_2, P1, P2, 
                                    ramp_type, d_max, t_max, 
                                    transitions, 
                                    dt, t_sim,
                                    m)

    return times, rs, vs
