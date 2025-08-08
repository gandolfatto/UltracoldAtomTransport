import numpy as np
from numba import jit, prange

from constants import *





@jit(nopython=True, fastmath=True, cache=True)
def lens_elements_filtered(z, 
                           lens_elements):

    """
    Gets lens elements in lens_elements whose position along the optical axis is <= z. 

    Parameters
    ----------
    z (float): 
        Position along the optical axis. 

    lens_elements (np.ndarray): 
        Lens elements for the optical system. len_elements has shape
        (number of elements, 2). e.g.,

        lens_elements = np.array([
                                    [pos of lens 1, foc length of lens 1],             
                                    [pos of lens 2, foc length of lens 2],
                                                    .
                                                    .
                                                    .
                                ])

    Returns
    -------
    lens_elements_z (np.ndarray): 
        Lens elements for the optical system (w/ pos <= z). 
    """ 

    num_eles = lens_elements.shape[0]
    lens_elements_filt = np.empty((num_eles, 2), 
                                  dtype=lens_elements.dtype)
     
    count = 0
    for i in range(num_eles):
        if lens_elements[i][0] <= z:
            
            lens_elements_filt[count][0] = lens_elements[i][0]
            lens_elements_filt[count][1] = lens_elements[i][1]
        
            count += 1

    lens_elements_z = lens_elements_filt[:count] 

    return lens_elements_z



@jit(nopython=True, fastmath=True, cache=True)
def prop_matrix(z, 
                lens_elements):

    """
    Computes the ABCD matrix up to position z along the optical axis.

    Parameters
    ----------
    z (float)

    lens_elements (np.ndarray)

    Returns
    -------
    M (np.ndarray): 
        ABCD matrix for optical system. 
    """ 

    lens_elements_z = lens_elements_filtered(z, 
                                             lens_elements)
    num_eles = lens_elements_z.shape[0]
    
    if num_eles == 0:
        M = np.array([
                      [1., z], 
                      [0., 1.]
                    ])

    else:
        if z < lens_elements_z[0][0]:
            M = np.array([
                          [1., z], 
                          [0., 1.]
                        ])
        else: 
            M = np.array([
                          [1., 0.], 
                          [0., 1.]
                        ])
            zi = 0
            for i in range(num_eles):
                z_prop = lens_elements_z[i][0]
                z_diff = z_prop - zi
                zi = z_prop

                mat = np.array([
                                [1., z], 
                                [-1/lens_elements_z[i][1], 1 - (z_diff/lens_elements_z[i][1])]
                            ]) 
                M = mat @ M

            M = np.array([[1., z - lens_elements_z[-1][0]], [0., 1.]]) @ M

    return M



@jit(nopython=True, fastmath=True, cache=True)
def q(z, 
      lens_elements, 
      r0, prop_dir, wavelength, w0):
    
    """
    Computes the complex beam parameter q at position z along the optical axis. 

    Parameters
    ----------
    z (float)

    lens_elements (np.ndarray)
 
    r0 (float): 
        Initial position of the beam along the optical axis. 

    prop_dir (float): 
        Propagation direction of the beam (+1. or -1.).

    wavelength (float):
        Wavelength of the beam. 
    
    w0 (float):
        Beam width at r0. 

    Returns
    -------
    q_new (complex): 
        Complex beam parameter. 
    """ 

    Zr = np.pi*w0**2/wavelength
    q0 = 1j*Zr 
    
    abcd = prop_matrix(r0 + prop_dir*z, 
                       lens_elements)
    A = abcd[0][0]
    B = abcd[0][1]
    C = abcd[1][0]
    D = abcd[1][1]

    q_new = (A*q0 + B)/(C*q0 + D)

    return q_new 



@jit(nopython=True, fastmath=True, cache=True)
def beam_width(z, 
               lens_elements, 
               r0, prop_dir, wavelength, w0):
    
    """
    Computes the beam width w at position z along the optical axis. 

    Parameters
    ----------
    z (float)

    lens_elements (np.ndarray)

    r0 (float)

    prop_dir (float)

    wavelength (float)

    w0 (float)

    Returns
    -------
    w (float): 
        Beam width at position z. 
    """ 

    q_val = q(z, 
              lens_elements, 
              r0, prop_dir, wavelength, w0) 
    w = np.sqrt(wavelength / (np.pi * np.imag(-1/q_val))) 

    return w



@jit(nopython=True, fastmath=True, cache=True)
def beam_width_deriv(z, 
                     lens_elements, 
                     r0, prop_dir, wavelength, w0):

    """
    Computes the derivative of the beam width with respect to z.

    Parameters
    ----------
    z (float)

    lens_elements (np.ndarray)

    r0 (float)

    prop_dir (float)

    wavelength (float)

    w0 (float)

    Returns
    -------
    w_prime (float): 
        Derivative of the beam width respect to z. 
        
        This is useful for computing the force on 
        the atoms due to the optical dipole trap (ODT).
    """ 

    Zr = np.imag( 
                    q(z, 
                      lens_elements, 
                      r0, prop_dir, wavelength, w0) 
                )
    w0 = np.sqrt(Zr * wavelength / np.pi)
    w_prime = w0*(z/Zr)*(1/np.sqrt(1 + (z/Zr)**2))

    return w_prime



@jit(nopython=True, fastmath=True, cache=True)
def I_field(x, y, z, 
            lens_elements, 
            r0, prop_dir, wavelength, w0, P):
    
    """
    Computes the intensity of the Gaussian beam. 

    Parameters
    ----------
    z (float)

    lens_elements (np.ndarray)

    r0 (float)

    prop_dir (float)

    wavelength (float)

    w0 (float)

    P (float): 
        Beam power (Watts). 

    Returns
    -------
    I (float): 
        Beam intensity. 
    """ 

    w = beam_width(z, 
                   lens_elements, 
                   r0, prop_dir, wavelength, w0)
    I = ((2*P)/(np.pi*w**2)) * np.exp(-2*((x**2 + y**2)/w**2))

    return I






@jit(nopython=True, fastmath=True, cache=True)
def z_latt(t, 
           d_max, t_max):

    """
    Computes the derivative of the beam width with respect to z.

    Parameters
    ----------
    t (float): 
        Time. 

    d_max (float): 
        Transport distance.

    t_max (float): 
        Transport time. 

    Returns
    -------
    z_lattice (float): 
        Position of the ODT minimum along the optical axis. 
    """ 

    t_reduced = t/t_max
    z_lattice = d_max*(10*(t_reduced)**3 - 15*(t_reduced)**4 + 6*(t_reduced)**5)

    return z_lattice


@jit(nopython=True, fastmath=True, cache=True)
def v_latt(t,
           d_max, t_max): 
    
    t_reduced = t/t_max
    v_lattice = 30*(d_max/t_max)*(t_reduced**2 - 2*t_reduced**3 + t_reduced**4)
    
    return v_lattice






@jit(nopython=True, fastmath=True, cache=True)
def trap_info(wavelength, 
              transition_wavelength, transition_linewidth):
    
    """
    Get trap info (detuning, dipole matrix element, atomic polarizability) associated 
    with a particular atomic transition. 

    Parameters
    ----------
    wavelength (float) 

    transition_wavelength (float): 
        Wavelength of the atomic transition.

    transition_linewidth (float): 
        Linewidth of the atomic transition. 

    Returns
    -------
    delta (float): 
        Detuning between incident beam and atomic transition.
    
    dip_element (float):
        Dipole matrix element associated incident beam and atomic transition. 
    
    alpha (complex):
        Atomic polarizability associated with incident beam and atomic transition. 
    """ 

    omega = 2*np.pi*c/wavelength
    omega_i = 2*np.pi*c/transition_wavelength
    
    delta = omega - omega_i
    dip_element = (3*np.pi*e0*hbar*c**3 / omega_i**3) * transition_linewidth
    alpha = -(dip_element / hbar) * ((delta - 1j*transition_linewidth /2) / ((delta**2 + 1j*(transition_linewidth/2)**2)))

    return delta, dip_element, alpha








@jit(nopython=True, fastmath=True, cache=True)
def I_lattice(x, y, z, t, 
              lens_elements, 
              r0_1, r0_2, prop_dir1, prop_dir2, wavelength, w0_1, w0_2, P1, P2, 
              d_max, t_max):
    
    """
    Computes the intensity of the optical lattice formed by two counter-propagating
    Gaussian beams at position (x, y, z) and time t. 

    Parameters
    ----------
    x (float): 
        Position along x-axis. 
    
    y (float): 
        Position along y-axis. 
    
    x (float): 
        Position along z-axis (optical axis).

    t (float): 
        Time. 
    
    lens_elements (np.ndarray): 
        Lens elements for the optical system. len_elements has shape
        (number of elements, 2). 

    r0_1(2) (float): 
        Initial position of the beam 1 (2) along the optical axis. 

    prop_dir1(2) (float): 
        Propagation direction of beam 1 (2) (+1. or -1.).

    wavelength (float):
        Wavelength of the beam (assumed to be the same for beam 1 and beam 2).  
    
    w0_1(2) (float):
        Beam 1 (2) width at r0. 
    
    P1(2) (float):
        Beam 1 (2) power (Watts). 
    
    d_max (float): 
        Transport distance.

    t_max (float): 
        Transport time. 

    Returns
    -------
    I_lattice (float): 
        Intensity of the optical lattice at position (x, y, z) at time t. 
    """ 
        
    z_lattice = z_latt(t, d_max, t_max)                                             

    I1 = I_field(x, y, z, 
                 lens_elements, 
                 r0_1, prop_dir1, wavelength, w0_1, P1)
    
    I2 = I_field(x, y, z, 
                 lens_elements, 
                 r0_2, prop_dir2, wavelength, w0_2, P2)

    k = 2*np.pi/wavelength
    I = I1 + I2 + 2*np.sqrt(I1*I2)*np.cos(2*k*(z - z_lattice))

    return I
    


@jit(nopython=True, fastmath=True, cache=True)
def U_lattice(x, y, z, t, lens_elements, 
              r0_1, r0_2, prop_dir1, prop_dir2, wavelength, w0_1, w0_2, P1, P2, 
              d_max, t_max,
              transitions):
    
    """
    Computes the intensity of the optical lattice formed by two counter-propagating
    Gaussian beams at position (x, y, z) and time t. 

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

    transitions (np.ndarray):
        Array of relevant atomic transitions. transitions has shape 
        (number of transitions, 2). e.g.,

        lens_elements = np.array([
                                    [transition 1 wavelength, transition 1 linewidth],             
                                    [transition 2 wavelength, transition 2 linewidth],
                                                    .
                                                    .
                                                    .
                                ])

    Returns
    -------
    U_lattice (float): 
        ODT potential of the optical lattice at position (x, y, z) and time t. 
    """ 

    I = I_lattice(x, y, z, t, 
                  lens_elements, 
                  r0_1, r0_2, prop_dir1, prop_dir2, wavelength, w0_1, w0_2, P1, P2, 
                  d_max, t_max)
    
    U = 0
    for transition in transitions:
        alpha = trap_info(wavelength, transition[0], transition[1])[2]
        U += (-alpha.real / (2*e0*c)) * I
    return U



@jit(nopython=True, fastmath=True, cache=True)
def F_lattice(x, y, z, t, 
              lens_elements, 
              r0_1, r0_2, prop_dir1, prop_dir2, wavelength, w0_1, w0_2, P1, P2, 
              d_max, t_max,
              transitions):
    
    """
    Computes the intensity of the optical lattice formed by two counter-propagating
    Gaussian beams at position (x, y, z) and time t. 

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

    Returns
    -------
    F_lattice (float): 
        Force on atoms at position (x, y, z) at time t due to ODT potential. 
    """ 

        
    z_lattice = z_latt(t, d_max, t_max)


    w1 = beam_width(z, 
                    lens_elements, 
                    r0_1, prop_dir1, wavelength, w0_1)
    w2 = beam_width(z, 
                    lens_elements, 
                    r0_2, prop_dir2, wavelength, w0_2)
    
    w_prime1 = beam_width_deriv(z, 
                                lens_elements, 
                                r0_1, prop_dir1, wavelength, w0_1)
    w_prime2 = beam_width_deriv(z, 
                                lens_elements, 
                                r0_2, prop_dir2, wavelength, w0_2)
    
    I1 = I_field(x, y, z, 
                 lens_elements, 
                 r0_1, prop_dir1, wavelength, w0_1, P1)
    I2 = I_field(x, y, z, 
                 lens_elements, 
                 r0_2, prop_dir2, wavelength, w0_2, P2)

    k = 2*np.pi/wavelength
    _dIdx = ((4*x)/(w1**2))*I1 + ((4*x)/(w2**2))*I2 + 4*x*((1/w1**2) + (1/w2**2)) * np.sqrt(I1*I2)
    _dIdy = ((4*y)/(w1**2))*I1 + ((4*y)/(w2**2))*I2 + 4*y*((1/w1**2) + (1/w2**2)) * np.sqrt(I1*I2)
    _dIdz = (I1*(2*w_prime1/w1)*(1-((x**2 + y**2)/w1**2)) + I2*(2*w_prime2/w2)*(1-((x**2 + y**2)/w2**2)) + np.sqrt(I1*I2)*((2*w_prime1/w1)*(1-((x**2 + y**2)/w1**2)) + (2*w_prime2/w2)*(1-((x**2 + y**2)/w2**2))))*np.cos(2*k*(z - z_lattice))  + 4*k*np.sin(2*k*(z - z_lattice))*np.sqrt(I1*I2)                      


    Fx = 0
    Fy = 0
    Fz = 0

    for transition in transitions:

        alpha = trap_info(wavelength, 
                          transition[0], transition[1]
                          )[2]
        Fx += (-alpha.real / (2*e0*c)) * _dIdx
        Fy += (-alpha.real / (2*e0*c)) * _dIdy
        Fz += (-alpha.real / (2*e0*c)) * _dIdz

    F = Fx, Fy, Fz
    return F



@jit(nopython=True, fastmath=True, cache=True)
def MC_step(pos, vel, t, 
            lens_elements, 
            r0_1, r0_2, prop_dir1, prop_dir2, wavelength, w0_1, w0_2, P1, P2, 
            d_max, t_max, 
            transitions,
            dt): 
    
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
                           d_max, t_max, 
                           transitions)

    dvx1, dvy1, dvz1 = (Fx/m)*dt, (g0 + Fy/m)*dt, (Fz/m)*dt
    dx1, dy1, dz1 = (vx1 + dvx1/2)*dt, (vy1 + dvy1/2)*dt, (vz1 + dvz1/2)*dt

    Fx, Fy, Fz = F_lattice(x1 + dx1, y1 + dy1, z1 + dz1, t1 + dt, 
                           lens_elements, 
                           r0_1, r0_2, prop_dir1, prop_dir2, wavelength, w0_1, w0_2, P1, P2, 
                           d_max, t_max, 
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
                 d_max, t_max, 
                 transitions, 
                 dt, t_sim):
    
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
                       d_max, t_max, 
                       transitions, 
                       dt)
        rs[i], vs[i] = r, v

    return rs, vs



@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def run_MC_sim(pos_load, vel_load, 
               lens_elements, 
               r0_1, r0_2, prop_dir1, prop_dir2, wavelength, w0_1, w0_2, P1, P2, 
               d_max, t_max, 
               transitions, 
               dt, t_sim, N_atoms, sigma_x, sigma_v):
    
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
                                    d_max, t_max, 
                                    transitions, 
                                    dt, t_sim)

    return times, rs, vs
