import numpy as np
from numba import jit

from constants import *
from ramp_profiles import *
from gaussian_beam import *



#=======================================================#
#                     LATTICE FUNCTIONS                 #   
#=======================================================# 
#
###



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
              ramp_type, d_max, t_max):
    
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
        
    z_lattice = z_latt(ramp_type, t, d_max, t_max)                                             

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
              ramp_type, d_max, t_max,
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
                  ramp_type, d_max, t_max)
    
    U = 0
    for transition in transitions:
        alpha = trap_info(wavelength, transition[0], transition[1])[2]
        U += (-alpha.real / (2*e0*c)) * I
    return U




@jit(nopython=True, fastmath=True, cache=True)
def F_lattice(x, y, z, t, 
              lens_elements, 
              r0_1, r0_2, prop_dir1, prop_dir2, wavelength, w0_1, w0_2, P1, P2, 
              ramp_type, d_max, t_max,
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

        
    z_lattice = z_latt(ramp_type, t, d_max, t_max)


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
