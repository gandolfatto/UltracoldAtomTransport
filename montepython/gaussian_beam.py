import numpy as np
from numba import jit

from constants import *



#=======================================================#
#                 GAUSSIAN BEAM PROPERTIES              #   
#=======================================================# 
#
###



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
                                    [pos of lens 1, focal length of lens 1],             
                                    [pos of lens 2, focal length of lens 2],
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
