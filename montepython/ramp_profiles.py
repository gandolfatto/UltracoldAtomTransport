import numpy as np
from numba import jit

from constants import *



#=======================================================#
#                       RAMP PROFILES                   #   
#=======================================================# 
#
###


@jit(nopython=True, fastmath=True, cache=True)
def z_latt(ramp_type,
           t, 
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
    
    if ramp_type == 0: 
        z_lattice = 0
    
    elif ramp_type == 1: 
        t_reduced = t/t_max
        z_lattice = d_max*(10*(t_reduced)**3 - 15*(t_reduced)**4 + 6*(t_reduced)**5)

    return z_lattice



@jit(nopython=True, fastmath=True, cache=True)
def v_latt(ramp_type,
           t,
           d_max, t_max): 
    
    if ramp_type == 0: 
        v_lattice = 0
        
    elif ramp_type == 1: 
        t_reduced = t/t_max
        v_lattice = 30*(d_max/t_max)*(t_reduced**2 - 2*t_reduced**3 + t_reduced**4)
        
    return v_lattice

###
