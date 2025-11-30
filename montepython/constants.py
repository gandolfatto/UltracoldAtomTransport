import numpy as np



# length
nm = 1e-9
um = 1e-6
mm = 1e-3
cm = 1e-2



# time/frequency
ms = 1e-3
us = 1e-6
ns = 1e-9

mHz = 1e-3
kHz = 1e3
MHz = 1e6
GHz = 1e9



# temperature
uk = 1e-6



# fundumental constants
hbar = 1.054e-34
kb   = 1.38e-23
c    = 2.997e8
e0   = 8.85e-12
amu  = 1.67e-27



#axial direction for long-range transport is in z-direction --> gravity acts on y-comp.
g0 = -9.81
g_transport = np.array([0, g0, 0])
#
