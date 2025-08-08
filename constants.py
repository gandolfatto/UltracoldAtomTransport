import numpy as np


cm = 1e-2
amu = 1.67e-27

nm = 1e-9
um = 1e-6
mm = 1e-3
ms = 1e-3

hbar = 1.054e-34
kb = 1.38e-23
c = 2.997e8
e0 = 8.85e-12

m = 171*amu

#axial direction for long-range transport is in z-direction --> gravity acts on y-comp.
g0 = -9.81
g_transport = np.array([0, g0, 0])
