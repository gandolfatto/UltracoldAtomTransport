###
#
##= ATOMIC SPECIES =##
#
###


### 
include("C:/Users/gabri/OneDrive/Documents/UltracoldAtomTransport/julian/consts.jl")
###


amu = 1.67e-27


### Yb-171
m = 171*amu

### GS --> Excited state transitions for Yb-171
transitions_Yb171 = [399*nm 2*pi*29.1*MHz;                # (6s^2) 1S0 --> (6s6p) 1P1
                     556*nm 2*pi*181.1*KHz]               # (6s^2) 1S0 --> (6s6p) 3P1

