###
#
##=  ATOMIC SPECIES  =##
#
###


### 
include("consts.jl")
###



#==========================================#
#                   YB-171                 # 
#==========================================#
# 
###
#
####   mass   ####
m = 171*amu

###   GS --> ES transitions for Yb-171   ###
transitions_Yb171 = [399*nm 2*pi*29.1*MHz;                # (6s^2) 1S0 --> (6s6p) 1P1
                     556*nm 2*pi*181.1*KHz]               # (6s^2) 1S0 --> (6s6p) 3P1

