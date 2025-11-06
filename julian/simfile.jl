###
#
##=  SIMULATION FILE  =##
#
###



###
using JLD2, FileIO
###



###
include("simfuncs.jl")
include("atomicspecies.jl")
include("lenspresets.jl")
###



# Wavelength
WL = 1064.0*nm

### Beam properties
beam1 = GaussianBeam(r01_A, +1.0, WL, w01_A, P1_A)
beam2 = GaussianBeam(r02_A, -1.0, WL, w02_A, P2_A)

### Ramp parameters
RT = "Minimum Jerk"
max_d = 30.0*cm
max_t = 50.0*ms
Ramp = RampProfile(RT, max_d, max_t)
###



### Simulation parameters
dt = 1.0*us
t_sim = 50.0*ms
N_atoms = 8
sigma_x = [0.02*mm 0.02*mm 0.02*mm]
T = 0.0*uK
boltz = sqrt(kb*T/m)
sigma_v = boltz .* [0.0 0.0 0.0]
par_heating_ON = false
sim_params = SimParams(dt, 
                       t_sim, 
                       N_atoms, 
                       sigma_x, 
                       sigma_v, 
                       par_heating_ON)
###






#==========================================#
#               RUN SIMULATION             # 
#==========================================#
# 
###
start = time()

times, rs, vs = run_MC_sim(pos_loadA, vel_loadA, 
                           lens_elements_A, 
                           beam1, beam2, 
                           Ramp, 
                           transitions_Yb171,
                           sim_params)

params = Dict(
    "position load" => pos_loadA, 
    "velocity load" => vel_loadA, 
    "lens elements" => lens_elements_A,
    "beam 1" => beam1,
    "beam 2" => beam2,
    "Ramping profile" => Ramp,
    "transitions" => transitions_Yb171, 
    "Simulation params" => sim_params
)

results = Dict(
    "times" => times, 
    "positions" => rs,
    "velocities" => vs
)

@save "sim_params.jl2d" params
@save "sim_results.jl2d" results


stop = time()
sim_time = round(stop-start, sigdigits=3)
println("The simulation for $(N_atoms) atoms took $(sim_time) seconds.")
###
