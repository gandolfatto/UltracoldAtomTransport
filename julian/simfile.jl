###
#
##= SIMULATION FILE =##
#
###


###
include("C:/Users/gabri/OneDrive/Documents/UltracoldAtomTransport/julian/simfuncs.jl")
###


### Loading position/velocity
x0, y0, z0 = 0.0, 0.0, 11.6*cm
pos_load = [x0; y0; z0]

vx0, vy0, vz0 = 0.0, 0.0, 0.0
vel_load = [vx0; vy0; vz0]


### Lens elements
lens_elements = [1.69*cm 18.0*cm; 
                 51.6*cm 18.0*cm]


### Beam properties
wavelength = 1064.0*nm              

r0_1 = 0.0
prop_dir1 = 1.0 
wavelength1 = wavelength
w0_1 = 362.0*nm
P1 = 100.0
beam1 = GaussianBeam(r0_1, prop_dir1, wavelength1, w0_1, P1)

r0_2 = 53.2*cm
prop_dir2 = -1.0 
wavelength2 = wavelength
w0_2 = 362.0*nm
P2 = 100.0
beam2 = GaussianBeam(r0_2, prop_dir2, wavelength2, w0_2, P2)


### Ramp parameters
t_max = 50.0*ms
d_max = 30.0*cm


### GS --> Excited state transitions for Yb-171
transitions = [399*nm 2*pi*29.1*MHz;                # (6s^2) 1S0 --> (6s6p) 1P1
               556*nm 2*pi*181.1*KHz]               # (6s^2) 1S0 --> (6s6p) 3P1


### Simulation parameters
dt = 1e-7
t_sim = 50.0*ms
N_atoms = 8
sigma_x = [0.02*mm 0.02*mm 0.02*mm]
T = 1e-6
Boltz_fac = sqrt(kb*T/m)
sigma_v = Boltz_fac .* [1.0 1.0 1.0]
sim_params = SimParams(dt, t_sim, N_atoms, sigma_x, sigma_v)




###################### 
#   RUN SIMULATION   # 
###################### 
start = time()

times, rs, vs = run_MC_sim(pos_load, vel_load, 
                           lens_elements, 
                           beam1, beam2, 
                           d_max, t_max,
                           transitions,
                           sim_params)

stop = time()
sim_time = round(stop - start, sigdigits=3)
println("The simulation for $(N_atoms) atoms took $(sim_time) seconds.")
