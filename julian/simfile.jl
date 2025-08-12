###
#
##= SIMULATION FILE =##
#
###


###
include("C:/Users/gabri/OneDrive/Documents/UltracoldAtomTransport/julian/simfuncs.jl")
include("C:/Users/gabri/OneDrive/Documents/UltracoldAtomTransport/julian/atomicspecies.jl")
include("C:/Users/gabri/OneDrive/Documents/UltracoldAtomTransport/julian/lenspresets.jl")
###

# wavelength
wavelength = 1064.0*nm

VF_mode = false

if VF_mode == false

    ### Loading position/velocity
    pos_load = pos_load_STATIC1
    vel_load = [0.0; 0.0; 0.0]

    ### Lens elements
    lens_elements = lens_elements_STATIC1

    ### Beam properties
    r0_1 = r0_1_STATIC1
    prop_dir1 = 1.0 
    wavelength1 = wavelength
    w0_1 = w0_1_STATIC1
    P1 = 100.0
    beam1 = GaussianBeam(r0_1, prop_dir1, wavelength1, w0_1, P1)

    r0_2 = r0_2_STATIC1
    prop_dir2 = -1.0 
    wavelength2 = wavelength
    w0_2 = w0_1_STATIC1
    P2 = 100.0
    beam2 = GaussianBeam(r0_2, prop_dir2, wavelength2, w0_2, P2)


elseif VF_mode == true

    ### Loading position/velocity
    pos_load = pos_load_VF
    vel_load = [0.0; 0.0; 0.0]

    ### Lens elements
    lens_elements = lens_elements_STATIC1

    ### Beam properties
    r0_1 = r0_1_VF 
    prop_dir1 = 1.0 
    wavelength1 = wavelength
    w0_1 = w0_1_VF
    P1 = 25.0
    beam1 = GaussianBeam(r0_1, prop_dir1, wavelength1, w0_1, P1)

    r0_2 = r0_2_VF
    prop_dir2 = -1.0 
    wavelength2 = wavelength
    w0_2 = w0_2_VF
    P2 = 25.0
    beam2 = GaussianBeam(r0_2, prop_dir2, wavelength2, w0_2, P2)
end 
###


### Atomic transitions
transitions = transitions_Yb171
###


### Ramp parameters
ramp_type = "Minimum Jerk"
d_max = 30.0*cm
t_max = 50.0*ms
Ramp = RampProfile(ramp_type, d_max, t_max)
###



### Simulation parameters
dt = 1e-5
t_sim = 1.0*ms
N_atoms = 8
sigma_x = [0.02*mm 0.02*mm 0.02*mm]
T = 10.0*uK
Boltz_fac = sqrt(kb*T/m)
sigma_v = Boltz_fac .* [1.0 1.0 1.0]
par_heating_ON = true
sim_params = SimParams(dt, t_sim, N_atoms, sigma_x, sigma_v, par_heating_ON)
###






############################################
#               RUN SIMULATION             # 
############################################
start = time()

times, rs, vs = run_MC_sim(pos_load, vel_load, 
                           lens_elements, 
                           beam1, beam2, 
                           Ramp, 
                           transitions,
                           sim_params)

stop = time()
sim_time = round(stop - start, sigdigits=3)
println("The simulation for $(N_atoms) atoms took $(sim_time) seconds.")
