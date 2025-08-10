
include("C:/Users/gabri/OneDrive/Documents/UltracoldAtomTransport/julian/simfuncs.jl")


x0, y0, z0 = 0.0, 0.0, 11.6*cm
pos_load = [x0; y0; z0]

vx0, vy0, vz0 = 0.0, 0.0, 0.0
vel_load = [vx0; vy0; vz0]

lens_elements = [1.69*cm 18.0*cm; 
                 51.6*cm 18.0*cm]

r0_1, r0_2 = 0.0, 53.2*cm
prop_dir1, prop_dir2 = 1.0, -1.0
wavelength = 1064.0*nm
w0_1, w0_2 = 362.0*um, 362.0*um
P1, P2 = 100.0, 100.0

# t = 0.005
t_max = 50.0*ms
d_max = 30.0*cm

dt = 1e-6
t_sim = 50.0*ms

transitions = [399*nm 2*pi*29.1*MHz;  
               556*nm 2*pi*181.1*KHz]


N_atoms = 8
sigma_x = [0.02*mm 0.02*mm 0.02*mm]
sigma_v = [0.02*mm 0.02*mm 0.02*mm]



start = time()

times, rs, vs = run_MC_sim(pos_load, vel_load, 
                           lens_elements, 
                           r0_1, r0_2, 
                           prop_dir1, prop_dir2,
                           wavelength, 
                           w0_1, w0_2, 
                           P1, P2,
                           d_max, t_max,
                           transitions,
                           dt,
                           t_sim,
                           N_atoms,
                           sigma_x, sigma_v)

stop = time()
sim_time = stop - start
println("The simulation for $(N_atoms) atoms took $(sim_time) seconds.")
       