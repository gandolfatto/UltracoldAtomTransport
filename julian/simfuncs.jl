###
#
##= SIMULATION FUNCTIONS =##
#
###

###
using Base.Threads
# using NPZ

include("C:/Users/gabri/OneDrive/Documents/UltracoldAtomTransport/julian/latticefuncs.jl")
###


### Create a simulation parameter structure
struct SimParams
    dt::Float64                 # time step
    t_sim::Float64              # simulation time
    N_atoms::Int64              # number of atoms to simulate
    sigma_x::Array{Float64}     # initial position distribution width
    sigma_v::Array{Float64}     # initial velocity distribution width
end



function MC_step(pos::Array{Float64}, vel::Array{Float64}, t::Float64, 
                 lens_eles::Array{Float64}, 
                 beam1::GaussianBeam, beam2::GaussianBeam,
                 d_max::Float64, t_max::Float64, 
                 transitions::Array{Float64},
                 dt::Float64)
    
    x, y, z = pos[1], pos[2], pos[3]
    vx, vy, vz = vel[1], vel[2], vel[3]

    x1, y1, z1 = x, y, z
    vx1, vy1, vz1 = vx, vy, vz
    t1 = t

    Fx, Fy, Fz = F_lattice(x1, y1, z1, t1, 
                           lens_eles, 
                           beam1, beam2,
                           d_max, t_max, 
                           transitions)

    dvx1, dvy1, dvz1 = (Fx/m)*dt, (g0 + Fy/m)*dt, (Fz/m)*dt
    dx1, dy1, dz1 = (vx1 + dvx1/2)*dt, (vy1 + dvy1/2)*dt, (vz1 + dvz1/2)*dt

    Fx, Fy, Fz = F_lattice(x1 + dx1, y1 + dy1, z1 + dz1, t1 + dt, 
                           lens_eles, 
                           beam1, beam2,
                           d_max, t_max, 
                           transitions)

    dvx2, dvy2, dvz2 = (Fx/m)*dt, (g0 + Fy/m)*dt, (Fz/m)*dt

    x1 += dx1
    y1 += dy1
    z1 += dz1

    vx1 += (dvx1 + dvx2)/2
    vy1 += (dvy1 + dvy2)/2
    vz1 += (dvz1 + dvz2)/2
    
    t1 += dt
            
    r1 = [x1; y1; z1]
    v1 = [vx1; vy1; vz1]

    return r1, v1
end 



function one_atom_sim(r0::Array{Float64}, v0::Array{Float64}, 
                      lens_eles::Array{Float64}, 
                      beam1::GaussianBeam, beam2::GaussianBeam, 
                      d_max::Float64, t_max::Float64, 
                      transitions::Array{Float64},
                      dt::Float64, t_sim::Float64)
        
    Nt = round(Int, t_sim/dt)

    rs = zeros(Nt+1, 3)
    vs = zeros(Nt+1, 3)

    rs[1, :] = r0
    vs[1, :] = v0

    r = copy(r0)
    v = copy(v0)

    for i in 1:Nt
        t = (i-1)*dt 
        r, v = MC_step(r, v, t, 
                       lens_eles, 
                       beam1, beam2,
                       d_max, t_max, 
                       transitions, 
                       dt)
        
        rs[i, :] = r
        vs[i, :] = v
    end
    return rs, vs
end 



function run_MC_sim(pos_load::Array{Float64}, vel_load::Array{Float64}, 
                    lens_eles::Array{Float64}, 
                    beam1::GaussianBeam, beam2::GaussianBeam, 
                    d_max::Float64, t_max::Float64, 
                    transitions::Array{Float64},
                    sim_params::SimParams)
    
    Nt = round(Int, sim_params.t_sim/sim_params.dt)
    times = range(0.0, sim_params.t_sim, step=sim_params.dt)

    rs = zeros(sim_params.N_atoms, Nt+1, 3)
    vs = zeros(sim_params.N_atoms, Nt+1, 3)


    r0_arr = [pos_load[j] + sim_params.sigma_x[j] * randn() for i in 1:sim_params.N_atoms, j in 1:3]
    v0_arr = [vel_load[j] + sim_params.sigma_v[j] * randn() for i in 1:sim_params.N_atoms, j in 1:3]

    @threads for i in 1:sim_params.N_atoms

        rs[i, :, :], vs[i, :, :] = one_atom_sim(r0_arr[i, :], v0_arr[i, :], 
                                                lens_eles, 
                                                beam1, beam2,
                                                d_max, t_max, 
                                                transitions, 
                                                sim_params.dt, 
                                                sim_params.t_sim)
    end
    return times, rs, vs
end 
