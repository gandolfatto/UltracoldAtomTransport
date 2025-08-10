using Base.Threads
# using NPZ


include("C:/Users/gabri/OneDrive/Documents/UltracoldAtomTransport/julian/latticefuncs.jl")


function MC_step(pos::Array{Float64},
                 vel::Array{Float64},  
                 t::Float64, 
                 lens_eles::Matrix{Float64}, 
                 r0_1::Float64, r0_2::Float64, 
                 prop_dir1::Float64, prop_dir2::Float64, 
                 wavelength::Float64,
                 w0_1:: Float64, w0_2::Float64, 
                 P1::Float64, P2::Float64, 
                 d_max::Float64, t_max::Float64, 
                 transitions::Matrix{Float64},
                 dt::Float64)
    
    x, y, z = pos[1], pos[2], pos[3]
    vx, vy, vz = vel[1], vel[2], vel[3]

    x1, y1, z1 = x, y, z
    vx1, vy1, vz1 = vx, vy, vz
    t1 = t

    Fx, Fy, Fz = F_lattice(x1, y1, z1, t1, 
                           lens_eles, 
                           r0_1, r0_2, 
                           prop_dir1, prop_dir2, 
                           wavelength, 
                           w0_1, w0_2, 
                           P1, P2, 
                           d_max, t_max, 
                           transitions)

    dvx1, dvy1, dvz1 = (Fx/m)*dt, (g0 + Fy/m)*dt, (Fz/m)*dt
    dx1, dy1, dz1 = (vx1 + dvx1/2)*dt, (vy1 + dvy1/2)*dt, (vz1 + dvz1/2)*dt

    Fx, Fy, Fz = F_lattice(x1 + dx1, y1 + dy1, z1 + dz1, t1 + dt, 
                           lens_eles, 
                           r0_1, r0_2, 
                           prop_dir1, prop_dir2, 
                           wavelength, 
                           w0_1, w0_2, 
                           P1, P2, 
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



function one_atom_sim(r0::Array{Float64}, 
                      v0::Array{Float64}, 
                      lens_eles::Matrix{Float64}, 
                      r0_1::Float64, r0_2::Float64, 
                      prop_dir1::Float64, prop_dir2::Float64, 
                      wavelength::Float64,
                      w0_1::Float64, w0_2::Float64, 
                      P1::Float64, P2::Float64, 
                      d_max::Float64, t_max::Float64, 
                      transitions::Matrix{Float64},
                      dt::Float64,
                      t_sim::Float64)
        
    Nt = round(Int, t_sim/dt)

    rs = zeros(Nt+1, 3)
    vs = zeros(Nt+1, 3)

    rs[1, :] = r0
    vs[1, :] = v0

    r = copy(r0)
    v = copy(v0)

    for i in 1:Nt
        t = (i-1)*dt 
        r, v = MC_step(r, v,
                       t, 
                       lens_eles, 
                       r0_1, r0_2, 
                       prop_dir1, prop_dir2, 
                       wavelength, 
                       w0_1, w0_2, 
                       P1, P2, 
                       d_max, t_max, 
                       transitions, 
                       dt)
        
        rs[i, :] = r
        vs[i, :] = v
    end
    return rs, vs
end 



function run_MC_sim(pos_load::Array{Float64},
                    vel_load::Array{Float64}, 
                    lens_eles::Matrix{Float64}, 
                    r0_1::Float64, r0_2::Float64, 
                    prop_dir1::Float64, prop_dir2::Float64, 
                    wavelength::Float64,
                    w0_1:: Float64, w0_2::Float64, 
                    P1::Float64, P2::Float64, 
                    d_max::Float64, t_max::Float64, 
                    transitions::Matrix{Float64},
                    dt::Float64,
                    t_sim::Float64,
                    N_atoms::Int64, 
                    sigma_x::Array{Float64}, 
                    sigma_v::Array{Float64})
    
    Nt = round(Int, t_sim/dt)
    times = range(0.0, t_sim, step=dt)

    rs = zeros(N_atoms, Nt+1, 3)
    vs = zeros(N_atoms, Nt+1, 3)

    r0_arr = [pos_load[j] + sigma_x[j] * randn() for i in 1:N_atoms, j in 1:3]
    v0_arr = [vel_load[j] + sigma_v[j] * randn() for i in 1:N_atoms, j in 1:3]

    @threads for i in 1:N_atoms

        rs[i, :, :], vs[i, :, :] = one_atom_sim(r0_arr[i, :], 
                                                v0_arr[i, :], 
                                                lens_elements, 
                                                r0_1, r0_2, 
                                                prop_dir1, prop_dir2, 
                                                wavelength, 
                                                w0_1, w0_2, 
                                                P1, P2, 
                                                d_max, t_max, 
                                                transitions, 
                                                dt, 
                                                t_sim)
    end
    return times, rs, vs
end 
