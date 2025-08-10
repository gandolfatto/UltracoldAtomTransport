
include("C:/Users/gabri/OneDrive/Documents/UltracoldAtomTransport/julian/gaussianbeam.jl")
include("C:/Users/gabri/OneDrive/Documents/UltracoldAtomTransport/julian/rampprofiles.jl")
include("C:/Users/gabri/OneDrive/Documents/UltracoldAtomTransport/julian/consts.jl")



function trap_info(wavelength::Float64, 
                   transition_wavelength::Float64, transition_linewidth::Float64)

    omega = 2*pi*c/wavelength
    omega_i = 2*pi*c/transition_wavelength
    
    delta = omega - omega_i
    dip_element = (3*pi*e0*hbar*c^3 / omega_i^3) * transition_linewidth
    alpha = -(dip_element / hbar) * ((delta - im*transition_linewidth /2) / ((delta^2 + im*(transition_linewidth/2)^2)))

    return delta, dip_element, alpha
end 



function I_lattice(x::Float64, y::Float64, z::Float64, t::Float64, 
                   lens_eles::Matrix{Float64}, 
                   r0_1::Float64, r0_2::Float64, 
                   prop_dir1::Float64, prop_dir2::Float64, 
                   wavelength::Float64,
                   w0_1:: Float64, w0_2::Float64, 
                   P1::Float64, P2::Float64, 
                   d_max::Float64, t_max::Float64)
        
    z_lattice = z_latt(t, d_max, t_max)                                             

    I1 = I_field(x, y, z, 
                 lens_eles, 
                 r0_1, prop_dir1, wavelength, w0_1, P1)
    
    I2 = I_field(x, y, z, 
                 lens_eles, 
                 r0_2, prop_dir2, wavelength, w0_2, P2)

    k = 2*pi/wavelength
    I = I1 + I2 + 2*sqrt(I1*I2)*cos(2*k*(z - z_lattice))

    return I
end 



function U_lattice(x::Float64, y::Float64, z::Float64, t::Float64, 
                   lens_eles::Matrix{Float64}, 
                   r0_1::Float64, r0_2::Float64, 
                   prop_dir1::Float64, prop_dir2::Float64, 
                   wavelength::Float64,
                   w0_1:: Float64, w0_2::Float64, 
                   P1::Float64, P2::Float64, 
                   d_max::Float64, t_max::Float64, 
                   transitions::Matrix{Float64})
    
    I = I_lattice(x, y, z, t, 
                  lens_eles, 
                  r0_1, r0_2, 
                  prop_dir1, prop_dir2, 
                  wavelength, 
                  w0_1, w0_2, 
                  P1, P2, 
                  d_max, t_max)
    
    U = 0
    for transition in eachrow(transitions)
        alpha = trap_info(wavelength, transition[1], transition[2])[3]
        U += (-real(alpha) / (2*e0*c)) * I
    return U
    end
end 




function F_lattice(x::Float64, y::Float64, z::Float64, t::Float64, 
                   lens_eles::Matrix{Float64}, 
                   r0_1::Float64, r0_2::Float64, 
                   prop_dir1::Float64, prop_dir2::Float64, 
                   wavelength::Float64,
                   w0_1:: Float64, w0_2::Float64, 
                   P1::Float64, P2::Float64, 
                   d_max::Float64, t_max::Float64, 
                   transitions::Matrix{Float64})
    
        
    z_lattice = z_latt(t, d_max, t_max)

    w1 = beam_width(z, 
                    lens_eles, 
                    r0_1, prop_dir1, wavelength, w0_1)
    w2 = beam_width(z, 
                    lens_eles, 
                    r0_2, prop_dir2, wavelength, w0_2)
    
    w_prime1 = beam_width_derivative(z, 
                                     lens_eles, 
                                     r0_1, prop_dir1, wavelength, w0_1)
    w_prime2 = beam_width_derivative(z, 
                                     lens_eles, 
                                     r0_2, prop_dir2, wavelength, w0_2)
    
    I1 = I_field(x, y, z, 
                 lens_eles, 
                 r0_1, prop_dir1, wavelength, w0_1, P1)
    I2 = I_field(x, y, z, 
                 lens_eles, 
                 r0_2, prop_dir2, wavelength, w0_2, P2)

    k = 2*pi/wavelength
    _dIdx = ((4*x)/(w1^2))*I1 + ((4*x)/(w2^2))*I2 + 4*x*((1/w1^2) + (1/w2^2)) * sqrt(I1*I2)
    _dIdy = ((4*y)/(w1^2))*I1 + ((4*y)/(w2^2))*I2 + 4*y*((1/w1^2) + (1/w2^2)) * sqrt(I1*I2)
    _dIdz = (I1*(2*w_prime1/w1)*(1-((x^2 + y^2)/w1^2)) + I2*(2*w_prime2/w2)*(1-((x^2 + y^2)/w2^2)) + sqrt(I1*I2)*((2*w_prime1/w1)*(1-((x^2 + y^2)/w1^2)) + (2*w_prime2/w2)*(1-((x^2 + y^2)/w2^2))))*cos(2*k*(z - z_lattice))  + 4*k*sin(2*k*(z - z_lattice))*sqrt(I1*I2)                      


    Fx = 0
    Fy = 0
    Fz = 0

    for transition in eachrow(transitions)

        alpha = trap_info(wavelength, transition[1], transition[2])[3]
        Fx += (-real(alpha) / (2*e0*c)) * _dIdx
        Fy += (-real(alpha) / (2*e0*c)) * _dIdy
        Fz += (-real(alpha) / (2*e0*c)) * _dIdz
    end

    return Fx, Fy, Fz
end 
