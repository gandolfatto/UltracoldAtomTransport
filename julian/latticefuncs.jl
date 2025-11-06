###
#
##=  LATTICE FUNCTIONS  =##
#
###



### 
include("gaussianbeam.jl")
include("rampprofiles.jl")
include("consts.jl")
###



###
function atomic_pol(wavelength::Float64, 
                    transitions::Array{Float64})

    alpha = 0.0
    for transition in eachrow(transitions)

        transition_wavelength, Gamma = transition[1], transition[2]

        omega = 2*pi*c/wavelength
        omega_i = 2*pi*c/transition_wavelength
        
        delta = omega - omega_i                                                                     # relative detuning
        dip_element = (3*pi*e0*hbar*c^3 / omega_i^3) * Gamma                                        # dipole matrix element
        alpha += -(dip_element / hbar) * ((delta - im*Gamma/2) / ((delta^2 + im*(Gamma/2)^2)))      # scalar atomic polarizability
    end 
    return alpha
end 
###



###
function I_lattice(x::Float64, y::Float64, z::Float64, t::Float64, 
                   lens_eles::Array{Float64}, 
                   beam1::GaussianBeam, beam2::GaussianBeam,
                   rp::RampProfile)
    
    k = 2*pi / beam1.wavelength
    z_lattice = z_latt(t, rp)                                             

    I1 = I_field(x, y, z, lens_eles, beam1)
    I2 = I_field(x, y, z, lens_eles, beam2)

    I = I1 + I2 + 2*sqrt(I1*I2)*cos(2*k*(z - z_lattice))
    return I
end 
###



###
function U_lattice(x::Float64, y::Float64, z::Float64, t::Float64, 
                   lens_eles::Array{Float64}, 
                   beam1::GaussianBeam, beam2::GaussianBeam,
                   rp::RampProfile, 
                   transitions::Array{Float64})
    
    I = I_lattice(x, y, z, t, 
                  lens_eles, 
                  beam1, beam2,
                  rp)
    
    alpha = atomic_pol(beam1.wavelength, transitions)

    U = (-real(alpha) / (2*e0*c)) * I
    return U
end 
###



###
function F_lattice(x::Float64, y::Float64, z::Float64, t::Float64, 
                   lens_eles::Array{Float64}, 
                   beam1::GaussianBeam, beam2::GaussianBeam,
                   rp::RampProfile,
                   transitions::Array{Float64})
    
    k = 2*pi / beam1.wavelength
    r2_perp = x^2 + y^2
    z_lattice = z_latt(t, rp)

    # beam 1
    w1 = beam_width(z, lens_eles, beam1)
    w_prime1 = beam_width_derivative(z, lens_eles, beam1)
    I1 = I_field(x, y, z, lens_eles, beam1)

    # beam 2
    w2 = beam_width(z, lens_eles, beam2)
    w_prime2 = beam_width_derivative(z, lens_eles, beam2)
    I2 = I_field(x, y, z, lens_eles, beam2)


    _dIdx = ((4*x)/(w1^2))*I1 + ((4*x)/(w2^2))*I2 + 4*x*((1/w1^2) + (1/w2^2)) * sqrt(I1*I2)
    _dIdy = ((4*y)/(w1^2))*I1 + ((4*y)/(w2^2))*I2 + 4*y*((1/w1^2) + (1/w2^2)) * sqrt(I1*I2)
    _dIdz = (I1*(2*w_prime1/w1)*(1-(r2_perp/w1^2)) + I2*(2*w_prime2/w2)*(1-(r2_perp/w2^2)) + sqrt(I1*I2)*((2*w_prime1/w1)*(1-(r2_perp/w1^2)) + (2*w_prime2/w2)*(1-(r2_perp/w2^2))))*cos(2*k*(z - z_lattice))  + 4*k*sin(2*k*(z - z_lattice))*sqrt(I1*I2)                      


    alpha = atomic_pol(beam1.wavelength, transitions)
    fac = -real(alpha) / (2*e0*c)
    Fx = fac * _dIdx
    Fy = fac * _dIdy
    Fz = fac * _dIdz

    return Fx, Fy, Fz
end 




