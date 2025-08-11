###
#
##= SIMULATED LOSS DUE QUANTUM MECHANICAL EFFECTS =##
#
###

###
include("C:/Users/gabri/OneDrive/Documents/UltracoldAtomTransport/julian/latticefuncs.jl")
###


### Parametric heating 

function axial_freq(x::Float64, y::Float64, z::Float64, t::Float64, 
                    lens_eles::Array{Float64}, 
                    beam1::GaussianBeam, beam2::GaussianBeam,
                    d_max::Float64, t_max::Float64, 
                    transitions::Array{Float64})
    
    U = U_lattice(x, y, z, t,
                  lens_eles, 
                  beam1, beam2,
                  d_max, t_max, 
                  transitions)
    
    omega_z = (1/beam1.wavelength) * sqrt(abs(2*U/m))
    return omega_z
end



function RIN(freq)
    return -130
end



function par_heating_rate(freq, beam1::GaussianBeam, beam2::GaussianBeam)
    
    RelIntensityNoise = RIN(2*freq)
    P = beam1.P + beam2.P
    S = 10^(RelIntensityNoise / 10) * P^2
    Gamma_par = pi^2 * freq^2 * S
    return Gamma_par
end
