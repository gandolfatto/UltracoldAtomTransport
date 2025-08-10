###
#
##= GAUSSIAN BEAM PROPERTIES =##
#
###


### Create a Gaussian beam structure
struct GaussianBeam
    r0::Float64                 # initial position of the Gaussian beam
    prop_dir::Float64           # propagation direction
    wavelength::Float64         # wavelength
    w0::Float64                 # beam width at r0
    P::Float64                  # beam power
end



function prop_matrix(z::Float64, 
                     lens_eles::Array{Float64})
    
    #=
    Compute the ray propagation (ABCD) matrix at position z along the optical
    axis for a given set of lens elements (specified by the lens_eles array). 
    =#
    
    lens_eles_z = lens_eles[lens_eles[:, 1] .<= z, :]   
    num_eles = size(lens_eles_z, 1)

    if num_eles == 0
        return [1.0  z; 
                0.0  1.0]

    elseif z < lens_eles_z[1, 1]  
        return [1.0  z; 
                0.0  1.0]

    else
        M = [1.0  0.0; 
             0.0  1.0]
        zi = 0.0
        for i in 1:num_eles                 
            z_prop = lens_eles_z[i, 1]      
            z_diff = z_prop - zi            
            zi = z_prop                   

            mat = [1.0  z_prop; 
                   -1.0/lens_eles_z[i, 2]  1.0-(z_diff / lens_eles_z[i, 2])]

            M = mat * M
        end
        return [1.0  z - lens_eles_z[end, 1]; 0.0  1.0] * M
    end
end



function q(z::Float64, 
           lens_eles::Array{Float64}, 
           beam::GaussianBeam)

    Zr = pi*beam.w0^2 / beam.wavelength
    q0 = Zr*im
    
    abcd = prop_matrix(beam.r0 + beam.prop_dir*z, lens_eles)
    A = abcd[1, 1]
    B = abcd[1, 2]
    C = abcd[2, 1]
    D = abcd[2, 2]

    q_val = (A*q0 + B)/(C*q0 + D) 
    return q_val
end 



function beam_width(z::Float64, 
                    lens_eles::Array{Float64}, 
                    beam::GaussianBeam)
    
    q_val = q(z, lens_eles, beam) 

    w = sqrt(beam.wavelength / (pi * imag(-1/q_val))) 
    return w
end



function beam_width_derivative(z::Float64, 
                               lens_eles::Array{Float64}, 
                               beam::GaussianBeam) 
    
    q_val = q(z, lens_eles, beam) 
    Zr = imag(q_val)
    waist = sqrt(Zr * beam.wavelength / pi)

    w_prime = waist*(z/Zr)*(1/sqrt(1 + (z/Zr)^2))
    return w_prime 
end 



function I_field(x::Float64, y::Float64, z::Float64, 
                 lens_eles::Array{Float64}, 
                 beam::GaussianBeam)

    w = beam_width(z, lens_eles, beam)
    
    I = ((2*beam.P)/(pi*w^2)) * exp(-2*((x^2 + y^2)/w^2))
    return I
end 
