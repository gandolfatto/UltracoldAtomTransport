###
#
##= RAMP PROFILES =##
#
###



###   Create a ramp profile structure   ###
#
###
struct RampProfile
    ramp_type::String   # ramp type
    d_max::Float64      # max distance
    t_max::Float64      # max time 
end
###



###
function z_latt(t::Float64, 
                rp::RampProfile)

    t_reduced = t/rp.t_max
    
    if rp.ramp_type == "None"
        z_lattice = 0

    elseif rp.ramp_type == "Linear"
        if t < rp.t_max/2
            z_lattice = 2*rp.d_max * t_reduced^2
        elseif t >= rp.t_max/2
            z_lattice = (-rp.d_max) + (4*rp.d_max * t_reduced) - (2*rp.d_max * t_reduced^2)
        end

    elseif rp.ramp_type == "Minimum Jerk"
        z_lattice = rp.d_max*(10*t_reduced^3 - 15*t_reduced^4 + 6*t_reduced^5)
    end 

    return z_lattice
end 
###



###
function v_latt(t::Float64, 
                rp::RampProfile)
    
    t_reduced = t/rp.t_max

    if rp.ramp_type == "None"
        v_lattice = 0

    elseif rp.ramp_type == "Linear"
        if t < rp.t_max/2
            v_lattice = (4*rp.d_max/rp.t_max^2)*t
        elseif t >= rp.t_max/2
            v_lattice = (4*rp.d_max/rp.t_max^2)*t - (4*rp.d_max/rp.t_max^2)*t
        end
    
    elseif rp.ramp_type == "Minimum Jerk"
        v_lattice = 30*(rp.d_max/rp.t_max)*(t_reduced^2 - 2*t_reduced^3 + t_reduced^4)
    end

    return v_lattice
end
###



###
function a_latt(t::Float64,
                rp::RampProfile)
    
    t_reduced = t/rp.t_max

    if rp.ramp_type == "None"
        a_lattice = 0

    elseif rp.ramp_type == "Linear"
        if t < rp.t_max/2
            a_lattice = 4*rp.d_max/rp.t_max^2
        elseif t >= rp.t_max/2
            a_lattice = -4*rp.d_max/rp.t_max^2
        end
    
    elseif rp.ramp_type == "Minimum Jerk" 
        a_lattice =  60*(rp.d_max/rp.t_max^2)*(t_reduced - 3*t_reduced^2 + 2*t_reduced^3)
    end 

    return a_lattice
end
