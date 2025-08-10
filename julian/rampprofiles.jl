
function z_latt(t::Float64, 
                d_max::Float64, t_max::Float64)

    t_reduced = t/t_max
    z_lattice = d_max*(10*t_reduced^3 - 15*t_reduced^4 + 6*t_reduced^5)

    return z_lattice
end 


function v_latt(t::Float64, 
                d_max::Float64, t_max::Float64)
    
    t_reduced = t/t_max
    v_lattice = 30*(d_max/t_max)*(t_reduced^2 - 2*t_reduced^3 + t_reduced^4)
    
    return v_lattice
end
