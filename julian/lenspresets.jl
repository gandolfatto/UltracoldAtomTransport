###
#
##= LENS CONFIGURATION PRESETS =##
#
###


### Static Lens Configurations
#
#=
Static Config 1 (symmetric static lens setup)
--------
A w0_1_STATIC1 = 362.0*um waist Gaussian beam is turned on at r0_1_STATIC1 = 0.0*cm. It encounters
a thin lens w/ focal length 18.0*cm at position z = 1.6*cm. The new beam waist/beam waist location 
is (155.0*um, 5.0*cm + 1.6*cm). For d_max = 30.0*cm transport, this configuration should maximize
the minimum trap depth along the optical axis. 
=#

pos_load_STATIC1 = [0.0 0.0 11.6*cm]

w0_1_STATIC1 = 362.0*um
w0_2_STATIC1 = 362.0*um
r0_1_STATIC1 = 0.0*cm
r0_2_STATIC1 = 53.2*cm

lens_elements_STATIC1 = [1.60*cm 18.0*cm; 
                         51.6*cm 18.0*cm]







                         
### Varifocal (VF) Lens Configurations
#
#=
VF Config 1 (hybrid VF lens + static lens setup)
=#

p1 = 5.0*cm     # position of first static lens 
f1 = 5.0*cm     # focal length of first static lens 
l  = 14.0*cm    # distance between p1 and VF lens 
f  = 30.0*cm    # focal length of central static lens (should be equal to d_max)

pos_load_VF = [0.0 0.0 79.0*cm]

w0_1_VF = 1.0*mm
w0_2_VF = 350.0*um
r0_1_VF = 0.0*cm
r0_2_VF = 114.0*cm  # = p1 + l + f + f + f + p1




function waist_VF(beam::GaussianBeam)
    
    M = (f*f1*beam.wavelength)/sqrt((f1 - l)^2 * (pi * beam.w0^2)^2 + (f1*beam.wavelength*l)^2)
    w = M*beam.w0
    return w
end

function focus_VF(t::Float64, 
                  d_max::Float64, t_max::Float64, 
                  beam::GaussianBeam)

    x0 = z_latt(t, d_max, t_max) + f
    vf = 1/(1/f - x0/f^2 - ( ((f1 - l)*(pi*beam.w0^2)^2 - l*(f1 * beam.wavelength)^2) / ( (f1 - l)^2 * (pi*beam.w0^2)^2 + (f1 * beam.wavelength * l)^2 ) ))
    return vf
end 

function lens_elements_func(t::Float64, 
                            d_max::Float64, t_max::Float64, 
                            beam::GaussianBeam)
    
    lens_elements_VF1 = [p1     f1; 
                         p1+l   focus_VF(t, d_max, t_max, beam);
                         p1+l+f f]
    return lens_elements_VF1
end 

