###
#
##= PLOTTING =##
#
###



###
using Plots
using LaTeXStrings

pyplot()


@load "sim_params.jl2d" params
@load "sim_results.jl2d" results


pos_load = params["position load"]
vel_load = params["velocity load"]
lens_elements = params["lens elements"]
beam1 = params["beam 1"]
beam2 = params["beam 2"]
Ramp = params["Ramping profile"]
transitions = params["transitions"]
sim_params = params["Simulation params"]


times = results["times"]
rs    = results["positions"]
vs    = results["velocities"]


include("lenspresets.jl")
###



###
function info_text(fs::Int64)

    texts = [
        (0.01, 0.90, text("Beam Parameters:", fontsize=fs, halign = :left)),
        ############################
        (0.1, 0.85, text("wavelength = $(WL/nm) nm", fontsize=fs, halign = :left)),
        (0.1, 0.80, text("(w_0(1), w_0(2)) = ($(w01_A/um) \u03BCm, $(w02_A/um) \u03BCm)", fontsize=fs, halign = :left)),
        (0.1, 0.75, text("(P(1), P(2)) = ($(P1_A) W, $(P2_A) W)", fontsize=fs, halign = :left)),

        (0.01, 0.65, text("Simulation Parameters:", fontsize=fs, halign = :left)),
        ############################
        (0.1, 0.60, text("N = $(N_atoms)", fontsize=fs, halign = :left)),
        (0.1, 0.55, text("T = $(T/uK) \u03BCK", fontsize=fs, halign = :left)),
        (0.1, 0.50, text("dt = $(1e6*dt) \u03BCs", fontsize=fs, halign = :left)),
        (0.1, 0.45, text("Sim time = $(t_sim/ms) ms", fontsize=fs, halign = :left)),
    
        (0.01, 0.35, text("Transport Parameters:", fontsize=fs, halign = :left)),
        ############################
        (0.1, 0.30, text("Transport distance = $(Ramp.d_max/mm) mm", fontsize=fs, halign = :left)),
        (0.1, 0.25, text("Transport time = $(Ramp.t_max/ms) ms", fontsize=fs, halign = :left)),        
    ]
    
    return texts 
end 
###



###
function trajectory_plot(r_max::Float64, z_max::Float64)
    
    xs = range(-r_max, r_max, length=100)
    ys = range(-r_max, r_max, length=100)
    zs = range(0, z_max, length=500)

    depth_xy = [-1e6 * U_lattice(x, y, pos_load[3], 0.0, 
                                 lens_elements, 
                                 beam1, beam2,
                                 Ramp, transitions)/kb 
                                 for x in xs, y in ys]
    depth_xz = [-1e6 * U_lattice(x, pos_load[2], z, 0.0, 
                                 lens_elements, 
                                 beam1, beam2,
                                 Ramp, transitions)/kb 
                                 for x in xs, z in zs]
    depth_yz = [-1e6 * U_lattice(pos_load[1], y, z, 0.0, 
                                 lens_elements, 
                                 beam1, beam2,
                                 Ramp, transitions)/kb 
                                 for y in ys, z in zs]
    custom_layout = @layout [
        grid(2, 2, widths = [0.25, 0.75])
    ]
    plt = plot(layout = custom_layout, size=(2400, 1200))

    c_xy = contour!(plt[1], xs .*1e3, ys .*1e3, 
                    depth_xy, 
                    colorbar=false, 
                    cmap=:Reds, 
                    xlabel="x (mm)",
                    ylabel="y (mm)", 
                    colorbar_title = "Trap depth (μK)",
                    fill=true)
    c_xz = contour!(plt[2], zs .*1e3, xs .*1e3, 
                    depth_xz, 
                    colorbar=true, 
                    cmap=:Reds, 
                    xlabel="z (mm)",
                    ylabel="x (mm)", 
                    colorbar_title = "Trap depth (μK)",
                    fill=true)
    c_yz = contour!(plt[4], zs .*1e3, ys .*1e3, 
                    depth_yz, 
                    colorbar=true, 
                    cmap=:Reds, 
                    xlabel="z (mm)", 
                    ylabel="y (mm)", 
                    colorbar_title = "Trap depth (μK)",
                    fill=true)

    plot!(plt[3], legend=false, framestyle=:none, ticks=nothing)
    annotate!(plt[3], [text for text in info_text(4)])


    i_show = 3
    for i in 1:N_atoms

        mask = (rs[i, :, 1] .>= -r_max) .& (rs[i, :, 1] .<= r_max) .& (rs[i, :, 2] .>= -r_max) .& (rs[i, :, 2] .<= r_max) .& (rs[i, :, 3] .>= 0) .& (rs[i, :, 3] .<= 53.2e-2)

        if (i != i_show)
            plot!(plt[1], 1e3 .*rs[i, :, 1][mask], 1e3 .*rs[i, :, 2][mask], color = :skyblue, linewidth=2, legend=false)
            plot!(plt[2], 1e3 .*rs[i, :, 3][mask], 1e3 .*rs[i, :, 1][mask], color = :skyblue, linewidth=2, legend=false)
            plot!(plt[4], 1e3 .*rs[i, :, 3][mask], 1e3 .*rs[i, :, 2][mask], color = :skyblue, linewidth=2, legend=false)
        end
    end

    mask_ishow = (rs[i_show, :, 1] .>= -r_max) .& (rs[i_show, :, 1] .<= r_max) .& (rs[i_show, :, 2] .>= -r_max) .& (rs[i_show, :, 2] .<= r_max) .& (rs[i_show, :, 3] .>= 0) .& (rs[i_show, :, 3] .<= 53.2e-2)

    plot!(plt[1], 1e3 .*rs[i_show, :, 1][mask_ishow], 1e3 .*rs[i_show, :, 2][mask_ishow], color = :blue, linewidth=4, legend=false)
    plot!(plt[2], 1e3 .*rs[i_show, :, 3][mask_ishow], 1e3 .*rs[i_show, :, 1][mask_ishow], color = :blue, linewidth=4, legend=false)
    plot!(plt[4], 1e3 .*rs[i_show, :, 3][mask_ishow], 1e3 .*rs[i_show, :, 2][mask_ishow], color = :blue, linewidth=4, legend=false)

end



trajectory_plot(5e-4, 53.2e-2)




