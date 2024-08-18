#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : 2d_dam_break.jl                                                            |
|  Description: Please run this file in VSCode with Julia ENV                              |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Test Case  : 2D dam break test.                                                         |
|  Reference  : [1] Huang, Peng, Shun-li Li, Hu Guo, and Zhi-ming Hao. “Large Deformation  |
|                   Failure Analysis of the Soil Slope Based on the Material Point Method.”|
|                   Computational Geosciences 19, no. 4 (August 2015): 951–63.             |
|                   https://doi.org/10.1007/s10596-015-9512-9.                             |
|               [2] Morris, J.P., Fox, P.J., Zhu, Y.: Modeling low reynolds number         |
|                   incompressible flows using SPH. J. Comput. Phys. 136, 214–226 (1997).  |
|  Notes      : [1] Don't use "uGIMP" basis for this case, it may break the simulation.    |
|               [2] Remeber to comment the lines about Δϵij[:, 4]/ϵij[:, 4] (2D) and       |
|                   Δϵij[:, [4,5,6]]/ϵij[:, [4,5,6]] (3D) in OS.jl and OS_d.jl.            |
|               [3] Comment CFL part in OS.jl and OS_d.jl.                                 |
|               [4] B and γ is useless in this case, Cs = 35m/s (from paper). Remember     |
|                   change this in taitwater.jl or taitwater_d.jl.                         |
|               [5] Analytical solution is from shallow water theory.                      |
+==========================================================================================#

init_grid_space_x = 4e-3
init_grid_space_y = 4e-3
init_grid_range_x = [-0.008, 2.508]
init_grid_range_y = [-0.008, 0.122]
init_mp_in_space  = 2
init_project_name = "2d_dam_break"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :taitwater
init_gravity      = -9.8
init_ζ            = 0
init_ρs           = 1e3
init_ν            = 0.48
init_Ks           = 2.15e9
init_E            = init_Ks*(3*(1-2*init_ν))
init_G            = init_E /(2*(1+  init_ν))
init_T            = 0.5
init_B            = 1.119e7
init_γ            = 7
init_Te           = 0
init_ΔT           = 0.1*min(init_grid_space_x, init_grid_space_y)/sqrt(init_E/init_ρs)
init_step         = floor(init_T/init_ΔT/300) |> Int64
init_step<10 ? init_step=1 : nothing
init_phase        = 1
init_NIC          = 4
init_basis        = :linear
iInt              = Int64
iFloat            = Float64

# parameters setup
args = Args2D{iInt, iFloat}(
    Ttol         = init_T,
    Te           = init_Te,
    ΔT           = init_ΔT,
    time_step    = :fixed,
    FLIP         = 1,
    PIC          = 0,
    ζ            = init_ζ,
    gravity      = init_gravity,
    project_name = init_project_name,
    project_path = init_project_path,
    constitutive = init_constitutive,
    animation    = false,
    hdf5         = false,
    hdf5_step    = init_step,
    vollock      = false,
    device       = :CUDA,
    coupling     = :OS,
    basis        = init_basis
)

# background grid setup
grid = Grid2D{iInt, iFloat}(
    range_x1 = init_grid_range_x[1],
    range_x2 = init_grid_range_x[2],
    range_y1 = init_grid_range_y[1],
    range_y2 = init_grid_range_y[2],
    space_x  = init_grid_space_x,
    space_y  = init_grid_space_y,
    phase    = init_phase,
    NIC      = init_NIC
)

# material points setup
range_x  = [0+grid.space_x/init_mp_in_space/2, 1.0-grid.space_x/init_mp_in_space/2]
range_y  = [0+grid.space_y/init_mp_in_space/2, 0.1-grid.space_y/init_mp_in_space/2]
space_x  = grid.space_x/init_mp_in_space
space_y  = grid.space_y/init_mp_in_space
num_x    = length(range_x[1]:space_x:range_x[2])
num_y    = length(range_y[1]:space_y:range_y[2])
x_tmp    = repeat((range_x[1]:space_x:range_x[2])', num_y, 1) |> vec
y_tmp    = repeat((range_y[1]:space_y:range_y[2]) , 1, num_x) |> vec
mp_num   = length(x_tmp)
mp_ρs    = ones(mp_num).*init_ρs
mp_layer = ones(mp_num)
mp_pos   = [x_tmp y_tmp]
mp_ν     = [init_ν]
mp_E     = [init_E]
mp_G     = [init_G]
mp_Ks    = [init_Ks]
mp_B     = [init_B]
mp_γ     = [init_γ]
mp       = Particle2D{iInt, iFloat}(space_x=space_x, space_y=space_y, pos=mp_pos, ν=mp_ν, 
    E=mp_E, G=mp_G, ρs=mp_ρs, Ks=mp_Ks, layer=mp_layer, NIC=init_NIC, phase=init_phase, 
    B=mp_B, γ=mp_γ)

# boundary condition nodes index
vx_idx  = zeros(iInt, grid.node_num)
vy_idx  = zeros(iInt, grid.node_num)
tmp_idx = findall(i->grid.pos[i, 1]≤0, 1:grid.node_num)
tmp_idy = findall(i->grid.pos[i, 2]≤0, 1:grid.node_num)
vx_idx[tmp_idx] .= 1
vy_idx[tmp_idy] .= 1
bc = VBoundary2D{iInt, iFloat}(
    Vx_s_Idx = vx_idx,
    Vx_s_Val = zeros(grid.node_num),
    Vy_s_Idx = vy_idx,
    Vy_s_Val = zeros(grid.node_num)
)

# MPM solver
simulate!(args, grid, mp, bc)

# post-processing
let 
    figfont = joinpath(assets, "fonts/tnr.ttf")
    fig = Figure(size=(850, 660), fontsize=20)
    ax1 = Axis(fig[1, 1], aspect=4, xticks=(0.2:0.2:2.0), yticks=(0.02:0.04:0.1), titlegap=20,
        title="Computational model in MPM (sMPM)",
        xlabel="𝑿 [𝑚]", ylabel="𝒀 [𝑚]", xticklabelfont=figfont, yticklabelfont=figfont, 
        titlefont=figfont)
    ax2 = Axis(fig[2, 1], aspect=4, xticks=(0.2:0.2:2.0), yticks=(0.02:0.04:0.1), titlegap=20,
        title="Comparison between MPM and analytical solutions of the dam break problem",
        xlabel="𝑿 [𝑚]", ylabel="𝒀 [𝑚]", xticklabelfont=figfont, yticklabelfont=figfont, 
        titlefont=figfont)
    
    pl0 = scatter!(ax1, mp.init, markersize=6, color=mp.init[:, 2], colormap=Reverse(:Blues_3))
    vlines!(ax1, [1], ymax=[0.86], color=:black, linewidth=5)

    pl1 = scatter!(ax2, mp.pos, markersize=4, color=:gray)    
    y = collect(0:0.001:0.1)
    x = @. (2*sqrt(0.1*9.8)-3*sqrt(y*9.8))*0.5+1
    analytical = vcat([x y], [0 0.1])
    pl2 = lines!(ax2, analytical, linewidth=8, color=:red, alpha=0.3)
    vlines!(ax2, [0.5], color=:blue, linewidth=2, linestyle=:dash)
    
    limits!(ax1, 0, 2.1, 0, 0.122)
    limits!(ax2, 0, 2.1, 0, 0.122)
    axislegend(ax2, [pl1, pl2], ["Particle", "Analytical solution"], "T=0.5s", position=:rt,
        labelfont=figfont, labelsize=16, titlefont=figfont)
    display(fig)
    save(joinpath(args.project_path, "$(args.project_name).png"), fig)
    @info "Figure saved in project path"
end