#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : 2d_collapse.jl                                                             |
|  Description: Please run this file in VSCode with Julia ENV                              |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Test Case  : 2D collapse in Drucker-Prager constitutive equation                        |
+==========================================================================================#

# 0.00250:  12800 pts
# 0.00181:  24200 pts
# 0.00125:  51200 pts
# 0.00050: 320000 pts
init_grid_space_x = 0.0025
init_grid_space_y = 0.0025
init_grid_range_x = [-0.1, 0.82]
init_grid_range_y = [-0.1, 0.12]
init_mp_in_space  = 2
init_project_name = "2d_collapse"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :druckerprager
init_vtk_step     = 1
init_gravity      = -9.8
init_ζ            = 0
init_ρs           = 2650
init_ν            = 0.3
init_Ks           = 7e5
init_E            = init_Ks*(3.0*(1.0.-2.0*init_ν))
init_G            = init_E/(2.0*(1.0+init_ν))
init_T            = 1
init_Te           = 0
init_ΔT           = 0.5*init_grid_space_x/sqrt(init_E/init_ρs)
init_step         = floor(init_T/init_ΔT/200) |> Int64
init_step<10 ? init_step=1 : nothing
init_σt           = 0
init_ϕ            = 19.8*π/180
init_c            = 0
init_ψ            = 0
init_NIC          = 16
init_basis        = :uGIMP
init_phase        = 1
iInt              = Int64
iFloat            = Float64

# parameters setup
args = Args2D{iInt, iFloat}(
    Ttol         = init_T,
    Te           = init_Te,
    ΔT           = init_ΔT,
    time_step    = :fixed,
    FLIP         = 1.0,
    PIC          = 0.0,
    ζ            = init_ζ,
    project_name = init_project_name,
    project_path = init_project_path,
    constitutive = init_constitutive,
    animation    = false,
    hdf5         = false,
    hdf5_step    = init_step,
    vollock      = false,
    device       = :CUDA,
    coupling     = :OS,
    basis        = :uGIMP
)

# background grid setup
grid = Grid2D{iInt, iFloat}(
    NIC      = init_NIC,
    phase    = init_phase,
    range_x1 = init_grid_range_x[1],
    range_x2 = init_grid_range_x[2],
    range_y1 = init_grid_range_y[1],
    range_y2 = init_grid_range_y[2],
    space_x  = init_grid_space_x,
    space_y  = init_grid_space_y
)

# material points setup
range_x    = [0+grid.space_x/init_mp_in_space/2, 0.2-grid.space_x/init_mp_in_space/2]
range_y    = [0+grid.space_y/init_mp_in_space/2, 0.1-grid.space_y/init_mp_in_space/2]
space_x    = grid.space_x/init_mp_in_space
space_y    = grid.space_y/init_mp_in_space
num_x      = length(range_x[1]:space_x:range_x[2])
num_y      = length(range_y[1]:space_y:range_y[2])
x_tmp      = repeat((range_x[1]:space_x:range_x[2])', num_y, 1) |> vec
y_tmp      = repeat((range_y[1]:space_y:range_y[2]) , 1, num_x) |> vec
mp_num     = length(x_tmp)
mp_layer   = ones(mp_num)
mp_ρs      = ones(mp_num).*init_ρs
mp_ν       = [init_ν]
mp_E       = [init_E]
mp_G       = [init_G]
mp_σt      = [init_σt]
mp_ϕ       = [init_ϕ]
mp_c       = [init_c]
mp_ψ       = [init_ψ]
mp_Ks      = [init_Ks]
mp         = Particle2D{iInt, iFloat}(space_x=space_x, space_y=space_y, pos=[x_tmp y_tmp],
    ν=mp_ν, E=mp_E, G=mp_G, σt=mp_σt, ϕ=mp_ϕ, c=mp_c, ψ=mp_ψ, ρs=mp_ρs, Ks=mp_Ks,
    NIC=init_NIC, layer=mp_layer, phase=init_phase)

# boundary condition nodes index
vx_idx = zeros(iInt, grid.node_num)
vy_idx = zeros(iInt, grid.node_num)
tmp_idx = findall(i->(grid.pos[i, 1]≤0.0||
                      grid.pos[i, 1]≥0.8||
                      grid.pos[i, 2]≤0), 1:grid.node_num)
tmp_idy = findall(i->(grid.pos[i, 2]≤0), 1:grid.node_num)
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

# visualization
@views @inbounds let
    x_ticks   = ["0", "100", "200", "300", "400", "500"]
    y_ticks   = ["0", "50", "100"]
    cpu_title = join([args.project_name, "_CPU"])
    gpu_title = join([args.project_name, "_GPU"])
    figfont   = joinpath(assets, "fonts/tnr.ttf")
    font_size = 20
    args.device==:CPU ? (figtitle=cpu_title) : (figtitle=gpu_title)
    fig = Figure(size=(660, 800), font=figfont, fontsize=font_size)
    set_theme!(Theme(Axis=(xticklabelfont=figfont, yticklabelfont=figfont, 
        titlegap=font_size, titlefont=figfont, aspect=DataAspect()), Scatter=(strokewidth=0,
        markersize=3), Colorbar=(ticklabelfont=figfont, spinewidth=0)))
    # Plot 1 ===============================================================================
    supertitle = Label(fig[0, :], figtitle, fontsize=22, font=figfont)
    axis_1 = Axis(fig[1, :][2, 1], ylabel=L"y\,[mm]", xticks=(0:0.1:0.5, x_ticks), 
        yticks=(0:0.05:0.1, y_ticks))
    color_data_1 = log10.(mp.epII.+1)
    plts_1 = scatter!(axis_1, mp.pos, color=color_data_1, colormap=:jet, marker=:circle, 
        label="Particles", colorrange=(0.0, 1.4))
    limits!(axis_1, -0.012, 0.52, -0.012, 0.12)
    Colorbar(fig[1, :][1, 1], plts_1, label=L"\epsilon_{II}", size=10, vertical=false,
        ticks = 0.1:0.4:1.4, ticklabelsize=16)
    # Plot 2 ===============================================================================
    failure = readdlm(joinpath(assets, "2d_collapse_experiments/failure.csv"), ',', Float64)
    surface = readdlm(joinpath(assets, "2d_collapse_experiments/surface.csv"), ',', Float64)
    ux = mp.pos[:, 1].-mp.init[:, 1]
    u1 = findall(i->ux[i]>0.001, 1:mp.num)
    u2 = findall(i->ux[i]≤0.001, 1:mp.num)
    ux[u1] .= -1; ux[u2] .= 1
    colors = ["#ef0000", "#00008e"]; cmap = cgrad(colors, 2; categorical = true)
    axis_2 = Axis(fig[2, :][2, 1], ylabel=L"y\,[mm]", xticks=(0:0.1:0.5, x_ticks), 
        yticks=(0:0.05:0.1, y_ticks))
    plts_2 = scatter!(axis_2, mp.pos, color=ux, colormap=cmap, marker=:circle,
        label="Particles")
    scatterlines!(axis_2, failure[:, 1], failure[:, 2], color=:green, marker=:utriangle,
        markersize=10, strokewidth=0, linewidth=2)
    scatterlines!(axis_2, surface[:, 1], surface[:, 2], color=:green, marker=:diamond,
        markersize=10, strokewidth=0, linewidth=2)
    limits!(axis_2, -0.012, 0.52, -0.012, 0.12)
    Colorbar(fig[2, :][1, 1], plts_2, label="\n ", size=10, ticklabelsize=16,
        ticks=(-1:1:1, [L">1mm", L"\Delta u", L"≤1mm"]), vertical=false)
    # Plot 3 ===============================================================================
    mp_line = Int64[]
    for i in 1:mp.num
        for j in collect(0:0.02:0.98)
            isapprox(mp.init[i, 2], j, atol=mp.space_y/1.9) ? push!(mp_line, i) : nothing
        end
        for k in collect(0:0.02:0.2)
            isapprox(mp.init[i, 1], k, atol=mp.space_x/1.9) ? push!(mp_line, i) : nothing
        end
        unique!(mp_line)
    end
    colors = ["#F9807d", "#00C1C8"]; cmap = cgrad(colors, 2; categorical = true)
    axis_3 = Axis(fig[3, :], xlabel=L"x\,[mm]", ylabel=L"y\,[mm]", aspect=DataAspect(),
        xticks=(0:0.1:0.5, x_ticks), yticks=(0:0.05:0.1, y_ticks))
    scatter!(axis_3, mp.pos, color=:gray, marker=:circle, label="Particles",)
    scatter!(axis_3, mp.pos[mp_line, :], color=:red, marker=:circle, label="Particles")
    limits!(axis_3, -0.012, 0.52, -0.012, 0.12)
    # ======================================================================================
    colsize!(fig.layout, 1, Aspect(1, 2.8))
    save(joinpath(rtsdir, args.project_name, "$(figtitle).png"), fig, px_per_unit=2)
    display(fig)
    @info "Figure saved in project path"
end