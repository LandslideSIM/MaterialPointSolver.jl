#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : 2d_terzaghi.jl                                                             |
|  Description: Please run this file in VSCode with Julia ENV                              |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Test Case  : Terzaghi consolidation test                                                |
+==========================================================================================#

# model configuration
init_grid_space_x = 0.1
init_grid_space_y = 0.1
init_grid_range_x = [-0.1, 0.3]
init_grid_range_y = [-0.1, 1.2]
init_mp_in_space  = 2
init_project_name = "2d_terzaghi"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :linearelastic
init_gravity      = 0
init_ζ            = 0
init_ρs           = 2650
init_ρw           = 1000
init_porosity     = 0.3
init_k            = 1e-3
init_ν            = 0.3
init_E            = 1e7
init_G            = init_E/(2*(1+  init_ν))
init_Kw           = 1e8
init_Ks           = init_E/(3*(1-2*init_ν))
init_T            = 0.5
init_Te           = 0
init_ΔT           = 1.4e-4
init_step         = floor(init_T/init_ΔT/150) |> Int64
init_step<10 ? init_step=1 : nothing
init_basis        = :linear
init_NIC          = 4
init_phase        = 2
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
    gravity      = -9.8,#init_gravity,
    project_name = init_project_name,
    project_path = init_project_path,
    constitutive = init_constitutive,
    animation    = true,
    hdf5         = true,
    hdf5_step    = 1,#init_step,
    vollock      = false,
    device       = :CPU,
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
    NIC      = init_NIC,
    phase    = init_phase
)
global trac_idx = findall(i->(grid.pos[i, 2]==1)&&(0≤grid.pos[i, 1]≤0.2), 1:grid.node_num)
global trac_nds = -200/length(trac_idx)

# material points setup
range_x     = [0+grid.space_x/init_mp_in_space/2, 0.2-grid.space_x/init_mp_in_space/2]
range_y     = [0+grid.space_y/init_mp_in_space/2,   1-grid.space_y/init_mp_in_space/2]
space_x     = grid.space_x/init_mp_in_space
space_y     = grid.space_y/init_mp_in_space
num_x       = length(range_x[1]:space_x:range_x[2])
num_y       = length(range_y[1]:space_y:range_y[2])
x_tmp       = repeat((range_x[1]:space_x:range_x[2])', num_y, 1) |> vec
y_tmp       = repeat((range_y[1]:space_y:range_y[2]) , 1, num_x) |> vec
pos         = hcat(x_tmp, y_tmp)
mp_num      = length(x_tmp)
mp_ρs       = ones(mp_num).*init_ρs
mp_ρw       = ones(mp_num).*init_ρw
mp_porosity = ones(mp_num).*init_porosity
mp_layer    = ones(mp_num)
mp_ν        = [init_ν]
mp_E        = [init_E]
mp_G        = [init_G]
mp_k        = [init_k]
mp_Ks       = [init_Ks]
mp_Kw       = [init_Kw]
mp          = Particle2D{iInt, iFloat}(space_x=space_x, space_y=space_y, pos=pos, ν=mp_ν, 
    E=mp_E, G=mp_G, ρs=mp_ρs, ρw=mp_ρw, Ks=mp_Ks, Kw=mp_Kw, k=mp_k, porosity=mp_porosity, 
    NIC=init_NIC, layer=mp_layer, phase=init_phase)
#mp.σw .= 1e3


# boundary condition nodes index
vx_idx  = zeros(iInt, grid.node_num)
vy_idx  = zeros(iInt, grid.node_num)
tmp_idx = findall(i->(grid.pos[i, 1]≤0||grid.pos[i, 1]≥0.2||
                      grid.pos[i, 2]≤0), 1:grid.node_num)
tmp_idy = findall(i->(grid.pos[i, 2]≤0), 1:grid.node_num)
vx_idx[tmp_idx] .= 1
vy_idx[tmp_idy] .= 1
bc = VBoundary2D{iInt, iFloat}(
    Vx_s_Idx = vx_idx,
    Vx_s_Val = zeros(grid.node_num),
    Vy_s_Idx = vy_idx,
    Vy_s_Val = zeros(grid.node_num),
    Vx_w_Idx = vx_idx,
    Vx_w_Val = zeros(grid.node_num),
    Vy_w_Idx = vy_idx,
    Vy_w_Val = zeros(grid.node_num)
)

# MPM solver
simulate!(args, grid, mp, bc)

# post-process
let
    figfont = joinpath(assets, "fonts/tnr.ttf")
    fig = Figure(size=(600, 700), font=figfont, fontsize=15)
    ax1 = Axis(fig[1, 1:4], xlabel="Normalized pore pressure 𝑝 [-]", xlabelfont=figfont, 
        ylabelfont=figfont, ylabel="Normalized depth 𝐻 [-]", xticklabelfont=figfont, 
        yticklabelfont=figfont, xticks=0:0.2:1, yticks=0:0.2:1, 
        title="Excess pore pressure isochrones")
    ax2 = Axis(fig[2, 1], aspect=DataAspect(), xlabel="𝑋 Axis [𝑚]", ylabel="𝑌 Axis [𝑚]", 
        xticklabelfont=figfont, yticklabelfont=figfont, xlabelfont=figfont, 
        ylabelfont=figfont, xticks=0:0.2:0.2, yticks=0:0.2:1, 
        title="Pore pressure distribution")
    ax3 = Axis(fig[2, 3], aspect=DataAspect(), xlabel="𝑋 Axis [𝑚]", ylabel="𝑌 Axis [𝑚]", 
        xticklabelfont=figfont, yticklabelfont=figfont, xlabelfont=figfont, 
        ylabelfont=figfont, xticks=0:0.2:0.2, yticks=0:0.2:1,
        title="𝑌-velocity distribution")

    p1 = scatter!(ax1, rand(10, 2), markersize=12)
    p2 = scatter!(ax2, mp.pos, color=mp.σw./1e3 , markersize=18, marker=:rect, 
        colormap=:jet)
    p3 = scatter!(ax3, mp.pos, color=mp.Vs[:, 2], markersize=18, marker=:rect, 
        colormap=:jet)

    Colorbar(fig[2, 2], p2, label=L"\sigma_w\ [[kPa]", size=5, ticklabelfont=figfont, 
        spinewidth=0)
    Colorbar(fig[2, 4], p3, label=L"Y-Vs\ [[m/s]", size=5, ticklabelfont=figfont, 
        spinewidth=0)

    limits!(ax1, -0.1, 1.1, -0.1, 1.1)
    limits!(ax2, -0.1, 0.3, -0.1, 1.1)
    limits!(ax3, -0.1, 0.3, -0.1, 1.1)
    display(fig)
    save(joinpath(args.project_path, "$(args.project_name).png"), fig)
    @info "Figure saved in project path"
end