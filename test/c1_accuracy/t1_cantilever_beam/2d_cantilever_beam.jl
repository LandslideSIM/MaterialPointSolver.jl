#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : 2d_cantilever_beam.jl                                                      |
|  Description: Hyperelastic (Neo-Hooken) cantilever beam test                             |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Notes      : This test can use two constitutive models, i.e., linearelastic and         |
|               hyperelastic. It's better to add comment for the objective stress update   |
|               in linearelastic material                                                  |
|  Reference  : https://www.mdpi.com/1996-1944/10/10/1150                                  |
+==========================================================================================#

using MaterialPointSolver
using HDF5
using CairoMakie
using DelimitedFiles
using CUDA

rtsdir = joinpath(homedir(), "Workbench/outputs")
assetsdir = MaterialPointSolver.assets_dir

# model configuration
init_basis        = :uGIMP
init_NIC          = 16
init_grid_space_x = 0.25
init_grid_space_y = 0.25
init_grid_range_x = [-3, 8]
init_grid_range_y = [-8, 4]
init_mp_in_space  = 3
init_project_name = "2d_cantilever_beam"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :hyperelastic
init_gravity      = -9.8
init_ζs           = 0
init_ρs           = 1.05e3
init_ν            = 0.3
init_E            = 1e6
init_G            = init_E/(2*(1+  init_ν))
init_Ks           = init_E/(3*(1-2*init_ν))
init_T            = 3
init_Te           = 0
init_ΔT           = 1e-4#5.7e-4#
init_step         = floor(init_T/init_ΔT/100) |> Int64
init_step<10 ? init_step=1 : nothing
init_phase        = 1
init_scheme       = :MUSL
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
    ζs           = init_ζs,
    gravity      = init_gravity,
    project_name = init_project_name,
    project_path = init_project_path,
    constitutive = init_constitutive,
    animation    = false,
    hdf5         = true,
    hdf5_step    = init_step,
    MVL          = false,
    device       = :CUDA,
    coupling     = :OS,
    scheme       = init_scheme,
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

# material points setup
range_x = [0+grid.space_x/init_mp_in_space/2, 4-grid.space_x/init_mp_in_space/2]
range_y = [0+grid.space_y/init_mp_in_space/2, 1-grid.space_y/init_mp_in_space/2]
space_x = grid.space_x/init_mp_in_space
space_y = grid.space_y/init_mp_in_space
num_x   = length(range_x[1]:space_x:range_x[2])
num_y   = length(range_y[1]:space_y:range_y[2])
x_tmp   = repeat((range_x[1]:space_x:range_x[2])', num_y, 1) |> vec
y_tmp   = repeat((range_y[1]:space_y:range_y[2]) , 1, num_x) |> vec
pos     = hcat(x_tmp, y_tmp)
mp_num  = length(x_tmp)
mp_ρs   = ones(mp_num).*init_ρs
mp      = Particle2D{iInt, iFloat}(space_x=space_x, space_y=space_y, pos=pos, ρs=mp_ρs, 
    NIC=init_NIC, phase=init_phase)

# particle property setup
mp_layer = ones(mp_num)
mp_ν     = [init_ν]
mp_E     = [init_E]
mp_G     = [init_G]
mp_Ks    = [init_Ks]
pts_attr = ParticleProperty{iInt, iFloat}(layer=mp_layer, ν=mp_ν, E=mp_E, G=mp_G, Ks=mp_Ks)

# boundary condition nodes index
vx_idx  = zeros(iInt, grid.node_num)
vy_idx  = zeros(iInt, grid.node_num)
tmp_idx = findall(i->(grid.pos[i, 1]≤0)&&(grid.pos[i, 2]≥-1), 1:grid.node_num)
tmp_idy = findall(i->(grid.pos[i, 1]≤0)&&(grid.pos[i, 2]≥-1), 1:grid.node_num)
vx_idx[tmp_idx] .= 1
vy_idx[tmp_idy] .= 1
bc = VBoundary2D{iInt, iFloat}(
    Vx_s_Idx = vx_idx,
    Vx_s_Val = zeros(grid.node_num),
    Vy_s_Idx = vy_idx,
    Vy_s_Val = zeros(grid.node_num),
)

# MPM solver
materialpointsolver!(args, grid, mp, pts_attr, bc)

# validation
let
    # results from MPMSolver
    fid = h5open(joinpath(args.project_path, "$(args.project_name).h5"), "r")
    idx = findall(i->mp.init[i, 2]==minimum(mp.init[:, 2])&&
                     mp.init[i, 1]==maximum(mp.init[:, 1]), 1:mp.num)[1]
    mpy = Float64[]
    ti  = Float64[]
    itr = read(fid["FILE_NUM"])-1 |> Int64
    iy  = read(fid["mp_init"])[:, 2][idx]
    for i in 1:itr
        push!(mpy, read(fid["group$(i)/mp_pos"])[:, 2][idx])
        push!(ti , read(fid["group$(i)/time"])  )
    end; mpy .-= iy
    close(fid)

    # results from FEM and MPM-CPDI
    filepath = joinpath(assetsdir, "data/2d_cantileverbeam_data")
    fem      = readdlm(joinpath(filepath, "FEM_cantilever_beam.csv"), ',')
    cpdi     = readdlm(joinpath(filepath, "MPM_CPDI.csv"           ), ',')
    ugimp    = readdlm(joinpath(filepath, "MPM_uGIMP.csv"          ), ',')

    # plot
    figfont = MaterialPointSolver.fontcmu
    fig = Figure(size=(640, 360), fontsize=20, font=figfont)
    axis = Axis(fig[1, 1], autolimitaspect=0.5, xticklabelfont=figfont, 
        title=":rb Particle Displacement [𝘮]", titlefont=figfont, yticklabelfont=figfont, 
        xticks=(0:1:3), yticks=(-3:1:0.5))
    p1 = scatterlines!(axis, ti, mpy, color="#05B9E2", markersize=10, marker=:rect, 
        label="MPMSolver", linewidth=1)
    p2 = scatterlines!(axis, fem[:, 1], fem[:, 2], color="#f27970", marker=:xcross, 
        markersize=10, label="FEM", linewidth=1)
    p3 = scatterlines!(axis, cpdi[:, 1], cpdi[:, 2], color="#BB9727", marker=:circle, 
        markersize=10, label="uGIMP", linewidth=1)
    p4 = scatterlines!(axis, ugimp[:, 1], ugimp[:, 2], color="#54b345", marker=:star6, 
        markersize=10, label="CPDI", linewidth=1)
    xlims!(axis, -0.3, 3.5)
    ylims!(axis, -3.5, 0.3)
    axislegend(axis, merge=true, unique=true, position=:rb, labelfont=figfont)
    display(fig)
    save(joinpath(rtsdir, args.project_name, "$(args.project_name).png"), fig)
    @info "Figure saved in project path"
end