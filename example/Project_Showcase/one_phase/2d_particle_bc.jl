#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : 2d_particle_bc.jl                                                          |
|  Description: 2D model to test the particles as the fixed boundary                       |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Extra Code : Please add these after the workflow function:                              |
|               ----------------------------------------------------------------           |
|               copyto!(mp.pos[fixed_idx, :], mp.init[fixed_idx, :])                       |                           
|               fill!(mp.J[fixed_idx], 1)                                                  |
|               fill!(mp.Vs[fixed_idx], 0)                                                 |
+==========================================================================================#

using MaterialPointSolver
using CairoMakie

rtsdir = joinpath(homedir(), "Workbench/outputs")

function testprocedure!(args    ::MODELARGS, 
                        grid    ::GRID, 
                        mp      ::PARTICLE, 
                        pts_attr::PROPERTY,
                        bc      ::BOUNDARY,
                        ΔT      ::T2,
                        Ti      ::T2,
                                ::Val{:OS},
                                ::Val{:MUSL}) where {T2}
    Ti < args.Te ? G = args.gravity / args.Te * Ti : G = args.gravity
    dev = getBackend(args)
    resetgridstatus_OS!(dev)(ndrange=grid.node_num, grid)
    args.device == :CPU && args.basis == :uGIMP ? 
        resetmpstatus_OS!(grid, mp, Val(args.basis)) :
        resetmpstatus_OS!(dev)(ndrange=mp.num, grid, mp, Val(args.basis))
    P2G_OS!(dev)(ndrange=mp.num, grid, mp, G)
    solvegrid_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT, args.ζs)
    doublemapping1_OS!(dev)(ndrange=mp.num, grid, mp, pts_attr, ΔT, args.FLIP, args.PIC)
    doublemapping2_OS!(dev)(ndrange=mp.num, grid, mp)
    doublemapping3_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT)
    G2P_OS!(dev)(ndrange=mp.num, grid, mp)
    if args.constitutive==:hyperelastic
        hyE!(dev)(ndrange=mp.num, mp, pts_attr)
    elseif args.constitutive==:linearelastic
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
    elseif args.constitutive==:druckerprager
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
        if Ti≥args.Te
            dpP!(dev)(ndrange=mp.num, mp, pts_attr)
        end
    elseif args.constitutive==:mohrcoulomb
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
        if Ti≥args.Te
            mcP!(dev)(ndrange=mp.num, mp, pts_attr)
        end
    elseif args.constitutive==:taitwater
        twP!(dev)(ndrange=mp.num, mp)
    end
    if args.MVL == true
        vollock1_OS!(dev)(ndrange=mp.num, grid, mp)
        vollock2_OS!(dev)(ndrange=mp.num, grid, mp)
    end
    copyto!(mp.pos[fixed_idx, :], mp.init[fixed_idx, :])
    fill!(mp.J[fixed_idx], T2(1))
    fill!(mp.Vs[fixed_idx], T2(0))
    return nothing
end

init_grid_space_x = 0.1
init_grid_space_y = 0.1
init_grid_range_x = [-1, 21]
init_grid_range_y = [-1, 21]
init_mp_in_space  = 2
init_project_name = "2d_particle_bc"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :linearelastic
init_gravity      = -9.8
init_ζs           = 0
init_ρs           = 1200
init_ν            = 0.4
init_Ks           = 9e6
init_E            = init_Ks*(3.0*(1-2*init_ν))
init_G            = init_E /(2.0*(1+  init_ν))
init_T            = 10
init_Te           = 0
init_ΔT           = 0.4*init_grid_space_x/sqrt(init_E/init_ρs)
init_step         = floor(init_T/init_ΔT/200) |> Int64
init_step<10 ? init_step=1 : nothing
init_basis        = :uGIMP
init_phase        = 1
init_NIC          = 16
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
    project_name = init_project_name,
    project_path = init_project_path,
    constitutive = init_constitutive,
    animation    = true,
    hdf5         = true,
    hdf5_step    = init_step,
    MVL          = true,
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
    NIC      = init_NIC,
    phase    = init_phase
)

# material points setup
range_x = [0+grid.space_x/init_mp_in_space/2, 20-grid.space_x/init_mp_in_space/2]
range_y = [0+grid.space_y/init_mp_in_space/2, 20-grid.space_y/init_mp_in_space/2]
space_x = grid.space_x/init_mp_in_space
space_y = grid.space_y/init_mp_in_space
num_x   = length(range_x[1]:space_x:range_x[2])
num_y   = length(range_y[1]:space_y:range_y[2])
x_tmp1  = repeat((range_x[1]:space_x:range_x[2])', num_y, 1) |> vec
y_tmp1  = repeat((range_y[1]:space_y:range_y[2]) , 1, num_x) |> vec
fixed   = findall(i->(( 0≤x_tmp1[i]≤10)&&((y_tmp1[i]≤ 10-x_tmp1[i]) ||
                                          (y_tmp1[i]≥ 10+x_tmp1[i]))||
                      (10≤x_tmp1[i]≤20)&&((y_tmp1[i]≤-10+x_tmp1[i]) ||
                                          (y_tmp1[i]≥ 30-x_tmp1[i]))), 1:length(x_tmp1))
ball    = findall(i->(4≤x_tmp1[i]≤6)&&((y_tmp1[i]-10)^2<(1-(x_tmp1[i]-5)^2)), 
                      1:length(x_tmp1))
x_tmp   = x_tmp1[[fixed; ball]]
y_tmp   = y_tmp1[[fixed; ball]]
mp_num  = length(x_tmp)
global fixed_idx = 1:length(fixed)
global  ball_idx = 1+length(fixed):mp_num
mp_ρs = ones(mp_num).*init_ρs
mp    = Particle2D{iInt, iFloat}(space_x=space_x, space_y=space_y, pos=[x_tmp y_tmp],
    ρs=mp_ρs, phase=init_phase)
mp.Vs[ball_idx, 1] .= 10
mp.Vs[ball_idx, 2] .= 10

# particle property setup
mp_layer = ones(mp_num)
mp_layer[fixed_idx] .= 1
mp_layer[ ball_idx] .= 2
mp_ν     = [init_ν, init_ν]
mp_E     = [init_E, init_E*100]
mp_G     = [init_G, init_G]
mp_Ks    = [init_Ks, init_Ks]
pts_attr = ParticleProperty{iInt, iFloat}(layer=mp_layer, ν=mp_ν, E=mp_E, G=mp_G, Ks=mp_Ks)

# boundary condition nodes index
vx_idx  = zeros(iInt, grid.node_num)
vy_idx  = zeros(iInt, grid.node_num) 
tmp_idx = findall(i->(grid.pos[i, 1]==0||grid.pos[i, 1]==20||
                      grid.pos[i, 2]==0||grid.pos[i, 2]==20), 1:grid.node_num)
tmp_idx = findall(i->(grid.pos[i, 1]==0||grid.pos[i, 1]==20||
                      grid.pos[i, 2]==0||grid.pos[i, 2]==20), 1:grid.node_num)
vx_idx[tmp_idx] .= 1
vy_idx[tmp_idx] .= 1
bc = VBoundary2D{iInt, iFloat}(
    Vx_s_Idx = vx_idx,
    Vx_s_Val = zeros(grid.node_num),
    Vy_s_Idx = vy_idx,
    Vy_s_Val = zeros(grid.node_num)
)

# MPM solver
materialpointsolver!(args, grid, mp, pts_attr, bc, workflow=testprocedure!)

# visualization
let
    figfont = MaterialPointSolver.fontcmu
    fig = Figure(size=(1105, 805), fontsize=30, font=figfont)
    axis1 = Axis(fig[1, 1], xlabel="X Axis", ylabel=L"Y Axis", title="Visualization",
        titlefont=figfont, titlegap=12, spinewidth=3, aspect=DataAspect(),
        xticks=(0:5:20), yticks=(0:5:20), xticklabelsize=30, yticklabelsize=30, 
        xticklabelfont=figfont, yticklabelfont=figfont, xlabelfont=figfont, 
        ylabelfont=figfont)
    scatter!(axis1, mp.pos, color=:blue, markersize=10)
    limits!(axis1, -1, 21, -1, 21)
    save(joinpath(rtsdir, args.project_name, "$(args.project_name).png"), fig)
    @info "Figure saved in project path"
    fig
end