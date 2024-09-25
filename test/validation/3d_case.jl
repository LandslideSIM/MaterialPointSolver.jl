#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : 3d_case.jl                                                                 |
|  Description: Case used to vaildate the functions                                        |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
+==========================================================================================#

using MaterialPointSolver
using KernelAbstractions
using CairoMakie
using CUDA

warmup(Val(:CUDA))

include(joinpath(@__DIR__, "funcs.jl"))
# 0.01000:   8000 pts
# 0.00660:  27000 pts
# 0.00500:  64000 pts
# 0.00400: 125000 pts | grid: x[-0.008, 0.808] y[-0.008, 0.056] z[-0.08, 0.108]
# 0.00333: 216000 pts
# 0.00250: 512000 pts
init_grid_space_x = 0.0025
init_grid_space_y = 0.0025
init_grid_space_z = 0.0025
init_grid_range_x = [-0.02, 0.07]
init_grid_range_y = [-0.02, 0.55]
init_grid_range_z = [-0.02, 0.12]
init_mp_in_space  = 2
init_project_name = "3d_case"
init_project_path = joinpath(@__DIR__, init_project_name)
init_constitutive = :druckerprager
init_gravity      = -9.8
init_ζs           = 0
init_ρs           = 2700
init_ν            = 0
init_E            = 1e6
init_Ks           = init_E/(3*(1-2*init_ν))
init_G            = init_E/(2*(1+  init_ν))
init_T            = 1
init_Te           = 0
init_ΔT           = 0.5*init_grid_space_x/sqrt((init_Ks+4/3*init_G)/init_ρs)
init_step         = floor(init_T / init_ΔT / 50)
init_σt           = 0
init_ϕ            = deg2rad(19.8)
init_c            = 0
init_ψ            = 0
init_basis        = :uGIMP
init_phase        = 1
init_NIC          = 64
init_scheme       = :USF
iInt              = Int64
iFloat            = Float64

# parameters setup
args = Args3D{iInt, iFloat}(
    Ttol         = init_T,
    ΔT           = init_ΔT,
    Te           = init_Te,
    time_step    = :fixed,
    FLIP         = 1,
    PIC          = 0,
    ζs           = init_ζs,
    gravity      = init_gravity,
    project_name = init_project_name,
    project_path = init_project_path,
    constitutive = init_constitutive,
    animation    = true,
    jld2         = false,
    jld2_step    = init_step,
    MVL          = true,
    device       = :CUDA,
    coupling     = :OS,
    progressbar  = true,
    scheme       = init_scheme,
    basis        = init_basis
)

# background grid setup
grid = Grid3D{iInt, iFloat}(
    NIC      = init_NIC,
    range_x1 = init_grid_range_x[1],
    range_x2 = init_grid_range_x[2],
    range_y1 = init_grid_range_y[1],
    range_y2 = init_grid_range_y[2],
    range_z1 = init_grid_range_z[1],
    range_z2 = init_grid_range_z[2],
    space_x  = init_grid_space_x,
    space_y  = init_grid_space_y,
    space_z  = init_grid_space_z,
    phase    = init_phase
)

# material points setup
space_x = grid.space_x / init_mp_in_space
space_y = grid.space_y / init_mp_in_space
space_z = grid.space_z / init_mp_in_space
x_tmp, y_tmp, z_tmp = meshbuilder(0 + space_x / 2 : space_x : 0.05 - space_x / 2,
                                  0 + space_y / 2 : space_y : 0.20 - space_y / 2,
                                  0 + space_z / 2 : space_z : 0.10 - space_z / 2)
mp_num = length(x_tmp)
mp_ρs  = ones(mp_num).*init_ρs
mp     = Particle3D{iInt, iFloat}(space_x=space_x, space_y=space_y, space_z=space_z,
    pos=[x_tmp y_tmp z_tmp], ρs=mp_ρs, NIC=init_NIC, phase=init_phase)

# particle property setup
mp_layer = ones(mp_num)
mp_ν     = [init_ν]
mp_E     = [init_E]
mp_G     = [init_G]
mp_σt    = [init_σt]
mp_ϕ     = [init_ϕ]
mp_c     = [init_c]
mp_ψ     = [init_ψ]
mp_Ks    = [init_Ks]
pts_attr = ParticleProperty{iInt, iFloat}(layer=mp_layer, ν=mp_ν, E=mp_E, G=mp_G, σt=mp_σt, 
    ϕ=mp_ϕ, c=mp_c, ψ=mp_ψ, Ks=mp_Ks)

# boundary condition nodes index
vx_idx  = zeros(iInt, grid.node_num)
vy_idx  = zeros(iInt, grid.node_num)
vz_idx  = zeros(iInt, grid.node_num)
tmp_idx = findall(i -> grid.pos[i, 1] ≤ 0 || grid.pos[i, 1] ≥ 0.05 ||
                       grid.pos[i, 3] ≤ 0 || grid.pos[i, 2] ≤ 0, 1:grid.node_num)
tmp_idy = findall(i -> grid.pos[i, 2] ≤ 0 || grid.pos[i, 3] ≤ 0, 1:grid.node_num)
tmp_idz = findall(i -> grid.pos[i, 3] ≤ 0, 1:grid.node_num)
vx_idx[tmp_idx] .= 1
vy_idx[tmp_idy] .= 1
vz_idx[tmp_idz] .= 1
bc = VBoundary3D{iInt, iFloat}(
    Vx_s_Idx = vx_idx,
    Vx_s_Val = zeros(grid.node_num),
    Vy_s_Idx = vy_idx,
    Vy_s_Val = zeros(grid.node_num),
    Vz_s_Idx = vz_idx,
    Vz_s_Val = zeros(grid.node_num)
)

# MPM solver
materialpointsolver!(args, grid, mp, pts_attr, bc, workflow=testprocedure!)

let
    figfont = MaterialPointSolver.tnr
    fig = Figure(size=(1200, 700), fonts=(; regular=figfont, bold=figfont), fontsize=30)
    ax = Axis3(fig[1, 1], xlabel=L"x\ ((m)", ylabel=L"y\ ((m)", zlabel=L"z\ ((m)", 
        aspect=:data, azimuth=0.2*π, elevation=0.1*π, xlabeloffset=60, zlabeloffset=80,
        protrusions=100, xticks=(0:0.04:0.04), height=450, width=950)
    pl1 = scatter!(ax, mp.pos, color=log10.(mp.epII.+1), colormap=:jet, markersize=3,
        colorrange=(0, 1))
    Colorbar(fig[1, 1], limits=(0, 1), colormap=:jet, size=16, ticks=0:0.5:1, spinewidth=0,
        label=L"log_{10}(\epsilon_{II}+1)", vertical=false, tellwidth=false, width=200,
        halign=:right, valign=:top, flipaxis=false)
    display(fig)
end
rm(init_project_path, recursive=true)