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
using CairoMakie
using CUDA

MaterialPointSolver.warmup(Val(:CUDA))

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
init_grid_range_y = [-0.02, 0.75]
init_grid_range_z = [-0.02, 0.12]
init_mp_in_space  = 2
init_T            = 1
init_ρs           = 2650
init_ν            = 0.3
init_Ks           = 7e5
init_Es           = init_Ks * (3 * (1 - 2 * init_ν))
init_Gs           = init_Es / (2 * (1 +     init_ν))
init_ΔT           = 0.5 * init_grid_space_x / sqrt(init_Es / init_ρs)
init_step         = floor(init_T / init_ΔT / 50)
init_ϕ            = deg2rad(19.8)
init_FP           = "FP64"
init_basis        = :uGIMP
init_NIC          = 64

# args setup
args = UserArgs3D(
    Ttol         = init_T,
    Te           = 0,
    ΔT           = init_ΔT,
    time_step    = :fixed,
    FLIP         = 1,
    PIC          = 0,
    constitutive = :druckerprager,
    basis        = init_basis,
    animation    = false,
    hdf5         = false,
    hdf5_step    = init_step,
    MVL          = false,
    device       = :CUDA,
    coupling     = :OS,
    scheme       = :MUSL,
    va           = :a,
    progressbar  = true,
    gravity      = -9.8,
    ζs           = 0,
    project_name = "3d_case",
    project_path = @__DIR__,
    ϵ            = init_FP
)

# grid setup
grid = UserGrid3D(
    ϵ     = init_FP,
    phase =  1,
    x1    = -0.02,
    x2    =  0.07,
    y1    = -0.02,
    y2    =  0.75,
    z1    = -0.02,
    z2    =  0.12,
    dx    =  init_grid_space_x,
    dy    =  init_grid_space_y,
    dz    =  init_grid_space_z,
    NIC   =  init_NIC
)

# material point setup
dx = grid.dx / init_mp_in_space
dy = grid.dy / init_mp_in_space
dz = grid.dz / init_mp_in_space
x_tmp, y_tmp, z_tmp = meshbuilder(0 + dx / 2 : dx : 0.05 - dx / 2,
                                  0 + dy / 2 : dy : 0.20 - dy / 2,
                                  0 + dz / 2 : dz : 0.10 - dz / 2)
mpρs = ones(length(x_tmp)) * init_ρs
mp = UserParticle3D(
    ϵ     = init_FP,
    phase = 1,
    NIC   = init_NIC,
    dx    = dx,
    dy    = dy,
    dz    = dz,
    ξ     = [x_tmp y_tmp z_tmp],
    ρs    = mpρs
)

# property setup
nid = ones(mp.np)
attr = UserProperty(
    ϵ   = init_FP,
    nid = nid,
    ν   = [init_ν],
    Es  = [init_Es],
    Gs  = [init_Gs],
    Ks  = [init_Ks],
    σt  = [0],
    ϕ   = [init_ϕ],
    ϕr  = [0],
    ψ   = [0],
    c   = [0],
    cr  = [0],
    Hp  = [0]
)

# boundary setup
vx_idx  = zeros(grid.ni)
vy_idx  = zeros(grid.ni)
vz_idx  = zeros(grid.ni)
tmp_idx = findall(i -> grid.ξ[i, 1] ≤ 0 || grid.ξ[i, 1] ≥ 0.05 ||
                       grid.ξ[i, 3] ≤ 0 || grid.ξ[i, 2] ≤ 0, 1:grid.ni)
tmp_idy = findall(i -> grid.ξ[i, 2] ≤ 0 || grid.ξ[i, 3] ≤ 0, 1:grid.ni)
tmp_idz = findall(i -> grid.ξ[i, 3] ≤ 0, 1:grid.ni)
vx_idx[tmp_idx] .= 1
vy_idx[tmp_idy] .= 1
vz_idx[tmp_idz] .= 1
bc = UserVBoundary3D(
    ϵ        = init_FP,
    vx_s_idx = vx_idx,
    vx_s_val = zeros(grid.ni),
    vy_s_idx = vy_idx,
    vy_s_val = zeros(grid.ni),
    vz_s_idx = vz_idx,
    vz_s_val = zeros(grid.ni)
)

# solver setup
materialpointsolver!(args, grid, mp, attr, bc)

let
    figfont = MaterialPointSolver.tnr
    fig = Figure(size=(1200, 700), fonts=(; regular=figfont, bold=figfont), fontsize=30)
    ax = Axis3(fig[1, 1], xlabel=L"x\ (m)", ylabel=L"y\ (m)", zlabel=L"z\ (m)", 
        aspect=:data, azimuth=0.2*π, elevation=0.1*π, xlabeloffset=60, zlabeloffset=80,
        protrusions=100, xticks=(0:0.04:0.04), height=450, width=950)
    pl1 = scatter!(ax, mp.ξ, color=log10.(mp.ϵq.+1), colormap=:jet, markersize=3,
        colorrange=(0, 1))
    Colorbar(fig[1, 1], limits=(0, 1), colormap=:jet, size=16, ticks=0:0.5:1, spinewidth=0,
        label=L"log_{10}(\epsilon_{II}+1)", vertical=false, tellwidth=false, width=200,
        halign=:right, valign=:top, flipaxis=false)
    display(fig)
end
#rm(joinpath(abspath(args.project_path), args.project_name), recursive=true, force=true)