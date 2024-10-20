#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : 2d_case.jl                                                                 |
|  Description: Case used to vaildate the functions                                        |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
+==========================================================================================#

using MaterialPointSolver
using CairoMakie
using CUDA

MaterialPointSolver.warmup(Val(:CUDA))

init_grid_space_x = 0.0025
init_grid_space_y = 0.0025
init_grid_range_x = [-0.025, 0.82]
init_grid_range_y = [-0.025, 0.12]
init_mp_in_space  = 2
init_T            = 1
init_ρs           = 2650
init_ν            = 0.3
init_Ks           = 7e5
init_Es           = init_Ks * (3 * (1 - 2 * init_ν))
init_Gs           = init_Es / (2 * (1 +     init_ν))
init_ΔT           = 0.5 * init_grid_space_x / sqrt(init_Es / init_ρs)
init_step         = floor(init_T / init_ΔT / 200)
init_ϕ            = deg2rad(19.8)
init_NIC          = 9
init_basis        = :uGIMP
init_ϵ            = "FP64"

# args setup
args = UserArgs2D(
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
    project_name = "2d_case",
    project_path = @__DIR__,
    ϵ            = init_ϵ
)

# grid setup
grid = UserGrid2D(
    ϵ     = init_ϵ,
    phase = 1,
    x1    = init_grid_range_x[1],
    x2    = init_grid_range_x[2],
    y1    = init_grid_range_y[1],
    y2    = init_grid_range_y[2],
    dx    = init_grid_space_x,
    dy    = init_grid_space_y,
    NIC   = init_NIC
)

# material point setup
dx = grid.dx / init_mp_in_space
dy = grid.dy / init_mp_in_space
pts = meshbuilder(0 + dx / 2 : dx : 0.2 - dx / 2,
                  0 + dy / 2 : dy : 0.1 - dy / 2)
mpρs = ones(size(pts, 1)) * init_ρs
mp = UserParticle2D(
    ϵ     = init_ϵ,
    phase = 1,
    NIC   = init_NIC,
    dx    = dx,
    dy    = dy,
    ξ     = pts,
    ρs    = mpρs
)

# property setup
nid = ones(mp.np)
attr = UserProperty(
    ϵ   = init_ϵ,
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
vx_idx = zeros(grid.ni)
vy_idx = zeros(grid.ni)
tmp_idx = findall(i -> grid.ξ[i, 1] ≤ 0.0 || grid.ξ[i, 1] ≥ 0.8 ||
                       grid.ξ[i, 2] ≤ 0, 1:grid.ni)
tmp_idy = findall(i -> grid.ξ[i, 2] ≤ 0, 1:grid.ni)
vx_idx[tmp_idx] .= 1
vy_idx[tmp_idy] .= 1
bc = UserVBoundary2D(
    ϵ        = init_ϵ,
    vx_s_idx = vx_idx,
    vx_s_val = zeros(grid.ni),
    vy_s_idx = vy_idx,
    vy_s_val = zeros(grid.ni)
)

# solver setup
materialpointsolver!(args, grid, mp, attr, bc)

let 
    figregular = MaterialPointSolver.tnr
    figbold = MaterialPointSolver.tnrb
    fig = Figure(size=(440, 142), fonts=(; regular=figregular, bold=figbold), fontsize=12,
        padding=0)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel=L"x\ (m)", ylabel=L"y\ (m)", 
        xticks=(0:0.1:0.5), yticks=(0:0.05:0.1))
    p1 = scatter!(ax, mp.ξ, color=log10.(mp.ϵq.+1), markersize=2, colormap=:turbo,
        colorrange=(0, 1.5))
    Colorbar(fig[1, 2], p1, spinewidth=0, label=L"log_{10}(\epsilon_{II}+1)", size=6)
    limits!(ax, -0.02, 0.52, -0.02, 0.12)
    display(fig)
end
rm(joinpath(abspath(args.project_path), args.project_name), recursive=true, force=true)