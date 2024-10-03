using MaterialPointSolver
using KernelAbstractions
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

# args setup
args = UserArgs2D(
    Ttol         = init_T,
    Te           = 0,
    ΔT           = init_ΔT,
    time_step    = :fixed,
    FLIP         = 1,
    PIC          = 0,
    constitutive = :druckerprager,
    basis        = :uGIMP,
    animation    = true,
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
    ϵ            = "FP64"
)

# grid setup
grid = UserGrid2D(
    ϵ     = "FP64",
    phase =  1,
    x1    = -0.025,
    x2    =  0.82,
    y1    = -0.025,
    y2    =  0.12,
    dx    =  0.0025,
    dy    =  0.0025,
    NIC   = 16
)

# material point setup
dx = grid.dx / init_mp_in_space
dy = grid.dy / init_mp_in_space
x_tmp, y_tmp = meshbuilder(0 + dx / 2 : dx : 0.2 - dx / 2,
                           0 + dy / 2 : dy : 0.1 - dy / 2)
mpρs = ones(length(x_tmp)) * init_ρs
mp = UserParticle2D(
    ϵ     = "FP64",
    phase = 1,
    NIC   = 16,
    dx    = dx,
    dy    = dy,
    ξ     = [x_tmp y_tmp],
    n     = [0],
    ρs    = mpρs
)

# property setup
nid = ones(mp.np)
attr = UserProperty(
    ϵ   = "FP64",
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
    ϵ        = "FP64",
    vx_s_idx = vx_idx,
    vx_s_val = zeros(grid.ni),
    vy_s_idx = vy_idx,
    vy_s_val = zeros(grid.ni)
)

# solver setup
materialpointsolver!(args, grid, mp, attr, bc)


using CairoMakie
let 
    figregular = MaterialPointSolver.tnr
    figbold = MaterialPointSolver.tnrb
    fig = Figure(size=(400, 142), fonts=(; regular=figregular, bold=figbold), fontsize=12,
        padding=0)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel=L"x\ (m)", ylabel=L"y\ (m)")
    p1 = scatter!(ax, mp.ξ, color=log10.(mp.ϵq.+1), markersize=1, colormap=:turbo,
        colorrange=(0, 1.5))
    Colorbar(fig[1, 2], p1, spinewidth=0, label=L"log_{10}(\epsilon_{II}+1)", size=6)
    limits!(ax, -0.02, 0.48, -0.02, 0.12)
    display(fig)
end
savevtu(args, grid, mp, attr)

# initmpstatus!(CPU())(ndrange=mp.np, grid, mp, Val(args.basis))
# # variables setup for the simulation 
# T2 = Float64
# T1 = Int64
# Ti = T2(0.0)
# ΔT = 0.5
# G = Ti < args.Te ? args.gravity / args.Te * Ti : args.gravity
# dev = getBackend(Val(args.device))


# resetgridstatus_OS!(dev)(ndrange=grid.ni, grid)
# args.device == :CPU && args.basis == :uGIMP ? 
#     resetmpstatus_OS_CPU!(dev)(ndrange=mp.np, grid, mp, Val(args.basis)) :
#     resetmpstatus_OS!(dev)(ndrange=mp.np, grid, mp, Val(args.basis))
# P2G_OS!(dev)(ndrange=mp.np, grid, mp, G)
# solvegrid_a_OS!(dev)(ndrange=grid.ni, grid, bc, ΔT, args.ζs)
# doublemapping1_a_OS!(dev)(ndrange=mp.np, grid, mp, attr, ΔT)
# doublemapping2_OS!(dev)(ndrange=mp.np, grid, mp)
# doublemapping3_OS!(dev)(ndrange=grid.ni, grid, bc, ΔT)
# G2P_OS!(dev)(ndrange=mp.np, grid, mp, ΔT)
# if args.constitutive == :hyperelastic
#     hyE!(dev)(ndrange=mp.np, mp, attr)
# elseif args.constitutive == :linearelastic
#     liE!(dev)(ndrange=mp.np, mp, attr)
# elseif args.constitutive == :druckerprager
#     liE!(dev)(ndrange=mp.np, mp, attr)
#     if Ti ≥ args.Te
#         dpP!(dev)(ndrange=mp.np, mp, attr)
#     end
# elseif args.constitutive == :mohrcoulomb
#     liE!(dev)(ndrange=mp.np, mp, attr)
#     if Ti ≥ args.Te
#         mcP!(dev)(ndrange=mp.np, mp, attr)
#     end
# end
# if args.MVL == true
#     vollock1_OS!(dev)(ndrange=mp.np, grid, mp)
#     vollock2_OS!(dev)(ndrange=mp.np, grid, mp)
# end