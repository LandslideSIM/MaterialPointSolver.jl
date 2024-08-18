#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : 2d_water_pure.jl                                                           |
|  Description: 2D Newtonian fluid test                                                    |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Reference  : [1] Li, Xinpo, Jun Yao, Yulian Sun, and Yong Wu. "Material point method    |
|                   analysis of fluid–structure interaction in geohazards." Natural        |
|                   Hazards 114, no. 3 (2022): 3425-3443,                                  |
|                   https://link.springer.com/article/10.1007/s11069-022-05526-1.          |
|               [2] Tain EOS https://pysph.readthedocs.io/en/1.0a1/reference/equations.html|
+==========================================================================================#

using MaterialPointSolver
using KernelAbstractions

rtsdir = joinpath(homedir(), "Workbench/outputs")

@kernel inbounds=true function testneF!(
    mp      ::KernelParticle2D{T1, T2},
    pts_attr::KernelParticleProperty{T1, T2}, 
    ΔT      ::T2
) where {T1, T2}
    # DV: dynamic viscosity
    # Cγ: 7 for pure water
    FNUM_1 = T2(1); FNUM_1t = T2(1/ΔT); Cγ = T2(7)
    FNUM_2 = T2(2); DV      = T2(1e-3)
    ix = @index(Global)
    if ix ≤ mp.num
        P = (pts_attr.Ks[pts_attr.layer[ix]]/Cγ)*((mp.ρs[ix]/mp.ρs_init[ix])^Cγ-FNUM_1)
        mp.σij[ix, 1] = -P
        mp.σij[ix, 2] = -P
        mp.σij[ix, 4] += FNUM_2*DV*mp.Δϵij_s[ix, 4]*FNUM_1t
        mp.σm[ix] = P
    end
end

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
    testneF!(dev)(ndrange=mp.num, mp, pts_attr, ΔT)
    if args.MVL == true
        vollock1_OS!(dev)(ndrange=mp.num, grid, mp)
        vollock2_OS!(dev)(ndrange=mp.num, grid, mp)
    end
    return nothing
end

init_grid_space_x = 0.005
init_grid_space_y = 0.005
init_grid_range_x = [-0.1, 0.7]
init_grid_range_y = [-0.1, 1.0]
init_mp_in_space  = 2
init_project_name = "2d_water_pure"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :newtonfluid
init_gravity      = -9.8
init_ζs           = 0
init_ρs           = 1e3
init_ν            = 0
init_Ks           = 5e6 # smaller than real value for smaller time step
init_E            = 1e6
init_G            = init_E/(2*(1+init_ν))
init_T            = 2
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
    ζs           = init_ζs,
    gravity      = init_gravity,
    project_name = init_project_name,
    project_path = init_project_path,
    constitutive = init_constitutive,
    animation    = true,
    hdf5         = true,
    hdf5_step    = init_step,
    MVL          = false,
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
range_x  = [0+grid.space_x/init_mp_in_space/2, 0.15-grid.space_x/init_mp_in_space/2]
range_y  = [0+grid.space_y/init_mp_in_space/2, 0.30-grid.space_y/init_mp_in_space/2]
space_x  = grid.space_x/init_mp_in_space
space_y  = grid.space_y/init_mp_in_space
num_x    = length(range_x[1]:space_x:range_x[2])
num_y    = length(range_y[1]:space_y:range_y[2])
x_tmp    = repeat((range_x[1]:space_x:range_x[2])', num_y, 1) |> vec
y_tmp    = repeat((range_y[1]:space_y:range_y[2]) , 1, num_x) |> vec
mp_num   = length(x_tmp)
mp_ρs    = ones(mp_num).*init_ρs
mp_pos   = [x_tmp y_tmp]
mp       = Particle2D{iInt, iFloat}(space_x=space_x, space_y=space_y, pos=mp_pos, ρs=mp_ρs,
    phase=init_phase, NIC=init_NIC)

# particle property setup
mp_ν     = [init_ν]
mp_E     = [init_E]
mp_G     = [init_G]
mp_Ks    = [init_Ks]
mp_layer = ones(mp_num)
pts_attr = ParticleProperty{iInt, iFloat}(layer=mp_layer, ν=mp_ν, E=mp_E, G=mp_G, Ks=mp_Ks)

# boundary condition nodes index
vx_idx  = zeros(iInt, grid.node_num)
vy_idx  = zeros(iInt, grid.node_num) 
tmp_idx = findall(i->(grid.pos[i, 1]≤0)||(grid.pos[i, 1]≥0.65), 1:grid.node_num)
tmp_idy = findall(i->(grid.pos[i, 2]≤0), 1:grid.node_num)
vx_idx[tmp_idx] .= 1
vy_idx[tmp_idy] .= 1
bc = VBoundary2D{iInt, iFloat}(
    Vx_s_Idx = vx_idx,
    Vx_s_Val = zeros(grid.node_num),
    Vy_s_Idx = vy_idx,
    Vy_s_Val = zeros(grid.node_num),
)

# MPM solver
materialpointsolver!(args, grid, mp, pts_attr, bc, workflow=testprocedure!)
savevtu(args, grid, mp, pts_attr)