#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : 3d_sand_new_damp.jl                                                        |
|  Description: 3D sand collapse test                                                      |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  References : [1] Cao, Chunhui, Jili Feng, and Zhigang Tao. "3D numerical simulation of  |
|                   landslides for the full high waste dump using SPH method." Advances in |
|                   Civil Engineering 2021 (2021): 1-16,                                   |
|                   https://www.hindawi.com/journals/ace/2021/8897826/                     |
+==========================================================================================#

using MaterialPointSolver
using KernelAbstractions

rtsdir = joinpath(homedir(), "Workbench/outputs")

@kernel inbounds=true function testsolvegrid_OS!(
    grid::    KernelGrid3D{T1, T2},
    bc  ::KernelBoundary3D{T1, T2},
    ΔT  ::T2,
    ζs  ::T2
) where {T1, T2}
    INUM_0 = T1(0); FNUM_0 = T2(0.0); FNUM_1 = T1(1.0)
    ix = @index(Global)
    if ix≤grid.node_num
        iszero(grid.Ms[ix]) ? Ms_denom=FNUM_0 : Ms_denom=FNUM_1/grid.Ms[ix]
        # compute nodal velocity
        Ps_1 = grid.Ps[ix, 1]
        Ps_2 = grid.Ps[ix, 2]
        Ps_3 = grid.Ps[ix, 3]
        grid.Vs[ix, 1] = Ps_1*Ms_denom
        grid.Vs[ix, 2] = Ps_2*Ms_denom
        grid.Vs[ix, 3] = Ps_3*Ms_denom
        # damping force for solid
        dampx = ζs*abs(grid.Fs[ix, 1])*sign(grid.Vs[ix, 1])
        dampy = ζs*abs(grid.Fs[ix, 2])*sign(grid.Vs[ix, 2])
        dampz = ζs*abs(grid.Fs[ix, 3])*sign(grid.Vs[ix, 3])
        # compute nodal total force for mixture
        Fs_x = grid.Fs[ix, 1]-dampx
        Fs_y = grid.Fs[ix, 2]-dampy
        Fs_z = grid.Fs[ix, 3]-dampz
        # update nodal velocity
        grid.Vs_T[ix, 1] = (Ps_1+Fs_x*ΔT)*Ms_denom
        grid.Vs_T[ix, 2] = (Ps_2+Fs_y*ΔT)*Ms_denom
        grid.Vs_T[ix, 3] = (Ps_3+Fs_z*ΔT)*Ms_denom
        # boundary condition
        bc.Vx_s_Idx[ix]≠INUM_0 ? grid.Vs_T[ix, 1]=bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix]≠INUM_0 ? grid.Vs_T[ix, 2]=bc.Vy_s_Val[ix] : nothing
        bc.Vz_s_Idx[ix]≠INUM_0 ? grid.Vs_T[ix, 3]=bc.Vz_s_Val[ix] : nothing
        # reset grid momentum
        grid.Ps[ix, 1] = FNUM_0
        grid.Ps[ix, 2] = FNUM_0
        grid.Ps[ix, 3] = FNUM_0
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
    testsolvegrid_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT, args.ζs)
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
    end
    if args.MVL == true
        vollock1_OS!(dev)(ndrange=mp.num, grid, mp)
        vollock2_OS!(dev)(ndrange=mp.num, grid, mp)
    end
    return nothing
end

init_grid_space_x = 0.005
init_grid_space_y = 0.005
init_grid_space_z = 0.005
init_grid_range_x = [-0.10, 0.10]
init_grid_range_y = [-0.10, 0.10]
init_grid_range_z = [-0.01, 0.11]
init_mp_in_space  = 2
init_project_name = "3d_sand_new_damp"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :druckerprager
init_gravity      = -9.8
init_ζs           = 0.1
init_ρs           = 1450
init_ν            = 0.3
init_E            = 2.016e7
init_Ks           = init_E/(3*(1-2*init_ν))
init_G            = init_E/(2*(1+  init_ν))
init_T            = 1
init_Te           = 0
init_ΔT           = 0.1*init_grid_space_x/sqrt((init_Ks+4/3*init_G)/init_ρs)
init_step         = floor(init_T/init_ΔT/20) |> Int64
init_step<10 ? init_step=1 : nothing
init_σt           = 0
init_ϕ            = 35*π/180
init_c            = 0
init_ψ            = 0
init_basis        = :uGIMP
init_phase        = 1
init_NIC          = 64
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
    hdf5         = true,
    hdf5_step    = init_step,
    MVL          = false,
    device       = :CUDA,
    coupling     = :OS,
    progressbar  = true,
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
range_x    = [-0.025+grid.space_x/init_mp_in_space/2, 0.025-grid.space_x/init_mp_in_space/2]
range_y    = [-0.025+grid.space_y/init_mp_in_space/2, 0.025-grid.space_y/init_mp_in_space/2]
range_z    = [ 0.000+grid.space_z/init_mp_in_space/2, 0.100-grid.space_z/init_mp_in_space/2]
space_x    = grid.space_x/init_mp_in_space
space_y    = grid.space_y/init_mp_in_space
space_z    = grid.space_z/init_mp_in_space
vx         = range_x[1]:space_x:range_x[2] |> collect
vy         = range_y[1]:space_y:range_y[2] |> collect
vz         = range_z[1]:space_z:range_z[2] |> collect
m, n, o    = length(vy), length(vx), length(vz)
vx         = reshape(vx, 1, n, 1)
vy         = reshape(vy, m, 1, 1)
vz         = reshape(vz, 1, 1, o)
om         = ones(Int, m)
on         = ones(Int, n)
oo         = ones(Int, o)
x_tmp      = vec(vx[om, :, oo])
y_tmp      = vec(vy[:, on, oo])
z_tmp      = vec(vz[om, on, :])
delidx     = findall(i->(x_tmp[i]^2+y_tmp[i]^2≥0.000625), 1:length(x_tmp))
deleteat!(x_tmp, delidx)
deleteat!(y_tmp, delidx)
deleteat!(z_tmp, delidx)
mp_num     = length(x_tmp)
mp_ρs      = ones(mp_num).*init_ρs
mp         = Particle3D{iInt, iFloat}(space_x=space_x, space_y=space_y, space_z=space_z,
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
tmp_idx = findall(i->grid.pos[i, 3]≤0, 1:grid.node_num)
tmp_idy = copy(tmp_idx)
tmp_idz = copy(tmp_idx)
vx_idx[tmp_idx] .= iInt(1)
vy_idx[tmp_idy] .= iInt(1)
vz_idx[tmp_idz] .= iInt(1)
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
savevtu(args, grid, mp, pts_attr)