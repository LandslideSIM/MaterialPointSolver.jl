#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : 3d_water.jl                                                                |
|  Description: Please run this file in VSCode with Julia ENV                              |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Test Case  : 2D water column collapse in Tait equation.                                 |
|  Reference  : [1] Weakly compressible SPH for free surface flows, Markus Becker Matthias | 
|                   Teschner, ACM SIGGRAPH Symposium on Computer Animation (2007),         |
|               [2] Towards a predictive multi-phase model for alpine mass movements and   |
|                   process cascades, A. Cicoira, L. Blatny, X. Li, B. Trottet, J. Gaume,  |
|                   Engineering Geology, 10.1016/j.enggeo.2022.106866.                     |
|  Notes      : [1] The parameter of "B" is very sensitive to the result, maybe break the  |
|                   simulation.                                                            |
|               [2] Don't use "uGIMP" basis for this case, it will break the simulation.   |
|               [3] Remeber to comment the lines about Δϵij[:, 4]/ϵij[:, 4] (2D) and       |
|                   Δϵij[:, [4,5,6]]/ϵij[:, [4,5,6]] (3D) in OS.jl and OS_d.jl.            |
|               [4] Comment CFL part in OS.jl and OS_d.jl.                                 |
|               [5] B is 1.119e7, γ is 7, simulation time is 4s.                           |
+==========================================================================================#

using KernelAbstractions
using MaterialPointSolver

rtsdir = joinpath(homedir(), "Workbench/outputs")

@kernel inbounds=true function tmp!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2}
) where {T1, T2}
    FNUM_0 = T2(0.0); FNUM_1 = T2(1.0)
    ix = @index(Global)
    if ix≤mp.num
        dF1 = dF2 = dF3 = dF4 = dF5 = dF6 = dF7 = dF8 = dF9 = FNUM_0
        for iy in Int32(1):Int32(mp.NIC)
            p2n = mp.p2n[ix, iy]
            ∂Nx = mp.∂Nx[ix, iy]; ds1 = grid.Δd_s[p2n, 1]
            ∂Ny = mp.∂Ny[ix, iy]; ds2 = grid.Δd_s[p2n, 2]
            ∂Nz = mp.∂Nz[ix, iy]; ds3 = grid.Δd_s[p2n, 3]
            # compute solid incremental deformation gradient
            dF1 += ds1*∂Nx; dF2 += ds1*∂Ny; dF3 += ds1*∂Nz
            dF4 += ds2*∂Nx; dF5 += ds2*∂Ny; dF6 += ds2*∂Nz
            dF7 += ds3*∂Nx; dF8 += ds3*∂Ny; dF9 += ds3*∂Nz
        end
        mp.∂Fs[ix, 1] = dF1; mp.∂Fs[ix, 2] = dF2; mp.∂Fs[ix, 3] = dF3
        mp.∂Fs[ix, 4] = dF4; mp.∂Fs[ix, 5] = dF5; mp.∂Fs[ix, 6] = dF6
        mp.∂Fs[ix, 7] = dF7; mp.∂Fs[ix, 8] = dF8; mp.∂Fs[ix, 9] = dF9
        # compute strain increment
        mp.Δϵij_s[ix, 1] = dF1
        mp.Δϵij_s[ix, 2] = dF5
        mp.Δϵij_s[ix, 3] = dF9
        # update strain tensor
        mp.ϵij_s[ix, 1] += dF1
        mp.ϵij_s[ix, 2] += dF5
        mp.ϵij_s[ix, 3] += dF9
        mp.ϵij_s[ix, 4] += dF2+dF4
        mp.ϵij_s[ix, 5] += dF6+dF8
        mp.ϵij_s[ix, 6] += dF3+dF7
        # deformation gradient matrix
        F1 = mp.F[ix, 1]; F2 = mp.F[ix, 2]; F3 = mp.F[ix, 3]
        F4 = mp.F[ix, 4]; F5 = mp.F[ix, 5]; F6 = mp.F[ix, 6]
        F7 = mp.F[ix, 7]; F8 = mp.F[ix, 8]; F9 = mp.F[ix, 9]        
        mp.F[ix, 1] = (dF1+FNUM_1)*F1+dF2*F4+dF3*F7
        mp.F[ix, 2] = (dF1+FNUM_1)*F2+dF2*F5+dF3*F8
        mp.F[ix, 3] = (dF1+FNUM_1)*F3+dF2*F6+dF3*F9
        mp.F[ix, 4] = (dF5+FNUM_1)*F4+dF4*F1+dF6*F7
        mp.F[ix, 5] = (dF5+FNUM_1)*F5+dF4*F2+dF6*F8
        mp.F[ix, 6] = (dF5+FNUM_1)*F6+dF4*F3+dF6*F9
        mp.F[ix, 7] = (dF9+FNUM_1)*F7+dF8*F4+dF7*F1
        mp.F[ix, 8] = (dF9+FNUM_1)*F8+dF8*F5+dF7*F2
        mp.F[ix, 9] = (dF9+FNUM_1)*F9+dF8*F6+dF7*F3
        # update jacobian value and particle volume
        mp.J[ix] = mp.F[ix, 1]*mp.F[ix, 5]*mp.F[ix, 9]+mp.F[ix, 2]*mp.F[ix, 6]*mp.F[ix, 7]+
                   mp.F[ix, 3]*mp.F[ix, 4]*mp.F[ix, 8]-mp.F[ix, 7]*mp.F[ix, 5]*mp.F[ix, 3]-
                   mp.F[ix, 8]*mp.F[ix, 6]*mp.F[ix, 1]-mp.F[ix, 9]*mp.F[ix, 4]*mp.F[ix, 2]
        mp.vol[ix] = mp.J[ix]*mp.vol_init[ix]
        mp.ρs[ ix] = mp.ρs_init[ix]/mp.J[ix]
    end
end

@kernel inbounds=true function testtwP!(
    mp::KernelParticle3D{T1, T2}
) where {T1, T2}
    FNUM_1 = T2(1)
    ix = @index(Global)
    if ix ≤ mp.num
        pid = mp.layer[ix]
        P   = mp.B[pid]*((mp.ρs[ix]/mp.ρs_init[ix])^mp.γ[pid]-FNUM_1)
        mp.σij[ix, 1] = -P
        mp.σij[ix, 2] = -P
        mp.σij[ix, 3] = -P
        # update mean stress tensor
        mp.σm[ix] = P
    end
end

function testprocedure!(args::MODELARGS, 
                        grid::GRID, 
                        mp  ::PARTICLE, 
                        bc  ::BOUNDARY,
                        ΔT  ::T2,
                        Ti  ::T2,
                            ::Val{:OS},
                            ::Val{:MUSL}) where {T2}
    Ti < args.Te ? G = args.gravity / args.Te * Ti : G = args.gravity
    dev = getBackend(args)
    resetgridstatus_OS!(dev)(ndrange=grid.node_num, grid)
    resetmpstatus_OS!(dev)(ndrange=mp.num, grid, mp, Val(args.basis))
    P2G_OS!(dev)(ndrange=mp.num, grid, mp, G)
    solvegrid_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT, args.ζs)
    doublemapping1_OS!(dev)(ndrange=mp.num, grid, mp, ΔT, args.FLIP, args.PIC)
    doublemapping2_OS!(dev)(ndrange=mp.num, grid, mp)
    doublemapping3_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT)
    tmp!(dev)(ndrange=mp.num, grid, mp)
    if args.constitutive==:hyperelastic
        hyE!(dev)(ndrange=mp.num, mp)
    elseif args.constitutive==:linearelastic
        liE!(dev)(ndrange=mp.num, mp)
    elseif args.constitutive==:druckerprager
        liE!(dev)(ndrange=mp.num, mp)
        if Ti≥args.Te
            dpP!(dev)(ndrange=mp.num, mp)
        end
    elseif args.constitutive==:mohrcoulomb
        liE!(dev)(ndrange=mp.num, mp)
        if Ti≥args.Te
            mcP!(dev)(ndrange=mp.num, mp)
        end
    elseif args.constitutive==:taitwater
        testtwP!(dev)(ndrange=mp.num, mp)
    end
    if args.MVL == true
        vollock1_OS!(dev)(ndrange=mp.num, grid, mp)
        vollock2_OS!(dev)(ndrange=mp.num, grid, mp)
    end
    return nothing
end

init_grid_space_x = 0.1
init_grid_space_y = 0.1
init_grid_space_z = 0.1
init_grid_range_x = [-1, 11]
init_grid_range_y = [-1,  5]
init_grid_range_z = [-1, 21]
init_mp_in_space  = 2
init_project_name = "3d_water"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :taitwater
init_gravity      = -9.8
init_ζs           = 0
init_ρs           = 1e3
init_ν            = 0.48
init_Ks           = 2.15e9
init_E            = init_Ks*(3*(1-2*init_ν))
init_G            = init_E /(2*(1+  init_ν))
init_T            = 4
init_B            = 1.119e7
init_γ            = 7
init_Te           = 0
init_ΔT           = 0.05*min(init_grid_space_x, init_grid_space_y)/sqrt(init_E/init_ρs)
init_step         = floor(init_T/init_ΔT/100) |> Int64
init_step<10 ? init_step=1 : nothing
init_phase        = 1
init_NIC          = 8
init_basis        = :linear
iInt              = Int32
iFloat            = Float32

# parameters setup
args = Args3D{iInt, iFloat}(
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
    basis        = init_basis,
    αT           = 0.05
)

# background grid setup
grid = Grid3D{iInt, iFloat}(
    range_x1 = init_grid_range_x[1],
    range_x2 = init_grid_range_x[2],
    range_y1 = init_grid_range_y[1],
    range_y2 = init_grid_range_y[2],
    range_z1 = init_grid_range_z[1],
    range_z2 = init_grid_range_z[2],
    space_x  = init_grid_space_x,
    space_y  = init_grid_space_y,
    space_z  = init_grid_space_z,
    phase    = init_phase,
    NIC      = init_NIC
)

# material points setup
range_x  = [0+grid.space_x/init_mp_in_space/2, 10-grid.space_x/init_mp_in_space/2]
range_y  = [0+grid.space_y/init_mp_in_space/2,  4-grid.space_y/init_mp_in_space/2]
range_z  = [0+grid.space_y/init_mp_in_space/2,  4-grid.space_y/init_mp_in_space/2]
space_x  = grid.space_x/init_mp_in_space
space_y  = grid.space_y/init_mp_in_space
space_z  = grid.space_z/init_mp_in_space
vx       = range_x[1]:space_x:range_x[2] |> collect
vy       = range_y[1]:space_y:range_y[2] |> collect
vz       = range_z[1]:space_z:range_z[2] |> collect
m, n, o  = length(vy), length(vx), length(vz)
vx       = reshape(vx, 1, n, 1)
vy       = reshape(vy, m, 1, 1)
vz       = reshape(vz, 1, 1, o)
om       = ones(Int, m)
on       = ones(Int, n)
oo       = ones(Int, o)
x_tmp    = vec(vx[om, :, oo])
y_tmp    = vec(vy[:, on, oo])
z_tmp    = vec(vz[om, on, :])
tmp_del  = findall(i-> x_tmp[i]≥2 && z_tmp[i]≥2, 1:length(x_tmp))
deleteat!(x_tmp, tmp_del)
deleteat!(y_tmp, tmp_del)
deleteat!(z_tmp, tmp_del)
mp_num   = length(x_tmp)
mp_ρs    = ones(mp_num).*init_ρs
mp_layer = ones(mp_num)
mp_pos   = [x_tmp y_tmp z_tmp]
mp_ν     = [init_ν]
mp_E     = [init_E]
mp_G     = [init_G]
mp_Ks    = [init_Ks]
mp_B     = [init_B]
mp_γ     = [init_γ]
mp       = Particle3D{iInt, iFloat}(space_x=space_x, space_y=space_y, space_z=space_z, 
    pos=mp_pos, ν=mp_ν, E=mp_E, G=mp_G, ρs=mp_ρs, Ks=mp_Ks, layer=mp_layer, NIC=init_NIC,
    phase=init_phase, B=mp_B, γ=mp_γ)

# boundary condition nodes index
vx_idx  = zeros(iInt, grid.node_num)
vy_idx  = zeros(iInt, grid.node_num) 
vz_idx  = zeros(iInt, grid.node_num)
tmp_idx = findall(i->((grid.pos[i, 1]≤0)||(grid.pos[i, 1]≥10)||(grid.pos[i, 3]≤0)), 1:grid.node_num)
tmp_idy = findall(i->((grid.pos[i, 2]≤0)||(grid.pos[i, 2]≥ 4)||(grid.pos[i, 3]≤0)), 1:grid.node_num)
tmp_idz = findall(i->((grid.pos[i, 3]≤0)||(grid.pos[i, 3]≥20)), 1:grid.node_num)
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
materialpointsolver!(args, grid, mp, bc, workflow=testprocedure!)