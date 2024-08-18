#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : 3d_retrogressive.jl                                                        |
|  Description: Retrogressive slope failure mechanisms                                     |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Note       : Friction algorithm is needed to apply on the bottom boundary, see          |
|               Wang, B., P. J. Vardon, and M. A. Hicks. "Investigation of retrogressive   |
|               and progressive slope failure mechanisms using the material point method." |
|               Computers and Geotechnics 78 (2016): 88-98.                                |
+==========================================================================================#

using HDF5
using KernelAbstractions
using MaterialPointSolver

rtsdir = joinpath(homedir(), "Workbench/outputs")

@kernel inbounds=true function testdpP!(
    mp      ::      KernelParticle3D{T1, T2},
    pts_attr::KernelParticleProperty{T1, T2}
) where {T1, T2}
    FNUM_13   = T2(1/3); FNUM_12 = T2(0.5); FNUM_S2 = T2(sqrt(2))
    FNUM_29   = T2(2/9); FNUM_6  = T2(6.0); FNUM_S3 = T2(sqrt(3))
    FNUM_3    = T2(3.0); FNUM_1  = T2(1.0); FNUM_H  = T2(5e4)
    INUM_2    = T1(2)  ; FNUM_0  = T2(0.0);
    FNUM_CMIN = T2(4e3)
    FNUM_CMAX = T2(2e4)
    ix = @index(Global)
    if ix≤mp.num
        σm  = mp.σm[ix]
        pid = pts_attr.layer[ix]
        ϕ   = pts_attr.ϕ[pid]
        ψ   = pts_attr.ψ[pid]
        σt  = pts_attr.σt[pid]
        G   = pts_attr.G[pid]
        Ks  = pts_attr.Ks[pid]
        c   = FNUM_CMAX-FNUM_H*mp.epII[ix]<FNUM_CMIN ? FNUM_CMIN : FNUM_CMAX-FNUM_H*mp.epII[ix]
        # drucker-prager
        τ  = sqrt(FNUM_12*(mp.sij[ix, 1]^INUM_2+mp.sij[ix, 2]^INUM_2+mp.sij[ix, 3]^INUM_2)+
                           mp.sij[ix, 4]^INUM_2+mp.sij[ix, 5]^INUM_2+mp.sij[ix, 6]^INUM_2)
        kϕ = (FNUM_6*c*cos(ϕ))/(FNUM_S3*(FNUM_3+sin(ϕ)))
        qϕ = (FNUM_6  *sin(ϕ))/(FNUM_S3*(FNUM_3+sin(ϕ)))
        qψ = (FNUM_6  *sin(ψ))/(FNUM_S3*(FNUM_3+sin(ψ)))
        σt = min(σt, kϕ/qϕ)
        αb = sqrt(FNUM_1+qϕ^INUM_2)-qϕ
        τb = kϕ-qϕ*σt
        fs = τ+qϕ*σm-kϕ # yield function considering shear failure
        ft = σm-σt      # yield function considering tensile failure
        BF = (τ-τb)-(αb*(σm-σt)) # BF is used to classify shear failure from tensile failure
        # determination of failure criteria
        ## shear failure correction
        if ((σm<σt)&&(fs>FNUM_0))||
           ((σm≥σt)&&(BF>FNUM_0))
            Δλs  = fs/(G+Ks*qϕ*qψ)
            tmp1 = σm-Ks*qψ*Δλs
            tmp2 = (kϕ-qϕ*tmp1)/τ
            mp.σij[ ix, 1]  = mp.sij[ix, 1]*tmp2+tmp1
            mp.σij[ ix, 2]  = mp.sij[ix, 2]*tmp2+tmp1
            mp.σij[ ix, 3]  = mp.sij[ix, 3]*tmp2+tmp1
            mp.σij[ ix, 4]  = mp.sij[ix, 4]*tmp2
            mp.σij[ ix, 5]  = mp.sij[ix, 5]*tmp2
            mp.σij[ ix, 6]  = mp.sij[ix, 6]*tmp2
            mp.epII[ix   ] += Δλs*sqrt(FNUM_13+FNUM_29*qψ^INUM_2)
            mp.epK[ ix   ] += Δλs*qψ
        end
        ## tensile failure correction
        if (σm≥σt)&&(BF≤FNUM_0)
            Δλt = ft/Ks
            mp.σij[ ix, 1]  = mp.sij[ix, 1]+σt
            mp.σij[ ix, 2]  = mp.sij[ix, 2]+σt
            mp.σij[ ix, 3]  = mp.sij[ix, 3]+σt
            mp.epII[ix   ] += Δλt*FNUM_13*FNUM_S2
            mp.epK[ ix   ] += Δλt
        end
        # update mean stress tensor
        σm = (mp.σij[ix, 1]+mp.σij[ix, 2]+mp.σij[ix, 3])*FNUM_13
        mp.σm[ix] = σm
        # update deviatoric stress tensor
        mp.sij[ix, 1] = mp.σij[ix, 1]-σm
        mp.sij[ix, 2] = mp.σij[ix, 2]-σm
        mp.sij[ix, 3] = mp.σij[ix, 3]-σm
        mp.sij[ix, 4] = mp.σij[ix, 4]
        mp.sij[ix, 5] = mp.σij[ix, 5]
        mp.sij[ix, 6] = mp.σij[ix, 6]
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
    liE!(dev)(ndrange=mp.num, mp, pts_attr)
    if Ti≥args.Te
        testdpP!(dev)(ndrange=mp.num, mp, pts_attr)
    end
    if args.MVL == true
        vollock1_OS!(dev)(ndrange=mp.num, grid, mp)
        vollock2_OS!(dev)(ndrange=mp.num, grid, mp)
    end
    return nothing
end

init_grid_space_x = 0.2
init_grid_space_y = 0.2
init_grid_space_z = 0.2
init_grid_range_x = [-1, 65]
init_grid_range_y = [-1, 6 ]
init_grid_range_z = [-1, 13]
init_mp_in_space  = 2
init_project_name = "3d_retrogressive"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :druckerprager
init_gravity      = -9.8
init_ζs           = 0.01
init_ρs           = 2700
init_ν            = 0.3
init_E            = 1e6
init_G            = init_E/(2*(1+  init_ν))
init_Ks           = init_E/(3*(1-2*init_ν))
init_T            = 20
init_Te           = 6
init_ΔT           = 0.5*min(init_grid_space_x, init_grid_space_y)/sqrt(init_E/init_ρs)
init_step         = floor(init_T/init_ΔT/20) |> Int64
init_step<10 ? init_step=1 : nothing
init_σt           = 0
init_ϕ1           = 20*π/180
init_ϕ2           = 14*π/180
init_NIC          = 64
init_basis        = :uGIMP
init_phase        = 1
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
    animation    = false,
    hdf5         = false,
    hdf5_step    = init_step,
    MVL          = true,
    device       = :CUDA,
    coupling     = :OS,
    basis        = init_basis
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
range_x = [0+grid.space_x/init_mp_in_space/2, 64-grid.space_x/init_mp_in_space/2]
range_y = [0+grid.space_y/init_mp_in_space/2,  5-grid.space_y/init_mp_in_space/2]
range_z = [0+grid.space_z/init_mp_in_space/2, 12-grid.space_z/init_mp_in_space/2]
space_x = grid.space_x/init_mp_in_space
space_y = grid.space_y/init_mp_in_space
space_z = grid.space_z/init_mp_in_space
vx      = range_x[1]:space_x:range_x[2] |> collect
vy      = range_y[1]:space_y:range_y[2] |> collect
vz      = range_z[1]:space_z:range_z[2] |> collect
m, n, o = length(vy), length(vx), length(vz)
vx      = reshape(vx, 1, n, 1)
vy      = reshape(vy, m, 1, 1)
vz      = reshape(vz, 1, 1, o)
om      = ones(Int, m)
on      = ones(Int, n)
oo      = ones(Int, o)
x_tmp   = vec(vx[om, :, oo])
y_tmp   = vec(vy[:, on, oo])
z_tmp   = vec(vz[om, on, :])
deleidx = findall(i->((27.8≤x_tmp[i]≤36.2)&&(z_tmp[i]≥39.8-x_tmp[i]))||
                     ((36.2<x_tmp[i]     )&&(z_tmp[i]≥3.6          )), 1:length(x_tmp))
deleteat!(x_tmp, deleidx)
deleteat!(y_tmp, deleidx)
deleteat!(z_tmp, deleidx)
mp_num = length(x_tmp)
mp_ρs  = ones(mp_num).*init_ρs
mp_pos = [x_tmp y_tmp z_tmp]
mp     = Particle3D{iInt, iFloat}(space_x=space_x, space_y=space_y, space_z=space_z, 
    pos=mp_pos, ρs=mp_ρs, NIC=init_NIC, phase=init_phase)

# particle property setup
mp_layer              = ones(mp_num)
layer2indx            = findall(i->z_tmp[i]≤3.6, 1:length(x_tmp))
mp_layer[layer2indx] .= 2
mp_ν     = [init_ν, init_ν]
mp_E     = [init_E, init_E]
mp_G     = [init_G, init_G]
mp_ϕ     = [init_ϕ1, init_ϕ2]
mp_Ks    = [init_Ks, init_Ks]
pts_attr = ParticleProperty{iInt, iFloat}(layer=mp_layer, ν=mp_ν, E=mp_E, G=mp_G, ϕ=mp_ϕ, 
    Ks=mp_Ks)

# boundary condition nodes index
vx_idx  = zeros(iInt, grid.node_num)
vy_idx  = zeros(iInt, grid.node_num) 
vz_idx  = zeros(iInt, grid.node_num)
tmp_idx = findall(i->(grid.pos[i, 1]≤0)||(grid.pos[i, 1]≥64), 1:grid.node_num)
tmp_idy = findall(i->(grid.pos[i, 2]≤0)||(grid.pos[i, 2]≥5 ), 1:grid.node_num)
tmp_idz = findall(i->(grid.pos[i, 3]≤0                     ), 1:grid.node_num)
vx_idx[tmp_idx] .= 1
vy_idx[tmp_idy] .= 1
vz_idx[tmp_idz] .= 1
bc = VBoundary3D{iInt, iFloat}(
    Vx_s_Idx = vx_idx,
    Vx_s_Val = zeros(grid.node_num),
    Vy_s_Idx = vy_idx,
    Vy_s_Val = zeros(grid.node_num),
    Vz_s_Idx = vz_idx,
    Vz_s_Val = zeros(grid.node_num),
)

# MPM solver
materialpointsolver!(args, grid, mp, pts_attr, bc, workflow=testprocedure!)
savevtu(args, grid, mp, pts_attr)