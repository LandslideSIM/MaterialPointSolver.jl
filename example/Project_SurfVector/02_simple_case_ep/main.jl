#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : main.jl                                                                    |
|  Description: To get the velocity vector on the slide surface (with ep surface)          |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
+==========================================================================================#

using MaterialPointSolver
using KernelAbstractions
using CUDA

rtsdir = joinpath(homedir(), "Workbench/outputs")

@kernel inbounds=true function newmodel!(mp::KernelParticle3D{T1, T2}) where {T1, T2}
    FNUM_12 = T2(0.5); FNUM_2 = T2(2.0); FNUM_S2 = T2(sqrt(2)); INUM_2  = T1(2)
    FNUM_43 = T2(4/3); FNUM_1 = T2(1.0); FNUM_S3 = T2(sqrt(3))
    FNUM_23 = T2(2/3); FNUM_3 = T2(3.0); FNUM_29 = T2(2/9)
    FNUM_13 = T2(1/3); FNUM_0 = T2(0.0); FNUM_6  = T2(6.0)
    ix = @index(Global)
    if ix≤mp.num
        pid = mp.layer[ix]
        Ks  = mp.Ks[pid]
        G   = mp.G[pid]
        # spin tensor
        ωxy = FNUM_12*(mp.∂Fs[ix, 4]-mp.∂Fs[ix, 2]) 
        ωyz = FNUM_12*(mp.∂Fs[ix, 8]-mp.∂Fs[ix, 6])
        ωxz = FNUM_12*(mp.∂Fs[ix, 7]-mp.∂Fs[ix, 3]) 
        # objective stress
        σij1 = mp.σij[ix, 1]
        σij2 = mp.σij[ix, 2]
        σij3 = mp.σij[ix, 3]
        σij4 = mp.σij[ix, 4]
        σij5 = mp.σij[ix, 5]
        σij6 = mp.σij[ix, 6]
        mp.σij[ix, 1] +=  FNUM_2*(σij4*ωxy+σij6*ωxz)
        mp.σij[ix, 2] += -FNUM_2*(σij4*ωxy-σij5*ωyz)
        mp.σij[ix, 3] += -FNUM_2*(σij6*ωxz+σij5*ωyz)
        mp.σij[ix, 4] +=  ωxy*(σij2-σij1)+ωxz*σij5+ωyz*σij6
        mp.σij[ix, 5] +=  ωyz*(σij3-σij2)-ωxz*σij4-ωxy*σij6
        mp.σij[ix, 6] +=  ωxz*(σij3-σij1)+ωxy*σij5-ωyz*σij4
        # linear elastic
        Dt = Ks+FNUM_43*G
        Dd = Ks-FNUM_23*G
        mp.σij[ix, 1] += Dt*mp.Δϵij_s[ix, 1]+Dd*mp.Δϵij_s[ix, 2]+Dd*mp.Δϵij_s[ix, 3]
        mp.σij[ix, 2] += Dd*mp.Δϵij_s[ix, 1]+Dt*mp.Δϵij_s[ix, 2]+Dd*mp.Δϵij_s[ix, 3]
        mp.σij[ix, 3] += Dd*mp.Δϵij_s[ix, 1]+Dd*mp.Δϵij_s[ix, 2]+Dt*mp.Δϵij_s[ix, 3]
        mp.σij[ix, 4] += G *mp.Δϵij_s[ix, 4]
        mp.σij[ix, 5] += G *mp.Δϵij_s[ix, 5]
        mp.σij[ix, 6] += G *mp.Δϵij_s[ix, 6]
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
        # plasticity
        if mp.layer[ix]==INUM_2
            pid = mp.layer[ix]
            σm  = mp.σm[ix]
            c   = mp.c[pid]
            ϕ   = mp.ϕ[pid]
            ψ   = mp.ψ[pid]
            σt  = mp.σt[pid]
            G   = mp.G[pid]
            Ks  = mp.Ks[pid]
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

    mp.pos[mp.layer.==1, :] .= mp.init[mp.layer.==1, :]

    doublemapping2_OS!(dev)(ndrange=mp.num, grid, mp)
    doublemapping3_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT)
    G2P_OS!(dev)(ndrange=mp.num, grid, mp)
    newmodel!(dev)(ndrange=mp.num, mp)
    if args.MVL == true
        vollock1_OS!(dev)(ndrange=mp.num, grid, mp)
        vollock2_OS!(dev)(ndrange=mp.num, grid, mp)
    end
    return nothing
end

init_grid_space_x = 1
init_grid_space_y = 1
init_grid_space_z = 1
init_grid_range_x = [-3, 113]
init_grid_range_y = [-3,  23]
init_grid_range_z = [ 7,  38]
init_mp_in_space  = 2
init_project_name = "3d_slumping_vector_ep"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :linearelastic
init_gravity      = -9.8
init_ζs           = 0
init_ρs           = 2100
init_ν            = 0.3
init_E            = 70e6
init_G            = init_E/(2*(1+  init_ν))
init_Ks           = init_E/(3*(1-2*init_ν))
init_T            = 7
init_Te           = 0
init_ΔT           = 0.5*min(init_grid_space_x, init_grid_space_y)/sqrt(init_E/init_ρs)
init_step         = floor(init_T/init_ΔT/100) |> Int64
init_step<10 ? init_step=1 : nothing
init_σt           = 27.48e3
init_ϕ            = 20*π/180
init_c            = 1e4
init_ψ            = 0*π/180
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
    animation    = true,
    hdf5         = true,
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
range_x    = [ 0+grid.space_x/init_mp_in_space/2, 110-grid.space_x/init_mp_in_space/2]
range_y    = [ 0+grid.space_y/init_mp_in_space/2,  20-grid.space_y/init_mp_in_space/2]
range_z    = [10+grid.space_z/init_mp_in_space/2,  35-grid.space_z/init_mp_in_space/2]
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
deleidx    = findall(i->(   x_tmp[i]≥50&&z_tmp[i]≥15         )||
                        (30≤x_tmp[i]≤50&&z_tmp[i]≥65-x_tmp[i]), 1:length(x_tmp))
deleteat!(x_tmp, deleidx)
deleteat!(y_tmp, deleidx)
deleteat!(z_tmp, deleidx)
mp_num   = length(x_tmp)
mp_ρs    = ones(mp_num).*init_ρs
mp_layer = ones(mp_num)
mp_pos   = [x_tmp y_tmp z_tmp]
a        = 3
b        = 2.8
x        = LinRange(30, 55, 100)
y        = LinRange( 0, 20, 100)
o        = (48, 10, 18)
softid = findall(i->(mp_pos[i, 3]≥((mp_pos[i, 1]-o[1])^2/a^2+(mp_pos[i, 2]-o[2])^2/b^2)/2+o[3])&&
                    (x[1]≤mp_pos[i, 1]≤x[end])&&
                    (y[1]≤mp_pos[i, 2]≤y[end]), 1:mp_num)

mp_layer[softid] .= 2
mp_ν     = [init_ν , init_ν ]
mp_E     = [init_E , init_E ]
mp_G     = [init_G , init_G ]
mp_σt    = [init_σt, init_σt]
mp_ϕ     = [init_ϕ , 10*π/180 ]
mp_c     = [init_c , 1e3 ]
mp_ψ     = [init_ψ , init_ψ ]
mp_Ks    = [init_Ks, init_Ks]
mp       = Particle3D{iInt, iFloat}(space_x=space_x, space_y=space_y, space_z=space_z, 
    pos=mp_pos, ν=mp_ν, E=mp_E, G=mp_G, σt=mp_σt, ϕ=mp_ϕ, c=mp_c, ψ=mp_ψ, ρs=mp_ρs, 
    Ks=mp_Ks, layer=mp_layer, NIC=init_NIC, phase=init_phase)

# boundary condition nodes index
vx_idx  = zeros(iInt, grid.node_num)
vy_idx  = zeros(iInt, grid.node_num) 
vz_idx  = zeros(iInt, grid.node_num)
tmp_idx = findall(i->((grid.pos[i, 1]≤0)||(grid.pos[i, 1]≥110)||
                      (grid.pos[i, 3]≤10)), 1:grid.node_num)
tmp_idy = findall(i->((grid.pos[i, 2]≤0)||(grid.pos[i, 2]≥20)||
                      (grid.pos[i, 3]≤10)), 1:grid.node_num)
tmp_idz = findall(i->( grid.pos[i, 3]≤10 ), 1:grid.node_num)
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
materialpointsolver!(args, grid, mp, bc, workflow=testprocedure!)
savevtu(args, grid, mp)

# post processing
using GLMakie
begin
    figfont = MaterialPointSolver.fontcmu
    fig = Figure(fonts=(; regular=figfont, bold=figfont))
    ax = Axis3(fig[1, 1], aspect=:data, xticks=(0:10:110), azimuth=-0.5π)
    scatter!(ax, mp.init, alpha=0.1, markersize=1)

    a = 3
    b = 2.8
    x = LinRange(30, 55, 100)
    y = LinRange( 0, 20, 100)
    o = (48, 10, 18)

    z = [((xt-o[1])^2/a^2 + (yt-o[2])^2/b^2)/2+o[3] for xt in x, yt in y]
    surface!(ax, x, y, z, alpha=0.5)
    redid = findall(i->(mp.init[i, 3]≥((mp.init[i, 1]-o[1])^2/a^2+(mp.init[i, 2]-o[2])^2/b^2)/2+o[3])&&
                       (x[1]≤mp.init[i, 1]≤x[end])&&
                       (y[1]≤mp.init[i, 2]≤y[end]), 1:mp.num)

    scatter!(ax, mp.init[redid, :], color=:red)
    display(fig)
end