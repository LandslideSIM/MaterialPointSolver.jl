#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : 2d_terzaghi.jl                                                             |
|  Description: Please run this file in VSCode with Julia ENV                              |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Test Case  : Terzaghi consolidation test                                                |
+==========================================================================================#

using MaterialPointSolver
using HDF5
using CairoMakie
using DelimitedFiles
using KernelAbstractions
using .Threads
initial_threads = Threads.nthreads()
Threads.nthreads() = 1

rtsdir = joinpath(homedir(), "Workbench/outputs")
assetsdir = MaterialPointSolver.assets_dir

@kernel inbounds=true function TS_resetgridstatus!(
    grid::KernelGrid2D{T1, T2}
) where {T1, T2}
    FNUM_0 = T2(0.0)
    ix = @index(Global)
    if ix≤grid.node_num
        if ix≤grid.cell_num
            grid.σm[ ix] = FNUM_0
            grid.vol[ix] = FNUM_0
        end
        grid.Ms[ix   ] = FNUM_0
        grid.Mi[ix   ] = FNUM_0
        grid.Mw[ix   ] = FNUM_0
        grid.Ps[ix, 1] = FNUM_0
        grid.Ps[ix, 2] = FNUM_0
        grid.Pw[ix, 1] = FNUM_0
        grid.Pw[ix, 2] = FNUM_0
        grid.Fs[ix, 1] = FNUM_0
        grid.Fs[ix, 2] = FNUM_0
        grid.Fw[ix, 1] = FNUM_0
        grid.Fw[ix, 2] = FNUM_0
    end
end

@kernel inbounds=true function TS_resetmpstatus!(
    grid::    KernelGrid2D{T1, T2},
    mp  ::KernelParticle2D{T1, T2},
        ::Val{:linear}
) where {T1, T2}
    FNUM_1 = T2(1.0)
    ix = @index(Global)
    if ix≤mp.num
        # update mass and momentum
        mp.Ms[ix] = mp.vol[ix]*mp.ρs[ix]
        mp.Mw[ix] = mp.vol[ix]*mp.ρw[ix]
        mp.Mi[ix] = mp.vol[ix]*((FNUM_1-mp.porosity[ix])*mp.ρs[ix]+
                                        mp.porosity[ix] *mp.ρw[ix])
        mp.Ps[ix, 1] = mp.Ms[ix]*mp.Vs[ix, 1]*(FNUM_1-mp.porosity[ix])
        mp.Ps[ix, 2] = mp.Ms[ix]*mp.Vs[ix, 2]*(FNUM_1-mp.porosity[ix])
        mp.Pw[ix, 1] = mp.Mw[ix]*mp.Vw[ix, 1]*        mp.porosity[ix]
        mp.Pw[ix, 2] = mp.Mw[ix]*mp.Vw[ix, 2]*        mp.porosity[ix]
        # compute particle to cell and particle to node index
        mp.p2c[ix] = cld(mp.pos[ix, 2]-grid.range_y1, grid.space_y)+
                     fld(mp.pos[ix, 1]-grid.range_x1, grid.space_x)*grid.cell_num_y
        for iy in Int32(1):Int32(mp.NIC)
            # p2n index
            mp.p2n[ix, iy] = grid.c2n[mp.p2c[ix], iy]
            # compute distance between particle and related nodes
            p2n = mp.p2n[ix, iy]
            Δdx = mp.pos[ix, 1]-grid.pos[p2n, 1]
            Δdy = mp.pos[ix, 2]-grid.pos[p2n, 2]
            # compute basis function
            Nx, dNx = linearBasis(Δdx, grid.space_x)
            Ny, dNy = linearBasis(Δdy, grid.space_y)
            mp.Ni[ ix, iy] =  Nx*Ny
            mp.∂Nx[ix, iy] = dNx*Ny # x-gradient shape function
            mp.∂Ny[ix, iy] = dNy*Nx # y-gradient shape function
        end
    end
end

@kernel inbounds=true function TS_resetmpstatus!(
    grid::    KernelGrid2D{T1, T2},
    mp  ::KernelParticle2D{T1, T2},
        ::Val{:uGIMP}
) where {T1, T2}
    FNUM_1 = T2(1.0)
    ix = @index(Global)
    if ix≤mp.num
        # update mass and momentum
        mp.Ms[ix] = mp.vol[ix]*mp.ρs[ix]
        mp.Mw[ix] = mp.vol[ix]*mp.ρw[ix]
        mp.Mi[ix] = mp.vol[ix]*((FNUM_1-mp.porosity[ix])*mp.ρs[ix]+
                                        mp.porosity[ix] *mp.ρw[ix])
        mp.Ps[ix, 1] = mp.Ms[ix]*mp.Vs[ix, 1]*(FNUM_1-mp.porosity[ix])
        mp.Ps[ix, 2] = mp.Ms[ix]*mp.Vs[ix, 2]*(FNUM_1-mp.porosity[ix])
        mp.Pw[ix, 1] = mp.Mw[ix]*mp.Vw[ix, 1]*        mp.porosity[ix]
        mp.Pw[ix, 2] = mp.Mw[ix]*mp.Vw[ix, 2]*        mp.porosity[ix]
        # compute particle to cell and particle to node index
        mp.p2c[ix] = cld(mp.pos[ix, 2]-grid.range_y1, grid.space_y)+
                     fld(mp.pos[ix, 1]-grid.range_x1, grid.space_x)*grid.cell_num_y
        for iy in Int32(1):Int32(mp.NIC)
            # p2n index
            mp.p2n[ix, iy] = grid.c2n[mp.p2c[ix], iy]
            # compute distance between particle and related nodes
            p2n = mp.p2n[ix, iy]
            Δdx = mp.pos[ix, 1]-grid.pos[p2n, 1]
            Δdy = mp.pos[ix, 2]-grid.pos[p2n, 2]
            # compute basis function
            Nx, dNx = uGIMPbasis(Δdx, grid.space_x, mp.space_x)
            Ny, dNy = uGIMPbasis(Δdy, grid.space_y, mp.space_y)
            mp.Ni[ ix, iy] =  Nx*Ny
            mp.∂Nx[ix, iy] = dNx*Ny # x-gradient shape function
            mp.∂Ny[ix, iy] = dNy*Nx # y-gradient shape function
        end
    end
end

@kernel inbounds=true function TS_P2G!(
    grid   ::    KernelGrid2D{T1, T2},
    mp     ::KernelParticle2D{T1, T2},
    gravity::T2
) where {T1, T2}
    FNUM_1 = T2(1.0)
    ix = @index(Global)
    if ix≤mp.num
        for iy in Int32(1):Int32(mp.NIC)
            Ni  = mp.Ni[ ix, iy]
            ∂Nx = mp.∂Nx[ix, iy]
            ∂Ny = mp.∂Ny[ix, iy]
            p2n = mp.p2n[ix, iy]
            vol = mp.vol[ix    ]
            # compute nodal mass
            #=xxx=#grid.Ms[p2n] += Ni*mp.Ms[ix]*(FNUM_1-mp.porosity[ix])
            #=xxx=#grid.Mi[p2n] += Ni*mp.Mw[ix]*        mp.porosity[ix]
            #=xxx=#grid.Mw[p2n] += Ni*mp.Mw[ix]
            # compute nodal momentum
            #=xxx=#grid.Ps[p2n, 1] += Ni*mp.Ps[ix, 1]
            #=xxx=#grid.Ps[p2n, 2] += Ni*mp.Ps[ix, 2]
            #=xxx=#grid.Pw[p2n, 1] += Ni*mp.Pw[ix, 1]
            #=xxx=#grid.Pw[p2n, 2] += Ni*mp.Pw[ix, 2]
            # compute nodal total force
            #=xxx=#grid.Fw[p2n, 1] += -vol*(∂Nx*mp.σw[ix])
            #=xxx=#grid.Fw[p2n, 2] += -vol*(∂Ny*mp.σw[ix])
            #=xxx=#grid.Fs[p2n, 1] += -vol*(∂Nx*(mp.σij[ix, 1]+mp.σw[ix])+∂Ny*mp.σij[ix, 4])
            #=xxx=#grid.Fs[p2n, 2] += -vol*(∂Ny*(mp.σij[ix, 2]+mp.σw[ix])+∂Nx*mp.σij[ix, 4])
        end
    end
end

@kernel inbounds=true function TS_solvegrid!(
    grid::    KernelGrid2D{T1, T2},
    bc  ::KernelBoundary2D{T1, T2},
    ζs  ::T2,
    ζw  ::T2
) where {T1, T2}
    INUM_0 = T1(0); INUM_2 = T1(2); FNUM_0 = T2(0.0); FNUM_1 = T2(1.0)
    ix = @index(Global)
    if ix≤grid.node_num
        iszero(grid.Ms[ix]) ? Ms_denom=FNUM_0 : Ms_denom=FNUM_1/grid.Ms[ix]
        iszero(grid.Mw[ix]) ? Mw_denom=FNUM_0 : Mw_denom=FNUM_1/grid.Mw[ix]
        # compute nodal velocity
        grid.Vs[ix, 1] = grid.Ps[ix, 1]*Ms_denom
        grid.Vs[ix, 2] = grid.Ps[ix, 2]*Ms_denom
        grid.Vw[ix, 1] = grid.Pw[ix, 1]*Mw_denom
        grid.Vw[ix, 2] = grid.Pw[ix, 2]*Mw_denom
        # compute damping force
        tmpw = sqrt(grid.Fw[ix, 1]^INUM_2+grid.Fw[ix, 2]^INUM_2)
        damp_w_x = ζw*tmpw*sign(grid.Vw[ix, 1])
        damp_w_y = ζw*tmpw*sign(grid.Vw[ix, 2])
        tmps = sqrt((grid.Fs[ix, 1]-grid.Fw[ix, 1])^INUM_2+
                    (grid.Fs[ix, 2]-grid.Fs[ix, 2])^INUM_2)
        damp_s_x = ζs*tmps*sign(grid.Vs[ix, 1])
        damp_s_y = ζs*tmps*sign(grid.Vs[ix, 2])
        # compute nodal acceleration
        grid.a_w[ix, 1] = Mw_denom*(grid.Fw[ix, 1]-damp_w_x)
        grid.a_w[ix, 2] = Mw_denom*(grid.Fw[ix, 2]-damp_w_y)
        grid.a_s[ix, 1] = Ms_denom*(grid.Fs[ix, 1]-damp_s_x-damp_w_x-grid.Mi[ix]*grid.a_w[ix, 1])
        grid.a_s[ix, 2] = Ms_denom*(grid.Fs[ix, 2]-damp_s_y-damp_w_y-grid.Mi[ix]*grid.a_w[ix, 2])
        # boundary condition
        bc.Vx_s_Idx[ix]≠INUM_0 ? grid.a_s[ix, 1]=bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix]≠INUM_0 ? grid.a_s[ix, 2]=bc.Vy_s_Val[ix] : nothing
        bc.Vx_w_Idx[ix]≠INUM_0 ? grid.a_w[ix, 1]=bc.Vx_w_Val[ix] : nothing
        bc.Vy_w_Idx[ix]≠INUM_0 ? grid.a_w[ix, 2]=bc.Vy_w_Val[ix] : nothing
        # reset grid momentum
        grid.Ps[ix, 1] = FNUM_0
        grid.Ps[ix, 2] = FNUM_0
        grid.Pw[ix, 1] = FNUM_0
        grid.Pw[ix, 2] = FNUM_0
    end
end

@kernel inbounds=true function TS_doublemapping1!(
    grid::    KernelGrid2D{T1, T2},
    mp  ::KernelParticle2D{T1, T2},
    ΔT  ::T2
) where {T1, T2}
    FNUM_0 = T2(0.0)
    ix = @index(Global)
    if ix≤mp.num
        # update particle velocity
        tmp_vx_s = tmp_vy_s = tmp_vx_w = tmp_vy_w = FNUM_0
        for iy in Int32(1):Int32(mp.NIC)
            Ni  = mp.Ni[ ix, iy]
            p2n = mp.p2n[ix, iy]
            tmp_vx_s += Ni*grid.a_s[p2n, 1]
            tmp_vy_s += Ni*grid.a_s[p2n, 2]
            tmp_vx_w += Ni*grid.a_w[p2n, 1]
            tmp_vy_w += Ni*grid.a_w[p2n, 2]
        end
        mp.Vs[ix, 1] += tmp_vx_s*ΔT
        mp.Vs[ix, 2] += tmp_vy_s*ΔT
        mp.Vw[ix, 1] += tmp_vx_w*ΔT
        mp.Vw[ix, 2] += tmp_vy_w*ΔT
    end
end

@kernel inbounds=true function TS_doublemapping2!(
    grid::    KernelGrid2D{T1, T2},
    mp  ::KernelParticle2D{T1, T2}
) where {T1, T2}
    FNUM_1 = T2(1.0)
    ix = @index(Global)
    if ix≤mp.num
        for iy in Int32(1):Int32(mp.NIC)
            Ni  = mp.Ni[ ix, iy]
            p2n = mp.p2n[ix, iy]
            # compute nodal momentum
            #=xxx=#grid.Ps[p2n, 1] += Ni*mp.Ms[ix]*(FNUM_1-mp.porosity[ix])*mp.Vs[ix, 1]
            #=xxx=#grid.Ps[p2n, 2] += Ni*mp.Ms[ix]*(FNUM_1-mp.porosity[ix])*mp.Vs[ix, 2]
            #=xxx=#grid.Pw[p2n, 1] += Ni*mp.Mw[ix]*        mp.porosity[ix] *mp.Vw[ix, 1]
            #=xxx=#grid.Pw[p2n, 2] += Ni*mp.Mw[ix]*        mp.porosity[ix] *mp.Vw[ix, 2]
        end
    end
end

@kernel inbounds=true function TS_doublemapping3!(
    grid::    KernelGrid2D{T1, T2},
    bc  ::KernelBoundary2D{T1, T2},
    ΔT  ::T2
) where {T1, T2}
    FNUM_0 = T2(0.0); FNUM_1 = T2(1.0); INUM_0 = T1(0)
    ix = @index(Global)
    if ix≤grid.node_num
        iszero(grid.Ms[ix]) ? Ms_denom=FNUM_0 : Ms_denom=FNUM_1/grid.Ms[ix]
        iszero(grid.Mw[ix]) ? Mw_denom=FNUM_0 : Mw_denom=FNUM_1/grid.Mw[ix]
        # compute nodal velocities
        grid.Vs[ix, 1] = grid.Ps[ix, 1]*Ms_denom
        grid.Vs[ix, 2] = grid.Ps[ix, 2]*Ms_denom
        grid.Vw[ix, 1] = grid.Pw[ix, 1]*Mw_denom
        grid.Vw[ix, 2] = grid.Pw[ix, 2]*Mw_denom
        # fixed Dirichlet nodes
        bc.Vx_s_Idx[ix]≠INUM_0 ? grid.Vs[ix, 1]=bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix]≠INUM_0 ? grid.Vs[ix, 2]=bc.Vy_s_Val[ix] : nothing
        bc.Vx_w_Idx[ix]≠INUM_0 ? grid.Vw[ix, 1]=bc.Vx_w_Val[ix] : nothing
        bc.Vy_w_Idx[ix]≠INUM_0 ? grid.Vw[ix, 2]=bc.Vy_w_Val[ix] : nothing
        # compute nodal displacement
        grid.Δd_s[ix, 1] = grid.Vs[ix, 1]*ΔT
        grid.Δd_s[ix, 2] = grid.Vs[ix, 2]*ΔT
        grid.Δd_w[ix, 1] = grid.Vw[ix, 1]*ΔT
        grid.Δd_w[ix, 2] = grid.Vw[ix, 2]*ΔT
    end
end

@kernel inbounds=true function TS_G2P!(
    grid::    KernelGrid2D{T1, T2},
    mp  ::KernelParticle2D{T1, T2}
) where {T1, T2}
    INUM_1 = T1(1); FNUM_0 = T2(0.0); FNUM_1 = T2(1.0)
    ix = @index(Global)
    if ix≤mp.num
        dFs1 = dFs2 = dFs3 = dFs4 = FNUM_0
        dFw1 = dFw2 = dFw3 = dFw4 = FNUM_0
        du1  = du2  = FNUM_0
        for iy in INUM_1:Int32(mp.NIC)
            p2n = mp.p2n[ix, iy]
            ∂Nx = mp.∂Nx[ix, iy]
            ∂Ny = mp.∂Ny[ix, iy]
            Ni  = mp.Ni[ ix, iy]
            # compute solid incremental deformation gradient
            dFs1 += grid.Δd_s[p2n, 1]*∂Nx
            dFs2 += grid.Δd_s[p2n, 1]*∂Ny
            dFs3 += grid.Δd_s[p2n, 2]*∂Nx
            dFs4 += grid.Δd_s[p2n, 2]*∂Ny
            dFw1 += grid.Δd_w[p2n, 1]*∂Nx
            dFw2 += grid.Δd_w[p2n, 1]*∂Ny
            dFw3 += grid.Δd_w[p2n, 2]*∂Nx
            dFw4 += grid.Δd_w[p2n, 2]*∂Ny
            du1  += grid.Δd_s[p2n, 1]*Ni
            du2  += grid.Δd_s[p2n, 2]*Ni
        end
        mp.pos[ix, 1] += du1
        mp.pos[ix, 2] += du2
        mp.∂Fs[ix, 1]  = dFs1
        mp.∂Fs[ix, 2]  = dFs2
        mp.∂Fs[ix, 3]  = dFs3
        mp.∂Fs[ix, 4]  = dFs4
        # compute strain increment
        mp.Δϵij_s[ix, 1] = dFs1
        mp.Δϵij_s[ix, 2] = dFs4
        mp.Δϵij_s[ix, 4] = dFs2+dFs3
        mp.Δϵij_w[ix, 1] = dFw1
        mp.Δϵij_w[ix, 2] = dFw4
        mp.Δϵij_w[ix, 4] = dFw2+dFw3
        # update strain tensor
        mp.ϵij_s[ix, 1] += dFs1
        mp.ϵij_s[ix, 2] += dFs4
        mp.ϵij_s[ix, 4] += dFs2+dFs3
        # update jacobian value and particle volume
        # mp.J[ix] = FNUM_1+mp.ϵij_s[ix, 1]+mp.ϵij_s[ix, 2]
        # update pore pressure
        Kw = mp.Kw[mp.layer[ix]]
        mp.σw[ix] += (Kw/mp.porosity[ix])*((FNUM_1-mp.porosity[ix])*(dFs1+dFs4)+
                                                   mp.porosity[ix] *(dFw1+dFw4))
        mp.porosity[ix] = FNUM_1-(FNUM_1-mp.porosity[ix])/mp.J[ix]
        # mp.vol[ix] *= FNUM_1+dFs1+dFs4



        # deformation gradient matrix
        F1 = mp.F[ix, 1]; F2 = mp.F[ix, 2]; F3 = mp.F[ix, 3]; F4 = mp.F[ix, 4]
        mp.F[ix, 1] = (dFs1+FNUM_1)*F1+dFs2*F3
        mp.F[ix, 2] = (dFs1+FNUM_1)*F2+dFs2*F4
        mp.F[ix, 3] = (dFs4+FNUM_1)*F3+dFs3*F1
        mp.F[ix, 4] = (dFs4+FNUM_1)*F4+dFs3*F2
        # update jacobian value and particle volume
        mp.J[  ix] = mp.F[ix, 1]*mp.F[ix, 4]-mp.F[ix, 2]*mp.F[ix, 3]
        mp.vol[ix] = mp.J[ix]*mp.vol_init[ix]
        # mp.ρs[ ix] = mp.ρs_init[ix]/mp.J[ix]
    end
end

function test!(args::MODELARGS, 
               grid::GRID, 
               mp  ::PARTICLE, 
               bc  ::BOUNDARY,
               ΔT  ::T2,
               Ti  ::T2,
                   ::Val{:TS},
                   ::Val{:USL}) where {T2}
    Ti<args.Te ? G=args.gravity/args.Te*Ti : G=args.gravity
    if Ti≤0.2
        nothing
    else
        T2==Float64 ? T1=Int64 : T1=Base.Int32
        idx2 = T1[7, 14]
        vy_idx = zeros(T1, grid.node_num)
        vy_idx[idx2] .= 1
        bc.Vy_w_Idx = vy_idx
    end
    dev = getBackend(args)
    TS_resetgridstatus!(dev)(ndrange=grid.node_num, grid)
    TS_resetmpstatus!(dev)(ndrange=mp.num, grid, mp, Val(args.basis))
    TS_P2G!(dev)(ndrange=mp.num, grid, mp, G)
    grid.Fs[trac_idx, 2] .+= -500
    TS_solvegrid!(dev)(ndrange=grid.node_num, grid, bc, args.ζs, args.ζw)
    TS_doublemapping1!(dev)(ndrange=mp.num, grid, mp, ΔT)
    TS_doublemapping2!(dev)(ndrange=mp.num, grid, mp)
    TS_doublemapping3!(dev)(ndrange=grid.node_num, grid, bc, ΔT)
    TS_G2P!(dev)(ndrange=mp.num, grid, mp)
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
        twP!(dev)(ndrange=mp.num, mp)
    end                              
    return nothing
end


# model configuration
init_grid_space_x = 0.2
init_grid_space_y = 0.2
init_grid_range_x = [-0, 0.2]
init_grid_range_y = [-0, 1.2]
init_mp_in_space  = 2
init_project_name = "2d_terzaghi"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :linearelastic
init_gravity      = 0
init_ζs           = 0.75
init_ζw           = 0.75
init_ρs           = 2650
init_ρw           = 1000
init_porosity     = 0.3
init_k            = 1e-3
init_ν            = 0
init_E            = 1e6
init_G            = init_E/(2*(1+  init_ν))
init_Kw           = 2.2e9
init_Ks           = init_E/(3*(1-2*init_ν))
init_T            = 1.2
init_Te           = 0
init_ΔT           = 1.95e-6
init_step         = floor(init_T/init_ΔT/150) |> Int64
init_step<10 ? init_step=1 : nothing
init_basis        = :linear
init_NIC          = 4
init_phase        = 2
init_Pw           = 1e4
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
    ζw           = init_ζw,
    gravity      = init_gravity,
    project_name = init_project_name,
    project_path = init_project_path,
    constitutive = init_constitutive,
    animation    = true,
    hdf5         = true,
    hdf5_step    = init_step,
    MVL          = false,
    device       = :CPU,
    coupling     = :TS,
    scheme       = :USL,
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
global trac_idx = iInt.(findall(i->(grid.pos[i, 2]==1)&&(0≤grid.pos[i, 1]≤0.2), 1:grid.node_num))
global trac_val = iFloat(init_Pw*0.2/length(trac_idx))

# material points setup
range_x     = [0+grid.space_x/init_mp_in_space/2, 0.2-grid.space_x/init_mp_in_space/2]
range_y     = [0+grid.space_y/init_mp_in_space/2,   1-grid.space_y/init_mp_in_space/2]
space_x     = grid.space_x/init_mp_in_space
space_y     = grid.space_y/init_mp_in_space
num_x       = length(range_x[1]:space_x:range_x[2])
num_y       = length(range_y[1]:space_y:range_y[2])
x_tmp       = repeat((range_x[1]:space_x:range_x[2])', num_y, 1) |> vec
y_tmp       = repeat((range_y[1]:space_y:range_y[2]) , 1, num_x) |> vec
pos         = hcat(x_tmp, y_tmp)
mp_num      = length(x_tmp)
mp_ρs       = ones(mp_num).*init_ρs
mp_ρw       = ones(mp_num).*init_ρw
mp_porosity = ones(mp_num).*init_porosity
mp_layer    = ones(mp_num)
mp_ν        = [init_ν]
mp_E        = [init_E]
mp_G        = [init_G]
mp_k        = [init_k]
mp_Ks       = [init_Ks]
mp_Kw       = [init_Kw]
mp          = Particle2D{iInt, iFloat}(space_x=space_x, space_y=space_y, pos=pos, ν=mp_ν, 
    E=mp_E, G=mp_G, ρs=mp_ρs, ρw=mp_ρw, Ks=mp_Ks, Kw=mp_Kw, k=mp_k, porosity=mp_porosity, 
    NIC=init_NIC, layer=mp_layer, phase=init_phase)
mp.σw .= -init_Pw

# boundary condition nodes index
vx_idx  = zeros(iInt, grid.node_num)
vy_idx  = zeros(iInt, grid.node_num)
tmp_idx = findall(i->(grid.pos[i, 1]≤0||grid.pos[i, 1]≥0.1||
                      grid.pos[i, 2]≤0), 1:grid.node_num)
tmp_idy = findall(i->(grid.pos[i, 2]≤0||grid.pos[i, 2]≥1.0), 1:grid.node_num)
#tmp_idy = findall(i->grid.pos[i, 2]≤0, 1:grid.node_num)
vx_idx[tmp_idx] .= 1
vy_idx[tmp_idy] .= 1
bc = VBoundary2D{iInt, iFloat}(
    Vx_s_Idx = vx_idx,
    Vx_s_Val = zeros(grid.node_num),
    Vy_s_Idx = vy_idx,
    Vy_s_Val = zeros(grid.node_num),
    Vx_w_Idx = vx_idx,
    Vx_w_Val = zeros(grid.node_num),
    Vy_w_Idx = vy_idx,
    Vy_w_Val = zeros(grid.node_num)
)

# MPM solver
materialpointsolver!(args, grid, mp, bc, workflow=test!)
Threads.nthreads() = initial_threads
# post-process
begin
    # helper functions =====================================================================
    function terzaghi(p0, Tv)
        num = 100
        H = 1
        Z = range(0, 1, length=num)
        data = zeros(num, 2)
        data[:, 2] .= Z
        # Cv = init_k/1e4*(1/init_E+init_porosity/init_Kw)
        # Tv = (Cv*Ttol)/H^2
        @inbounds for i in 1:num
            p = 0.0
            for m in 1:2:1e4
                p += 4*p0/π*(1/m)*sin((m*π*data[i, 2])/(2*H))*exp((-m^2)*((π/2)^2)*Tv)
            end
            data[num+1-i, 1] = p/p0
        end
        return data
    end

    function consolidation()
        num = 1000
        dat = zeros(num, 2)    
        dat[:, 1] .= collect(range(0, 10, length=num))
        @inbounds for i in 1:num
            tmp = 0.0
            for m in 1:2:1e4
                tmp += (8/π^2)*(1/m^2)*exp(-(m*π/2)^2*dat[i, 1]) 
            end
            dat[i, 2] = 1-tmp
        end
        return dat
    end
    # figure setup =========================================================================
    figfont = MaterialPointSolver.fonttnr
    fig = Figure(size=(1000, 450), fontsize=15, fonts=(; regular=figfont, bold=figfont))
    titlay = fig[0, :] = GridLayout()
    Label(titlay[1, :], "Two-phase single-point MPM (𝑣-𝑤)\n2D Terzaghi Consolidation Test", 
        fontsize=20, tellwidth=false, halign=:center, justification=:center, lineheight=1.2)
    layout = fig[1, 1] = GridLayout()
    gd1 = layout[1, 1]
    gd2 = layout[1, 2]
    gd3 = layout[1, 4]
    cb1 = layout[1, 3]
    cb2 = layout[1, 5]
    colsize!(layout, 1, 348)
    # axis setup ===========================================================================
    ax1 = Axis(gd1, xlabel="Normalized pore pressure 𝑝 [-]", 
    ylabel="Normalized depth 𝐻 [-]", xticks=0:0.2:1, yticks=0:0.2:1, 
    title="Excess pore pressure isochrones", aspect=1)
    ax2 = Axis(gd2, aspect=DataAspect(), xlabel="𝑋 Axis [𝑚]", ylabel="𝑌 Axis [𝑚]", 
        xticks=0:0.2:0.2, yticks=0:0.2:1.2, title="Pore pressure distribution")
    ax3 = Axis(gd3, aspect=DataAspect(), xlabel="𝑋 Axis [𝑚]", ylabel="𝑌 Axis [𝑚]", 
        xticks=0:0.2:0.2, yticks=0:0.2:1.2, title="𝑌-velocity distribution")
    limits!(ax1, -0.1, 1.1, -0.1, 1.1)
    limits!(ax2, -0.1, 0.3, -0.1, 1.3)
    limits!(ax3, -0.1, 0.3, -0.1, 1.3)
    # plot setup ===========================================================================
    p11 = lines!(ax1, terzaghi(init_Pw, 0.1), color=:black, linewidth=1, 
        label="Analytical solution")
    p12 = lines!(ax1, terzaghi(init_Pw, 0.3), color=:black, linewidth=1)
    p13 = lines!(ax1, terzaghi(init_Pw, 0.5), color=:black, linewidth=1)
    p14 = lines!(ax1, terzaghi(init_Pw, 0.7), color=:black, linewidth=1)
    fid = h5open(joinpath(args.project_path, "$(args.project_name).h5"), "r")
    num = mp.num/length(unique(mp.init[:, 1])) |> Int
    mp_rst = zeros(num, 2, 4)
    timeset = [39, 64, 89, 114]
    for i in eachindex(timeset)
        c_pp = fid["group$(timeset[i])/pp"] |> read
        mp_rst[:, 1, i] .= reverse(c_pp[1:num])./-init_Pw
        mp_rst[:, 2, i] .= reverse(mp.init[1:num, 2])
    end
    close(fid)
    p15 = scatterlines!(ax1, mp_rst[:, :, 1], linewidth=0.5, markersize=6, color=:red, 
        marker=:star8, strokewidth=0, label="MPM solution")
    p16 = scatterlines!(ax1, mp_rst[:, :, 2], linewidth=0.5, markersize=6, color=:red, 
        marker=:star8, strokewidth=0)
    p17 = scatterlines!(ax1, mp_rst[:, :, 3], linewidth=0.5, markersize=6, color=:red, 
        marker=:star8, strokewidth=0)
    p18 = scatterlines!(ax1, mp_rst[:, :, 4], linewidth=0.5, markersize=6, color=:red, 
        marker=:star8, strokewidth=0)
    axislegend(ax1, merge=true, labelsize=10, padding=(10, 6, 0, 0))
    #---------------------------------------------------------------------------------------
    p21 = scatter!(ax2, mp.pos, color=mp.σw./init_Pw , markersize=18, marker=:rect, 
        colormap=:turbo)
    #---------------------------------------------------------------------------------------
    p31 = scatter!(ax3, mp.pos, color=mp.Vs[:, 2], markersize=18, marker=:rect, 
        colormap=:turbo)
    # colorbar setup =======================================================================
    Colorbar(cb1, p21, label=L"\sigma_w\ [[kPa]", size=5, spinewidth=0, vertical=true, 
        height=Relative(1/1.5))
    Colorbar(cb2, p31, label=L"Y-Vs\ [[m/s]", size=5, spinewidth=0, vertical=true, 
        height=Relative(1/1.5))
    # save figure ==========================================================================
    display(fig)
    save(joinpath(args.project_path, "$(args.project_name).png"), fig)
    @info "Figure saved in project path"
end