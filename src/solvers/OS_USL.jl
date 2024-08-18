#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : OS_USL.jl                                                                  |
|  Description: Defaule USL (update stress last) implementation of USL for one-phase       | 
|               single-point MPM                                                           |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 01. solvegrid_USL_OS! [2D]                                                 |
|               02. solvegrid_USL_OS! [3D]                                                 |
|               03. procedure!        [2D & 3D]                                            |
+==========================================================================================#

export solvegrid_USL_OS!

"""
    solvegrid_USL_OS!(grid::KernelGrid2D{T1, T2}, bc::KernelBoundary2D{T1, T2}, ΔT::T2, ζs::T2)

Description:
---
1. Solve equations on grid.
2. Add boundary condition.

I/0 accesses:
---
- read  → grid.node_num* 7
- write → grid.node_num* 4
- total → grid.node_num*11
"""
@kernel inbounds=true function solvegrid_USL_OS!(
    grid::    KernelGrid2D{T1, T2},
    bc  ::KernelBoundary2D{T1, T2},
    ΔT  ::T2,
    ζs  ::T2
) where {T1, T2}
    INUM_0 = T1(0  ); INUM_2 = T1(2)
    FNUM_0 = T2(0.0); FNUM_1 = T2(1.0)
    ix = @index(Global)
    if ix≤grid.node_num
        iszero(grid.Ms[ix]) ? Ms_denom=FNUM_0 : Ms_denom=FNUM_1/grid.Ms[ix]
        # compute nodal velocity
        grid.Vs[ix, 1] = grid.Ps[ix, 1]*Ms_denom
        grid.Vs[ix, 2] = grid.Ps[ix, 2]*Ms_denom
        # damping force for solid
        dampvs = -ζs*sqrt(grid.Fs[ix, 1]^INUM_2+grid.Fs[ix, 2]^INUM_2)
        # compute nodal total force for mixture
        Fs_x = grid.Fs[ix, 1]+dampvs*sign(grid.Vs[ix, 1])
        Fs_y = grid.Fs[ix, 2]+dampvs*sign(grid.Vs[ix, 2])
        # update nodal velocity
        grid.Vs_T[ix, 1] = (grid.Ps[ix, 1]+Fs_x*ΔT)*Ms_denom
        grid.Vs_T[ix, 2] = (grid.Ps[ix, 2]+Fs_y*ΔT)*Ms_denom
        # boundary condition
        bc.Vx_s_Idx[ix]≠INUM_0 ? grid.Vs_T[ix, 1]=bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix]≠INUM_0 ? grid.Vs_T[ix, 2]=bc.Vy_s_Val[ix] : nothing
        # compute nodal displacement
        grid.Δd_s[ix, 1] = grid.Vs_T[ix, 1]*ΔT
        grid.Δd_s[ix, 2] = grid.Vs_T[ix, 2]*ΔT
    end
end

"""
    solvegrid_USL_OS!(grid::KernelGrid3D{T1, T2}, bc::KernelBoundary3D{T1, T2}, ΔT::T2, ζs::T2)

Description:
---
1. Solve equations on grid.
2. Add boundary condition.

I/0 accesses:
---
- read  → grid.node_num*10
- write → grid.node_num* 6
- total → grid.node_num*16
"""
@kernel inbounds=true function solvegrid_USL_OS!(
    grid::    KernelGrid3D{T1, T2},
    bc  ::KernelBoundary3D{T1, T2},
    ΔT  ::T2,
    ζs  ::T2
) where {T1, T2}
    INUM_0 = T1(0); INUM_2 = T1(2); FNUM_0 = T2(0.0); FNUM_1 = T1(1.0)
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
        dampvs = -ζs*sqrt(grid.Fs[ix, 1]^INUM_2+
                          grid.Fs[ix, 2]^INUM_2+
                          grid.Fs[ix, 3]^INUM_2)
        # compute nodal total force for mixture
        Fs_x = grid.Fs[ix, 1]+dampvs*sign(grid.Vs[ix, 1])
        Fs_y = grid.Fs[ix, 2]+dampvs*sign(grid.Vs[ix, 2])
        Fs_z = grid.Fs[ix, 3]+dampvs*sign(grid.Vs[ix, 3])
        # update nodal velocity
        grid.Vs_T[ix, 1] = (Ps_1+Fs_x*ΔT)*Ms_denom
        grid.Vs_T[ix, 2] = (Ps_2+Fs_y*ΔT)*Ms_denom
        grid.Vs_T[ix, 3] = (Ps_3+Fs_z*ΔT)*Ms_denom
        # boundary condition
        bc.Vx_s_Idx[ix]≠INUM_0 ? grid.Vs_T[ix, 1]=bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix]≠INUM_0 ? grid.Vs_T[ix, 2]=bc.Vy_s_Val[ix] : nothing
        bc.Vz_s_Idx[ix]≠INUM_0 ? grid.Vs_T[ix, 3]=bc.Vz_s_Val[ix] : nothing
        # compute nodal displacement
        grid.Δd_s[ix, 1] = grid.Vs_T[ix, 1]*ΔT
        grid.Δd_s[ix, 2] = grid.Vs_T[ix, 2]*ΔT
        grid.Δd_s[ix, 3] = grid.Vs_T[ix, 3]*ΔT
    end
end

function procedure!(args    ::MODELARGS, 
                    grid    ::GRID, 
                    mp      ::PARTICLE, 
                    pts_attr::PROPERTY,
                    bc      ::BOUNDARY,
                    ΔT      ::T2,
                    Ti      ::T2,
                            ::Val{:OS},
                            ::Val{:USL}) where {T2}
    Ti<args.Te ? G=args.gravity/args.Te*Ti : G=args.gravity
    dev = getBackend(args)
    resetgridstatus_OS!(dev)(ndrange=grid.node_num, grid)
    args.device == :CPU && args.basis == :uGIMP ? 
        resetmpstatus_OS_CPU!(dev)(ndrange=mp.num, grid, mp, Val(args.basis)) :
        resetmpstatus_OS!(dev)(ndrange=mp.num, grid, mp, Val(args.basis))
    P2G_OS!(dev)(ndrange=mp.num, grid, mp, G)
    solvegrid_USL_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT, args.ζs)
    doublemapping1_OS!(dev)(ndrange=mp.num, grid, mp, pts_attr, ΔT, args.FLIP, args.PIC)
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
    if args.MVL==true
        vollock1_OS!(dev)(ndrange=mp.num, grid, mp)
        vollock2_OS!(dev)(ndrange=mp.num, grid, mp)
    end                                  
    return nothing
end