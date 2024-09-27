#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : OS_USL.jl                                                                  |
|  Description: Defaule USL (update stress last) implementation of USL for one-phase       | 
|               single-point MPM                                                           |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
+==========================================================================================#

export solvegrid_USL_OS!
export solvegrid_a_USL_OS!

"""
    solvegrid_USL_OS!(grid::KernelGrid2D{T1, T2}, bc::KernelBoundary2D{T1, T2}, ΔT::T2, ζs::T2)

Description:
---
1. Solve equations on grid.
2. Add boundary condition.
"""
@kernel inbounds=true function solvegrid_USL_OS!(
    grid::    KernelGrid2D{T1, T2},
    bc  ::KernelBoundary2D{T1, T2},
    ΔT  ::T2,
    ζs  ::T2
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ grid.node_num
        Ms_denom = grid.Ms[ix] < eps(T2) ? T2(0.0) : inv(grid.Ms[ix])
        # compute nodal velocity
        grid.Vs[ix, 1] = grid.Ps[ix, 1] * Ms_denom
        grid.Vs[ix, 2] = grid.Ps[ix, 2] * Ms_denom
        # damping force for solid
        dampvs = -ζs * sqrt(grid.Fs[ix, 1] * grid.Fs[ix, 1] + 
                            grid.Fs[ix, 2] * grid.Fs[ix, 2])
        # compute nodal total force for mixture
        Fs_x = grid.Fs[ix, 1] + dampvs * sign(grid.Vs[ix, 1])
        Fs_y = grid.Fs[ix, 2] + dampvs * sign(grid.Vs[ix, 2])
        # update nodal velocity
        grid.Vs_T[ix, 1] = (grid.Ps[ix, 1] + Fs_x * ΔT) * Ms_denom
        grid.Vs_T[ix, 2] = (grid.Ps[ix, 2] + Fs_y * ΔT) * Ms_denom
        # boundary condition
        bc.Vx_s_Idx[ix] ≠ T1(0) ? grid.Vs_T[ix, 1] = bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix] ≠ T1(0) ? grid.Vs_T[ix, 2] = bc.Vy_s_Val[ix] : nothing
        # compute nodal displacement
        grid.Δd_s[ix, 1] = grid.Vs_T[ix, 1] * ΔT
        grid.Δd_s[ix, 2] = grid.Vs_T[ix, 2] * ΔT
    end
end

"""
    solvegrid_USL_OS!(grid::KernelGrid3D{T1, T2}, bc::KernelBoundary3D{T1, T2}, ΔT::T2, ζs::T2)

Description:
---
1. Solve equations on grid.
2. Add boundary condition.
"""
@kernel inbounds=true function solvegrid_USL_OS!(
    grid::    KernelGrid3D{T1, T2},
    bc  ::KernelBoundary3D{T1, T2},
    ΔT  ::T2,
    ζs  ::T2
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ grid.node_num
        Ms_denom = grid.Ms[ix] < eps(T2) ? T2(0.0) : inv(grid.Ms[ix])
        # compute nodal velocity
        Ps_1 = grid.Ps[ix, 1]
        Ps_2 = grid.Ps[ix, 2]
        Ps_3 = grid.Ps[ix, 3]
        grid.Vs[ix, 1] = Ps_1 * Ms_denom
        grid.Vs[ix, 2] = Ps_2 * Ms_denom
        grid.Vs[ix, 3] = Ps_3 * Ms_denom
        # damping force for solid
        dampvs = -ζs * sqrt(grid.Fs[ix, 1] * grid.Fs[ix, 1] +
                            grid.Fs[ix, 2] * grid.Fs[ix, 2] +
                            grid.Fs[ix, 3] * grid.Fs[ix, 3])
        # compute nodal total force for mixture
        Fs_x = grid.Fs[ix, 1] + dampvs * sign(grid.Vs[ix, 1])
        Fs_y = grid.Fs[ix, 2] + dampvs * sign(grid.Vs[ix, 2])
        Fs_z = grid.Fs[ix, 3] + dampvs * sign(grid.Vs[ix, 3])
        # update nodal velocity
        grid.Vs_T[ix, 1] = (Ps_1 + Fs_x * ΔT) * Ms_denom
        grid.Vs_T[ix, 2] = (Ps_2 + Fs_y * ΔT) * Ms_denom
        grid.Vs_T[ix, 3] = (Ps_3 + Fs_z * ΔT) * Ms_denom
        # boundary condition
        bc.Vx_s_Idx[ix] ≠ T1(0) ? grid.Vs_T[ix, 1] = bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix] ≠ T1(0) ? grid.Vs_T[ix, 2] = bc.Vy_s_Val[ix] : nothing
        bc.Vz_s_Idx[ix] ≠ T1(0) ? grid.Vs_T[ix, 3] = bc.Vz_s_Val[ix] : nothing
        # compute nodal displacement
        grid.Δd_s[ix, 1] = grid.Vs_T[ix, 1] * ΔT
        grid.Δd_s[ix, 2] = grid.Vs_T[ix, 2] * ΔT
        grid.Δd_s[ix, 3] = grid.Vs_T[ix, 3] * ΔT
    end
end

function procedure!(
    args    ::MODELARGS, 
    grid    ::GRID, 
    mp      ::PARTICLE, 
    pts_attr::PROPERTY,
    bc      ::BOUNDARY,
    ΔT      ::T2,
    Ti      ::T2,
            ::Val{:OS},
            ::Val{:USL},
            ::Val{:v}
) where {T2}
    G = Ti < args.Te ? args.gravity / args.Te * Ti : args.gravity
    dev = getBackend(Val(args.device))
    resetgridstatus_OS!(dev)(ndrange=grid.node_num, grid)
    args.device == :CPU && args.basis == :uGIMP ? 
        resetmpstatus_OS_CPU!(dev)(ndrange=mp.num, grid, mp, Val(args.basis)) :
        resetmpstatus_OS!(dev)(ndrange=mp.num, grid, mp, Val(args.basis))
    P2G_OS!(dev)(ndrange=mp.num, grid, mp, G)
    solvegrid_USL_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT, args.ζs)
    doublemapping1_OS!(dev)(ndrange=mp.num, grid, mp, pts_attr, ΔT, args.FLIP, args.PIC)
    G2P_OS!(dev)(ndrange=mp.num, grid, mp, ΔT)
    if args.constitutive == :hyperelastic
        hyE!(dev)(ndrange=mp.num, mp, pts_attr)
    elseif args.constitutive == :linearelastic
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
    elseif args.constitutive == :druckerprager
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
        if Ti ≥ args.Te
            dpP!(dev)(ndrange=mp.num, mp, pts_attr)
        end
    elseif args.constitutive==:mohrcoulomb
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
        if Ti ≥ args.Te
            mcP!(dev)(ndrange=mp.num, mp, pts_attr)
        end
    end
    if args.MVL == true
        vollock1_OS!(dev)(ndrange=mp.num, grid, mp)
        vollock2_OS!(dev)(ndrange=mp.num, grid, mp)
    end                                  
    return nothing
end


@kernel inbounds = true function solvegrid_a_USL_OS!(
    grid::    KernelGrid2D{T1, T2},
    bc  ::KernelBoundary2D{T1, T2},
    ΔT  ::T2,
    ζs  ::T2
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ grid.node_num
        Ms_denom = grid.Ms[ix] < eps(T2) ? T2(0.0) : inv(grid.Ms[ix])
        # boundary condition
        bc.Vx_s_Idx[ix] ≠ T1(0) ? grid.Ps[ix, 1] = bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix] ≠ T1(0) ? grid.Ps[ix, 2] = bc.Vy_s_Val[ix] : nothing
        # compute nodal velocity
        grid.Vs[ix, 1] = grid.Ps[ix, 1] * Ms_denom
        grid.Vs[ix, 2] = grid.Ps[ix, 2] * Ms_denom
        # damping force for solid
        dampvs = -ζs * sqrt(grid.Fs[ix, 1] * grid.Fs[ix, 1] + 
                            grid.Fs[ix, 2] * grid.Fs[ix, 2])
        # compute nodal total force for mixture
        Fs_x = grid.Fs[ix, 1] + dampvs * sign(grid.Vs[ix, 1])
        Fs_y = grid.Fs[ix, 2] + dampvs * sign(grid.Vs[ix, 2])
        # update nodal velocity
        grid.Vs[ix, 1] += Fs_x * ΔT * Ms_denom
        grid.Vs[ix, 2] += Fs_y * ΔT * Ms_denom
        # update nodal acceleration
        grid.a_s[ix, 1] = Fs_x * Ms_denom
        grid.a_s[ix, 2] = Fs_y * Ms_denom
        # boundary condition
        bc.Vx_s_Idx[ix] ≠ T1(0) ? grid.Vs[ix, 1] = bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix] ≠ T1(0) ? grid.Vs[ix, 2] = bc.Vy_s_Val[ix] : nothing
        bc.Vx_s_Idx[ix] ≠ T1(0) ? grid.a_s[ix, 1] = bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix] ≠ T1(0) ? grid.a_s[ix, 2] = bc.Vy_s_Val[ix] : nothing
        # compute nodal displacement
        grid.Δd_s[ix, 1] = grid.Vs[ix, 1] * ΔT
        grid.Δd_s[ix, 2] = grid.Vs[ix, 2] * ΔT
    end
end

@kernel inbounds=true function solvegrid_a_USL_OS!(
    grid::    KernelGrid3D{T1, T2},
    bc  ::KernelBoundary3D{T1, T2},
    ΔT  ::T2,
    ζs  ::T2
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ grid.node_num
        Ms_denom = grid.Ms[ix] < eps(T2) ? T2(0.0) : inv(grid.Ms[ix])
        # boundary condition
        bc.Vx_s_Idx[ix] ≠ T1(0) ? grid.Ps[ix, 1] = bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix] ≠ T1(0) ? grid.Ps[ix, 2] = bc.Vy_s_Val[ix] : nothing
        bc.Vz_s_Idx[ix] ≠ T1(0) ? grid.Ps[ix, 3] = bc.Vz_s_Val[ix] : nothing
        # compute nodal velocity
        grid.Vs[ix, 1] = grid.Ps[ix, 1] * Ms_denom
        grid.Vs[ix, 2] = grid.Ps[ix, 2] * Ms_denom
        grid.Vs[ix, 3] = grid.Ps[ix, 3] * Ms_denom
        # damping force for solid
        dampvs = -ζs * sqrt(grid.Fs[ix, 1] * grid.Fs[ix, 1] + 
                            grid.Fs[ix, 2] * grid.Fs[ix, 2] + 
                            grid.Fs[ix, 3] * grid.Fs[ix, 3])
        # compute nodal total force for mixture
        Fs_x = grid.Fs[ix, 1] + dampvs * sign(grid.Vs[ix, 1])
        Fs_y = grid.Fs[ix, 2] + dampvs * sign(grid.Vs[ix, 2])
        Fs_z = grid.Fs[ix, 3] + dampvs * sign(grid.Vs[ix, 3])
        # update nodal velocity
        grid.Vs[ix, 1] += Fs_x * ΔT * Ms_denom
        grid.Vs[ix, 2] += Fs_y * ΔT * Ms_denom
        grid.Vs[ix, 3] += Fs_z * ΔT * Ms_denom
        # update nodal acceleration
        grid.a_s[ix, 1] = Fs_x * Ms_denom
        grid.a_s[ix, 2] = Fs_y * Ms_denom
        grid.a_s[ix, 3] = Fs_z * Ms_denom
        # boundary condition
        bc.Vx_s_Idx[ix] ≠ T1(0) ? grid.Vs[ix, 1] = bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix] ≠ T1(0) ? grid.Vs[ix, 2] = bc.Vy_s_Val[ix] : nothing
        bc.Vz_s_Idx[ix] ≠ T1(0) ? grid.Vs[ix, 3] = bc.Vz_s_Val[ix] : nothing
        bc.Vx_s_Idx[ix] ≠ T1(0) ? grid.a_s[ix, 1] = bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix] ≠ T1(0) ? grid.a_s[ix, 2] = bc.Vy_s_Val[ix] : nothing
        bc.Vz_s_Idx[ix] ≠ T1(0) ? grid.a_s[ix, 3] = bc.Vz_s_Val[ix] : nothing
        # compute nodal displacement
        grid.Δd_s[ix, 1] = grid.Vs[ix, 1] * ΔT
        grid.Δd_s[ix, 2] = grid.Vs[ix, 2] * ΔT
        grid.Δd_s[ix, 3] = grid.Vs[ix, 3] * ΔT
    end
end

function procedure!(
    args    ::MODELARGS, 
    grid    ::GRID, 
    mp      ::PARTICLE, 
    pts_attr::PROPERTY,
    bc      ::BOUNDARY,
    ΔT      ::T2,
    Ti      ::T2,
            ::Val{:OS},
            ::Val{:USL},
            ::Val{:a}
) where {T2}
    G = Ti < args.Te ? args.gravity / args.Te * Ti : args.gravity
    dev = getBackend(Val(args.device))
    resetgridstatus_OS!(dev)(ndrange=grid.node_num, grid)
    args.device == :CPU && args.basis == :uGIMP ? 
        resetmpstatus_OS_CPU!(dev)(ndrange=mp.num, grid, mp, Val(args.basis)) :
        resetmpstatus_OS!(dev)(ndrange=mp.num, grid, mp, Val(args.basis))
    P2G_OS!(dev)(ndrange=mp.num, grid, mp, G)
    solvegrid_a_USL_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT, args.ζs)
    doublemapping1_a_OS!(dev)(ndrange=mp.num, grid, mp, pts_attr, ΔT)
    G2P_OS!(dev)(ndrange=mp.num, grid, mp, ΔT)
    if args.constitutive == :hyperelastic
        hyE!(dev)(ndrange=mp.num, mp, pts_attr)
    elseif args.constitutive == :linearelastic
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
    elseif args.constitutive == :druckerprager
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
        if Ti ≥ args.Te
            dpP!(dev)(ndrange=mp.num, mp, pts_attr)
        end
    elseif args.constitutive==:mohrcoulomb
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
        if Ti ≥ args.Te
            mcP!(dev)(ndrange=mp.num, mp, pts_attr)
        end
    end
    if args.MVL == true
        vollock1_OS!(dev)(ndrange=mp.num, grid, mp)
        vollock2_OS!(dev)(ndrange=mp.num, grid, mp)
    end                                  
    return nothing
end