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
    solvegrid_USL_OS!(grid::DeviceGrid2D{T1, T2}, bc::DeviceVBoundary2D{T1, T2}, ΔT::T2, ζs::T2)

Description:
---
1. Solve equations on grid.
2. Add boundary condition.
"""
@kernel inbounds=true function solvegrid_USL_OS!(
    grid::     DeviceGrid2D{T1, T2},
    bc  ::DeviceVBoundary2D{T1, T2},
    ΔT  ::T2,
    ζs  ::T2
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ grid.ni
        ms_denom = grid.ms[ix] < eps(T2) ? T2(0.0) : inv(grid.ms[ix])
        # compute nodal velocity
        grid.vs[ix, 1] = grid.ps[ix, 1] * ms_denom
        grid.vs[ix, 2] = grid.ps[ix, 2] * ms_denom
        # damping force for solid
        dampvs = -ζs * sqrt(grid.fs[ix, 1] * grid.fs[ix, 1] + 
                            grid.fs[ix, 2] * grid.fs[ix, 2])
        # compute nodal total force for mixture
        Fs_x = grid.fs[ix, 1] + dampvs * sign(grid.vs[ix, 1])
        Fs_y = grid.fs[ix, 2] + dampvs * sign(grid.vs[ix, 2])
        # update nodal velocity
        grid.vsT[ix, 1] = (grid.ps[ix, 1] + Fs_x * ΔT) * ms_denom
        grid.vsT[ix, 2] = (grid.ps[ix, 2] + Fs_y * ΔT) * ms_denom
        # boundary condition
        bc.vx_s_idx[ix] ≠ T1(0) ? grid.vsT[ix, 1] = bc.vx_s_val[ix] : nothing
        bc.vy_s_idx[ix] ≠ T1(0) ? grid.vsT[ix, 2] = bc.vy_s_val[ix] : nothing
        # compute nodal displacement
        grid.Δus[ix, 1] = grid.vsT[ix, 1] * ΔT
        grid.Δus[ix, 2] = grid.vsT[ix, 2] * ΔT
    end
end

"""
    solvegrid_USL_OS!(grid::DeviceGrid3D{T1, T2}, bc::DeviceVBoundary3D{T1, T2}, ΔT::T2, ζs::T2)

Description:
---
1. Solve equations on grid.
2. Add boundary condition.
"""
@kernel inbounds=true function solvegrid_USL_OS!(
    grid::     DeviceGrid3D{T1, T2},
    bc  ::DeviceVBoundary3D{T1, T2},
    ΔT  ::T2,
    ζs  ::T2
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ grid.ni
        ms_denom = grid.ms[ix] < eps(T2) ? T2(0.0) : inv(grid.ms[ix])
        # compute nodal velocity
        ps_1 = grid.ps[ix, 1]
        ps_2 = grid.ps[ix, 2]
        ps_3 = grid.ps[ix, 3]
        grid.vs[ix, 1] = ps_1 * ms_denom
        grid.vs[ix, 2] = ps_2 * ms_denom
        grid.vs[ix, 3] = ps_3 * ms_denom
        # damping force for solid
        dampvs = -ζs * sqrt(grid.fs[ix, 1] * grid.fs[ix, 1] +
                            grid.fs[ix, 2] * grid.fs[ix, 2] +
                            grid.fs[ix, 3] * grid.fs[ix, 3])
        # compute nodal total force for mixture
        Fs_x = grid.fs[ix, 1] + dampvs * sign(grid.vs[ix, 1])
        Fs_y = grid.fs[ix, 2] + dampvs * sign(grid.vs[ix, 2])
        Fs_z = grid.fs[ix, 3] + dampvs * sign(grid.vs[ix, 3])
        # update nodal velocity
        grid.vsT[ix, 1] = (ps_1 + Fs_x * ΔT) * ms_denom
        grid.vsT[ix, 2] = (ps_2 + Fs_y * ΔT) * ms_denom
        grid.vsT[ix, 3] = (ps_3 + Fs_z * ΔT) * ms_denom
        # boundary condition
        bc.vx_s_idx[ix] ≠ T1(0) ? grid.vsT[ix, 1] = bc.vx_s_val[ix] : nothing
        bc.vy_s_idx[ix] ≠ T1(0) ? grid.vsT[ix, 2] = bc.vy_s_val[ix] : nothing
        bc.vz_s_idx[ix] ≠ T1(0) ? grid.vsT[ix, 3] = bc.vz_s_val[ix] : nothing
        # compute nodal displacement
        grid.Δus[ix, 1] = grid.vsT[ix, 1] * ΔT
        grid.Δus[ix, 2] = grid.vsT[ix, 2] * ΔT
        grid.Δus[ix, 3] = grid.vsT[ix, 3] * ΔT
    end
end

function procedure!(
    args::DeviceArgs{T1, T2}, 
    grid::DeviceGrid{T1, T2}, 
    mp  ::DeviceParticle{T1, T2}, 
    attr::DeviceProperty{T1, T2},
    bc  ::DeviceVBoundary{T1, T2},
    ΔT  ::T2,
    Ti  ::T2,
        ::Val{:OS},
        ::Val{:USL},
        ::Val{:v}
) where {T1, T2}
    G = Ti < args.Te ? args.gravity / args.Te * Ti : args.gravity
    dev = getBackend(Val(args.device))
    resetgridstatus_OS!(dev)(ndrange=grid.ni, grid)
    args.device == :CPU && args.basis == :uGIMP ? 
        resetmpstatus_OS_CPU!(dev)(ndrange=mp.np, grid, mp, Val(args.basis)) :
        resetmpstatus_OS!(dev)(ndrange=mp.np, grid, mp, Val(args.basis))
    P2G_OS!(dev)(ndrange=mp.np, grid, mp, G)
    solvegrid_USL_OS!(dev)(ndrange=grid.ni, grid, bc, ΔT, args.ζs)
    doublemapping1_OS!(dev)(ndrange=mp.np, grid, mp, attr, ΔT, args.FLIP, args.PIC)
    G2P_OS!(dev)(ndrange=mp.np, grid, mp, ΔT)
    if args.constitutive == :hyperelastic
        hyE!(dev)(ndrange=mp.np, mp, attr)
    elseif args.constitutive == :linearelastic
        liE!(dev)(ndrange=mp.np, mp, attr)
    elseif args.constitutive == :druckerprager
        liE!(dev)(ndrange=mp.np, mp, attr)
        if Ti ≥ args.Te
            dpP!(dev)(ndrange=mp.np, mp, attr)
        end
    elseif args.constitutive==:mohrcoulomb
        liE!(dev)(ndrange=mp.np, mp, attr)
        if Ti ≥ args.Te
            mcP!(dev)(ndrange=mp.np, mp, attr)
        end
    end
    if args.MVL == true
        vollock1_OS!(dev)(ndrange=mp.np, grid, mp)
        vollock2_OS!(dev)(ndrange=mp.np, grid, mp)
    end                                  
    return nothing
end


@kernel inbounds = true function solvegrid_a_USL_OS!(
    grid::     DeviceGrid2D{T1, T2},
    bc  ::DeviceVBoundary2D{T1, T2},
    ΔT  ::T2,
    ζs  ::T2
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ grid.ni
        ms_denom = grid.ms[ix] < eps(T2) ? T2(0.0) : inv(grid.ms[ix])
        # boundary condition
        bc.vx_s_idx[ix] ≠ T1(0) ? grid.ps[ix, 1] = bc.vx_s_val[ix] : nothing
        bc.vy_s_idx[ix] ≠ T1(0) ? grid.ps[ix, 2] = bc.vy_s_val[ix] : nothing
        # compute nodal velocity
        grid.vs[ix, 1] = grid.ps[ix, 1] * ms_denom
        grid.vs[ix, 2] = grid.ps[ix, 2] * ms_denom
        # damping force for solid
        dampvs = -ζs * sqrt(grid.fs[ix, 1] * grid.fs[ix, 1] + 
                            grid.fs[ix, 2] * grid.fs[ix, 2])
        # compute nodal total force for mixture
        Fs_x = grid.fs[ix, 1] + dampvs * sign(grid.vs[ix, 1])
        Fs_y = grid.fs[ix, 2] + dampvs * sign(grid.vs[ix, 2])
        # update nodal velocity
        grid.vs[ix, 1] += Fs_x * ΔT * ms_denom
        grid.vs[ix, 2] += Fs_y * ΔT * ms_denom
        # update nodal acceleration
        grid.as[ix, 1] = Fs_x * ms_denom
        grid.as[ix, 2] = Fs_y * ms_denom
        # boundary condition
        bc.vx_s_idx[ix] ≠ T1(0) ? grid.vs[ix, 1] = bc.vx_s_val[ix] : nothing
        bc.vy_s_idx[ix] ≠ T1(0) ? grid.vs[ix, 2] = bc.vy_s_val[ix] : nothing
        bc.vx_s_idx[ix] ≠ T1(0) ? grid.as[ix, 1] = bc.vx_s_val[ix] : nothing
        bc.vy_s_idx[ix] ≠ T1(0) ? grid.as[ix, 2] = bc.vy_s_val[ix] : nothing
        # compute nodal displacement
        grid.Δus[ix, 1] = grid.vs[ix, 1] * ΔT
        grid.Δus[ix, 2] = grid.vs[ix, 2] * ΔT
    end
end

@kernel inbounds=true function solvegrid_a_USL_OS!(
    grid::    DeviceGrid3D{T1, T2},
    bc  ::DeviceVBoundary3D{T1, T2},
    ΔT  ::T2,
    ζs  ::T2
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ grid.ni
        ms_denom = grid.ms[ix] < eps(T2) ? T2(0.0) : inv(grid.ms[ix])
        # boundary condition
        bc.vx_s_idx[ix] ≠ T1(0) ? grid.ps[ix, 1] = bc.vx_s_val[ix] : nothing
        bc.vy_s_idx[ix] ≠ T1(0) ? grid.ps[ix, 2] = bc.vy_s_val[ix] : nothing
        bc.vz_s_idx[ix] ≠ T1(0) ? grid.ps[ix, 3] = bc.vz_s_val[ix] : nothing
        # compute nodal velocity
        grid.vs[ix, 1] = grid.ps[ix, 1] * ms_denom
        grid.vs[ix, 2] = grid.ps[ix, 2] * ms_denom
        grid.vs[ix, 3] = grid.ps[ix, 3] * ms_denom
        # damping force for solid
        dampvs = -ζs * sqrt(grid.fs[ix, 1] * grid.fs[ix, 1] + 
                            grid.fs[ix, 2] * grid.fs[ix, 2] + 
                            grid.fs[ix, 3] * grid.fs[ix, 3])
        # compute nodal total force for mixture
        Fs_x = grid.fs[ix, 1] + dampvs * sign(grid.vs[ix, 1])
        Fs_y = grid.fs[ix, 2] + dampvs * sign(grid.vs[ix, 2])
        Fs_z = grid.fs[ix, 3] + dampvs * sign(grid.vs[ix, 3])
        # update nodal velocity
        grid.vs[ix, 1] += Fs_x * ΔT * ms_denom
        grid.vs[ix, 2] += Fs_y * ΔT * ms_denom
        grid.vs[ix, 3] += Fs_z * ΔT * ms_denom
        # update nodal acceleration
        grid.as[ix, 1] = Fs_x * ms_denom
        grid.as[ix, 2] = Fs_y * ms_denom
        grid.as[ix, 3] = Fs_z * ms_denom
        # boundary condition
        bc.vx_s_idx[ix] ≠ T1(0) ? grid.vs[ix, 1] = bc.vx_s_val[ix] : nothing
        bc.vy_s_idx[ix] ≠ T1(0) ? grid.vs[ix, 2] = bc.vy_s_val[ix] : nothing
        bc.vz_s_idx[ix] ≠ T1(0) ? grid.vs[ix, 3] = bc.vz_s_val[ix] : nothing
        bc.vx_s_idx[ix] ≠ T1(0) ? grid.as[ix, 1] = bc.vx_s_val[ix] : nothing
        bc.vy_s_idx[ix] ≠ T1(0) ? grid.as[ix, 2] = bc.vy_s_val[ix] : nothing
        bc.vz_s_idx[ix] ≠ T1(0) ? grid.as[ix, 3] = bc.vz_s_val[ix] : nothing
        # compute nodal displacement
        grid.Δus[ix, 1] = grid.vs[ix, 1] * ΔT
        grid.Δus[ix, 2] = grid.vs[ix, 2] * ΔT
        grid.Δus[ix, 3] = grid.vs[ix, 3] * ΔT
    end
end

function procedure!(
    args::     DeviceArgs{T1, T2}, 
    grid::     DeviceGrid{T1, T2}, 
    mp  :: DeviceParticle{T1, T2}, 
    attr:: DeviceProperty{T1, T2},
    bc  ::DeviceVBoundary{T1, T2},
    ΔT  ::T2,
    Ti  ::T2,
        ::Val{:OS},
        ::Val{:USL},
        ::Val{:a}
) where {T1, T2}
    G = Ti < args.Te ? args.gravity / args.Te * Ti : args.gravity
    dev = getBackend(Val(args.device))
    resetgridstatus_OS!(dev)(ndrange=grid.ni, grid)
    args.device == :CPU && args.basis == :uGIMP ? 
        resetmpstatus_OS_CPU!(dev)(ndrange=mp.np, grid, mp, Val(args.basis)) :
        resetmpstatus_OS!(dev)(ndrange=mp.np, grid, mp, Val(args.basis))
    P2G_OS!(dev)(ndrange=mp.np, grid, mp, G)
    solvegrid_a_USL_OS!(dev)(ndrange=grid.ni, grid, bc, ΔT, args.ζs)
    doublemapping1_a_OS!(dev)(ndrange=mp.np, grid, mp, attr, ΔT)
    G2P_OS!(dev)(ndrange=mp.np, grid, mp, ΔT)
    if args.constitutive == :hyperelastic
        hyE!(dev)(ndrange=mp.np, mp, attr)
    elseif args.constitutive == :linearelastic
        liE!(dev)(ndrange=mp.np, mp, attr)
    elseif args.constitutive == :druckerprager
        liE!(dev)(ndrange=mp.np, mp, attr)
        if Ti ≥ args.Te
            dpP!(dev)(ndrange=mp.np, mp, attr)
        end
    elseif args.constitutive==:mohrcoulomb
        liE!(dev)(ndrange=mp.np, mp, attr)
        if Ti ≥ args.Te
            mcP!(dev)(ndrange=mp.np, mp, attr)
        end
    end
    if args.MVL == true
        vollock1_OS!(dev)(ndrange=mp.np, grid, mp)
        vollock2_OS!(dev)(ndrange=mp.np, grid, mp)
    end                                  
    return nothing
end