#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : utils.jl                                                                   |
|  Description: some useful functions can be used in the MPM procedure                     |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 01. initmpstatus! [2/3D linear/uGIMP basis]                                |
|               02. solvegrid_USL_OS! [2/3D]                                               |
+==========================================================================================#

export initmpstatus!
export solvegrid_USL_OS!

"""
    initmpstatus!(grid::KernelGrid2D{T1, T2}, mp::KernelParticle2D{T1, T2}, Val{:linear})

Description:
---
This function will setup the particle to node and particle to cell index for 2D MPM solver 
    with linear basis functions.
"""
@kernel inbounds=true function initmpstatus!(
    grid::    KernelGrid2D{T1, T2}, 
    mp  ::KernelParticle2D{T1, T2},
        ::Val{:linear}
) where {T1, T2}
    ix = @index(Global)
    mp.p2c[ix] = unsafe_trunc(T1, 
        cld(mp.pos[ix, 2] - grid.range_y1, grid.space_y) +
        fld(mp.pos[ix, 1] - grid.range_x1, grid.space_x) * grid.cell_num_y)
    @KAunroll for iy in Int32(1):Int32(4)
        p2n = getP2N_linear(grid, mp.p2c[ix], iy)
        mp.p2n[ix, iy] = p2n
        # compute distance between particle and related nodes
        Δdx = mp.pos[ix, 1] - grid.pos[mp.p2n[ix, iy], 1]
        Δdy = mp.pos[ix, 2] - grid.pos[mp.p2n[ix, iy], 2]
        # compute basis function
        Nx, dNx = linearBasis(Δdx, grid.space_x)
        Ny, dNy = linearBasis(Δdy, grid.space_y)
        mp.Ni[ ix, iy] =  Nx * Ny # shape function
        mp.∂Nx[ix, iy] = dNx * Ny # x-gradient shape function
        mp.∂Ny[ix, iy] = dNy * Nx # y-gradient shape function
    end
end

"""
    initmpstatus!(grid::KernelGrid3D{T1, T2}, mp::KernelParticle3D{T1, T2}, Val{:linear})

Description:
---
This function will setup the particle to node and particle to cell index for 3D MPM solver.
    with linear basis functions.
"""
@kernel inbounds=true function initmpstatus!(
    grid::    KernelGrid3D{T1, T2}, 
    mp  ::KernelParticle3D{T1, T2},
        ::Val{:linear}
) where {T1, T2}
    ix = @index(Global)
    # compute particle to cell and particle to node index
    mp.p2c[ix] = unsafe_trunc(T1, 
        cld(mp.pos[ix, 2] - grid.range_y1, grid.space_y) +
        fld(mp.pos[ix, 3] - grid.range_z1, grid.space_z) * 
            grid.cell_num_y * grid.cell_num_x +
        fld(mp.pos[ix, 1] - grid.range_x1, grid.space_x) * grid.cell_num_y)
    @KAunroll for iy in Int32(1):Int32(8)
        p2n = getP2N_linear(grid, mp.p2c[ix], iy)
        mp.p2n[ix, iy] = p2n
        # compute distance between particle and related nodes
        Δdx = mp.pos[ix, 1] - grid.pos[mp.p2n[ix, iy], 1]
        Δdy = mp.pos[ix, 2] - grid.pos[mp.p2n[ix, iy], 2]
        Δdz = mp.pos[ix, 3] - grid.pos[mp.p2n[ix, iy], 3]
        # compute basis function
        Nx, dNx = linearBasis(Δdx, grid.space_x)
        Ny, dNy = linearBasis(Δdy, grid.space_y)
        Nz, dNz = linearBasis(Δdz, grid.space_z)
        mp.Ni[ ix, iy] =  Nx * Ny * Nz
        mp.∂Nx[ix, iy] = dNx * Ny * Nz # x-gradient shape function
        mp.∂Ny[ix, iy] = dNy * Nx * Nz # y-gradient shape function
        mp.∂Nz[ix, iy] = dNz * Nx * Ny # z-gradient shape function
    end
end

"""
    initmpstatus!(grid::KernelGrid2D{T1, T2}, mp::KernelParticle2D{T1, T2}, Val{:uGIMP})

Description:
---
This function will setup the particle to node and particle to cell index for 2D MPM solver 
    with uGIMP basis functions.
"""
@kernel inbounds=true function initmpstatus!(
    grid::    KernelGrid2D{T1, T2}, 
    mp  ::KernelParticle2D{T1, T2},
        ::Val{:uGIMP}
) where {T1, T2}
    ix = @index(Global)
    mp.p2c[ix] = unsafe_trunc(T1,
        cld(mp.pos[ix, 2] - grid.range_y1, grid.space_y) +
        fld(mp.pos[ix, 1] - grid.range_x1, grid.space_x) * grid.cell_num_y)
    @KAunroll for iy in Int32(1):Int32(16)
        p2n = getP2N_uGIMP(grid, mp.p2c[ix], iy)
        mp.p2n[ix, iy] = p2n
        # compute distance between particle and related nodes
        p2n = mp.p2n[ix, iy]
        Δdx = mp.pos[ix, 1] - grid.pos[p2n, 1]
        Δdy = mp.pos[ix, 2] - grid.pos[p2n, 2]
        # compute basis function
        Nx, dNx = uGIMPbasis(Δdx, grid.space_x, mp.space_x)
        Ny, dNy = uGIMPbasis(Δdy, grid.space_y, mp.space_y)
        mp.Ni[ ix, iy] =  Nx * Ny
        mp.∂Nx[ix, iy] = dNx * Ny # x-gradient shape function
        mp.∂Ny[ix, iy] = dNy * Nx # y-gradient shape function
    end
end

"""
    initmpstatus!(grid::KernelGrid3D{T1, T2}, mp::KernelParticle3D{T1, T2}, Val{:uGIMP})

Description:
---
This function will setup the particle to node and particle to cell index for 3D MPM solver 
    with uGIMP basis functions.
"""
@kernel inbounds=true function initmpstatus!(
    grid::    KernelGrid3D{T1, T2}, 
    mp  ::KernelParticle3D{T1, T2},
        ::Val{:uGIMP}
) where {T1, T2}
    ix = @index(Global)
    mp.p2c[ix] = unsafe_trunc(T1,
        cld(mp_pos_2 - grid.range_y1, grid.space_y) +
        fld(mp_pos_3 - grid.range_z1, grid.space_z) * 
            grid.cell_num_y * grid.cell_num_x +
        fld(mp_pos_1 - grid.range_x1, grid.space_x) * grid.cell_num_y)
    @KAunroll for iy in Int32(1):Int32(64)
        p2n = getP2N_uGIMP(grid, mp.p2c[ix], iy)
        mp.p2n[ix, iy] = p2n
        # compute distance betwe en particle and related nodes
        Δdx = mp_pos_1 - grid.pos[p2n, 1]
        Δdy = mp_pos_2 - grid.pos[p2n, 2]
        Δdz = mp_pos_3 - grid.pos[p2n, 3]
        # compute basis function
        Nx, dNx = uGIMPbasis(Δdx, grid.space_x, mp.space_x)
        Ny, dNy = uGIMPbasis(Δdy, grid.space_y, mp.space_y)
        Nz, dNz = uGIMPbasis(Δdz, grid.space_z, mp.space_z)
        mp.Ni[ ix, iy] =  Nx * Ny * Nz
        mp.∂Nx[ix, iy] = dNx * Ny * Nz # x-gradient basis function
        mp.∂Ny[ix, iy] = dNy * Nx * Nz # y-gradient basis function
        mp.∂Nz[ix, iy] = dNz * Nx * Ny # z-gradient basis function
    end
end

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
        iszero(grid.Ms[ix]) ? Ms_denom=T2(0.0) : Ms_denom=T2(1.0)/grid.Ms[ix]
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
        bc.Vx_s_Idx[ix]≠T1(0) ? grid.Vs_T[ix, 1]=bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix]≠T1(0) ? grid.Vs_T[ix, 2]=bc.Vy_s_Val[ix] : nothing
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
        iszero(grid.Ms[ix]) ? Ms_denom=T2(0.0) : Ms_denom=T2(1.0)/grid.Ms[ix]
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
        bc.Vx_s_Idx[ix]≠T1(0) ? grid.Vs_T[ix, 1]=bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix]≠T1(0) ? grid.Vs_T[ix, 2]=bc.Vy_s_Val[ix] : nothing
        bc.Vz_s_Idx[ix]≠T1(0) ? grid.Vs_T[ix, 3]=bc.Vz_s_Val[ix] : nothing
        # compute nodal displacement
        grid.Δd_s[ix, 1] = grid.Vs_T[ix, 1] * ΔT
        grid.Δd_s[ix, 2] = grid.Vs_T[ix, 2] * ΔT
        grid.Δd_s[ix, 3] = grid.Vs_T[ix, 3] * ΔT
    end
end