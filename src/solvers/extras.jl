#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : extras.jl                                                                  |
|  Description: some useful functions can be used in the MPM procedure                     |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 01. initmpstatus! [2/3D linear/uGIMP basis]                                |
+==========================================================================================#

export initmpstatus!

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
    @KAunroll for iy in Int32(1):Int32(mp.NIC)
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
    @KAunroll for iy in Int32(1):Int32(mp.NIC)
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
    @KAunroll for iy in Int32(1):Int32(mp.NIC)
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
        cld(mp.pos[ix, 2] - grid.range_y1, grid.space_y) +
        fld(mp.pos[ix, 3] - grid.range_z1, grid.space_z) * 
            grid.cell_num_y * grid.cell_num_x +
        fld(mp.pos[ix, 1] - grid.range_x1, grid.space_x) * grid.cell_num_y)
    @KAunroll for iy in Int32(1):Int32(mp.NIC)
        p2n = getP2N_uGIMP(grid, mp.p2c[ix], iy)
        mp.p2n[ix, iy] = p2n
        # compute distance betwe en particle and related nodes
        Δdx = mp.pos[ix, 1] - grid.pos[p2n, 1]
        Δdy = mp.pos[ix, 2] - grid.pos[p2n, 2]
        Δdz = mp.pos[ix, 3] - grid.pos[p2n, 3]
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