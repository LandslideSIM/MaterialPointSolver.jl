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
    initmpstatus!(grid::DeviceGrid2D{T1, T2}, mp::DeviceParticle2D{T1, T2}, Val{:linear})

Description:
---
This function will setup the particle to node and particle to cell index for 2D MPM solver 
    with linear basis functions.
"""
@kernel inbounds=true function initmpstatus!(
    grid::    DeviceGrid2D{T1, T2}, 
    mp  ::DeviceParticle2D{T1, T2},
        ::Val{:linear}
) where {T1, T2}
    ix = @index(Global)
    mp.p2c[ix] = unsafe_trunc(T1,
        cld(mp.ξ[ix, 2] - grid.y1, grid.dy) +
        fld(mp.ξ[ix, 1] - grid.x1, grid.dx) * grid.ncy)
    @KAunroll for iy in Int32(1):Int32(mp.NIC)
        p2n = getP2N_linear(grid, mp.p2c[ix], iy)
        mp.p2n[ix, iy] = p2n
        # compute distance between particle and related nodes
        Δdx = mp.ξ[ix, 1] - grid.ξ[mp.p2n[ix, iy], 1]
        Δdy = mp.ξ[ix, 2] - grid.ξ[mp.p2n[ix, iy], 2]
        # compute basis function
        Nx, dNx = linearBasis(Δdx, grid.dx)
        Ny, dNy = linearBasis(Δdy, grid.dy)
        mp.Nij[ix, iy] =  Nx * Ny # shape function
        mp.∂Nx[ix, iy] = dNx * Ny # x-gradient shape function
        mp.∂Ny[ix, iy] = dNy * Nx # y-gradient shape function
    end
end

"""
    initmpstatus!(grid::DeviceGrid3D{T1, T2}, mp::DeviceParticle3D{T1, T2}, Val{:linear})

Description:
---
This function will setup the particle to node and particle to cell index for 3D MPM solver.
    with linear basis functions.
"""
@kernel inbounds=true function initmpstatus!(
    grid::    DeviceGrid3D{T1, T2}, 
    mp  ::DeviceParticle3D{T1, T2},
        ::Val{:linear}
) where {T1, T2}
    ix = @index(Global)
    # compute particle to cell and particle to node index
    mp.p2c[ix] = unsafe_trunc(T1, 
        cld(mp.ξ[ix, 2] - grid.y1, grid.dy) +
        fld(mp.ξ[ix, 3] - grid.z1, grid.dz) * grid.ncy * grid.ncx +
        fld(mp.ξ[ix, 1] - grid.x1, grid.dx) * grid.ncy)
    @KAunroll for iy in Int32(1):Int32(mp.NIC)
        p2n = getP2N_linear(grid, mp.p2c[ix], iy)
        mp.p2n[ix, iy] = p2n
        # compute distance between particle and related nodes
        Δdx = mp.ξ[ix, 1] - grid.ξ[mp.p2n[ix, iy], 1]
        Δdy = mp.ξ[ix, 2] - grid.ξ[mp.p2n[ix, iy], 2]
        Δdz = mp.ξ[ix, 3] - grid.ξ[mp.p2n[ix, iy], 3]
        # compute basis function
        Nx, dNx = linearBasis(Δdx, grid.dx)
        Ny, dNy = linearBasis(Δdy, grid.dy)
        Nz, dNz = linearBasis(Δdz, grid.dz)
        mp.Nij[ix, iy] =  Nx * Ny * Nz
        mp.∂Nx[ix, iy] = dNx * Ny * Nz # x-gradient shape function
        mp.∂Ny[ix, iy] = dNy * Nx * Nz # y-gradient shape function
        mp.∂Nz[ix, iy] = dNz * Nx * Ny # z-gradient shape function
    end
end

"""
    initmpstatus!(grid::DeviceGrid2D{T1, T2}, mp::DeviceParticle2D{T1, T2}, Val{:uGIMP})

Description:
---
This function will setup the particle to node and particle to cell index for 2D MPM solver 
    with uGIMP basis functions.
"""
@kernel inbounds=true function initmpstatus!(
    grid::    DeviceGrid2D{T1, T2}, 
    mp  ::DeviceParticle2D{T1, T2},
        ::Val{:uGIMP}
) where {T1, T2}
    ix = @index(Global)
    mp.p2c[ix] = unsafe_trunc(T1,
        cld(mp.ξ[ix, 2] - grid.y1, grid.dy) +
        fld(mp.ξ[ix, 1] - grid.x1, grid.dx) * grid.ncy)
    # reset Nij, ∂Nx, ∂Ny
    @KAunroll for iy in Int32(1):Int32(mp.NIC)
        mp.Nij[ix, iy] = T2(0.0)
        mp.∂Nx[ix, iy] = T2(0.0)
        mp.∂Ny[ix, iy] = T2(0.0)
    end
    viy = T1(1)
    @KAunroll for iy in Int32(1):Int32(16)
        # p2n index
        p2n = getP2N_uGIMP(grid, mp.p2c[ix], iy)
        # compute distance between particle and related nodes
        Δdx = mp.ξ[ix, 1] - grid.ξ[p2n, 1]
        Δdy = mp.ξ[ix, 2] - grid.ξ[p2n, 2]
        # compute basis function
        if abs(Δdx) < (grid.dx + T2(0.5) * mp.dx) &&
           abs(Δdy) < (grid.dy + T2(0.5) * mp.dy)
            Nx, dNx = uGIMPbasis(Δdx, grid.dx, mp.dx)
            Ny, dNy = uGIMPbasis(Δdy, grid.dy, mp.dy)
            mp.Nij[ix, viy] =  Nx * Ny
            mp.∂Nx[ix, viy] = dNx * Ny # x-gradient shape function
            mp.∂Ny[ix, viy] = dNy * Nx # y-gradient shape function
            mp.p2n[ix, viy] = p2n
            viy += T1(1)
        end
        viy > mp.NIC && break
    end
end

"""
    initmpstatus!(grid::DeviceGrid3D{T1, T2}, mp::DeviceParticle3D{T1, T2}, Val{:uGIMP})

Description:
---
This function will setup the particle to node and particle to cell index for 3D MPM solver 
    with uGIMP basis functions.
"""
@kernel inbounds=true function initmpstatus!(
    grid::    DeviceGrid3D{T1, T2}, 
    mp  ::DeviceParticle3D{T1, T2},
        ::Val{:uGIMP}
) where {T1, T2}
    ix = @index(Global)
    mpξ1 = mp.ξ[ix, 1]
    mpξ2 = mp.ξ[ix, 2]
    mpξ3 = mp.ξ[ix, 3]
    mp.p2c[ix] = unsafe_trunc(T1,
        cld(mpξ2 - grid.y1, grid.dy) +
        fld(mpξ3 - grid.z1, grid.dz) * grid.ncy * grid.ncx +
        fld(mpξ1 - grid.x1, grid.dx) * grid.ncy)
    @KAunroll for iy in Int32(1):Int32(mp.NIC)
        mp.Nij[ix, iy] = T2(0.0)
        mp.∂Nx[ix, iy] = T2(0.0)
        mp.∂Ny[ix, iy] = T2(0.0)
        mp.∂Nz[ix, iy] = T2(0.0)
    end
    viy = T1(1)
    @KAunroll for iy in Int32(1):Int32(64)
        # p2n index
        p2n = getP2N_uGIMP(grid, mp.p2c[ix], iy)
        # compute distance betwe en particle and related nodes
        Δdx = mpξ1 - grid.ξ[p2n, 1]
        Δdy = mpξ2 - grid.ξ[p2n, 2]
        Δdz = mpξ3 - grid.ξ[p2n, 3]
        # compute basis function
        if abs(Δdx) < (grid.dx + T2(0.5) * mp.dx) &&
           abs(Δdy) < (grid.dy + T2(0.5) * mp.dy) &&
           abs(Δdz) < (grid.dz + T2(0.5) * mp.dz)
            Nx, dNx = uGIMPbasis(Δdx, grid.dx, mp.dx)
            Ny, dNy = uGIMPbasis(Δdy, grid.dy, mp.dy)
            Nz, dNz = uGIMPbasis(Δdz, grid.dz, mp.dz)
            mp.Nij[ix, viy] =  Nx * Ny * Nz
            mp.∂Nx[ix, viy] = dNx * Ny * Nz # x-gradient basis function
            mp.∂Ny[ix, viy] = dNy * Nx * Nz # y-gradient basis function
            mp.∂Nz[ix, viy] = dNz * Nx * Ny # z-gradient basis function
            mp.p2n[ix, viy] = p2n
            viy += T1(1)
        end
        viy > mp.NIC && break
    end
end