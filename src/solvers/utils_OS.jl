#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : utils_OS.jl                                                                |
|  Description: Basic computing functions for one-phase single-point MPM, and these        |
|               functions are mainly used for MUSL update scheme.                          |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 01. resetgridstatus_OS!     [2D]                                           |
|               02. resetgridstatus_OS!     [3D]                                           |
|               03. resetmpstatus_OS!       [2D, linear basis]                             |
|               04. resetmpstatus_OS!       [3D, linear basis]                             |
|               05. resetmpstatus_OS!       [2D,  uGIMP basis]                             |
|               06. resetmpstatus_OS_CPU!   [2D,  uGIMP basis]                             |
|               07. resetmpstatus_OS!       [3D,  uGIMP basis]                             |
|               08. resetmpstatus_OS_CPU!   [3D,  uGIMP basis]                             |
|               09. P2G_OS!                 [2D]                                           |
|               10. P2G_OS!                 [3D]                                           |
|               11. solvegrid_OS!           [2D]                                           |
|               12. solvegrid_a_OS!         [2D, acceleration]                             |
|               13. solvegrid_OS!           [3D]                                           |
|               14. solvegrid_a_OS!         [3D, acceleration]                             |
|               15. doublemapping1_OS!      [2D]                                           |
|               16. doublemapping1_a_OS!    [2D]                                           |
|               17. doublemapping1_OS!      [3D]                                           |
|               18. doublemapping1_a_OS!    [3D]                                           |
|               19. doublemapping2_OS!      [2D]                                           |
|               20. doublemapping2_OS!      [3D]                                           |
|               21. doublemapping3_OS!      [2D]                                           |
|               22. doublemapping3_OS!      [3D]                                           |
|               23. G2P_OS!                 [2D]                                           |
|               24. G2P_OS!                 [3D]                                           |
|               25. vollock1_OS!            [2D]                                           |
|               26. vollock1_OS!            [3D]                                           |
|               27. vollock2_OS!            [2D]                                           |
|               28. vollock2_OS!            [3D]                                           |
+==========================================================================================#

export resetgridstatus_OS!
export resetmpstatus_OS!, resetmpstatus_OS_CPU!
export P2G_OS! 
export solvegrid_OS!, solvegrid_a_OS!
export doublemapping1_OS!, doublemapping1_a_OS!
export doublemapping2_OS!
export doublemapping3_OS!
export G2P_OS! 
export vollock1_OS!
export vollock2_OS!

"""
    resetgridstatus_OS!(grid::DeviceGrid2D{T1, T2})

Description:
---
Reset some variables for the grid.
"""
@kernel inbounds = true function resetgridstatus_OS!(
    grid::DeviceGrid2D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ grid.ni
        if ix ≤ grid.nc
            grid.σm[ix] = T2(0.0)
            grid.Ω[ix]  = T2(0.0)
        end
        grid.ms[ix]    = T2(0.0)
        grid.ps[ix, 1] = T2(0.0)
        grid.ps[ix, 2] = T2(0.0)
        grid.fs[ix, 1] = T2(0.0)
        grid.fs[ix, 2] = T2(0.0)
    end
end

"""
    resetgridstatus_OS!(grid::DeviceGrid3D{T1, T2})

Description:
---
Reset some variables for the grid.
"""
@kernel inbounds = true function resetgridstatus_OS!(
    grid::DeviceGrid3D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ grid.ni
        if ix ≤ grid.nc
            grid.σm[ix] = T2(0.0)
            grid.Ω[ix]  = T2(0.0)
        end
        grid.ms[ix]    = T2(0.0)
        grid.ps[ix, 1] = T2(0.0)
        grid.ps[ix, 2] = T2(0.0)
        grid.ps[ix, 3] = T2(0.0)
        grid.fs[ix, 1] = T2(0.0)
        grid.fs[ix, 2] = T2(0.0)
        grid.fs[ix, 3] = T2(0.0)
    end
end

@kernel inbounds = true function resetmpstatus_OS!(
    grid::    DeviceGrid2D{T1, T2},
    mp  ::DeviceParticle2D{T1, T2},
        ::Val{:linear}
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ mp.np
        # update momentum and mass
        mp.ms[ix]    = mp.Ω[ix]  * mp.ρs[ix]
        mp.ps[ix, 1] = mp.ms[ix] * mp.vs[ix, 1]
        mp.ps[ix, 2] = mp.ms[ix] * mp.vs[ix, 2]
        # compute particle to cell and particle to node index
        mp.p2c[ix] = unsafe_trunc(T1,
            cld(mp.ξ[ix, 2] - grid.y1, grid.dy) +
            fld(mp.ξ[ix, 1] - grid.x1, grid.dx) * grid.ncy)
        @KAunroll for iy in Int32(1):Int32(mp.NIC)
            p2n = getP2N_linear(grid, mp.p2c[ix], iy)
            mp.p2n[ix, iy] = p2n
            # compute distance between particle and related nodes
            Δdx = mp.ξ[ix, 1] - grid.ξ[p2n, 1]
            Δdy = mp.ξ[ix, 2] - grid.ξ[p2n, 2]
            # compute basis function
            Nx, dNx = linearBasis(Δdx, grid.dx)
            Ny, dNy = linearBasis(Δdy, grid.dy)
            mp.Nij[ix, iy] =  Nx * Ny
            mp.∂Nx[ix, iy] = dNx * Ny # x-gradient shape function
            mp.∂Ny[ix, iy] = dNy * Nx # y-gradient shape function
        end
    end
end

@kernel inbounds = true function resetmpstatus_OS!(
    grid::    DeviceGrid3D{T1, T2},
    mp  ::DeviceParticle3D{T1, T2},
        ::Val{:linear}
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ mp.np
        # update momentum and mass
        mp.ms[ix]    = mp.Ω[ix]  * mp.ρs[ix]
        mp.ps[ix, 1] = mp.ms[ix] * mp.vs[ix, 1]
        mp.ps[ix, 2] = mp.ms[ix] * mp.vs[ix, 2]
        mp.ps[ix, 3] = mp.ms[ix] * mp.vs[ix, 3]
        # compute particle to cell and particle to node index
        mp.p2c[ix] = unsafe_trunc(T1, 
            cld(mp.ξ[ix, 2] - grid.y1, grid.dy) +
            fld(mp.ξ[ix, 3] - grid.z1, grid.dz) * grid.ncy * grid.ncx +
            fld(mp.ξ[ix, 1] - grid.x1, grid.dx) * grid.ncy)
        @KAunroll for iy in Int32(1):Int32(mp.NIC)
            p2n = getP2N_linear(grid, mp.p2c[ix], iy)
            mp.p2n[ix, iy] = p2n
            # compute distance between particle and related nodes
            Δdx = mp.ξ[ix, 1] - grid.ξ[p2n, 1]
            Δdy = mp.ξ[ix, 2] - grid.ξ[p2n, 2]
            Δdz = mp.ξ[ix, 3] - grid.ξ[p2n, 3]
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
end

"""
    resetmpstatus_OS!(grid::DeviceGrid2D{T1, T2}, mp::DeviceParticle2D{T1, T2}, 
        ::Val{:uGIMP})

Description:
---
1. Get topology between particle and grid.
2. Compute the value of basis function (uGIMP).
3. Update particle mass and momentum.
"""
@kernel inbounds = true function resetmpstatus_OS!(
    grid::    DeviceGrid2D{T1, T2},
    mp  ::DeviceParticle2D{T1, T2},
        ::Val{:uGIMP}
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ mp.np
        smem = @localmem T2 Int32(13)
        smem[1]  = grid.dx
        smem[2]  = mp.dx
        smem[3]  = T2(1.0) / (T2(4.0) * grid.dx * mp.dx)
        smem[4]  = T2(1.0) / (grid.dx * mp.dx)
        smem[5]  = T2(1.0) / grid.dx
        smem[6]  = T2(0.5) * mp.dx
        smem[7]  = grid.dy
        smem[8]  = mp.dy
        smem[9]  = T2(1.0) / (T2(4.0) * grid.dy * mp.dy)
        smem[10] = T2(1.0) / (grid.dy * mp.dy)
        smem[11] = T2(1.0) / grid.dy
        smem[12] = T2(0.5) * mp.dy
        # update mass and momentum
        mp.ms[ix]    = mp.Ω[ix]  * mp.ρs[ix]
        mp.ps[ix, 1] = mp.ms[ix] * mp.vs[ix, 1]
        mp.ps[ix, 2] = mp.ms[ix] * mp.vs[ix, 2]
        # p2c index
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
                Nx, dNx = uGIMPbasisx(Δdx, smem)
                Ny, dNy = uGIMPbasisy(Δdy, smem)
                mp.Nij[ix, viy] =  Nx * Ny
                mp.∂Nx[ix, viy] = dNx * Ny # x-gradient shape function
                mp.∂Ny[ix, viy] = dNy * Nx # y-gradient shape function
                mp.p2n[ix, viy] = p2n
                viy += T1(1)
            end
            viy > mp.NIC && break
        end
    end
end

@kernel inbounds = true function resetmpstatus_OS_CPU!(
    grid::    DeviceGrid2D{T1, T2},
    mp  ::DeviceParticle2D{T1, T2},
        ::Val{:uGIMP}
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ mp.np
        # update mass and momentum
        mp.ms[ix]    = mp.Ω[ix]  * mp.ρs[ix]
        mp.ps[ix, 1] = mp.ms[ix] * mp.vs[ix, 1]
        mp.ps[ix, 2] = mp.ms[ix] * mp.vs[ix, 2]
        # p2c index
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
end

"""
    resetmpstatus_OS!(grid::DeviceGrid3D{T1, T2}, mp::DeviceParticle3D{T1, T2},
        ::Val{:uGIMP})

Description:
---
1. Get topology between particle and grid.
2. Compute the value of basis function (uGIMP).
3. Update particle mass and momentum.
"""
@kernel inbounds = true function resetmpstatus_OS!(
    grid::    DeviceGrid3D{T1, T2},
    mp  ::DeviceParticle3D{T1, T2},
        ::Val{:uGIMP}
) where {T1, T2}
    ix = @index(Global)
    smem = @localmem T2 Int32(19)
    smem[1]  = grid.dx
    smem[2]  = mp.dx
    smem[3]  = T2(1.0) / (T2(4.0) * grid.dx * mp.dx)
    smem[4]  = T2(1.0) / (grid.dx * mp.dx)
    smem[5]  = T2(1.0) / grid.dx
    smem[6]  = T2(0.5) * mp.dx
    smem[7]  = grid.dy
    smem[8]  = mp.dy
    smem[9]  = T2(1.0) / (T2(4.0) * grid.dy * mp.dy)
    smem[10] = T2(1.0) / (grid.dy * mp.dy)
    smem[11] = T2(1.0) / grid.dy
    smem[12] = T2(0.5) * mp.dy
    smem[13] = grid.dz
    smem[14] = mp.dz
    smem[15] = T2(1.0) / (T2(4.0) * grid.dz * mp.dz)
    smem[16] = T2(1.0) / (grid.dz* mp.dz)
    smem[17] = T2(1.0) / grid.dz
    smem[18] = T2(0.5) * mp.dz
    if ix ≤ mp.np
        mpξ1 = mp.ξ[ix, 1]
        mpξ2 = mp.ξ[ix, 2]
        mpξ3 = mp.ξ[ix, 3]
        mpms = mp.Ω[ix] * mp.ρs[ix]
        # update particle mass and momentum
        mp.ms[ix]    = mpms
        mp.ps[ix, 1] = mpms * mp.vs[ix, 1]
        mp.ps[ix, 2] = mpms * mp.vs[ix, 2]
        mp.ps[ix, 3] = mpms * mp.vs[ix, 3]
        # p2c index
        mp.p2c[ix] = unsafe_trunc(T1,
            cld(mpξ2 - grid.y1, grid.dy) +
            fld(mpξ3 - grid.z1, grid.dz) * grid.ncy * grid.ncx +
            fld(mpξ1 - grid.x1, grid.dx) * grid.ncy)
        # reset Nij, ∂Nx, ∂Ny, ∂Nz
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
            # compute distance between particle and related nodes
            Δdx = mpξ1 - grid.ξ[p2n, 1]
            Δdy = mpξ2 - grid.ξ[p2n, 2]
            Δdz = mpξ3 - grid.ξ[p2n, 3]
            # compute basis function
            if abs(Δdx) < (grid.dx + T2(0.5) * mp.dx) &&
               abs(Δdy) < (grid.dy + T2(0.5) * mp.dy) &&
               abs(Δdz) < (grid.dz + T2(0.5) * mp.dz)
                Nx, dNx = uGIMPbasisx(Δdx, smem)
                Ny, dNy = uGIMPbasisy(Δdy, smem)
                Nz, dNz = uGIMPbasisz(Δdz, smem)
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
end

@kernel inbounds = true function resetmpstatus_OS_CPU!(
    grid::    DeviceGrid3D{T1, T2},
    mp  ::DeviceParticle3D{T1, T2},
        ::Val{:uGIMP}
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ mp.np
        mpξ1 = mp.ξ[ix, 1]
        mpξ2 = mp.ξ[ix, 2]
        mpξ3 = mp.ξ[ix, 3]
        mpms = mp.Ω[ix] * mp.ρs[ix]
        # update particle mass and momentum
        mp.ms[ix]    = mpms
        mp.ps[ix, 1] = mpms * mp.vs[ix, 1]
        mp.ps[ix, 2] = mpms * mp.vs[ix, 2]
        mp.ps[ix, 3] = mpms * mp.vs[ix, 3]
        # p2c index
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
end

"""
    P2G_OS!(grid::DeviceGrid2D{T1, T2}, mp::DeviceParticle2D{T1, T2}, gravity::T2)

Description:
---
P2G procedure for scattering the mass, momentum, and forces from particles to grid.
"""
@kernel inbounds = true function P2G_OS!(
    grid   ::    DeviceGrid2D{T1, T2},
    mp     ::DeviceParticle2D{T1, T2},
    gravity::T2
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ mp.np
        @KAunroll for iy in Int32(1):Int32(mp.NIC)
            Ni = mp.Nij[ix, iy]
            if Ni ≠ T2(0.0)
                ∂Nx = mp.∂Nx[ix, iy]
                ∂Ny = mp.∂Ny[ix, iy]
                p2n = mp.p2n[ix, iy]
                vol = mp.Ω[ix]
                NiM = mp.ms[ix] * Ni
                # compute nodal mass
                @KAatomic grid.ms[p2n] += NiM
                # compute nodal momentum
                @KAatomic grid.ps[p2n, 1] += Ni * mp.ps[ix, 1]
                @KAatomic grid.ps[p2n, 2] += Ni * mp.ps[ix, 2]
                # compute nodal total force for solid
                @KAatomic grid.fs[p2n, 1] += -vol * (∂Nx * mp.σij[ix, 1]  + 
                                                     ∂Ny * mp.σij[ix, 4])
                @KAatomic grid.fs[p2n, 2] += -vol * (∂Ny * mp.σij[ix, 2]  + 
                                                     ∂Nx * mp.σij[ix, 4]) + NiM * gravity
            end
        end
    end
end

"""
    P2G_OS!(grid::DeviceGrid3D{T1, T2}, mp::DeviceParticle3D{T1, T2}, gravity::T2)

Description:
---
P2G procedure for scattering the mass, momentum, and forces from particles to grid.
"""
@kernel inbounds = true function P2G_OS!(
    grid   ::    DeviceGrid3D{T1, T2},
    mp     ::DeviceParticle3D{T1, T2},
    gravity::T2
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ mp.np
        @KAunroll for iy in Int32(1):Int32(mp.NIC)
            Ni = mp.Nij[ix, iy]
            if Ni ≠ T2(0.0)
                ∂Nx = mp.∂Nx[ix, iy]
                ∂Ny = mp.∂Ny[ix, iy]
                ∂Nz = mp.∂Nz[ix, iy]
                p2n = mp.p2n[ix, iy]
                vol = mp.Ω[ix]
                NiM = mp.ms[ix] * Ni
                # compute nodal mass
                @KAatomic grid.ms[p2n] += NiM
                # compute nodal momentum
                @KAatomic grid.ps[p2n, 1] += Ni * mp.ps[ix, 1]
                @KAatomic grid.ps[p2n, 2] += Ni * mp.ps[ix, 2]
                @KAatomic grid.ps[p2n, 3] += Ni * mp.ps[ix, 3]
                # compute nodal total force for solid
                @KAatomic grid.fs[p2n, 1] += -vol * (∂Nx * mp.σij[ix, 1]  + 
                                                     ∂Ny * mp.σij[ix, 4]  + 
                                                     ∂Nz * mp.σij[ix, 6])
                @KAatomic grid.fs[p2n, 2] += -vol * (∂Ny * mp.σij[ix, 2]  + 
                                                     ∂Nx * mp.σij[ix, 4]  + 
                                                     ∂Nz * mp.σij[ix, 5])
                @KAatomic grid.fs[p2n, 3] += -vol * (∂Nz * mp.σij[ix, 3]  + 
                                                     ∂Nx * mp.σij[ix, 6]  + 
                                                     ∂Ny * mp.σij[ix, 5]) + NiM * gravity
            end
        end
    end
end


"""
    solvegrid_OS!(grid::DeviceGrid2D{T1, T2}, bc::DeviceVBoundary2D{T1, T2}, ΔT::T2, ζs::T2)

Description:
---
1. Solve equations on grid.
2. Add boundary condition.
"""
@kernel inbounds = true function solvegrid_OS!(
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
        # reset grid momentum
        grid.ps[ix, 1] = T2(0.0)
        grid.ps[ix, 2] = T2(0.0)
    end
end

"""
    solvegrid_a_OS!(grid::DeviceGrid2D{T1, T2}, bc::DeviceVBoundary2D{T1, T2}, ΔT::T2, 
        ζs::T2)

Description:
---
1. Solve equations on grid.
2. Add boundary condition.
3. Update particle velocity based on the acceleration.
"""
@kernel inbounds = true function solvegrid_a_OS!(
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
        grid.vsT[ix, 1] = grid.vs[ix, 1] + Fs_x * ΔT * ms_denom
        grid.vsT[ix, 2] = grid.vs[ix, 2] + Fs_y * ΔT * ms_denom
        # update nodal acceleration
        grid.as[ix, 1] = Fs_x * ms_denom
        grid.as[ix, 2] = Fs_y * ms_denom
        # boundary condition
        bc.vx_s_idx[ix] ≠ T1(0) ? grid.vsT[ix, 1] = bc.vx_s_val[ix] : nothing
        bc.vy_s_idx[ix] ≠ T1(0) ? grid.vsT[ix, 2] = bc.vy_s_val[ix] : nothing
        bc.vx_s_idx[ix] ≠ T1(0) ? grid.as[ix, 1] = bc.vx_s_val[ix] : nothing
        bc.vy_s_idx[ix] ≠ T1(0) ? grid.as[ix, 2] = bc.vy_s_val[ix] : nothing
        # reset grid momentum
        grid.ps[ix, 1] = T2(0.0)
        grid.ps[ix, 2] = T2(0.0)
    end
end

"""
    solvegrid_a_OS!(grid::DeviceGrid3D{T1, T2}, bc::DeviceVBoundary3D{T1, T2}, ΔT::T2, 
        ζs::T2)

Description:
---
1. Solve equations on grid.
2. Add boundary condition.
3. Update particle velocity based on the acceleration.
"""
@kernel inbounds = true function solvegrid_a_OS!(
    grid::     DeviceGrid3D{T1, T2},
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
        grid.vsT[ix, 1] = grid.vs[ix, 1] + Fs_x * ΔT * ms_denom
        grid.vsT[ix, 2] = grid.vs[ix, 2] + Fs_y * ΔT * ms_denom
        grid.vsT[ix, 3] = grid.vs[ix, 3] + Fs_z * ΔT * ms_denom
        # update nodal acceleration
        grid.as[ix, 1] = Fs_x * ms_denom
        grid.as[ix, 2] = Fs_y * ms_denom
        grid.as[ix, 3] = Fs_z * ms_denom
        # boundary condition
        bc.vx_s_idx[ix] ≠ T1(0) ? grid.vsT[ix, 1] = bc.vx_s_val[ix] : nothing
        bc.vy_s_idx[ix] ≠ T1(0) ? grid.vsT[ix, 2] = bc.vy_s_val[ix] : nothing
        bc.vz_s_idx[ix] ≠ T1(0) ? grid.vsT[ix, 3] = bc.vz_s_val[ix] : nothing
        bc.vx_s_idx[ix] ≠ T1(0) ? grid.as[ix, 1] = bc.vx_s_val[ix] : nothing
        bc.vy_s_idx[ix] ≠ T1(0) ? grid.as[ix, 2] = bc.vy_s_val[ix] : nothing
        bc.vz_s_idx[ix] ≠ T1(0) ? grid.as[ix, 3] = bc.vz_s_val[ix] : nothing
        # reset grid momentum
        grid.ps[ix, 1] = T2(0.0)
        grid.ps[ix, 2] = T2(0.0)
        grid.ps[ix, 3] = T2(0.0)
    end
end

"""
    solvegrid_OS!(grid::DeviceGrid3D{T1, T2}, bc::DeviceVBoundary3D{T1, T2}, ΔT::T2, ζs::T2)

Description:
---
1. Solve equations on grid.
2. Add boundary condition.
"""
@kernel inbounds = true function solvegrid_OS!(
    grid::     DeviceGrid3D{T1, T2},
    bc  ::DeviceVBoundary3D{T1, T2},
    ΔT  ::T2,
    ζs  ::T2
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ grid.ni
        ms_denom = grid.ms[ix] < eps(T2) ? T2(0.0) : inv(grid.ms[ix])
        # compute nodal velocity
        Ps_1 = grid.ps[ix, 1]
        Ps_2 = grid.ps[ix, 2]
        Ps_3 = grid.ps[ix, 3]
        grid.vs[ix, 1] = Ps_1 * ms_denom
        grid.vs[ix, 2] = Ps_2 * ms_denom
        grid.vs[ix, 3] = Ps_3 * ms_denom
        # damping force for solid
        dampvs = -ζs * sqrt(grid.fs[ix, 1] * grid.fs[ix, 1] +
                            grid.fs[ix, 2] * grid.fs[ix, 2] +
                            grid.fs[ix, 3] * grid.fs[ix, 3])
        # compute nodal total force for mixture
        Fs_x = grid.fs[ix, 1] + dampvs * sign(grid.vs[ix, 1])
        Fs_y = grid.fs[ix, 2] + dampvs * sign(grid.vs[ix, 2])
        Fs_z = grid.fs[ix, 3] + dampvs * sign(grid.vs[ix, 3])
        # update nodal velocity
        grid.vsT[ix, 1] = (Ps_1 + Fs_x * ΔT) * ms_denom
        grid.vsT[ix, 2] = (Ps_2 + Fs_y * ΔT) * ms_denom
        grid.vsT[ix, 3] = (Ps_3 + Fs_z * ΔT) * ms_denom
        # boundary condition
        bc.vx_s_idx[ix] ≠ T1(0) ? grid.vsT[ix, 1] = bc.vx_s_val[ix] : nothing
        bc.vy_s_idx[ix] ≠ T1(0) ? grid.vsT[ix, 2] = bc.vy_s_val[ix] : nothing
        bc.vz_s_idx[ix] ≠ T1(0) ? grid.vsT[ix, 3] = bc.vz_s_val[ix] : nothing
        # reset grid momentum
        grid.ps[ix, 1] = T2(0.0)
        grid.ps[ix, 2] = T2(0.0)
        grid.ps[ix, 3] = T2(0.0)
    end
end

"""
    doublemapping1_OS!(grid::DeviceGrid2D{T1, T2}, mp::DeviceParticle2D{T1, T2},
        attr::DeviceProperty{T1, T2}, ΔT::T2, FLIP::T2, PIC::T2)

Description:
---
1. Solve equations on grid.
2. Compute CFL conditions.
"""
@kernel inbounds = true function doublemapping1_OS!(
    grid::    DeviceGrid2D{T1, T2},
    mp  ::DeviceParticle2D{T1, T2},
    attr::  DeviceProperty{T1, T2},
    ΔT  ::T2,
    FLIP::T2,
    PIC ::T2
) where {T1, T2}
    # 4/3 = 1.333333
    ix = @index(Global)
    # update particle position & velocity
    if ix ≤ mp.np
        nid = attr.nid[ix]
        Ks  = attr.Ks[nid]
        Gs  = attr.Gs[nid]
        tmp_vx_s1 = tmp_vx_s2 = T2(0.0)
        tmp_vy_s1 = tmp_vy_s2 = T2(0.0)
        tmp_pos_x = tmp_pos_y = T2(0.0)
        # update particle position
        @KAunroll for iy in Int32(1):Int32(mp.NIC)
            Ni = mp.Nij[ix, iy]
            if Ni ≠ T2(0.0)
                p2n = mp.p2n[ix, iy]
                tmp_pos_x += Ni *  grid.vsT[p2n, 1]
                tmp_pos_y += Ni *  grid.vsT[p2n, 2]
                tmp_vx_s1 += Ni * (grid.vsT[p2n, 1] - grid.vs[p2n, 1])
                tmp_vx_s2 += Ni *  grid.vsT[p2n, 1]
                tmp_vy_s1 += Ni * (grid.vsT[p2n, 2] - grid.vs[p2n, 2])
                tmp_vy_s2 += Ni *  grid.vsT[p2n, 2]
            end
        end
        mp.ξ[ix, 1] += ΔT * tmp_pos_x
        mp.ξ[ix, 2] += ΔT * tmp_pos_y
        # update particle velocity
        mp.vs[ix, 1] = FLIP * (mp.vs[ix, 1] + tmp_vx_s1) + PIC * tmp_vx_s2
        mp.vs[ix, 2] = FLIP * (mp.vs[ix, 2] + tmp_vy_s1) + PIC * tmp_vy_s2
        # update particle momentum
        mp.ps[ix, 1] = mp.ms[ix] * mp.vs[ix, 1]
        mp.ps[ix, 2] = mp.ms[ix] * mp.vs[ix, 2]
        # # update CFL conditions
        # sqr = sqrt((Ks + Gs * T2(1.333333)) / mp.ρs[ix]) # 4/3 ≈ 1.333333
        # cd_sx = grid.dx / (sqr + abs(mp.vs[ix, 1]))
        # cd_sy = grid.dy / (sqr + abs(mp.vs[ix, 2]))
        # mp.cfl[ix] = min(cd_sx, cd_sy)
    end
end

@kernel inbounds = true function doublemapping1_a_OS!(
    grid::    DeviceGrid2D{T1, T2},
    mp  ::DeviceParticle2D{T1, T2},
    attr::  DeviceProperty{T1, T2},
    ΔT  ::T2,
    FLIP::T2,
    PIC ::T2
) where {T1, T2}
    # 4/3 = 1.333333
    ix = @index(Global)
    # update particle position & velocity
    if ix ≤ mp.np
        nid = attr.nid[ix]
        Ks  = attr.Ks[nid]
        Gs  = attr.Gs[nid]
        tmp_vx = tmp_vy = tmp_px = tmp_py = tmp_Tx = tmp_Ty = T2(0.0)
        # update particle position
        @KAunroll for iy in Int32(1):Int32(mp.NIC)
            Ni = mp.Nij[ix, iy]
            if Ni ≠ T2(0.0)
                p2n = mp.p2n[ix, iy]
                tmp_px += Ni * grid.vsT[p2n, 1]
                tmp_py += Ni * grid.vsT[p2n, 2]
                tmp_vx += Ni * grid.as[p2n, 1]
                tmp_vy += Ni * grid.as[p2n, 2]
                tmp_Tx += Ni * grid.vs[p2n, 1]
                tmp_Ty += Ni * grid.vs[p2n, 2]
            end
        end
        mp.ξ[ix, 1] += tmp_px * ΔT
        mp.ξ[ix, 2] += tmp_py * ΔT
        # update particle velocity
        mp.vs[ix, 1] = FLIP * mp.vs[ix, 1] + PIC * tmp_Tx + tmp_vx * ΔT
        mp.vs[ix, 2] = FLIP * mp.vs[ix, 2] + PIC * tmp_Ty + tmp_vy * ΔT
        # update particle momentum
        mp.ps[ix, 1] = mp.ms[ix] * mp.vs[ix, 1]
        mp.ps[ix, 2] = mp.ms[ix] * mp.vs[ix, 2]
        # # update CFL conditions
        # sqr = sqrt((Ks + Gs * T2(1.333333)) / mp.ρs[ix]) # 4/3 ≈ 1.333333
        # cd_sx = grid.dx / (sqr + abs(mp.vs[ix, 1]))
        # cd_sy = grid.dy / (sqr + abs(mp.vs[ix, 2]))
        # mp.cfl[ix] = min(cd_sx, cd_sy)
    end
end

"""
    doublemapping1_OS!(grid::DeviceGrid3D{T1, T2}, mp::DeviceParticle3D{T1, T2}, 
        attr::DeviceProperty{T1, T2}, ΔT::T2, FLIP::T2, PIC::T2)

Description:
---
1. Solve equations on grid.
2. Compute CFL conditions.
"""
@kernel inbounds = true function doublemapping1_OS!(
    grid::    DeviceGrid3D{T1, T2},
    mp  ::DeviceParticle3D{T1, T2},
    attr::  DeviceProperty{T1, T2},
    ΔT  ::T2,
    FLIP::T2,
    PIC ::T2
) where {T1, T2}
    ix = @index(Global)
    # update particle position & velocity
    if ix ≤ mp.np
        nid = attr.nid[ix]
        Ks  = attr.Ks[nid]
        Gs  = attr.Gs[nid]
        tmp_vx_s1 = tmp_vx_s2 = tmp_vy_s1 = T2(0.0)
        tmp_vy_s2 = tmp_vz_s1 = tmp_vz_s2 = T2(0.0)
        tmp_pos_x = tmp_pos_y = tmp_pos_z = T2(0.0)
        # update particle position
        @KAunroll for iy in Int32(1):Int32(mp.NIC)
            Ni = mp.Nij[ix, iy]
            if Ni ≠ T2(0.0)
                p2n   = mp.p2n[ix, iy]
                Vs_T1 = grid.vsT[p2n, 1]
                Vs_T2 = grid.vsT[p2n, 2]
                Vs_T3 = grid.vsT[p2n, 3]
                tmp_pos_x += Ni *  Vs_T1
                tmp_pos_y += Ni *  Vs_T2
                tmp_pos_z += Ni *  Vs_T3
                tmp_vx_s1 += Ni * (Vs_T1 - grid.vs[p2n, 1])
                tmp_vx_s2 += Ni *  Vs_T1
                tmp_vy_s1 += Ni * (Vs_T2 - grid.vs[p2n, 2])
                tmp_vy_s2 += Ni *  Vs_T2
                tmp_vz_s1 += Ni * (Vs_T3 - grid.vs[p2n, 3])
                tmp_vz_s2 += Ni *  Vs_T3
            end
        end
        # update particle position
        mp.ξ[ix, 1] += ΔT * tmp_pos_x
        mp.ξ[ix, 2] += ΔT * tmp_pos_y
        mp.ξ[ix, 3] += ΔT * tmp_pos_z
        # update particle velocity
        mp.vs[ix, 1] = FLIP * (mp.vs[ix, 1] + tmp_vx_s1) + PIC * tmp_vx_s2
        mp.vs[ix, 2] = FLIP * (mp.vs[ix, 2] + tmp_vy_s1) + PIC * tmp_vy_s2
        mp.vs[ix, 3] = FLIP * (mp.vs[ix, 3] + tmp_vz_s1) + PIC * tmp_vz_s2
        # update particle momentum
        mp.ps[ix, 1] = mp.ms[ix] * mp.vs[ix, 1]
        mp.ps[ix, 2] = mp.ms[ix] * mp.vs[ix, 2]
        mp.ps[ix, 3] = mp.ms[ix] * mp.vs[ix, 3]
        # update CFL conditions
        # sqr = sqrt((Ks + Gs * T2(1.333333)) / mp.ρs[ix])
        # cd_sx = grid.dx / (sqr + abs(mp.vs[ix, 1]))
        # cd_sy = grid.dy / (sqr + abs(mp.vs[ix, 2]))
        # cd_sz = grid.dz / (sqr + abs(mp.vs[ix, 3]))
        # mp.cfl[ix] = min(cd_sx, cd_sy, cd_sz)
    end
end

@kernel inbounds = true function doublemapping1_a_OS!(
    grid::    DeviceGrid3D{T1, T2},
    mp  ::DeviceParticle3D{T1, T2},
    attr::  DeviceProperty{T1, T2},
    ΔT  ::T2,
    FLIP::T2,
    PIC ::T2
) where {T1, T2}
    # 4/3 = 1.333333
    ix = @index(Global)
    # update particle position & velocity
    if ix ≤ mp.np
        nid = attr.nid[ix]
        Ks  = attr.Ks[nid]
        Gs  = attr.Gs[nid]
        tmp_vx = tmp_vy = tmp_vz = T2(0.0)
        tmp_px = tmp_py = tmp_pz = T2(0.0)
        tmp_Tx = tmp_Ty = tmp_Tz = T2(0.0)
        # update particle position
        @KAunroll for iy in Int32(1):Int32(mp.NIC)
            Ni = mp.Nij[ix, iy]
            if Ni ≠ T2(0.0)
                p2n = mp.p2n[ix, iy]
                tmp_px += Ni * grid.vsT[p2n, 1]
                tmp_py += Ni * grid.vsT[p2n, 2]
                tmp_pz += Ni * grid.vsT[p2n, 3]
                tmp_vx += Ni * grid.as[p2n, 1]
                tmp_vy += Ni * grid.as[p2n, 2]
                tmp_vz += Ni * grid.as[p2n, 3]
                tmp_Tx += Ni * grid.vs[p2n, 1]
                tmp_Ty += Ni * grid.vs[p2n, 2]
                tmp_Tz += Ni * grid.vs[p2n, 3]
            end
        end
        mp.ξ[ix, 1] += tmp_px * ΔT
        mp.ξ[ix, 2] += tmp_py * ΔT
        mp.ξ[ix, 3] += tmp_pz * ΔT
        # update particle velocity
        mp.vs[ix, 1] = FLIP * mp.vs[ix, 1] + PIC * tmp_Tx + tmp_vx * ΔT
        mp.vs[ix, 2] = FLIP * mp.vs[ix, 2] + PIC * tmp_Ty + tmp_vy * ΔT
        mp.vs[ix, 3] = FLIP * mp.vs[ix, 3] + PIC * tmp_Tz + tmp_vz * ΔT
        # update particle momentum
        mp.ps[ix, 1] = mp.ms[ix] * mp.vs[ix, 1]
        mp.ps[ix, 2] = mp.ms[ix] * mp.vs[ix, 2]
        mp.ps[ix, 3] = mp.ms[ix] * mp.vs[ix, 3]
        # # update CFL conditions
        # sqr = sqrt((Ks + Gs * T2(1.333333)) / mp.ρs[ix])
        # cd_sx = grid.dx / (sqr + abs(mp.vs[ix, 1]))
        # cd_sy = grid.dy / (sqr + abs(mp.vs[ix, 2]))
        # cd_sz = grid.dz / (sqr + abs(mp.vs[ix, 3]))
        # mp.cfl[ix] = min(cd_sx, cd_sy, cd_sz)
    end
end

"""
    doublemapping2_OS!(grid::DeviceGrid2D{T1, T2}, mp::DeviceParticle2D{T1, T2})

Description:
---
Scatter momentum from particles to grid.
"""
@kernel inbounds = true function doublemapping2_OS!(
    grid::    DeviceGrid2D{T1, T2},
    mp  ::DeviceParticle2D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ mp.np
        # update particle position & velocity
        @KAunroll for iy in Int32(1):Int32(mp.NIC)
            Ni = mp.Nij[ix, iy]
            if Ni ≠ T2(0.0)
                p2n = mp.p2n[ix, iy]
                @KAatomic grid.ps[p2n, 1] += mp.ps[ix, 1] * Ni
                @KAatomic grid.ps[p2n, 2] += mp.ps[ix, 2] * Ni
            end
        end
    end
end

"""
    doublemapping2_OS!(grid::DeviceGrid3D{T1, T2}, mp::DeviceParticle3D{T1, T2})

Description:
---
Scatter momentum from particles to grid.
"""
@kernel inbounds = true function doublemapping2_OS!(
    grid::    DeviceGrid3D{T1, T2},
    mp  ::DeviceParticle3D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ mp.np
        # update particle position & velocity
        @KAunroll for iy in Int32(1):Int32(mp.NIC)
            Ni = mp.Nij[ix, iy]
            if Ni ≠ T2(0.0)
                p2n = mp.p2n[ix, iy]
                @KAatomic grid.ps[p2n, 1] += mp.ps[ix, 1] * Ni
                @KAatomic grid.ps[p2n, 2] += mp.ps[ix, 2] * Ni
                @KAatomic grid.ps[p2n, 3] += mp.ps[ix, 3] * Ni
            end
        end
    end
end

"""
    doublemapping3_OS!(grid::DeviceGrid2D{T1, T2}, bc::DeviceVBoundary2D{T1, T2}, ΔT::T2)

Description:
---
Solve equations on grid.
"""
@kernel inbounds = true function doublemapping3_OS!(
    grid::     DeviceGrid2D{T1, T2},
    bc  ::DeviceVBoundary2D{T1, T2},
    ΔT  ::T2
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ grid.ni
        ms_denom = grid.ms[ix] < eps(T2) ? T2(0.0) : inv(grid.ms[ix])
        # compute nodal velocities
        grid.vs[ix, 1] = grid.ps[ix, 1] * ms_denom
        grid.vs[ix, 2] = grid.ps[ix, 2] * ms_denom
        # fixed Dirichlet nodes
        bc.vx_s_idx[ix] ≠ T1(0) ? grid.vs[ix, 1] = bc.vx_s_val[ix] : nothing
        bc.vy_s_idx[ix] ≠ T1(0) ? grid.vs[ix, 2] = bc.vy_s_val[ix] : nothing
        # compute nodal displacement
        grid.Δus[ix, 1] = grid.vs[ix, 1] * ΔT
        grid.Δus[ix, 2] = grid.vs[ix, 2] * ΔT
    end
end

"""
    doublemapping3_OS!(grid::DeviceGrid3D{T1, T2}, bc::KernelVBoundary3D{T1, T2}, ΔT::T2)

Description:
---
Solve equations on grid.
"""
@kernel inbounds = true function doublemapping3_OS!(
    grid::     DeviceGrid3D{T1, T2},
    bc  ::DeviceVBoundary3D{T1, T2},
    ΔT  ::T2
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ grid.ni
        ms_denom = grid.ms[ix] < eps(T2) ? T2(0.0) : inv(grid.ms[ix])
        # compute nodal velocities
        grid.vs[ix, 1] = grid.ps[ix, 1] * ms_denom
        grid.vs[ix, 2] = grid.ps[ix, 2] * ms_denom
        grid.vs[ix, 3] = grid.ps[ix, 3] * ms_denom
        # fixed Dirichlet nodes
        bc.vx_s_idx[ix] ≠ T1(0) ? grid.vs[ix, 1] = bc.vx_s_val[ix] : nothing
        bc.vy_s_idx[ix] ≠ T1(0) ? grid.vs[ix, 2] = bc.vy_s_val[ix] : nothing
        bc.vz_s_idx[ix] ≠ T1(0) ? grid.vs[ix, 3] = bc.vz_s_val[ix] : nothing
        # compute nodal displacement
        grid.Δus[ix, 1] = grid.vs[ix, 1] * ΔT
        grid.Δus[ix, 2] = grid.vs[ix, 2] * ΔT
        grid.Δus[ix, 3] = grid.vs[ix, 3] * ΔT
    end
end

"""
    G2P_OS!(grid::DeviceGrid2D{T1, T2}, mp::DeviceParticle2D{T1, T2})

Description:
---
Update particle information.
"""
@kernel inbounds = true function G2P_OS!(
    grid::    DeviceGrid2D{T1, T2},
    mp  ::DeviceParticle2D{T1, T2},
    ΔT  ::T2
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ mp.np
        ΔT_1 = inv(ΔT)
        dF1 = dF2 = dF3 = dF4 = T2(0.0)
        @KAunroll for iy in Int32(1):Int32(mp.NIC)
            if mp.Nij[ix, iy] ≠ T2(0.0)
                p2n = mp.p2n[ix, iy]
                ∂Nx = mp.∂Nx[ix, iy]
                ∂Ny = mp.∂Ny[ix, iy]
                # compute solid incremental deformation gradient
                dF1 += grid.Δus[p2n, 1] * ∂Nx
                dF2 += grid.Δus[p2n, 1] * ∂Ny
                dF3 += grid.Δus[p2n, 2] * ∂Nx
                dF4 += grid.Δus[p2n, 2] * ∂Ny
            end
        end
        mp.ΔFs[ix, 1] = dF1
        mp.ΔFs[ix, 2] = dF2
        mp.ΔFs[ix, 3] = dF3
        mp.ΔFs[ix, 4] = dF4
        # strain rate (Second Invariant of Strain Rate Tensor)
        dϵxx = dF1 * ΔT_1
        dϵyy = dF4 * ΔT_1
        dϵxy = T2(0.5) * (dF2 + dF3) * ΔT_1
        mp.ϵv[ix] = sqrt(dϵxx * dϵxx + dϵyy * dϵyy + T2(2.0) * dϵxy * dϵxy)
        # compute strain increment 
        mp.Δϵijs[ix, 1] = dF1
        mp.Δϵijs[ix, 2] = dF4
        mp.Δϵijs[ix, 4] = dF2 + dF3
        # update strain tensor
        mp.ϵijs[ix, 1] += dF1
        mp.ϵijs[ix, 2] += dF4
        mp.ϵijs[ix, 4] += dF2 + dF3
        # deformation gradient matrix
        F1 = mp.F[ix, 1]; F2 = mp.F[ix, 2]; F3 = mp.F[ix, 3]; F4 = mp.F[ix, 4]      
        mp.F[ix, 1] = (dF1 + T2(1.0)) * F1 + dF2 * F3
        mp.F[ix, 2] = (dF1 + T2(1.0)) * F2 + dF2 * F4
        mp.F[ix, 3] = (dF4 + T2(1.0)) * F3 + dF3 * F1
        mp.F[ix, 4] = (dF4 + T2(1.0)) * F4 + dF3 * F2
        # update jacobian value and particle volume
        J = mp.F[ix, 1] * mp.F[ix, 4] - mp.F[ix, 2] * mp.F[ix, 3]
        mp.Ω[ix]  = J * mp.Ω0[ix]
        mp.ρs[ix] = mp.ρs0[ix] / J
    end
end

"""
    G2P_OS!(grid::DeviceGrid3D{T1, T2}, mp::DeviceParticle3D{T1, T2})

Description:
---
Update particle information.
"""
@kernel inbounds = true function G2P_OS!(
    grid::    DeviceGrid3D{T1, T2},
    mp  ::DeviceParticle3D{T1, T2},
    ΔT  ::T2
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ mp.np
        ΔT_1 = T2(1.0) / ΔT
        dF1 = dF2 = dF3 = dF4 = dF5 = dF6 = dF7 = dF8 = dF9 = T2(0.0)
        @KAunroll for iy in Int32(1):Int32(mp.NIC)
            if mp.Nij[ix, iy] ≠ T2(0.0)
                p2n = mp.p2n[ix, iy]
                ∂Nx = mp.∂Nx[ix, iy]; ds1 = grid.Δus[p2n, 1]
                ∂Ny = mp.∂Ny[ix, iy]; ds2 = grid.Δus[p2n, 2]
                ∂Nz = mp.∂Nz[ix, iy]; ds3 = grid.Δus[p2n, 3]
                # compute solid incremental deformation gradient
                dF1 += ds1 * ∂Nx; dF2 += ds1 * ∂Ny; dF3 += ds1 * ∂Nz
                dF4 += ds2 * ∂Nx; dF5 += ds2 * ∂Ny; dF6 += ds2 * ∂Nz
                dF7 += ds3 * ∂Nx; dF8 += ds3 * ∂Ny; dF9 += ds3 * ∂Nz
            end
        end
        mp.ΔFs[ix, 1] = dF1; mp.ΔFs[ix, 2] = dF2; mp.ΔFs[ix, 3] = dF3
        mp.ΔFs[ix, 4] = dF4; mp.ΔFs[ix, 5] = dF5; mp.ΔFs[ix, 6] = dF6
        mp.ΔFs[ix, 7] = dF7; mp.ΔFs[ix, 8] = dF8; mp.ΔFs[ix, 9] = dF9
        # strain rate (Second Invariant of Strain Rate Tensor)
        dϵxx = dF1 * ΔT_1
        dϵyy = dF5 * ΔT_1
        dϵzz = dF9 * ΔT_1
        dϵxy = T2(0.5) * (dF2 + dF4) * ΔT_1
        dϵyz = T2(0.5) * (dF6 + dF8) * ΔT_1
        dϵxz = T2(0.5) * (dF3 + dF7) * ΔT_1
        mp.ϵv[ix] = sqrt(dϵxx * dϵxx + dϵyy * dϵyy + dϵzz * dϵzz + 
            T2(2.0) * (dϵxy * dϵxy + dϵyz * dϵyz + dϵxz * dϵxz))
        # compute strain increment
        mp.Δϵijs[ix, 1] = dF1
        mp.Δϵijs[ix, 2] = dF5
        mp.Δϵijs[ix, 3] = dF9
        mp.Δϵijs[ix, 4] = dF2 + dF4
        mp.Δϵijs[ix, 5] = dF6 + dF8
        mp.Δϵijs[ix, 6] = dF3 + dF7
        # update strain tensor
        mp.ϵijs[ix, 1] += dF1
        mp.ϵijs[ix, 2] += dF5
        mp.ϵijs[ix, 3] += dF9
        mp.ϵijs[ix, 4] += dF2 + dF4
        mp.ϵijs[ix, 5] += dF6 + dF8
        mp.ϵijs[ix, 6] += dF3 + dF7
        # deformation gradient matrix
        F1 = mp.F[ix, 1]; F2 = mp.F[ix, 2]; F3 = mp.F[ix, 3]
        F4 = mp.F[ix, 4]; F5 = mp.F[ix, 5]; F6 = mp.F[ix, 6]
        F7 = mp.F[ix, 7]; F8 = mp.F[ix, 8]; F9 = mp.F[ix, 9]        
        mp.F[ix, 1] = (dF1 + T2(1.0)) * F1 + dF2 * F4 + dF3 * F7
        mp.F[ix, 2] = (dF1 + T2(1.0)) * F2 + dF2 * F5 + dF3 * F8
        mp.F[ix, 3] = (dF1 + T2(1.0)) * F3 + dF2 * F6 + dF3 * F9
        mp.F[ix, 4] = (dF5 + T2(1.0)) * F4 + dF4 * F1 + dF6 * F7
        mp.F[ix, 5] = (dF5 + T2(1.0)) * F5 + dF4 * F2 + dF6 * F8
        mp.F[ix, 6] = (dF5 + T2(1.0)) * F6 + dF4 * F3 + dF6 * F9
        mp.F[ix, 7] = (dF9 + T2(1.0)) * F7 + dF8 * F4 + dF7 * F1
        mp.F[ix, 8] = (dF9 + T2(1.0)) * F8 + dF8 * F5 + dF7 * F2
        mp.F[ix, 9] = (dF9 + T2(1.0)) * F9 + dF8 * F6 + dF7 * F3
        # update jacobian value and particle volume
        J = mp.F[ix, 1] * mp.F[ix, 5] * mp.F[ix, 9] + 
            mp.F[ix, 2] * mp.F[ix, 6] * mp.F[ix, 7] +
            mp.F[ix, 3] * mp.F[ix, 4] * mp.F[ix, 8] - 
            mp.F[ix, 7] * mp.F[ix, 5] * mp.F[ix, 3] -
            mp.F[ix, 8] * mp.F[ix, 6] * mp.F[ix, 1] - 
            mp.F[ix, 9] * mp.F[ix, 4] * mp.F[ix, 2] 
        mp.Ω[ix]  = J * mp.Ω0[ix]
        mp.ρs[ix] = mp.ρs0[ix] / J
    end
end

"""
    vollock1_OS!(grid::DeviceGrid2D{T1, T2}, mp::DeviceParticle2D{T1, T2})

Description:
---
Mapping mean stress and volume from particle to grid.
"""
@kernel inbounds = true function vollock1_OS!(
    grid::    DeviceGrid2D{T1, T2},
    mp  ::DeviceParticle2D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ mp.np
        p2c = mp.p2c[ix]
        vol = mp.Ω[ix]
        @KAatomic grid.σm[p2c] += vol * mp.σm[ix]
        @KAatomic grid.Ω[p2c]  += vol
    end
end

"""
    vollock1_OS!(grid::DeviceGrid3D{T1, T2}, mp::DeviceParticle3D{T1, T2})

Description:
---
Mapping mean stress and volume from particle to grid.
"""
@kernel inbounds = true function vollock1_OS!(
    grid::    DeviceGrid3D{T1, T2},
    mp  ::DeviceParticle3D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ mp.np
        p2c = mp.p2c[ix]
        vol = mp.Ω[ix]
        @KAatomic grid.σm[p2c] += vol * mp.σm[ix]
        @KAatomic grid.Ω[p2c]  += vol
    end
end

"""
    vollock2_OS!(grid::DeviceGrid2D{T1, T2}, mp::DeviceParticle2D{T1, T2})

Description:
---
Mapping back mean stress and volume from grid to particle.
"""
@kernel inbounds = true function vollock2_OS!(
    grid::    DeviceGrid2D{T1, T2},
    mp  ::DeviceParticle2D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ mp.np
        p2c = mp.p2c[ix]
        σm  = grid.σm[p2c] / grid.Ω[p2c]
        mp.σij[ix, 1] = mp.sij[ix, 1] + σm
        mp.σij[ix, 2] = mp.sij[ix, 2] + σm
        mp.σij[ix, 3] = mp.sij[ix, 3] + σm
        mp.σij[ix, 4] = mp.sij[ix, 4]
        # update mean stress tensor
        σm = (mp.σij[ix, 1] + mp.σij[ix, 2] + mp.σij[ix, 3]) * T2(0.333333) # 1/3 ≈ 0.333333
        mp.σm[ix] = σm
        # update deviatoric stress tensor
        mp.sij[ix, 1] = mp.σij[ix, 1] - σm
        mp.sij[ix, 2] = mp.σij[ix, 2] - σm
        mp.sij[ix, 3] = mp.σij[ix, 3] - σm
        mp.sij[ix, 4] = mp.σij[ix, 4]
    end
end

"""
    vollock2_OS!(grid::DeviceGrid3D{T1, T2}, mp::DeviceParticle3D{T1, T2})

Description:
---
Mapping back mean stress and volume from grid to particle.
"""
@kernel inbounds = true function vollock2_OS!(
    grid::    DeviceGrid3D{T1, T2},
    mp  ::DeviceParticle3D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix <= mp.np
        p2c = mp.p2c[ix]
        σm  = grid.σm[p2c] / grid.Ω[p2c]
        mp.σij[ix, 1] = mp.sij[ix, 1] + σm
        mp.σij[ix, 2] = mp.sij[ix, 2] + σm
        mp.σij[ix, 3] = mp.sij[ix, 3] + σm
        mp.σij[ix, 4] = mp.sij[ix, 4]
        mp.σij[ix, 5] = mp.sij[ix, 5]
        mp.σij[ix, 6] = mp.sij[ix, 6]
        # update mean stress tensor
        σm = (mp.σij[ix, 1] + mp.σij[ix, 2] + mp.σij[ix, 3]) * T2(0.333333) # 1/3 ≈ 0.333333
        mp.σm[ix] = σm
        # update deviatoric stress tensor
        mp.sij[ix, 1] = mp.σij[ix, 1] - σm
        mp.sij[ix, 2] = mp.σij[ix, 2] - σm
        mp.sij[ix, 3] = mp.σij[ix, 3] - σm
        mp.sij[ix, 4] = mp.σij[ix, 4]
        mp.sij[ix, 5] = mp.σij[ix, 5]
        mp.sij[ix, 6] = mp.σij[ix, 6]
    end
end