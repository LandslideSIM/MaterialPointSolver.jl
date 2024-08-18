#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : OS.jl                                                                      |
|  Description: Basic computing functions for one-phase single-point MPM                   |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 01. resetgridstatus_OS! [2D]                                               |
|               02. resetgridstatus_OS! [3D]                                               |
|               03. resetmpstatus_OS!   [2D, linear basis]                                 |
|               04. resetmpstatus_OS!   [3D, linear basis]                                 |
|               05. resetmpstatus_OS!   [2D,  uGIMP basis]                                 |
|               06. resetmpstatus_OS!   [3D,  uGIMP basis]                                 |
|               07. P2G_OS!             [2D]                                               |
|               08. P2G_OS!             [3D]                                               |
|               09. solvegrid_OS!       [2D]                                               |
|               10. solvegrid_OS!       [3D]                                               |
|               11. doublemapping1_OS!  [2D]                                               |
|               12. doublemapping1_OS!  [3D]                                               |
|               13. doublemapping2_OS!  [2D]                                               |
|               14. doublemapping2_OS!  [3D]                                               |
|               15. doublemapping3_OS!  [2D]                                               |
|               16. doublemapping3_OS!  [3D]                                               |
|               17. G2P_OS!             [2D]                                               |
|               18. G2P_OS!             [3D]                                               |
|               19. vollock1_OS!        [2D]                                               |
|               20. vollock1_OS!        [3D]                                               |
|               21. vollock2_OS!        [2D]                                               |
|               22. vollock2_OS!        [3D]                                               |
+==========================================================================================#

export resetgridstatus_OS!, resetmpstatus_OS!, resetmpstatus_OS_CPU!, P2G_OS!, 
    solvegrid_OS!, doublemapping1_OS!, doublemapping2_OS!, doublemapping3_OS!, G2P_OS!, 
    vollock1_OS!, vollock2_OS!


"""
    resetgridstatus_OS!(grid::KernelGrid2D{T1, T2})

Description:
---
Reset some variables for the grid.
"""
@kernel inbounds = true function resetgridstatus_OS!(
    grid::KernelGrid2D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix <= grid.node_num
        if ix <= grid.cell_num
            grid.σm[ix] = T2(0.0)
            grid.vol[ix] = T2(0.0)
        end
        grid.Ms[ix] = T2(0.0)
        grid.Ps[ix, 1] = T2(0.0)
        grid.Ps[ix, 2] = T2(0.0)
        grid.Fs[ix, 1] = T2(0.0)
        grid.Fs[ix, 2] = T2(0.0)
    end
end

"""
    resetgridstatus_OS!(grid::KernelGrid3D{T1, T2})

Description:
---
Reset some variables for the grid.
"""
@kernel inbounds = true function resetgridstatus_OS!(
    grid::KernelGrid3D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix <= grid.node_num
        if ix <= grid.cell_num
            grid.σm[ix] = T2(0.0)
            grid.vol[ix] = T2(0.0)
        end
        grid.Ms[ix] = T2(0.0)
        grid.Ps[ix, 1] = T2(0.0)
        grid.Ps[ix, 2] = T2(0.0)
        grid.Ps[ix, 3] = T2(0.0)
        grid.Fs[ix, 1] = T2(0.0)
        grid.Fs[ix, 2] = T2(0.0)
        grid.Fs[ix, 3] = T2(0.0)
    end
end

@kernel inbounds = true function resetmpstatus_OS!(
    grid::    KernelGrid2D{T1, T2},
    mp  ::KernelParticle2D{T1, T2},
        ::Val{:linear}
) where {T1, T2}
    ix = @index(Global)
    if ix <= mp.num
        # update momentum and mass
        mp.Ms[ix] = mp.vol[ix] * mp.ρs[ix]
        mp.Ps[ix, 1] = mp.Ms[ix] * mp.Vs[ix, 1]
        mp.Ps[ix, 2] = mp.Ms[ix] * mp.Vs[ix, 2]
        # compute particle to cell and particle to node index
        mp.p2c[ix] = unsafe_trunc(T1, 
            cld(mp.pos[ix, 2] - grid.range_y1, grid.space_y) +
            fld(mp.pos[ix, 1] - grid.range_x1, grid.space_x) * 
                grid.cell_num_y)
        Base.Cartesian.@nexprs 4 iy -> begin
            p2n = getP2N_linear(grid, mp.p2c[ix], Int32(iy))
            mp.p2n[ix, iy] = p2n
            # compute distance between particle and related nodes
            Δdx = mp.pos[ix, 1] - grid.pos[mp.p2n[ix, iy], 1]
            Δdy = mp.pos[ix, 2] - grid.pos[mp.p2n[ix, iy], 2]
            # compute basis function
            Nx, dNx = linearBasis(Δdx, grid.space_x)
            Ny, dNy = linearBasis(Δdy, grid.space_y)
            mp.Ni[ix, iy] = Nx * Ny # shape function
            mp.∂Nx[ix, iy] = dNx * Ny # x-gradient shape function
            mp.∂Ny[ix, iy] = dNy * Nx # y-gradient shape function
        end
    end
end

@kernel inbounds = true function resetmpstatus_OS!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2},
        ::Val{:linear}
) where {T1, T2}
    ix = @index(Global)
    if ix <= mp.num
        # update momentum and mass
        mp.Ms[ix] = mp.vol[ix] * mp.ρs[ix]
        mp.Ps[ix, 1] = mp.Ms[ix] * mp.Vs[ix, 1]
        mp.Ps[ix, 2] = mp.Ms[ix] * mp.Vs[ix, 2]
        mp.Ps[ix, 3] = mp.Ms[ix] * mp.Vs[ix, 3]
        # compute particle to cell and particle to node index
        mp.p2c[ix] = unsafe_trunc(T1, 
            cld(mp.pos[ix, 2] - grid.range_y1, grid.space_y) +
            fld(mp.pos[ix, 3] - grid.range_z1, grid.space_z) * 
                grid.cell_num_y * grid.cell_num_x +
            fld(mp.pos[ix, 1] - grid.range_x1, grid.space_x) * 
                grid.cell_num_y)
        Base.Cartesian.@nexprs 8 iy -> begin
            p2n = getP2N_linear(grid, mp.p2c[ix], Int32(iy))
            mp.p2n[ix, iy] = p2n
            # compute distance between particle and related nodes
            Δdx = mp.pos[ix, 1] - grid.pos[mp.p2n[ix, iy], 1]
            Δdy = mp.pos[ix, 2] - grid.pos[mp.p2n[ix, iy], 2]
            Δdz = mp.pos[ix, 3] - grid.pos[mp.p2n[ix, iy], 3]
            # compute basis function
            Nx, dNx = linearBasis(Δdx, grid.space_x)
            Ny, dNy = linearBasis(Δdy, grid.space_y)
            Nz, dNz = linearBasis(Δdz, grid.space_z)
            mp.Ni[ix, iy] = Nx * Ny * Nz
            mp.∂Nx[ix, iy] = dNx * Ny * Nz # x-gradient shape function
            mp.∂Ny[ix, iy] = dNy * Nx * Nz # y-gradient shape function
            mp.∂Nz[ix, iy] = dNz * Nx * Ny # z-gradient shape function
        end
    end
end

"""
    resetmpstatus_OS!(grid::KernelGrid2D{T1, T2}, mp::KernelParticle2D{T1, T2}, ::Val{:uGIMP})

Description:
---
1. Get topology between particle and grid.
2. Compute the value of basis function (uGIMP).
3. Update particle mass and momentum.
"""
@kernel inbounds = true function resetmpstatus_OS!(
    grid::    KernelGrid2D{T1, T2},
    mp  ::KernelParticle2D{T1, T2},
        ::Val{:uGIMP}
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ mp.num
        smem = @localmem T2 Int32(13)
        smem[1]  = grid.space_x
        smem[2]  = mp.space_x
        smem[3]  = T2(1.0) / (T2(4.0) * grid.space_x * mp.space_x)
        smem[4]  = T2(1.0) / (grid.space_x * mp.space_x)
        smem[5]  = T2(1.0) / grid.space_x
        smem[6]  = T2(0.5) * mp.space_x
        smem[7]  = grid.space_x
        smem[8]  = mp.space_x
        smem[9]  = T2(1.0) / (T2(4.0) * grid.space_y * mp.space_y)
        smem[10] = T2(1.0) / (grid.space_y * mp.space_y)
        smem[11] = T2(1.0) / grid.space_y
        smem[12] = T2(0.5) * mp.space_y
        # update mass and momentum
        mp.Ms[ix] = mp.vol[ix] * mp.ρs[ix]
        mp.Ps[ix, 1] = mp.Ms[ix] * mp.Vs[ix, 1]
        mp.Ps[ix, 2] = mp.Ms[ix] * mp.Vs[ix, 2]
        # p2c index
        mp.p2c[ix] = unsafe_trunc(T1,
            cld(mp.pos[ix, 2] - grid.range_y1, grid.space_y) +
            fld(mp.pos[ix, 1] - grid.range_x1, grid.space_x) * 
                grid.cell_num_y)
        Base.Cartesian.@nexprs 16 iy -> begin
            # p2n index
            p2n = getP2N_uGIMP(grid, mp.p2c[ix], Int32(iy))
            mp.p2n[ix, iy] = p2n
            # compute distance between particle and related nodes
            p2n = mp.p2n[ix, iy]
            Δdx = mp.pos[ix, 1] - grid.pos[p2n, 1]
            Δdy = mp.pos[ix, 2] - grid.pos[p2n, 2]
            # compute basis function
            Nx, dNx = uGIMPbasisx(Δdx, smem)
            Ny, dNy = uGIMPbasisy(Δdy, smem)
            mp.Ni[ix, iy] = Nx * Ny
            mp.∂Nx[ix, iy] = dNx * Ny # x-gradient shape function
            mp.∂Ny[ix, iy] = dNy * Nx # y-gradient shape function
        end
    end
end

@kernel inbounds = true function resetmpstatus_OS_CPU!(
    grid::    KernelGrid2D{T1, T2},
    mp  ::KernelParticle2D{T1, T2},
        ::Val{:uGIMP}
) where {T1, T2}
    ix = @index(Global)
    # update mass and momentum
    mp.Ms[ix] = mp.vol[ix] * mp.ρs[ix]
    mp.Ps[ix, 1] = mp.Ms[ix] * mp.Vs[ix, 1]
    mp.Ps[ix, 2] = mp.Ms[ix] * mp.Vs[ix, 2]
    # p2c index
    mp.p2c[ix] = unsafe_trunc(T1,
        cld(mp.pos[ix, 2] - grid.range_y1, grid.space_y) +
        fld(mp.pos[ix, 1] - grid.range_x1, grid.space_x) * 
            grid.cell_num_y)
    for iy in Int32(1):Int32(mp.NIC)
        # p2n index
        p2n = getP2N_uGIMP(grid, mp.p2c[ix], Int32(iy))
        mp.p2n[ix, iy] = p2n
        # compute distance between particle and related nodes
        p2n = mp.p2n[ix, iy]
        Δdx = mp.pos[ix, 1] - grid.pos[p2n, 1]
        Δdy = mp.pos[ix, 2] - grid.pos[p2n, 2]
        # compute basis function
        Nx, dNx = uGIMPbasis(Δdx, grid.space_x, mp.space_x)
        Ny, dNy = uGIMPbasis(Δdy, grid.space_y, mp.space_y)
        mp.Ni[ix, iy] = Nx * Ny
        mp.∂Nx[ix, iy] = dNx * Ny # x-gradient shape function
        mp.∂Ny[ix, iy] = dNy * Nx # y-gradient shape function
    end
end

"""
    resetmpstatus_OS!(grid::KernelGrid3D{T1, T2}, mp::KernelParticle3D{T1, T2} ::Val{:uGIMP})

Description:
---
1. Get topology between particle and grid.
2. Compute the value of basis function (uGIMP).
3. Update particle mass and momentum.
"""
@kernel inbounds = true function resetmpstatus_OS!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2},
        ::Val{:uGIMP}
) where {T1, T2}
    ix = @index(Global)
    smem = @localmem T2 Int32(19)
    smem[1]  = grid.space_x
    smem[2]  = mp.space_x
    smem[3]  = T2(1.0) / (T2(4.0) * grid.space_x * mp.space_x)
    smem[4]  = T2(1.0) / (grid.space_x * mp.space_x)
    smem[5]  = T2(1.0) / grid.space_x
    smem[6]  = T2(0.5) * mp.space_x
    smem[7]  = grid.space_y
    smem[8]  = mp.space_y
    smem[9]  = T2(1.0) / (T2(4.0) * grid.space_y * mp.space_y)
    smem[10] = T2(1.0) / (grid.space_y * mp.space_y)
    smem[11] = T2(1.0) / grid.space_y
    smem[12] = T2(0.5) * mp.space_y
    smem[13] = grid.space_z
    smem[14] = mp.space_z
    smem[15] = T2(1.0) / (T2(4.0) * grid.space_z * mp.space_z)
    smem[16] = T2(1.0) / (grid.space_z* mp.space_z)
    smem[17] = T2(1.0) / grid.space_z
    smem[18] = T2(0.5) * mp.space_z
    if ix ≤ mp.num
        # update particle mass and momentum
        mp_Ms = mp.vol[ix] * mp.ρs[ix]
        mp.Ms[ix] = mp_Ms
        mp.Ps[ix, 1] = mp_Ms * mp.Vs[ix, 1]
        mp.Ps[ix, 2] = mp_Ms * mp.Vs[ix, 2]
        mp.Ps[ix, 3] = mp_Ms * mp.Vs[ix, 3]
        # get temp variables
        mp_pos_1 = mp.pos[ix, 1]
        mp_pos_2 = mp.pos[ix, 2]
        mp_pos_3 = mp.pos[ix, 3]
        # p2c index
        mp.p2c[ix] = unsafe_trunc(T1,
            cld(mp_pos_2 - grid.range_y1, grid.space_y) +
            fld(mp_pos_3 - grid.range_z1, grid.space_z) * 
                grid.cell_num_y * grid.cell_num_x +
            fld(mp_pos_1 - grid.range_x1, grid.space_x) * grid.cell_num_y)
        Base.Cartesian.@nexprs 64 iy -> begin
            # p2n index
            p2n = getP2N_uGIMP(grid, mp.p2c[ix], Int32(iy))
            mp.p2n[ix, iy] = p2n
            # compute distance betwe en particle and related nodes
            Δdx = mp_pos_1 - grid.pos[p2n, 1]
            Δdy = mp_pos_2 - grid.pos[p2n, 2]
            Δdz = mp_pos_3 - grid.pos[p2n, 3]
            # compute basis function
            Nx, dNx = uGIMPbasisx(Δdx, smem)
            Ny, dNy = uGIMPbasisy(Δdy, smem)
            Nz, dNz = uGIMPbasisz(Δdz, smem)
            mp.Ni[ix, iy] = Nx * Ny * Nz
            mp.∂Nx[ix, iy] = dNx * Ny * Nz # x-gradient basis function
            mp.∂Ny[ix, iy] = dNy * Nx * Nz # y-gradient basis function
            mp.∂Nz[ix, iy] = dNz * Nx * Ny # z-gradient basis function
        end
    end
end

@kernel inbounds = true function resetmpstatus_OS_CPU!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2},
        ::Val{:uGIMP}
) where {T1, T2}
    ix = @index(Global)
    # update particle mass and momentum
    mp_Ms = mp.vol[ix] * mp.ρs[ix]
    mp.Ms[ix] = mp_Ms
    mp.Ps[ix, 1] = mp_Ms * mp.Vs[ix, 1]
    mp.Ps[ix, 2] = mp_Ms * mp.Vs[ix, 2]
    mp.Ps[ix, 3] = mp_Ms * mp.Vs[ix, 3]
    # get temp variables
    mp_pos_1 = mp.pos[ix, 1]
    mp_pos_2 = mp.pos[ix, 2]
    mp_pos_3 = mp.pos[ix, 3]
    # p2c index
    mp.p2c[ix] = unsafe_trunc(T1,
        cld(mp_pos_2 - grid.range_y1, grid.space_y) +
        fld(mp_pos_3 - grid.range_z1, grid.space_z) * 
            grid.cell_num_y * grid.cell_num_x +
        fld(mp_pos_1 - grid.range_x1, grid.space_x) * grid.cell_num_y)
    for iy in Int32(1):Int32(mp.NIC)
        # p2n index
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
        mp.Ni[ix, iy] = Nx * Ny * Nz
        mp.∂Nx[ix, iy] = dNx * Ny * Nz # x-gradient basis function
        mp.∂Ny[ix, iy] = dNy * Nx * Nz # y-gradient basis function
        mp.∂Nz[ix, iy] = dNz * Nx * Ny # z-gradient basis function
    end
end

"""
    P2G_OS!(grid::KernelGrid2D{T1, T2}, mp::KernelParticle2D{T1, T2}, gravity::T2)

Description:
---
P2G procedure for scattering the mass, momentum, and forces from particles to grid.
"""
@kernel inbounds = true function P2G_OS!(
    grid   ::    KernelGrid2D{T1, T2},
    mp     ::KernelParticle2D{T1, T2},
    gravity::T2
) where {T1, T2}
    ix = @index(Global)
    if ix <= mp.num
        for iy in Int32(1):Int32(mp.NIC)
            Ni = mp.Ni[ix, iy]
            if Ni != T2(0.0)
                ∂Nx = mp.∂Nx[ix, iy]
                ∂Ny = mp.∂Ny[ix, iy]
                p2n = mp.p2n[ix, iy]
                vol = mp.vol[ix]
                NiM = mp.Ms[ix] * Ni
                # compute nodal mass
                @KAatomic grid.Ms[p2n] += NiM
                # compute nodal momentum
                @KAatomic grid.Ps[p2n, 1] += Ni * mp.Ps[ix, 1]
                @KAatomic grid.Ps[p2n, 2] += Ni * mp.Ps[ix, 2]
                # compute nodal total force for solid
                @KAatomic grid.Fs[p2n, 1] += -vol * (∂Nx * mp.σij[ix, 1] + 
                                                     ∂Ny * mp.σij[ix, 4])
                @KAatomic grid.Fs[p2n, 2] += -vol * (∂Ny * mp.σij[ix, 2] + 
                                                     ∂Nx * mp.σij[ix, 4]) + NiM * gravity
            end
        end
    end
end

"""
    P2G_OS!(grid::KernelGrid3D{T1, T2}, mp::KernelParticle3D{T1, T2}, gravity::T2)

Description:
---
P2G procedure for scattering the mass, momentum, and forces from particles to grid.
"""
@kernel inbounds = true function P2G_OS!(
    grid   ::    KernelGrid3D{T1, T2},
    mp     ::KernelParticle3D{T1, T2},
    gravity::T2
) where {T1, T2}
    ix = @index(Global)
    if ix <= mp.num
        for iy in Int32(1):Int32(mp.NIC)
            Ni = mp.Ni[ix, iy]
            if Ni != T2(0.0)
                ∂Nx = mp.∂Nx[ix, iy]
                ∂Ny = mp.∂Ny[ix, iy]
                ∂Nz = mp.∂Nz[ix, iy]
                p2n = mp.p2n[ix, iy]
                vol = mp.vol[ix]
                NiM = mp.Ms[ix] * Ni
                # compute nodal mass
                @KAatomic grid.Ms[p2n] += NiM
                # compute nodal momentum
                @KAatomic grid.Ps[p2n, 1] += Ni * mp.Ps[ix, 1]
                @KAatomic grid.Ps[p2n, 2] += Ni * mp.Ps[ix, 2]
                @KAatomic grid.Ps[p2n, 3] += Ni * mp.Ps[ix, 3]
                # compute nodal total force for solid
                @KAatomic grid.Fs[p2n, 1] += -vol * (∂Nx * mp.σij[ix, 1] + 
                                                     ∂Ny * mp.σij[ix, 4] + 
                                                     ∂Nz * mp.σij[ix, 6])
                @KAatomic grid.Fs[p2n, 2] += -vol * (∂Ny * mp.σij[ix, 2] + 
                                                     ∂Nx * mp.σij[ix, 4] + 
                                                     ∂Nz * mp.σij[ix, 5])
                @KAatomic grid.Fs[p2n, 3] += -vol * (∂Nz * mp.σij[ix, 3] + 
                                                     ∂Nx * mp.σij[ix, 6] + 
                                                     ∂Ny * mp.σij[ix, 5]) + NiM * gravity
            end
        end
    end
end


"""
    solvegrid_OS!(grid::KernelGrid2D{T1, T2}, bc::KernelBoundary2D{T1, T2}, ΔT::T2, ζs::T2)

Description:
---
1. Solve equations on grid.
2. Add boundary condition.
"""
@kernel inbounds = true function solvegrid_OS!(
    grid::    KernelGrid2D{T1, T2},
    bc  ::KernelBoundary2D{T1, T2},
    ΔT  ::T2,
    ζs  ::T2
) where {T1, T2}
    ix = @index(Global)
    if ix <= grid.node_num && grid.Ms[ix] != Int32(0)
        Ms_denom = T2(1.0) / grid.Ms[ix]
        # compute nodal velocity
        grid.Vs[ix, 1] = grid.Ps[ix, 1] * Ms_denom
        grid.Vs[ix, 2] = grid.Ps[ix, 2] * Ms_denom
        # damping force for solid
        dampvs = -ζs * sqrt(grid.Fs[ix, 1]^T1(2) + grid.Fs[ix, 2]^T1(2))
        # compute nodal total force for mixture
        Fs_x = grid.Fs[ix, 1] + dampvs * sign(grid.Vs[ix, 1])
        Fs_y = grid.Fs[ix, 2] + dampvs * sign(grid.Vs[ix, 2])
        # update nodal velocity
        grid.Vs_T[ix, 1] = (grid.Ps[ix, 1] + Fs_x * ΔT) * Ms_denom
        grid.Vs_T[ix, 2] = (grid.Ps[ix, 2] + Fs_y * ΔT) * Ms_denom
        # boundary condition
        bc.Vx_s_Idx[ix] == T1(1) ? grid.Vs_T[ix, 1] = bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix] == T1(1) ? grid.Vs_T[ix, 2] = bc.Vy_s_Val[ix] : nothing
        # reset grid momentum
        grid.Ps[ix, 1] = T2(0.0)
        grid.Ps[ix, 2] = T2(0.0)
    end
end

"""
    solvegrid_OS!(grid::KernelGrid3D{T1, T2}, bc::KernelBoundary3D{T1, T2}, ΔT::T2, ζs::T2)

Description:
---
1. Solve equations on grid.
2. Add boundary condition.
"""
@kernel inbounds = true function solvegrid_OS!(
    grid::    KernelGrid3D{T1, T2},
    bc  ::KernelBoundary3D{T1, T2},
    ΔT  ::T2,
    ζs  ::T2
) where {T1, T2}
    ix = @index(Global)
    if ix <= grid.node_num && grid.Ms[ix] != Int32(0)
        Ms_denom = T2(1.0) / grid.Ms[ix]
        # compute nodal velocity
        Ps_1 = grid.Ps[ix, 1]
        Ps_2 = grid.Ps[ix, 2]
        Ps_3 = grid.Ps[ix, 3]
        grid.Vs[ix, 1] = Ps_1 * Ms_denom
        grid.Vs[ix, 2] = Ps_2 * Ms_denom
        grid.Vs[ix, 3] = Ps_3 * Ms_denom
        # damping force for solid
        dampvs = -ζs * sqrt(grid.Fs[ix, 1]^T1(2) +
                            grid.Fs[ix, 2]^T1(2) +
                            grid.Fs[ix, 3]^T1(2))
        # compute nodal total force for mixture
        Fs_x = grid.Fs[ix, 1] + dampvs * sign(grid.Vs[ix, 1])
        Fs_y = grid.Fs[ix, 2] + dampvs * sign(grid.Vs[ix, 2])
        Fs_z = grid.Fs[ix, 3] + dampvs * sign(grid.Vs[ix, 3])
        # update nodal velocity
        grid.Vs_T[ix, 1] = (Ps_1 + Fs_x * ΔT) * Ms_denom
        grid.Vs_T[ix, 2] = (Ps_2 + Fs_y * ΔT) * Ms_denom
        grid.Vs_T[ix, 3] = (Ps_3 + Fs_z * ΔT) * Ms_denom
        # boundary condition
        bc.Vx_s_Idx[ix] == T1(1) ? grid.Vs_T[ix, 1] = bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix] == T1(1) ? grid.Vs_T[ix, 2] = bc.Vy_s_Val[ix] : nothing
        bc.Vz_s_Idx[ix] == T1(1) ? grid.Vs_T[ix, 3] = bc.Vz_s_Val[ix] : nothing
        # reset grid momentum
        grid.Ps[ix, 1] = T2(0.0)
        grid.Ps[ix, 2] = T2(0.0)
        grid.Ps[ix, 3] = T2(0.0)
    end
end

"""
    doublemapping1_OS!(grid::KernelGrid2D{T1, T2}, mp::KernelParticle2D{T1, T2},
        pts_attr::KernelParticleProperty{T1, T2}, ΔT::T2, FLIP::T2, PIC::T2)

Description:
---
1. Solve equations on grid.
2. Compute CFL conditions.
"""
@kernel inbounds = true function doublemapping1_OS!(
    grid    ::          KernelGrid2D{T1, T2},
    mp      ::      KernelParticle2D{T1, T2},
    pts_attr::KernelParticleProperty{T1, T2},
    ΔT      ::T2,
    FLIP    ::T2,
    PIC     ::T2
) where {T1, T2}
    ix = @index(Global)
    # update particle position & velocity
    if ix <= mp.num
        pid = pts_attr.layer[ix]
        Ks  = pts_attr.Ks[pid]
        G   = pts_attr.G[pid]
        tmp_vx_s1 = tmp_vx_s2 = T2(0.0)
        tmp_vy_s1 = tmp_vy_s2 = T2(0.0)
        tmp_pos_x = tmp_pos_y = T2(0.0)
        # update particle position
        for iy in Int32(1):Int32(mp.NIC)
            Ni = mp.Ni[ix, iy]
            if Ni != T2(0.0)
                p2n = mp.p2n[ix, iy]
                tmp_pos_x += Ni* grid.Vs_T[p2n, 1]
                tmp_pos_y += Ni* grid.Vs_T[p2n, 2]
                tmp_vx_s1 += Ni*(grid.Vs_T[p2n, 1]-grid.Vs[p2n, 1])
                tmp_vx_s2 += Ni* grid.Vs_T[p2n, 1]
                tmp_vy_s1 += Ni*(grid.Vs_T[p2n, 2]-grid.Vs[p2n, 2])
                tmp_vy_s2 += Ni* grid.Vs_T[p2n, 2]
            end
        end
        mp.pos[ix, 1] += ΔT*tmp_pos_x
        mp.pos[ix, 2] += ΔT*tmp_pos_y
        # update particle velocity
        mp.Vs[ix, 1] = FLIP*(mp.Vs[ix, 1]+tmp_vx_s1)+PIC*tmp_vx_s2
        mp.Vs[ix, 2] = FLIP*(mp.Vs[ix, 2]+tmp_vy_s1)+PIC*tmp_vy_s2
        # update particle momentum
        Vs_1 = mp.Vs[ix, 1]
        Vs_2 = mp.Vs[ix, 2]
        Ms   = mp.Ms[ix]
        mp.Ps[ix, 1] = Ms*Vs_1
        mp.Ps[ix, 2] = Ms*Vs_2
        # update CFL conditions
        sqr = sqrt((Ks+G*T2(1.333333))/mp.ρs[ix]) # 4/3 ≈ 1.333333
        cd_sx = grid.space_x/(sqr+abs(Vs_1))
        cd_sy = grid.space_y/(sqr+abs(Vs_2))
        mp.cfl[ix] = min(cd_sx, cd_sy)
    end
end

"""
    doublemapping1_OS!(grid::KernelGrid3D{T1, T2}, mp::KernelParticle3D{T1, T2}, 
        pts_attr::KernelParticleProperty{T1, T2}, ΔT::T2, FLIP::T2, PIC::T2)

Description:
---
1. Solve equations on grid.
2. Compute CFL conditions.
"""
@kernel inbounds=true function doublemapping1_OS!(
    grid    ::          KernelGrid3D{T1, T2},
    mp      ::      KernelParticle3D{T1, T2},
    pts_attr::KernelParticleProperty{T1, T2},
    ΔT      ::T2,
    FLIP    ::T2,
    PIC     ::T2
) where {T1, T2}
    ix = @index(Global)
    # update particle position & velocity
    if ix <= mp.num
        pid = pts_attr.layer[ix]
        Ks  = pts_attr.Ks[pid]
        G   = pts_attr.G[pid]
        tmp_vx_s1 = tmp_vx_s2 = tmp_vy_s1 = T2(0.0)
        tmp_vy_s2 = tmp_vz_s1 = tmp_vz_s2 = T2(0.0)
        tmp_pos_x = tmp_pos_y = tmp_pos_z = T2(0.0)
        # update particle position
        for iy in Int32(1):Int32(mp.NIC)
            Ni = mp.Ni[ix, iy]
            if Ni != T2(0.0)
                p2n   = mp.p2n[ix, iy]
                Vs_T1 = grid.Vs_T[p2n, 1]
                Vs_T2 = grid.Vs_T[p2n, 2]
                Vs_T3 = grid.Vs_T[p2n, 3]
                tmp_pos_x += Ni* Vs_T1
                tmp_pos_y += Ni* Vs_T2
                tmp_pos_z += Ni* Vs_T3
                tmp_vx_s1 += Ni*(Vs_T1-grid.Vs[p2n, 1])
                tmp_vx_s2 += Ni* Vs_T1
                tmp_vy_s1 += Ni*(Vs_T2-grid.Vs[p2n, 2])
                tmp_vy_s2 += Ni* Vs_T2
                tmp_vz_s1 += Ni*(Vs_T3-grid.Vs[p2n, 3])
                tmp_vz_s2 += Ni* Vs_T3
            end
        end
        # update particle position
        mp.pos[ix, 1] += ΔT*tmp_pos_x
        mp.pos[ix, 2] += ΔT*tmp_pos_y
        mp.pos[ix, 3] += ΔT*tmp_pos_z
        # update particle velocity
        mp.Vs[ix, 1] = FLIP*(mp.Vs[ix, 1]+tmp_vx_s1)+PIC*tmp_vx_s2
        mp.Vs[ix, 2] = FLIP*(mp.Vs[ix, 2]+tmp_vy_s1)+PIC*tmp_vy_s2
        mp.Vs[ix, 3] = FLIP*(mp.Vs[ix, 3]+tmp_vz_s1)+PIC*tmp_vz_s2
        # update particle momentum
        Vs_1 = mp.Vs[ix, 1]
        Vs_2 = mp.Vs[ix, 2]
        Vs_3 = mp.Vs[ix, 3]
        Ms   = mp.Ms[ix]
        mp.Ps[ix, 1] = Ms*Vs_1
        mp.Ps[ix, 2] = Ms*Vs_2
        mp.Ps[ix, 3] = Ms*Vs_3
        # update CFL conditions
        sqr = sqrt((Ks+G*T2(1.333333))/mp.ρs[ix])
        cd_sx = grid.space_x/(sqr+abs(Vs_1))
        cd_sy = grid.space_y/(sqr+abs(Vs_2))
        cd_sz = grid.space_z/(sqr+abs(Vs_3))
        mp.cfl[ix] = min(cd_sx, cd_sy, cd_sz)
    end
end

"""
    doublemapping2_OS!(grid::KernelGrid2D{T1, T2}, mp::KernelParticle2D{T1, T2})

Description:
---
Scatter momentum from particles to grid.
"""
@kernel inbounds=true function doublemapping2_OS!(
    grid::    KernelGrid2D{T1, T2},
    mp  ::KernelParticle2D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix <= mp.num
        # update particle position & velocity
        for iy in Int32(1):Int32(mp.NIC)
            Ni = mp.Ni[ix, iy]
            if Ni != T2(0.0)
                p2n = mp.p2n[ix, iy]
                @KAatomic grid.Ps[p2n, 1] += mp.Ps[ix, 1]*Ni
                @KAatomic grid.Ps[p2n, 2] += mp.Ps[ix, 2]*Ni
            end
        end
    end
end

"""
    doublemapping2_OS!(grid::KernelGrid3D{T1, T2}, mp::KernelParticle3D{T1, T2})

Description:
---
Scatter momentum from particles to grid.
"""
@kernel inbounds=true function doublemapping2_OS!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix <= mp.num
        # update particle position & velocity
        for iy in Int32(1):Int32(mp.NIC)
            Ni = mp.Ni[ix, iy]
            if Ni != T2(0.0)
                p2n = mp.p2n[ix, iy]
                @KAatomic grid.Ps[p2n, 1] += mp.Ps[ix, 1]*Ni
                @KAatomic grid.Ps[p2n, 2] += mp.Ps[ix, 2]*Ni
                @KAatomic grid.Ps[p2n, 3] += mp.Ps[ix, 3]*Ni
            end
        end
    end
end

"""
    doublemapping3_OS!(grid::KernelGrid2D{T1, T2}, bc::KernelBoundary2D{T1, T2}, ΔT::T2)

Description:
---
Solve equations on grid.
"""
@kernel inbounds=true function doublemapping3_OS!(
    grid::    KernelGrid2D{T1, T2},
    bc  ::KernelBoundary2D{T1, T2},
    ΔT  ::T2
) where {T1, T2}
    ix = @index(Global)
    if ix <= grid.node_num && grid.Ms[ix] != Int32(0)
        Ms_denom = T2(1.0) / grid.Ms[ix]
        # compute nodal velocities
        grid.Vs[ix, 1] = grid.Ps[ix, 1]*Ms_denom
        grid.Vs[ix, 2] = grid.Ps[ix, 2]*Ms_denom
        # fixed Dirichlet nodes
        bc.Vx_s_Idx[ix] == T1(1) ? grid.Vs[ix, 1]=bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix] == T1(1) ? grid.Vs[ix, 2]=bc.Vy_s_Val[ix] : nothing
        # compute nodal displacement
        grid.Δd_s[ix, 1] = grid.Vs[ix, 1]*ΔT
        grid.Δd_s[ix, 2] = grid.Vs[ix, 2]*ΔT
    end
end

"""
    doublemapping3_OS!(grid::KernelGrid3D{T1, T2}, bc::KernelVBoundary3D{T1, T2}, ΔT::T2)

Description:
---
Solve equations on grid.
"""
@kernel inbounds=true function doublemapping3_OS!(
    grid::    KernelGrid3D{T1, T2},
    bc  ::KernelBoundary3D{T1, T2},
    ΔT  ::T2
) where {T1, T2}
    ix = @index(Global)
    if ix <= grid.node_num && grid.Ms[ix] != Int32(0)
        Ms_denom = T2(1.0) / grid.Ms[ix]
        # compute nodal velocities
        grid.Vs[ix, 1] = grid.Ps[ix, 1]*Ms_denom
        grid.Vs[ix, 2] = grid.Ps[ix, 2]*Ms_denom
        grid.Vs[ix, 3] = grid.Ps[ix, 3]*Ms_denom
        # fixed Dirichlet nodes
        bc.Vx_s_Idx[ix]==T1(1) ? grid.Vs[ix, 1]=bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix]==T1(1) ? grid.Vs[ix, 2]=bc.Vy_s_Val[ix] : nothing
        bc.Vz_s_Idx[ix]==T1(1) ? grid.Vs[ix, 3]=bc.Vz_s_Val[ix] : nothing
        # compute nodal displacement
        grid.Δd_s[ix, 1] = grid.Vs[ix, 1]*ΔT
        grid.Δd_s[ix, 2] = grid.Vs[ix, 2]*ΔT
        grid.Δd_s[ix, 3] = grid.Vs[ix, 3]*ΔT
    end
end

"""
    G2P_OS!(grid::KernelGrid2D{T1, T2}, mp::KernelParticle2D{T1, T2})

Description:
---
Update particle information.
"""
@kernel inbounds=true function G2P_OS!(
    grid::    KernelGrid2D{T1, T2},
    mp  ::KernelParticle2D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix <= mp.num
        dF1 = dF2 = dF3 = dF4 = T2(0.0)
        for iy in Int32(1):Int32(mp.NIC)
            if mp.Ni[ix, iy] != T2(0.0)
                p2n = mp.p2n[ix, iy]
                ∂Nx = mp.∂Nx[ix, iy]
                ∂Ny = mp.∂Ny[ix, iy]
                # compute solid incremental deformation gradient
                dF1 += grid.Δd_s[p2n, 1]*∂Nx
                dF2 += grid.Δd_s[p2n, 1]*∂Ny
                dF3 += grid.Δd_s[p2n, 2]*∂Nx
                dF4 += grid.Δd_s[p2n, 2]*∂Ny
            end
        end
        mp.ΔFs[ix, 1] = dF1
        mp.ΔFs[ix, 2] = dF2
        mp.ΔFs[ix, 3] = dF3
        mp.ΔFs[ix, 4] = dF4
        # compute strain increment 
        mp.Δϵij_s[ix, 1] = dF1
        mp.Δϵij_s[ix, 2] = dF4
        mp.Δϵij_s[ix, 4] = dF2+dF3
        # update strain tensor
        mp.ϵij_s[ix, 1] += dF1
        mp.ϵij_s[ix, 2] += dF4
        mp.ϵij_s[ix, 4] += dF2+dF3
        # deformation gradient matrix
        F1 = mp.F[ix, 1]; F2 = mp.F[ix, 2]; F3 = mp.F[ix, 3]; F4 = mp.F[ix, 4]      
        mp.F[ix, 1] = (dF1+T2(1.0))*F1+dF2*F3
        mp.F[ix, 2] = (dF1+T2(1.0))*F2+dF2*F4
        mp.F[ix, 3] = (dF4+T2(1.0))*F3+dF3*F1
        mp.F[ix, 4] = (dF4+T2(1.0))*F4+dF3*F2
        # update jacobian value and particle volume
        mp.J[  ix] = mp.F[ix, 1]*mp.F[ix, 4]-mp.F[ix, 2]*mp.F[ix, 3]
        mp.vol[ix] = mp.J[ix]*mp.vol_init[ix]
        mp.ρs[ ix] = mp.ρs_init[ix]/mp.J[ix]
    end
end

"""
    G2P_OS!(grid::KernelGrid3D{T1, T2}, mp::KernelParticle3D{T1, T2})

Description:
---
Update particle information.
"""
@kernel inbounds=true function G2P_OS!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix <= mp.num
        dF1 = dF2 = dF3 = dF4 = dF5 = dF6 = dF7 = dF8 = dF9 = T2(0.0)
        for iy in Int32(1):Int32(mp.NIC)
            if mp.Ni[ix, iy] != T2(0.0)
                p2n = mp.p2n[ix, iy]
                ∂Nx = mp.∂Nx[ix, iy]; ds1 = grid.Δd_s[p2n, 1]
                ∂Ny = mp.∂Ny[ix, iy]; ds2 = grid.Δd_s[p2n, 2]
                ∂Nz = mp.∂Nz[ix, iy]; ds3 = grid.Δd_s[p2n, 3]
                # compute solid incremental deformation gradient
                dF1 += ds1*∂Nx; dF2 += ds1*∂Ny; dF3 += ds1*∂Nz
                dF4 += ds2*∂Nx; dF5 += ds2*∂Ny; dF6 += ds2*∂Nz
                dF7 += ds3*∂Nx; dF8 += ds3*∂Ny; dF9 += ds3*∂Nz
            end
        end
        mp.ΔFs[ix, 1] = dF1; mp.ΔFs[ix, 2] = dF2; mp.ΔFs[ix, 3] = dF3
        mp.ΔFs[ix, 4] = dF4; mp.ΔFs[ix, 5] = dF5; mp.ΔFs[ix, 6] = dF6
        mp.ΔFs[ix, 7] = dF7; mp.ΔFs[ix, 8] = dF8; mp.ΔFs[ix, 9] = dF9
        # compute strain increment
        mp.Δϵij_s[ix, 1] = dF1
        mp.Δϵij_s[ix, 2] = dF5
        mp.Δϵij_s[ix, 3] = dF9
        mp.Δϵij_s[ix, 4] = dF2+dF4
        mp.Δϵij_s[ix, 5] = dF6+dF8
        mp.Δϵij_s[ix, 6] = dF3+dF7
        # update strain tensor
        mp.ϵij_s[ix, 1] += dF1
        mp.ϵij_s[ix, 2] += dF5
        mp.ϵij_s[ix, 3] += dF9
        mp.ϵij_s[ix, 4] += dF2+dF4
        mp.ϵij_s[ix, 5] += dF6+dF8
        mp.ϵij_s[ix, 6] += dF3+dF7
        # deformation gradient matrix
        F1 = mp.F[ix, 1]; F2 = mp.F[ix, 2]; F3 = mp.F[ix, 3]
        F4 = mp.F[ix, 4]; F5 = mp.F[ix, 5]; F6 = mp.F[ix, 6]
        F7 = mp.F[ix, 7]; F8 = mp.F[ix, 8]; F9 = mp.F[ix, 9]        
        mp.F[ix, 1] = (dF1+T2(1.0))*F1+dF2*F4+dF3*F7
        mp.F[ix, 2] = (dF1+T2(1.0))*F2+dF2*F5+dF3*F8
        mp.F[ix, 3] = (dF1+T2(1.0))*F3+dF2*F6+dF3*F9
        mp.F[ix, 4] = (dF5+T2(1.0))*F4+dF4*F1+dF6*F7
        mp.F[ix, 5] = (dF5+T2(1.0))*F5+dF4*F2+dF6*F8
        mp.F[ix, 6] = (dF5+T2(1.0))*F6+dF4*F3+dF6*F9
        mp.F[ix, 7] = (dF9+T2(1.0))*F7+dF8*F4+dF7*F1
        mp.F[ix, 8] = (dF9+T2(1.0))*F8+dF8*F5+dF7*F2
        mp.F[ix, 9] = (dF9+T2(1.0))*F9+dF8*F6+dF7*F3
        # update jacobian value and particle volume
        mp.J[ix] = mp.F[ix, 1]*mp.F[ix, 5]*mp.F[ix, 9]+mp.F[ix, 2]*mp.F[ix, 6]*mp.F[ix, 7]+
                   mp.F[ix, 3]*mp.F[ix, 4]*mp.F[ix, 8]-mp.F[ix, 7]*mp.F[ix, 5]*mp.F[ix, 3]-
                   mp.F[ix, 8]*mp.F[ix, 6]*mp.F[ix, 1]-mp.F[ix, 9]*mp.F[ix, 4]*mp.F[ix, 2]
        mp.vol[ix] = mp.J[ix]*mp.vol_init[ix]
        mp.ρs[ ix] = mp.ρs_init[ix]/mp.J[ix]
    end
end

"""
    vollock1_OS!(grid::KernelGrid2D{T1, T2}, mp::KernelParticle2D{T1, T2})

Description:
---
Mapping mean stress and volume from particle to grid.
"""
@kernel inbounds=true function vollock1_OS!(
    grid::    KernelGrid2D{T1, T2},
    mp  ::KernelParticle2D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix <= mp.num
        p2c = mp.p2c[ix]
        vol = mp.vol[ix]
        @KAatomic grid.σm[ p2c] += vol*mp.σm[ix]
        @KAatomic grid.vol[p2c] += vol
    end
end

"""
    vollock1_OS!(grid::KernelGrid3D{T1, T2}, mp::KernelParticle3D{T1, T2})

Description:
---
Mapping mean stress and volume from particle to grid.
"""
@kernel inbounds=true function vollock1_OS!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix <= mp.num
        p2c = mp.p2c[ix]
        vol = mp.vol[ix]
        @KAatomic grid.σm[ p2c] += vol*mp.σm[ix]
        @KAatomic grid.vol[p2c] += vol
    end
end

"""
    vollock2_OS!(grid::KernelGrid2D{T1, T2}, mp::KernelParticle2D{T1, T2})

Description:
---
Mapping back mean stress and volume from grid to particle.
"""
@kernel inbounds=true function vollock2_OS!(
    grid::    KernelGrid2D{T1, T2},
    mp  ::KernelParticle2D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix <= mp.num
        p2c = mp.p2c[ix]
        σm  = grid.σm[p2c]/grid.vol[p2c]
        mp.σij[ix, 1] = mp.sij[ix, 1]+σm
        mp.σij[ix, 2] = mp.sij[ix, 2]+σm
        mp.σij[ix, 3] = mp.sij[ix, 3]+σm
        mp.σij[ix, 4] = mp.sij[ix, 4]
        # update mean stress tensor
        σm = (mp.σij[ix, 1]+mp.σij[ix, 2]+mp.σij[ix, 3])*T2(0.333333) # 1/3 ≈ 0.333333
        mp.σm[ix] = σm
        # update deviatoric stress tensor
        mp.sij[ix, 1] = mp.σij[ix, 1]-σm
        mp.sij[ix, 2] = mp.σij[ix, 2]-σm
        mp.sij[ix, 3] = mp.σij[ix, 3]-σm
        mp.sij[ix, 4] = mp.σij[ix, 4]
    end
end

"""
    vollock2_OS!(grid::KernelGrid3D{T1, T2}, mp::KernelParticle3D{T1, T2})

Description:
---
Mapping back mean stress and volume from grid to particle.
"""
@kernel inbounds=true function vollock2_OS!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix <= mp.num
        p2c = mp.p2c[ix]
        σm  = grid.σm[p2c]/grid.vol[p2c]
        mp.σij[ix, 1] = mp.sij[ix, 1]+σm
        mp.σij[ix, 2] = mp.sij[ix, 2]+σm
        mp.σij[ix, 3] = mp.sij[ix, 3]+σm
        mp.σij[ix, 4] = mp.sij[ix, 4]
        mp.σij[ix, 5] = mp.sij[ix, 5]
        mp.σij[ix, 6] = mp.sij[ix, 6]
        # update mean stress tensor
        σm = (mp.σij[ix, 1]+mp.σij[ix, 2]+mp.σij[ix, 3])*T2(0.333333) # 1/3 ≈ 0.333333
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