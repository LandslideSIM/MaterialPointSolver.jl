#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : TS.jl                                                                      |
|  Description: Basic computing functions for two-phase single-point MPM                   |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 01. resetgridstatus_TS! [2D]                                               |
|               02. resetgridstatus_TS! [3D]                                               |
|               03. resetmpstatus_TS!   [2D, linear basis]                                 |
|               04. resetmpstatus_TS!   [3D, linear basis]                                 |
|               05. resetmpstatus_TS!   [2D,  uGIMP basis]                                 |
|               06. resetmpstatus_TS!   [3D,  uGIMP basis]                                 |
|               07. P2G_TS!             [2D]                                               |
|               08. P2G_TS!             [3D]                                               |
|               09. solvegrid_TS!       [2D]                                               |
|               10. solvegrid_TS!       [3D]                                               |
|               11. doublemapping1_TS!  [2D]                                               |
|               12. doublemapping1_TS!  [3D]                                               |
|               13. doublemapping2_TS!  [2D]                                               |
|               14. doublemapping2_TS!  [3D]                                               |
|               15. doublemapping3_TS!  [2D]                                               |
|               16. doublemapping3_TS!  [3D]                                               |
|               17. G2P_TS!             [2D]                                               |
|               18. G2P_TS!             [3D]                                               |
|               19. vollock1_TS!        [2D]                                               |
|               20. vollock1_TS!        [3D]                                               |
|               21. vollock2_TS!        [2D]                                               |
|               22. vollock2_TS!        [3D]                                               |
+==========================================================================================#

export resetgridstatus_TS!, resetmpstatus_TS!, P2G_TS!, solvegrid_TS!, 
    doublemapping1_TS!, doublemapping2_TS!, doublemapping3_TS!, G2P_TS!, 
    vollock1_TS!, vollock2_TS!

"""
    resetgridstatus_TS!(grid::KernelGrid2D{T1, T2})

Description:
---
Reset some variables for the grid.
"""
@kernel inbounds=true function resetgridstatus_TS!(
    grid::KernelGrid2D{T1, T2}
) where {T1, T2}
    FNUM_0 = T2(0.0)
    ix = @index(Global)
    if ix≤grid.node_num
        if ix≤grid.cell_num
            grid.σm[ix] = FNUM_0
            grid.vol[ix] = FNUM_0
        end
        grid.Ms[ix] = FNUM_0
        grid.Mi[ix] = FNUM_0
        grid.Mw[ix] = FNUM_0
        grid.Ps[ix, 1] = FNUM_0
        grid.Ps[ix, 2] = FNUM_0
        grid.Pw[ix, 1] = FNUM_0
        grid.Pw[ix, 2] = FNUM_0
        grid.Fs[ix, 1] = FNUM_0
        grid.Fs[ix, 2] = FNUM_0
        grid.Fw[ix, 1] = FNUM_0
        grid.Fw[ix, 2] = FNUM_0
        grid.Fdrag[ix, 1] = FNUM_0
        grid.Fdrag[ix, 2] = FNUM_0
    end
end

"""
    resetgridstatus_TS!(grid::KernelGrid3D{T1, T2})

Description:
---
Reset some variables for the grid.
"""
@kernel inbounds=true function resetgridstatus_TS!(
    grid::KernelGrid3D{T1, T2}
) where {T1, T2}
    FNUM_0 = T2(0.0)
    ix = @index(Global)
    if ix≤grid.node_num
        if ix≤grid.cell_num
            grid.σm[ix] = FNUM_0
            grid.vol[ix] = FNUM_0
        end
        grid.Ms[ix] = FNUM_0
        grid.Mi[ix] = FNUM_0
        grid.Mw[ix] = FNUM_0
        grid.Ps[ix, 1] = FNUM_0
        grid.Ps[ix, 2] = FNUM_0
        grid.Ps[ix, 3] = FNUM_0
        grid.Pw[ix, 1] = FNUM_0
        grid.Pw[ix, 2] = FNUM_0
        grid.Pw[ix, 3] = FNUM_0
        grid.Fs[ix, 1] = FNUM_0
        grid.Fs[ix, 2] = FNUM_0
        grid.Fs[ix, 3] = FNUM_0
        grid.Fw[ix, 1] = FNUM_0
        grid.Fw[ix, 2] = FNUM_0
        grid.Fw[ix, 3] = FNUM_0
        grid.Fdrag[ix, 1] = FNUM_0
        grid.Fdrag[ix, 2] = FNUM_0
        grid.Fdrag[ix, 3] = FNUM_0
    end
end

@kernel inbounds=true function resetmpstatus_TS!(
    grid::    KernelGrid2D{T1, T2},
    mp  ::KernelParticle2D{T1, T2},
        ::Val{:linear}
) where {T1, T2}
    FNUM_1 = T2(1.0)
    ix = @index(Global)
    if ix≤mp.num
        # update momentum and mass
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
            mp.p2n[ix, iy] = grid.c2n[mp.p2c[ix], iy]
            # compute distance between particle and related nodes
            Δdx = mp.pos[ix, 1]-grid.pos[mp.p2n[ix, iy], 1]
            Δdy = mp.pos[ix, 2]-grid.pos[mp.p2n[ix, iy], 2]
            # compute basis function
            Nx, dNx = linearBasis(Δdx, grid.space_x)
            Ny, dNy = linearBasis(Δdy, grid.space_y)
            mp.Ni[ ix, iy] =  Nx*Ny # shape function
            mp.∂Nx[ix, iy] = dNx*Ny # x-gradient shape function
            mp.∂Ny[ix, iy] = dNy*Nx # y-gradient shape function
        end
    end
end

@kernel inbounds=true function resetmpstatus_TS!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2},
        ::Val{:linear}
) where {T1, T2}
    FNUM_1 = T2(1.0)
    ix = @index(Global)
    if ix≤mp.num
        # update momentum and mass
        mp.Ms[ix] = mp.vol[ix]*mp.ρs[ix]
        mp.Mw[ix] = mp.vol[ix]*mp.ρw[ix]
        mp.Mi[ix] = mp.vol[ix]*((FNUM_1-mp.porosity[ix])*mp.ρs[ix]+
                                        mp.porosity[ix] *mp.ρw[ix])
        mp.Ps[ix, 1] = mp.Ms[ix]*mp.Vs[ix, 1]*(FNUM_1-mp.porosity[ix])
        mp.Ps[ix, 2] = mp.Ms[ix]*mp.Vs[ix, 2]*(FNUM_1-mp.porosity[ix])
        mp.Ps[ix, 3] = mp.Ms[ix]*mp.Vs[ix, 3]*(FNUM_1-mp.porosity[ix])
        mp.Pw[ix, 1] = mp.Mw[ix]*mp.Vw[ix, 1]*        mp.porosity[ix]
        mp.Pw[ix, 2] = mp.Mw[ix]*mp.Vw[ix, 2]*        mp.porosity[ix]
        mp.Pw[ix, 3] = mp.Mw[ix]*mp.Vw[ix, 3]*        mp.porosity[ix]
        # compute particle to cell and particle to node index
        mp.p2c[ix] = cld(mp.pos[ix, 2]-grid.range_y1, grid.space_y)+
                     fld(mp.pos[ix, 3]-grid.range_z1, grid.space_z)*grid.cell_num_y*grid.cell_num_x+
                     fld(mp.pos[ix, 1]-grid.range_x1, grid.space_x)*grid.cell_num_y
        for iy in Int32(1):Int32(mp.NIC)
            mp.p2n[ix, iy] = grid.c2n[mp.p2c[ix], iy]
            # compute distance between particle and related nodes
            Δdx = mp.pos[ix, 1]-grid.pos[mp.p2n[ix, iy], 1]
            Δdy = mp.pos[ix, 2]-grid.pos[mp.p2n[ix, iy], 2]
            Δdz = mp.pos[ix, 3]-grid.pos[mp.p2n[ix, iy], 3]
            # compute basis function
            Nx, dNx = linearBasis(Δdx, grid.space_x)
            Ny, dNy = linearBasis(Δdy, grid.space_y)
            Nz, dNz = linearBasis(Δdz, grid.space_z)
            mp.Ni[ ix, iy] =  Nx*Ny*Nz
            mp.∂Nx[ix, iy] = dNx*Ny*Nz # x-gradient shape function
            mp.∂Ny[ix, iy] = dNy*Nx*Nz # y-gradient shape function
            mp.∂Nz[ix, iy] = dNz*Nx*Ny # z-gradient shape function
        end
    end
end

"""
    resetmpstatus_TS!(grid::KernelGrid2D{T1, T2}, mp::KernelParticle2D{T1, T2},
       ::Val{:uGIMP})

Description:
---
1. Get topology between particle and grid.
2. Compute the value of basis function (uGIMP).
3. Update particle mass and momentum.

I/0 accesses:
---
- read  → mp.num* 7 + mp.num*2*mp.NIC
- write → mp.num* 3 + mp.num*3*mp.NIC
- total → mp.num*10 + mp.num*3*mp.NIC
"""
@kernel inbounds=true function resetmpstatus_TS!(
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
        # p2c index
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

"""
    resetmpstatus_TS!(grid::KernelGrid3D{T1, T2}, mp::KernelParticle3D{T1, T2}, ::Val{:uGIMP})

Description:
---
1. Get topology between particle and grid.
2. Compute the value of basis function (uGIMP).
3. Update particle mass and momentum.

I/0 accesses:
---
- read  → mp.num* 9 + mp.num*3*mp.NIC
- write → mp.num* 4 + mp.num*4*mp.NIC
- total → mp.num*13 + mp.num*4*mp.NIC
"""
@kernel inbounds=true function resetmpstatus_TS!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2},
        ::Val{:uGIMP}
) where {T1, T2}
    FNUM_1 = T2(1.0)
    ix = @index(Global)   
    if ix≤mp.num
        # update particle mass and momentum
        mp.Ms[ix] = mp.vol[ix]*mp.ρs[ix]
        mp.Mw[ix] = mp.vol[ix]*mp.ρw[ix]
        mp.Mi[ix] = mp.vol[ix]*((FNUM_1-mp.porosity[ix])*mp.ρs[ix]+
                                        mp.porosity[ix] *mp.ρw[ix])
        mp.Ps[ix, 1] = mp.Ms[ix]*mp.Vs[ix, 1]*(FNUM_1-mp.porosity[ix])
        mp.Ps[ix, 2] = mp.Ms[ix]*mp.Vs[ix, 2]*(FNUM_1-mp.porosity[ix])
        mp.Ps[ix, 3] = mp.Ms[ix]*mp.Vs[ix, 3]*(FNUM_1-mp.porosity[ix])
        mp.Pw[ix, 1] = mp.Mw[ix]*mp.Vw[ix, 1]*        mp.porosity[ix]
        mp.Pw[ix, 2] = mp.Mw[ix]*mp.Vw[ix, 2]*        mp.porosity[ix]
        mp.Pw[ix, 3] = mp.Mw[ix]*mp.Vw[ix, 3]*        mp.porosity[ix]
        # get temp variables
        mp_pos_1 = mp.pos[ix, 1]
        mp_pos_2 = mp.pos[ix, 2]
        mp_pos_3 = mp.pos[ix, 3]
        # p2c index
        mp.p2c[ix] = cld(mp_pos_2-grid.range_y1, grid.space_y)+
                     fld(mp_pos_3-grid.range_z1, grid.space_z)*grid.cell_num_y*grid.cell_num_x+
                     fld(mp_pos_1-grid.range_x1, grid.space_x)*grid.cell_num_y
        for iy in Int32(1):Int32(mp.NIC)
            # p2n index
            mp.p2n[ix, iy] = grid.c2n[mp.p2c[ix], iy]
            # compute distance between particle and related nodes
            p2n = mp.p2n[ix, iy]
            Δdx = mp_pos_1-grid.pos[p2n, 1]
            Δdy = mp_pos_2-grid.pos[p2n, 2]
            Δdz = mp_pos_3-grid.pos[p2n, 3]
            # compute basis function
            Nx, dNx = uGIMPbasis(Δdx, grid.space_x, mp.space_x)
            Ny, dNy = uGIMPbasis(Δdy, grid.space_y, mp.space_y)
            Nz, dNz = uGIMPbasis(Δdz, grid.space_z, mp.space_z)
            mp.Ni[ ix, iy] =  Nx*Ny*Nz
            mp.∂Nx[ix, iy] = dNx*Ny*Nz # x-gradient basis function
            mp.∂Ny[ix, iy] = dNy*Nx*Nz # y-gradient basis function
            mp.∂Nz[ix, iy] = dNz*Nx*Ny # z-gradient basis function
        end
    end
end

"""
    P2G_TS!(grid::KernelGrid2D{T1, T2}, mp::KernelParticle2D{T1, T2}, 
        pts_attr::KernelParticleProperty{T1, T2}, gravity::T2)

Description:
---
P2G procedure for scattering the mass, momentum, and forces from particles to grid.
"""
@kernel inbounds=true function P2G_TS!(
    grid    ::          KernelGrid2D{T1, T2},
    mp      ::      KernelParticle2D{T1, T2},
    pts_attr::KernelParticleProperty{T1, T2},
    gravity ::T2
) where {T1, T2}
    FNUM_1 = T2(1.0)
    ix = @index(Global)
    if ix≤mp.num
        for iy in Int32(1):Int32(mp.NIC)
            Ni  = mp.Ni[ix, iy]
            ∂Nx = mp.∂Nx[ix, iy]
            ∂Ny = mp.∂Ny[ix, iy]
            p2n = mp.p2n[ix, iy]
            vol = mp.vol[ix]
            tmp_drag = (mp.porosity[ix]*mp.Mw[ix]*gravity)/pts_attr.k[pts_attr.layer[ix]]
            # compute nodal mass
            @KAatomic grid.Ms[p2n] += Ni*mp.Ms[ix]*(FNUM_1-mp.porosity[ix])
            @KAatomic grid.Mi[p2n] += Ni*mp.Mw[ix]*        mp.porosity[ix]
            @KAatomic grid.Mw[p2n] += Ni*mp.Mw[ix]
            # compute nodal momentum
            @KAatomic grid.Ps[p2n, 1] += Ni*mp.Ps[ix, 1]
            @KAatomic grid.Ps[p2n, 2] += Ni*mp.Ps[ix, 2]
            @KAatomic grid.Pw[p2n, 1] += Ni*mp.Pw[ix, 1]
            @KAatomic grid.Pw[p2n, 2] += Ni*mp.Pw[ix, 2]
            # compute nodal total force
            @KAatomic grid.Fw[p2n, 1] += -vol*(∂Nx*mp.σw[ix])
            @KAatomic grid.Fw[p2n, 2] += -vol*(∂Ny*mp.σw[ix])+Ni*mp.Mw[ix]*gravity
            @KAatomic grid.Fdrag[p2n, 1] += Ni*tmp_drag*(mp.Vw[ix, 1]-mp.Vs[ix, 1])
            @KAatomic grid.Fdrag[p2n, 2] += Ni*tmp_drag*(mp.Vw[ix, 2]-mp.Vs[ix, 2])
            @KAatomic grid.Fs[p2n, 1] += -vol*(∂Nx*mp.σij[ix, 1]+∂Ny*mp.σij[ix, 4])
            @KAatomic grid.Fs[p2n, 2] += -vol*(∂Ny*mp.σij[ix, 2]+∂Nx*mp.σij[ix, 4])+
                Ni*mp.Mi[ix]*gravity
        end
    end
end

"""
    P2G_TS!(grid::KernelGrid3D{T1, T2}, mp::KernelParticle3D{T1, T2}, 
        pts_attr::KernelParticleProperty{T1, T2}, gravity::T2)

Description:
---
P2G procedure for scattering the mass, momentum, and forces from particles to grid.
"""
@kernel inbounds=true function P2G_TS!(
    grid    ::          KernelGrid3D{T1, T2},
    mp      ::      KernelParticle3D{T1, T2},
    pts_attr::KernelParticleProperty{T1, T2},
    gravity ::T2
) where {T1, T2}
    FNUM_1 = T2(1.0)
    ix = @index(Global)
    if ix≤mp.num
        for iy in Int32(1):Int32(mp.NIC)
            Ni  = mp.Ni[ix, iy]
            ∂Nx = mp.∂Nx[ix, iy]
            ∂Ny = mp.∂Ny[ix, iy]
            ∂Nz = mp.∂Nz[ix, iy]
            p2n = mp.p2n[ix, iy]
            vol = mp.vol[ix]
            tmp_drag = (mp.porosity[ix]*mp.Mw[ix]*gravity)/pts_attr.k[pts_attr.layer[ix]]
            # compute nodal mass
            @KAatomic grid.Ms[p2n] += Ni*mp.Ms[ix]*(FNUM_1-mp.porosity[ix])
            @KAatomic grid.Mi[p2n] += Ni*mp.Mw[ix]*        mp.porosity[ix]
            @KAatomic grid.Mw[p2n] += Ni*mp.Mw[ix]
            # compute nodal momentum
            @KAatomic grid.Ps[p2n, 1] += Ni*mp.Ps[ix, 1]
            @KAatomic grid.Ps[p2n, 2] += Ni*mp.Ps[ix, 2]
            @KAatomic grid.Ps[p2n, 3] += Ni*mp.Ps[ix, 3]
            @KAatomic grid.Pw[p2n, 1] += Ni*mp.Pw[ix, 1]
            @KAatomic grid.Pw[p2n, 2] += Ni*mp.Pw[ix, 2]
            @KAatomic grid.Pw[p2n, 3] += Ni*mp.Pw[ix, 3]
            # compute nodal total force
            @KAatomic grid.Fw[p2n, 1] += -vol*(∂Nx*mp.σw[ix])
            @KAatomic grid.Fw[p2n, 2] += -vol*(∂Ny*mp.σw[ix])
            @KAatomic grid.Fw[p2n, 3] += -vol*(∂Nz*mp.σw[ix])+Ni*mp.Mw[ix]*gravity
            @KAatomic grid.Fdrag[p2n, 1] += Ni*tmp_drag*(mp.Vw[ix, 1]-mp.Vs[ix, 1])
            @KAatomic grid.Fdrag[p2n, 2] += Ni*tmp_drag*(mp.Vw[ix, 2]-mp.Vs[ix, 2])
            @KAatomic grid.Fdrag[p2n, 3] += Ni*tmp_drag*(mp.Vw[ix, 3]-mp.Vs[ix, 3])
            @KAatomic grid.Fs[p2n, 1] += -vol*(∂Nx*mp.σij[ix, 1]+∂Ny*mp.σij[ix, 4]+∂Nz*mp.σij[ix, 6])
            @KAatomic grid.Fs[p2n, 2] += -vol*(∂Ny*mp.σij[ix, 2]+∂Nx*mp.σij[ix, 4]+∂Nz*mp.σij[ix, 5])
            @KAatomic grid.Fs[p2n, 3] += -vol*(∂Nz*mp.σij[ix, 3]+∂Nx*mp.σij[ix, 6]+∂Ny*mp.σij[ix, 5])+
                Ni*mp.Mi[ix]*gravity
        end
    end
end


"""
    solvegrid_TS!(grid::KernelGrid2D{T1, T2}, bc::KernelBoundary2D{T1, T2}, ΔT::T2, ζ::T2)

Description:
---
1. Solve equations on grid.
2. Add boundary condition.
"""
@kernel inbounds=true function solvegrid_TS!(
    grid::    KernelGrid2D{T1, T2},
    bc  ::KernelBoundary2D{T1, T2},
    ΔT  ::T2,
    ζs  ::T2,
    ζw  ::T2
) where {T1, T2}
    INUM_0 = T1(0); INUM_2 = T1(2); FNUM_0 = T2(0.0); FNUM_1 = T2(1.0)
    ix = @index(Global)
    if ix≤grid.node_num
        iszero(grid.Ms[ix]) ? Ms_denom=FNUM_0 : Ms_denom=FNUM_1/grid.Ms[ix]
        iszero(grid.Mi[ix]) ? Mi_denom=FNUM_0 : Mi_denom=FNUM_1/grid.Mi[ix]
        iszero(grid.Mw[ix]) ? Mw_denom=FNUM_0 : Mw_denom=FNUM_1/grid.Mw[ix]
        # compute nodal velocity
        grid.Vs[ix, 1] = grid.Ps[ix, 1]*Ms_denom
        grid.Vs[ix, 2] = grid.Ps[ix, 2]*Ms_denom
        grid.Vw[ix, 1] = grid.Pw[ix, 1]*Mi_denom
        grid.Vw[ix, 2] = grid.Pw[ix, 2]*Mi_denom
        # compute damping force
        tmp_damp = -ζw*sqrt(grid.Fw[ix, 1]^INUM_2+grid.Fw[ix, 2]^INUM_2)
        damp_w_x = tmp_damp*sign(grid.Vw[ix, 1])
        damp_w_y = tmp_damp*sign(grid.Vw[ix, 2])
        tmp_damp = -ζs*sqrt((grid.Fs[ix, 1]-grid.Fw[ix, 1])^INUM_2+
                            (grid.Fs[ix, 2]-grid.Fw[ix, 2])^INUM_2)
        damp_s_x = tmp_damp*sign(grid.Vs[ix, 1])
        damp_s_y = tmp_damp*sign(grid.Vs[ix, 2])
        # compute node acceleration
        grid.a_w[ix, 1] = Mw_denom*(grid.Fw[ix, 1]+damp_w_x-grid.Fdrag[ix, 1])
        grid.a_w[ix, 2] = Mw_denom*(grid.Fw[ix, 2]+damp_w_y-grid.Fdrag[ix, 2])
        grid.a_s[ix, 1] = Ms_denom*(-grid.Mi[ix]*grid.a_w[ix, 1]+grid.Fs[ix, 1]+damp_w_x+damp_s_x)
        grid.a_s[ix, 2] = Ms_denom*(-grid.Mi[ix]*grid.a_w[ix, 2]+grid.Fs[ix, 2]+damp_w_y+damp_s_y)
        # update nodal temp velocity
        grid.Vs_T[ix, 1] = grid.Vs[ix, 1]+grid.a_s[ix, 1]*ΔT
        grid.Vs_T[ix, 2] = grid.Vs[ix, 2]+grid.a_s[ix, 2]*ΔT
        grid.Vw_T[ix, 1] = grid.Vw[ix, 1]+grid.a_w[ix, 1]*ΔT
        grid.Vw_T[ix, 2] = grid.Vw[ix, 2]+grid.a_w[ix, 2]*ΔT
        # apply boundary condition
        bc.Vx_s_Idx[ix]≠INUM_0 ? grid.Vs_T[ix, 1]=bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix]≠INUM_0 ? grid.Vs_T[ix, 2]=bc.Vy_s_Val[ix] : nothing
        bc.Vx_w_Idx[ix]≠INUM_0 ? grid.Vw_T[ix, 1]=bc.Vx_w_Val[ix] : nothing
        bc.Vy_w_Idx[ix]≠INUM_0 ? grid.Vw_T[ix, 2]=bc.Vy_w_Val[ix] : nothing
        # reset grid momentum
        grid.Ps[ix, 1] = FNUM_0
        grid.Ps[ix, 2] = FNUM_0
        grid.Pw[ix, 1] = FNUM_0
        grid.Pw[ix, 2] = FNUM_0
    end
end

"""
    solvegrid_TS!(grid::KernelGrid3D{T1, T2}, bc::KernelBoundary3D{T1, T2}, ΔT::T2, ζ::T2)

Description:
---
1. Solve equations on grid.
2. Add boundary condition.
"""
@kernel inbounds=true function solvegrid_TS!(
    grid::    KernelGrid3D{T1, T2},
    bc  ::KernelBoundary3D{T1, T2},
    ΔT  ::T2,
    ζs  ::T2,
    ζw  ::T2
) where {T1, T2}
    INUM_0 = T1(0); INUM_2 = T1(2); FNUM_0 = T2(0.0); FNUM_1 = T1(1.0)
    ix = @index(Global)
    if ix≤grid.node_num
        iszero(grid.Ms[ix]) ? Ms_denom=FNUM_0 : Ms_denom=FNUM_1/grid.Ms[ix]
        iszero(grid.Mi[ix]) ? Mi_denom=FNUM_0 : Mi_denom=FNUM_1/grid.Mi[ix]
        iszero(grid.Mw[ix]) ? Mw_denom=FNUM_0 : Mw_denom=FNUM_1/grid.Mw[ix]
        # compute nodal velocity
        grid.Vs[ix, 1] = grid.Ps[ix, 1]*Ms_denom
        grid.Vs[ix, 2] = grid.Ps[ix, 2]*Ms_denom
        grid.Vs[ix, 3] = grid.Ps[ix, 3]*Ms_denom
        grid.Vw[ix, 1] = grid.Pw[ix, 1]*Mi_denom
        grid.Vw[ix, 2] = grid.Pw[ix, 2]*Mi_denom
        grid.Vw[ix, 3] = grid.Pw[ix, 3]*Mi_denom
        # compute damping force
        tmp_damp = -ζw*sqrt(grid.Fw[ix, 1]^INUM_2+
                            grid.Fw[ix, 2]^INUM_2+
                            grid.Fw[ix, 3]^INUM_2)
        damp_w_x = tmp_damp*sign(grid.Vw[ix, 1])
        damp_w_y = tmp_damp*sign(grid.Vw[ix, 2])
        damp_w_z = tmp_damp*sign(grid.Vw[ix, 3])
        tmp_damp = -ζs*sqrt((grid.Fs[ix, 1]-grid.Fw[ix, 1])^INUM_2+
                            (grid.Fs[ix, 2]-grid.Fw[ix, 2])^INUM_2+
                            (grid.Fs[ix, 3]-grid.Fw[ix, 3])^INUM_2)
        damp_s_x = tmp_damp*sign(grid.Vs[ix, 1])
        damp_s_y = tmp_damp*sign(grid.Vs[ix, 2])
        damp_s_z = tmp_damp*sign(grid.Vs[ix, 3])
        # compute node acceleration
        grid.a_w[ix, 1] = Mw_denom*(grid.Fw[ix, 1]+damp_w_x-grid.Fdrag[ix, 1])
        grid.a_w[ix, 2] = Mw_denom*(grid.Fw[ix, 2]+damp_w_y-grid.Fdrag[ix, 2])
        grid.a_w[ix, 3] = Mw_denom*(grid.Fw[ix, 3]+damp_w_z-grid.Fdrag[ix, 3])
        grid.a_s[ix, 1] = Ms_denom*(-grid.Mi[ix]*grid.a_w[ix, 1]+grid.Fs[ix, 1]+damp_w_x+damp_s_x)
        grid.a_s[ix, 2] = Ms_denom*(-grid.Mi[ix]*grid.a_w[ix, 2]+grid.Fs[ix, 2]+damp_w_y+damp_s_y)
        grid.a_s[ix, 3] = Ms_denom*(-grid.Mi[ix]*grid.a_w[ix, 3]+grid.Fs[ix, 3]+damp_w_z+damp_s_z)
        # update nodal temp velocity
        grid.Vs_T[ix, 1] = grid.Vs[ix, 1]+grid.a_s[ix, 1]*ΔT
        grid.Vs_T[ix, 2] = grid.Vs[ix, 2]+grid.a_s[ix, 2]*ΔT
        grid.Vs_T[ix, 3] = grid.Vs[ix, 3]+grid.a_s[ix, 3]*ΔT
        grid.Vw_T[ix, 1] = grid.Vw[ix, 1]+grid.a_w[ix, 1]*ΔT
        grid.Vw_T[ix, 2] = grid.Vw[ix, 2]+grid.a_w[ix, 2]*ΔT
        grid.Vw_T[ix, 3] = grid.Vw[ix, 3]+grid.a_w[ix, 3]*ΔT
        # apply boundary condition
        bc.Vx_s_Idx[ix]≠INUM_0 ? grid.Vs_T[ix, 1]=bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix]≠INUM_0 ? grid.Vs_T[ix, 2]=bc.Vy_s_Val[ix] : nothing
        bc.Vz_s_Idx[ix]≠INUM_0 ? grid.Vs_T[ix, 3]=bc.Vz_s_Val[ix] : nothing
        bc.Vx_w_Idx[ix]≠INUM_0 ? grid.Vw_T[ix, 1]=bc.Vx_w_Val[ix] : nothing
        bc.Vy_w_Idx[ix]≠INUM_0 ? grid.Vw_T[ix, 2]=bc.Vy_w_Val[ix] : nothing
        bc.Vz_w_Idx[ix]≠INUM_0 ? grid.Vw_T[ix, 3]=bc.Vz_w_Val[ix] : nothing
        # reset grid momentum
        grid.Ps[ix, 1] = FNUM_0
        grid.Ps[ix, 2] = FNUM_0
        grid.Ps[ix, 3] = FNUM_0
        grid.Pw[ix, 1] = FNUM_0
        grid.Pw[ix, 2] = FNUM_0
        grid.Pw[ix, 3] = FNUM_0
    end
end

"""
    doublemapping1_TS!(grid::KernelGrid2D{T1, T2}, mp::KernelParticle2D{T1, T2}, 
        pts_attr::KernelParticleProperty{T1, T2}, ΔT::T2, FLIP::T2, PIC::T2)

Description:
---
1. Solve equations on grid.
2. Compute CFL conditions.
"""
@kernel inbounds=true function doublemapping1_TS!(
    grid    ::          KernelGrid2D{T1, T2},
    mp      ::      KernelParticle2D{T1, T2},
    pts_attr::KernelParticleProperty{T1, T2},
    ΔT      ::T2,
    FLIP    ::T2,
    PIC     ::T2
) where {T1, T2}
    FNUM_0 = T2(0.0); FNUM_1 = T2(1.0); FNUM_43 = T2(4/3)
    ix = @index(Global)
    # update particle position & velocity
    if ix≤mp.num
        pid = pts_attr.layer[ix]
        Ks  = pts_attr.Ks[pid]
        G   = pts_attr.G[pid]
        tmp_vx_s1 = tmp_vx_s2 = tmp_vy_s1 = tmp_vy_s2 = FNUM_0
        tmp_vx_w1 = tmp_vx_w2 = tmp_vy_w1 = tmp_vy_w2 = FNUM_0
        tmp_pos_x = tmp_pos_y = FNUM_0
        for iy in Int32(1):Int32(mp.NIC)
            Ni  = mp.Ni[ ix, iy]
            p2n = mp.p2n[ix, iy]
            tmp_pos_x += Ni* grid.Vs_T[p2n, 1]
            tmp_pos_y += Ni* grid.Vs_T[p2n, 2]
            tmp_vx_s1 += Ni*(grid.Vs_T[p2n, 1]-grid.Vs[p2n, 1])
            tmp_vx_s2 += Ni* grid.Vs_T[p2n, 1]
            tmp_vy_s1 += Ni*(grid.Vs_T[p2n, 2]-grid.Vs[p2n, 2])
            tmp_vy_s2 += Ni* grid.Vs_T[p2n, 2]
            tmp_vx_w1 += Ni*(grid.Vw_T[p2n, 1]-grid.Vw[p2n, 1])
            tmp_vx_w2 += Ni* grid.Vw_T[p2n, 1]
            tmp_vy_w1 += Ni*(grid.Vw_T[p2n, 2]-grid.Vw[p2n, 2])
            tmp_vy_w2 += Ni* grid.Vw_T[p2n, 2]
        end
        # update particle position
        mp.pos[ix, 1] += ΔT*tmp_pos_x
        mp.pos[ix, 2] += ΔT*tmp_pos_y
        # update particle velocity
        mp.Vs[ix, 1] = FLIP*(mp.Vs[ix, 1]+tmp_vx_s1)+PIC*tmp_vx_s2
        mp.Vs[ix, 2] = FLIP*(mp.Vs[ix, 2]+tmp_vy_s1)+PIC*tmp_vy_s2
        mp.Vw[ix, 1] = FLIP*(mp.Vw[ix, 1]+tmp_vx_w1)+PIC*tmp_vx_w2
        mp.Vw[ix, 2] = FLIP*(mp.Vw[ix, 2]+tmp_vy_w1)+PIC*tmp_vy_w2
        # update particle momentum
        mp.Ps[ix, 1] = mp.Ms[ix]*mp.Vs[ix, 1]*(FNUM_1-mp.porosity[ix])
        mp.Ps[ix, 2] = mp.Ms[ix]*mp.Vs[ix, 2]*(FNUM_1-mp.porosity[ix])
        mp.Pw[ix, 1] = mp.Mw[ix]*mp.Vw[ix, 1]*        mp.porosity[ix]
        mp.Pw[ix, 2] = mp.Mw[ix]*mp.Vw[ix, 2]*        mp.porosity[ix]
        # update CFL conditions
        sqr = sqrt((Ks+G*FNUM_43)/mp.ρs[ix])
        cd_sx = grid.space_x/(sqr+abs(Vs_1))
        cd_sy = grid.space_y/(sqr+abs(Vs_2))
        mp.cfl[ix] = min(cd_sx, cd_sy)
    end
end

"""
    doublemapping1_TS!(grid::KernelGrid3D{T1, T2}, mp::KernelParticle3D{T1, T2}, 
        pts_attr::KernelParticleProperty{T1, T2}, ΔT::T2, FLIP::T2, PIC::T2)

Description:
---
1. Solve equations on grid.
2. Compute CFL conditions.
"""
@kernel inbounds=true function doublemapping1_TS!(
    grid    ::          KernelGrid3D{T1, T2},
    mp      ::      KernelParticle3D{T1, T2},
    pts_attr::KernelParticleProperty{T1, T2},
    ΔT      ::T2,
    FLIP    ::T2,
    PIC     ::T2
) where {T1, T2}
    FNUM_0 = T2(0.0); FNUM_1 = T2(1.0); FNUM_43 = T2(4/3)
    ix = @index(Global)
    # update particle position & velocity
    if ix≤mp.num
        pid = pts_attr.layer[ix]
        Ks  = pts_attr.Ks[pid]
        G   = pts_attr.G[pid]
        tmp_vx_s1 = tmp_vx_s2 = tmp_vy_s1 = tmp_vy_s2 = tmp_vz_s1 = tmp_vz_s2 = FNUM_0
        tmp_vx_w1 = tmp_vx_w2 = tmp_vy_w1 = tmp_vy_w2 = tmp_vz_w1 = tmp_vz_w2 = FNUM_0
        tmp_pos_x = tmp_pos_y = tmp_pos_z = FNUM_0
        for iy in Int32(1):Int32(mp.NIC)
            Ni  = mp.Ni[ ix, iy]
            p2n = mp.p2n[ix, iy]
            tmp_pos_x += Ni* grid.Vs_T[p2n, 1]
            tmp_pos_y += Ni* grid.Vs_T[p2n, 2]
            tmp_pos_z += Ni* grid.Vs_T[p2n, 3]
            tmp_vx_s1 += Ni*(grid.Vs_T[p2n, 1]-grid.Vs[p2n, 1])
            tmp_vx_s2 += Ni* grid.Vs_T[p2n, 1]
            tmp_vy_s1 += Ni*(grid.Vs_T[p2n, 2]-grid.Vs[p2n, 2])
            tmp_vy_s2 += Ni* grid.Vs_T[p2n, 2]
            tmp_vz_s1 += Ni*(grid.Vs_T[p2n, 3]-grid.Vs[p2n, 3])
            tmp_vz_s2 += Ni* grid.Vs_T[p2n, 3]
            tmp_vx_w1 += Ni*(grid.Vw_T[p2n, 1]-grid.Vw[p2n, 1])
            tmp_vx_w2 += Ni* grid.Vw_T[p2n, 1]
            tmp_vy_w1 += Ni*(grid.Vw_T[p2n, 2]-grid.Vw[p2n, 2])
            tmp_vy_w2 += Ni* grid.Vw_T[p2n, 2]
            tmp_vz_w1 += Ni*(grid.Vw_T[p2n, 3]-grid.Vw[p2n, 3])
            tmp_vz_w2 += Ni* grid.Vw_T[p2n, 3]
        end
        # update particle position
        mp.pos[ix, 1] += ΔT*tmp_pos_x
        mp.pos[ix, 2] += ΔT*tmp_pos_y
        mp.pos[ix, 3] += ΔT*tmp_pos_z
        # update particle velocity
        mp.Vs[ix, 1] = FLIP*(mp.Vs[ix, 1]+tmp_vx_s1)+PIC*tmp_vx_s2
        mp.Vs[ix, 2] = FLIP*(mp.Vs[ix, 2]+tmp_vy_s1)+PIC*tmp_vy_s2
        mp.Vs[ix, 3] = FLIP*(mp.Vs[ix, 3]+tmp_vz_s1)+PIC*tmp_vz_s2
        mp.Vw[ix, 1] = FLIP*(mp.Vw[ix, 1]+tmp_vx_w1)+PIC*tmp_vx_w2
        mp.Vw[ix, 2] = FLIP*(mp.Vw[ix, 2]+tmp_vy_w1)+PIC*tmp_vy_w2
        mp.Vw[ix, 3] = FLIP*(mp.Vw[ix, 3]+tmp_vz_w1)+PIC*tmp_vz_w2
        # update particle momentum
        mp.Ps[ix, 1] = mp.Ms[ix]*mp.Vs[ix, 1]*(FNUM_1-mp.porosity[ix])
        mp.Ps[ix, 2] = mp.Ms[ix]*mp.Vs[ix, 2]*(FNUM_1-mp.porosity[ix])
        mp.Ps[ix, 3] = mp.Ms[ix]*mp.Vs[ix, 3]*(FNUM_1-mp.porosity[ix])
        mp.Pw[ix, 1] = mp.Mw[ix]*mp.Vw[ix, 1]*        mp.porosity[ix]
        mp.Pw[ix, 2] = mp.Mw[ix]*mp.Vw[ix, 2]*        mp.porosity[ix]
        mp.Pw[ix, 3] = mp.Mw[ix]*mp.Vw[ix, 3]*        mp.porosity[ix]
        # update CFL conditions
        sqr = sqrt((Ks+G*FNUM_43)/mp.ρs[ix])
        cd_sx = grid.space_x/(sqr+abs(Vs_1))
        cd_sy = grid.space_y/(sqr+abs(Vs_2))
        mp.cfl[ix] = min(cd_sx, cd_sy)
    end
end

"""
    doublemapping2_TS!(grid::KernelGrid2D{T1, T2}, mp::KernelParticle2D{T1, T2})

Description:
---
Scatter momentum from particles to grid.
"""
@kernel inbounds=true function doublemapping2_TS!(
    grid::    KernelGrid2D{T1, T2},
    mp  ::KernelParticle2D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix≤mp.num
        # update grid momentum
        for iy in Int32(1):Int32(mp.NIC)
            Ni  = mp.Ni[ ix, iy]
            p2n = mp.p2n[ix, iy]
            @KAatomic grid.Ps[p2n, 1] += mp.Ps[ix, 1]*Ni
            @KAatomic grid.Ps[p2n, 2] += mp.Ps[ix, 2]*Ni
            @KAatomic grid.Pw[p2n, 1] += mp.Pw[ix, 1]*Ni
            @KAatomic grid.Pw[p2n, 2] += mp.Pw[ix, 2]*Ni
        end
    end
end

"""
    doublemapping2_TS!(grid::KernelGrid3D{T1, T2}, mp::KernelParticle3D{T1, T2})

Description:
---
Scatter momentum from particles to grid.
"""
@kernel inbounds=true function doublemapping2_TS!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix≤mp.num
        # update grid momentum
        for iy in Int32(1):Int32(mp.NIC)
            Ni  = mp.Ni[ ix, iy]
            p2n = mp.p2n[ix, iy]
            @KAatomic grid.Ps[p2n, 1] += mp.Ps[ix, 1]*Ni
            @KAatomic grid.Ps[p2n, 2] += mp.Ps[ix, 2]*Ni
            @KAatomic grid.Ps[p2n, 3] += mp.Ps[ix, 3]*Ni
            @KAatomic grid.Pw[p2n, 1] += mp.Pw[ix, 1]*Ni
            @KAatomic grid.Pw[p2n, 2] += mp.Pw[ix, 2]*Ni
            @KAatomic grid.Pw[p2n, 3] += mp.Pw[ix, 3]*Ni
        end
    end
end

"""
    doublemapping3_TS!(grid::KernelGrid2D{T1, T2}, bc::KernelBoundary2D{T1, T2}, ΔT::T2)

Description:
---
Solve equations on grid.
"""
@kernel inbounds=true function doublemapping3_TS!(
    grid::    KernelGrid2D{T1, T2},
    bc  ::KernelBoundary2D{T1, T2},
    ΔT  ::T2
) where {T1, T2}
    INUM_0 = T1(0); FNUM_0 = T2(0.0); FNUM_1 = T2(1.0)
    ix = @index(Global)
    if ix≤grid.node_num
        iszero(grid.Ms[ix]) ? Ms_denom = FNUM_0 : Ms_denom = FNUM_1/grid.Ms[ix]
        iszero(grid.Mi[ix]) ? Mi_denom = FNUM_0 : Mi_denom = FNUM_1/grid.Mi[ix]
        # compute nodal velocities
        grid.Vs[ix, 1] = grid.Ps[ix, 1]*Ms_denom
        grid.Vs[ix, 2] = grid.Ps[ix, 2]*Ms_denom
        grid.Vw[ix, 1] = grid.Pw[ix, 1]*Mi_denom
        grid.Vw[ix, 2] = grid.Pw[ix, 2]*Mi_denom
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

"""
    doublemapping3_TS!(grid::KernelGrid3D{T1, T2}, bc::KernelVBoundary3D{T1, T2}, ΔT::T2)

Description:
---
Solve equations on grid.
"""
@kernel inbounds=true function doublemapping3_TS!(
    grid::    KernelGrid3D{T1, T2},
    bc  ::KernelBoundary3D{T1, T2},
    ΔT  ::T2
) where {T1, T2}
    INUM_0 = T1(0); FNUM_0 = T2(0.0); FNUM_1 = T2(1.0)
    ix = @index(Global)
    if ix≤grid.node_num
        iszero(grid.Ms[ix]) ? Ms_denom = FNUM_0 : Ms_denom = FNUM_1/grid.Ms[ix]
        iszero(grid.Mi[ix]) ? Mi_denom = FNUM_0 : Mi_denom = FNUM_1/grid.Mi[ix]
        # compute nodal velocities
        grid.Vs[ix, 1] = grid.Ps[ix, 1]*Ms_denom
        grid.Vs[ix, 2] = grid.Ps[ix, 2]*Ms_denom
        grid.Vs[ix, 3] = grid.Ps[ix, 3]*Ms_denom
        grid.Vw[ix, 1] = grid.Pw[ix, 1]*Mi_denom
        grid.Vw[ix, 2] = grid.Pw[ix, 2]*Mi_denom
        grid.Vw[ix, 3] = grid.Pw[ix, 3]*Mi_denom
        # fixed Dirichlet nodes
        bc.Vx_s_Idx[ix]≠INUM_0 ? grid.Vs[ix, 1]=bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix]≠INUM_0 ? grid.Vs[ix, 2]=bc.Vy_s_Val[ix] : nothing
        bc.Vz_s_Idx[ix]≠INUM_0 ? grid.Vs[ix, 3]=bc.Vz_s_Val[ix] : nothing
        bc.Vx_w_Idx[ix]≠INUM_0 ? grid.Vw[ix, 1]=bc.Vx_w_Val[ix] : nothing
        bc.Vy_w_Idx[ix]≠INUM_0 ? grid.Vw[ix, 2]=bc.Vy_w_Val[ix] : nothing
        bc.Vz_w_Idx[ix]≠INUM_0 ? grid.Vw[ix, 3]=bc.Vz_w_Val[ix] : nothing
        # compute nodal displacement
        grid.Δd_s[ix, 1] = grid.Vs[ix, 1]*ΔT
        grid.Δd_s[ix, 2] = grid.Vs[ix, 2]*ΔT
        grid.Δd_s[ix, 3] = grid.Vs[ix, 3]*ΔT
        grid.Δd_w[ix, 1] = grid.Vw[ix, 1]*ΔT
        grid.Δd_w[ix, 2] = grid.Vw[ix, 2]*ΔT
        grid.Δd_w[ix, 3] = grid.Vw[ix, 3]*ΔT
    end
end

"""
    G2P_TS!(grid::KernelGrid2D{T1, T2}, mp::KernelParticle2D{T1, T2}, pts_attr::KernelParticleProperty{T1, T2})

Description:
---
Update particle information.
"""
@kernel inbounds=true function G2P_TS!(
    grid    ::          KernelGrid2D{T1, T2},
    mp      ::      KernelParticle2D{T1, T2},
    pts_attr::KernelParticleProperty{T1, T2}
) where {T1, T2}
    FNUM_0 = T2(0.0); FNUM_1 = T2(1.0)
    ix = @index(Global)
    if ix≤mp.num
        dFs1 = dFs2 = dFs3 = dFs4 = dFw1 = dFw2 = dFw3 = dFw4 = FNUM_0
        for iy in Int32(1):Int32(mp.NIC)
            p2n = mp.p2n[ix, iy]
            ∂Nx = mp.∂Nx[ix, iy]
            ∂Ny = mp.∂Ny[ix, iy]
            # compute solid incremental deformation gradient
            dFs1 += grid.Δd_s[p2n, 1]*∂Nx
            dFs2 += grid.Δd_s[p2n, 1]*∂Ny
            dFs3 += grid.Δd_s[p2n, 2]*∂Nx
            dFs4 += grid.Δd_s[p2n, 2]*∂Ny
            dFw1 += grid.Δd_w[p2n, 1]*∂Nx
            dFw2 += grid.Δd_w[p2n, 1]*∂Ny
            dFw3 += grid.Δd_w[p2n, 2]*∂Nx
            dFw4 += grid.Δd_w[p2n, 2]*∂Ny
        end
        mp.∂Fs[ix, 1] = dFs1
        mp.∂Fs[ix, 2] = dFs2
        mp.∂Fs[ix, 3] = dFs3
        mp.∂Fs[ix, 4] = dFs4
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
        mp.ϵij_w[ix, 1] += dFw1
        mp.ϵij_w[ix, 2] += dFw4
        mp.ϵij_w[ix, 4] += dFw2+dFw3
        # deformation gradient matrix
        F1 = mp.F[ix, 1]; F2 = mp.F[ix, 2]; F3 = mp.F[ix, 3]; F4 = mp.F[ix, 4]      
        mp.F[ix, 1] = (dFs1+FNUM_1)*F1+dFs2*F3
        mp.F[ix, 2] = (dFs1+FNUM_1)*F2+dFs2*F4
        mp.F[ix, 3] = (dFs4+FNUM_1)*F3+dFs3*F1
        mp.F[ix, 4] = (dFs4+FNUM_1)*F4+dFs3*F2
        # update jacobian value and particle volume
        mp.J[ix] = mp.F[ix, 1]*mp.F[ix, 4]-mp.F[ix, 2]*mp.F[ix, 3]
        ΔJ = mp.J[ix]*mp.vol_init[ix]/mp.vol[ix]
        mp.vol[ix] = mp.J[ix]*mp.vol_init[ix]
        mp.ρs[ ix] = mp.ρs_init[ix]/mp.J[ix]
        mp.ρw[ ix] = mp.ρw_init[ix]/mp.J[ix]
        # update pore pressure and porosity
        mp.σw[ix] += (pts_attr.Kw[pts_attr.layer[ix]]/mp.porosity[ix])*(
                (FNUM_1-mp.porosity[ix])*(dFs1+dFs4)+mp.porosity[ix] *(dFw1+dFw4))
        mp.porosity[ix] = FNUM_1-(FNUM_1-mp.porosity[ix])/ΔJ
    end
end

"""
    G2P_TS!(grid::KernelGrid3D{T1, T2}, mp::KernelParticle3D{T1, T2}, pts_attr::KernelParticleProperty{T1, T2})

Description:
---
Update particle information.
"""
@kernel inbounds=true function G2P_TS!(
    grid    ::          KernelGrid3D{T1, T2},
    mp      ::      KernelParticle3D{T1, T2},
    pts_attr::KernelParticleProperty{T1, T2}
) where {T1, T2}
    FNUM_0 = T2(0.0); FNUM_1 = T2(1.0)
    ix = @index(Global)
    if ix≤mp.num
        dFs1 = dFs2 = dFs3 = dFs4 = dFs5 = dFs6 = dFs7 = dFs8 = dFs9 = FNUM_0
        dFw1 = dFw2 = dFw3 = dFw4 = dFw5 = dFw6 = dFw7 = dFw8 = dFw9 = FNUM_0
        for iy in Int32(1):Int32(mp.NIC)
            p2n = mp.p2n[ix, iy]
            ∂Nx = mp.∂Nx[ix, iy]; ds1 = grid.Δd_s[p2n, 1]; dw1 = grid.Δd_w[p2n, 1]
            ∂Ny = mp.∂Ny[ix, iy]; ds2 = grid.Δd_s[p2n, 2]; dw2 = grid.Δd_w[p2n, 2]
            ∂Nz = mp.∂Nz[ix, iy]; ds3 = grid.Δd_s[p2n, 3]; dw3 = grid.Δd_w[p2n, 3]
            # compute incremental deformation gradient
            dFs1 += ds1*∂Nx; dFs2 += ds1*∂Ny; dFs3 += ds1*∂Nz
            dFs4 += ds2*∂Nx; dFs5 += ds2*∂Ny; dFs6 += ds2*∂Nz
            dFs7 += ds3*∂Nx; dFs8 += ds3*∂Ny; dFs9 += ds3*∂Nz
            dFw1 += dw1*∂Nx; dFw2 += dw1*∂Ny; dFw3 += dw1*∂Nz
            dFw4 += dw2*∂Nx; dFw5 += dw2*∂Ny; dFw6 += dw2*∂Nz
            dFw7 += dw3*∂Nx; dFw8 += dw3*∂Ny; dFw9 += dw3*∂Nz
        end
        mp.∂Fs[ix, 1] = dFs1; mp.∂Fs[ix, 2] = dFs2; mp.∂Fs[ix, 3] = dFs3
        mp.∂Fs[ix, 4] = dFs4; mp.∂Fs[ix, 5] = dFs5; mp.∂Fs[ix, 6] = dFs6
        mp.∂Fs[ix, 7] = dFs7; mp.∂Fs[ix, 8] = dFs8; mp.∂Fs[ix, 9] = dFs9
        # compute strain increment
        mp.Δϵij_s[ix, 1] = dFs1
        mp.Δϵij_s[ix, 2] = dFs5
        mp.Δϵij_s[ix, 3] = dFs9
        mp.Δϵij_s[ix, 4] = dFs2+dFs4
        mp.Δϵij_s[ix, 5] = dFs6+dFs8
        mp.Δϵij_s[ix, 6] = dFs3+dFs7
        mp.Δϵij_w[ix, 1] = dFw1
        mp.Δϵij_w[ix, 2] = dFw5
        mp.Δϵij_w[ix, 3] = dFw9
        mp.Δϵij_w[ix, 4] = dFw2+dFw4
        mp.Δϵij_w[ix, 5] = dFw6+dFw8
        mp.Δϵij_w[ix, 6] = dFw3+dFw7
        # update strain tensor
        mp.ϵij_s[ix, 1] += dFs1
        mp.ϵij_s[ix, 2] += dFs5
        mp.ϵij_s[ix, 3] += dFs9
        mp.ϵij_s[ix, 4] += dFs2+dFs4
        mp.ϵij_s[ix, 5] += dFs6+dFs8
        mp.ϵij_s[ix, 6] += dFs3+dFs7
        mp.ϵij_w[ix, 1] += dFs1
        mp.ϵij_w[ix, 2] += dFs5
        mp.ϵij_w[ix, 3] += dFs9
        mp.ϵij_w[ix, 4] += dFs2+dFs4
        mp.ϵij_w[ix, 5] += dFs6+dFs8
        mp.ϵij_w[ix, 6] += dFs3+dFs7
        # deformation gradient matrix
        F1 = mp.F[ix, 1]; F2 = mp.F[ix, 2]; F3 = mp.F[ix, 3]
        F4 = mp.F[ix, 4]; F5 = mp.F[ix, 5]; F6 = mp.F[ix, 6]
        F7 = mp.F[ix, 7]; F8 = mp.F[ix, 8]; F9 = mp.F[ix, 9]        
        mp.F[ix, 1] = (dFs1+FNUM_1)*F1+dFs2*F4+dFs3*F7
        mp.F[ix, 2] = (dFs1+FNUM_1)*F2+dFs2*F5+dFs3*F8
        mp.F[ix, 3] = (dFs1+FNUM_1)*F3+dFs2*F6+dFs3*F9
        mp.F[ix, 4] = (dFs5+FNUM_1)*F4+dFs4*F1+dFs6*F7
        mp.F[ix, 5] = (dFs5+FNUM_1)*F5+dFs4*F2+dFs6*F8
        mp.F[ix, 6] = (dFs5+FNUM_1)*F6+dFs4*F3+dFs6*F9
        mp.F[ix, 7] = (dFs9+FNUM_1)*F7+dFs8*F4+dFs7*F1
        mp.F[ix, 8] = (dFs9+FNUM_1)*F8+dFs8*F5+dFs7*F2
        mp.F[ix, 9] = (dFs9+FNUM_1)*F9+dFs8*F6+dFs7*F3
        # update jacobian value and particle volume
        mp.J[ix] = mp.F[ix, 1]*mp.F[ix, 5]*mp.F[ix, 9]+mp.F[ix, 2]*mp.F[ix, 6]*mp.F[ix, 7]+
                   mp.F[ix, 3]*mp.F[ix, 4]*mp.F[ix, 8]-mp.F[ix, 7]*mp.F[ix, 5]*mp.F[ix, 3]-
                   mp.F[ix, 8]*mp.F[ix, 6]*mp.F[ix, 1]-mp.F[ix, 9]*mp.F[ix, 4]*mp.F[ix, 2]
        ΔJ = mp.J[ix]*mp.vol_init[ix]/mp.vol[ix]
        mp.vol[ix] = mp.J[ix]*mp.vol_init[ix]
        mp.ρs[ ix] = mp.ρs_init[ix]/mp.J[ix]
        mp.ρw[ ix] = mp.ρw_init[ix]/mp.J[ix]
        # update pore pressure and porosity
        mp.σw[ix] += (pts_attr.Kw[pts_attr.layer[ix]]/mp.porosity[ix])*(
            (FNUM_1-mp.porosity[ix])*(dFs1+dFs5+dFs9)+mp.porosity[ix] *(dFw1+dFw5+dFw9))
        mp.porosity[ix] = FNUM_1-(FNUM_1-mp.porosity[ix])/ΔJ
    end
end

"""
    vollock1_TS!(grid::KernelGrid2D{T1, T2}, mp::KernelParticle2D{T1, T2})

Description:
---
Mapping mean stress and volume from particle to grid.
"""
@kernel inbounds=true function vollock1_TS!(
    grid::    KernelGrid2D{T1, T2},
    mp  ::KernelParticle2D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix≤mp.num
        p2c = mp.p2c[ix]
        vol = mp.vol[ix]
        @KAatomic grid.σm[ p2c] += vol*mp.σm[ix]
        @KAatomic grid.vol[p2c] += vol
    end
end

"""
    vollock1_TS!(grid::KernelGrid3D{T1, T2}, mp::KernelParticle3D{T1, T2})

Description:
---
Mapping mean stress and volume from particle to grid.
"""
@kernel inbounds=true function vollock1_TS!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix≤mp.num
        p2c = mp.p2c[ix]
        vol = mp.vol[ix]
        @KAatomic grid.σm[ p2c] += vol*mp.σm[ix]
        @KAatomic grid.vol[p2c] += vol
    end
end

"""
    vollock2_TS!(grid::KernelGrid2D{T1, T2}, mp::KernelParticle2D{T1, T2})

Description:
---
Mapping back mean stress and volume from grid to particle.
"""
@kernel inbounds=true function vollock2_TS!(
    grid::    KernelGrid2D{T1, T2},
    mp  ::KernelParticle2D{T1, T2}
) where {T1, T2}
    FNUM_13 = T2(1/3)
    ix = @index(Global)
    if ix≤mp.num
        p2c = mp.p2c[ix]
        σm  = grid.σm[p2c]/grid.vol[p2c]
        mp.σij[ix, 1] = mp.sij[ix, 1]+σm
        mp.σij[ix, 2] = mp.sij[ix, 2]+σm
        mp.σij[ix, 3] = mp.sij[ix, 3]+σm
        mp.σij[ix, 4] = mp.sij[ix, 4]
        # update mean stress tensor
        σm = (mp.σij[ix, 1]+mp.σij[ix, 2]+mp.σij[ix, 3])*FNUM_13
        mp.σm[ix] = σm
        # update deviatoric stress tensor
        mp.sij[ix, 1] = mp.σij[ix, 1]-σm
        mp.sij[ix, 2] = mp.σij[ix, 2]-σm
        mp.sij[ix, 3] = mp.σij[ix, 3]-σm
        mp.sij[ix, 4] = mp.σij[ix, 4]
    end
end

"""
    vollock2_TS!(grid::KernelGrid3D{T1, T2}, mp::KernelParticle3D{T1, T2})

Description:
---
Mapping back mean stress and volume from grid to particle.
"""
@kernel inbounds=true function vollock2_TS!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2}
) where {T1, T2}
    FNUM_13 = T2(1/3)
    ix = @index(Global)
    if ix≤mp.num
        p2c = mp.p2c[ix]
        σm  = grid.σm[p2c]/grid.vol[p2c]
        mp.σij[ix, 1] = mp.sij[ix, 1]+σm
        mp.σij[ix, 2] = mp.sij[ix, 2]+σm
        mp.σij[ix, 3] = mp.sij[ix, 3]+σm
        mp.σij[ix, 4] = mp.sij[ix, 4]
        mp.σij[ix, 5] = mp.sij[ix, 5]
        mp.σij[ix, 6] = mp.sij[ix, 6]
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