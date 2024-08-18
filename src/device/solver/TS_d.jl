#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : TS_d.jl (two-phase single-point)                                           |
|  Description: Kernel functions for the computing in MPM cycle on GPU.                    |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 01. kernel_TS01!() [2D]                                                    |
|               02. kernel_TS01!() [3D]                                                    |
|               03. kernel_TS02!() [2D, linear basis]                                      |
|               04. kernel_TS02!() [3D, linear basis]                                      |
|               05. kernel_TS02!() [2D, uGIMP basis]                                       |
|               06. kernel_TS02!() [3D, uGIMP basis]                                       |
|               07. kernel_TS03!() [2D]                                                    |
|               08. kernel_TS03!() [3D]                                                    |
|               09. kernel_TS04!() [2D]                                                    |
|               10. kernel_TS04!() [3D]                                                    |
|               11. kernel_TS05!() [2D]                                                    |
|               12. kernel_TS05!() [3D]                                                    |
|               13. kernel_TS06!() [2D]                                                    |
|               14. kernel_TS06!() [3D]                                                    |
|               15. kernel_TS07!() [2D]                                                    |
|               16. kernel_TS07!() [3D]                                                    |
|               17. kernel_TS08!() [2D]                                                    |
|               18. kernel_TS08!() [3D]                                                    |
|               19. kernel_TS09!() [2D]                                                    |
|               20. kernel_TS09!() [3D]                                                    |
|               21. kernel_TS10!() [2D]                                                    |
|               22. kernel_TS10!() [3D]                                                    |
|               23. procedure!()   [2D & 3D]                                               |
+==========================================================================================#

function kernel_TS01!(cu_grid::KernelGrid2D{T1, T2}) where {T1, T2}
    FNUM_0 = T2(0); INUM_1 = T1(1)
    # GPU global threads index
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x # thread x_dim index
    # kernel function
    if ix≤cu_grid.node_num
        if ix≤cu_grid.cell_num
            cu_grid.σm[ix]  = FNUM_0
            cu_grid.σw[ix]  = FNUM_0
            cu_grid.vol[ix] = FNUM_0
        end
        cu_grid.Ms[ix]       = FNUM_0
        cu_grid.Mw[ix]       = FNUM_0
        cu_grid.Mi[ix]       = FNUM_0
        cu_grid.Ps[ix, 1]    = FNUM_0
        cu_grid.Ps[ix, 2]    = FNUM_0
        cu_grid.Pw[ix, 1]    = FNUM_0
        cu_grid.Pw[ix, 2]    = FNUM_0
        cu_grid.Fw[ix, 1]    = FNUM_0
        cu_grid.Fw[ix, 2]    = FNUM_0
        cu_grid.Fs[ix, 1]    = FNUM_0
        cu_grid.Fs[ix, 2]    = FNUM_0
        cu_grid.Fdrag[ix, 1] = FNUM_0
        cu_grid.Fdrag[ix, 2] = FNUM_0
    end
    return nothing
end

function kernel_TS01!(cu_grid::KernelGrid3D{T1, T2}) where {T1, T2}
    FNUM_0 = T2(0); INUM_1 = T1(1)
    # GPU global threads index
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x # thread x_dim index
    # kernel function
    if ix≤cu_grid.node_num
        if ix≤cu_grid.cell_num
            cu_grid.σm[ix]  = FNUM_0
            cu_grid.σw[ix]  = FNUM_0
            cu_grid.vol[ix] = FNUM_0
        end
        cu_grid.Ms[ix]       = FNUM_0
        cu_grid.Mw[ix]       = FNUM_0
        cu_grid.Mi[ix]       = FNUM_0
        cu_grid.Ps[ix, 1]    = FNUM_0
        cu_grid.Ps[ix, 2]    = FNUM_0
        cu_grid.Ps[ix, 3]    = FNUM_0
        cu_grid.Pw[ix, 1]    = FNUM_0
        cu_grid.Pw[ix, 2]    = FNUM_0
        cu_grid.Pw[ix, 3]    = FNUM_0
        cu_grid.Fw[ix, 1]    = FNUM_0
        cu_grid.Fw[ix, 2]    = FNUM_0
        cu_grid.Fw[ix, 3]    = FNUM_0
        cu_grid.Fs[ix, 1]    = FNUM_0
        cu_grid.Fs[ix, 2]    = FNUM_0
        cu_grid.Fs[ix, 3]    = FNUM_0
        cu_grid.Fdrag[ix, 1] = FNUM_0
        cu_grid.Fdrag[ix, 2] = FNUM_0
        cu_grid.Fdrag[ix, 3] = FNUM_0
    end
    return nothing
end

function kernel_TS02!(cu_grid::    KernelGrid2D{T1, T2},
                      cu_mp  ::KernelParticle2D{T1, T2},
                             ::Val{:linear}) where {T1, T2}
    FNUM_1 = T2(1.0); INUM_1 = T1(1)
    # GPU global threads index
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x # thread x_dim index
    iy = (blockIdx().y-INUM_1)*blockDim().y+threadIdx().y # thread y_dim index
    # kernel function
    if (ix≤cu_mp.num)&&(iy≤cu_mp.NIC)
        # update momentum
        cu_mp.Ms[ix] = cu_mp.vol[ix]*cu_mp.ρs[ix]
        cu_mp.Mw[ix] = cu_mp.vol[ix]*cu_mp.ρw[ix]
        cu_mp.Mi[ix] = cu_mp.vol[ix]*((FNUM_1-cu_mp.porosity[ix])*cu_mp.ρs[ix]+
                                              cu_mp.porosity[ix] *cu_mp.ρw[ix])
        cu_mp.Ps[ix, 1] = cu_mp.Ms[ix]*cu_mp.Vs[ix, 1]
        cu_mp.Ps[ix, 2] = cu_mp.Ms[ix]*cu_mp.Vs[ix, 2]
        cu_mp.Pw[ix, 1] = cu_mp.Mw[ix]*cu_mp.Vw[ix, 1]
        cu_mp.Pw[ix, 2] = cu_mp.Mw[ix]*cu_mp.Vw[ix, 2]
        # compute particle to cell and particle to node index
        cu_mp.p2c[ix] = cld(cu_mp.pos[ix, 2]-cu_grid.range_y1, cu_grid.space_y)+
                        fld(cu_mp.pos[ix, 1]-cu_grid.range_x1, cu_grid.space_x)*
                        cu_grid.cell_num_y
        cu_mp.p2n[ix, iy] = cu_grid.c2n[cu_mp.p2c[ix], iy]
        # compute the value of shape function
        Δdx = cu_mp.pos[ix, 1]-cu_grid.pos[cu_mp.p2n[ix, iy], 1]
        Δdy = cu_mp.pos[ix, 2]-cu_grid.pos[cu_mp.p2n[ix, iy], 2]
        # compute basis function
        Nx, dNx = linear_basis(Δdx, cu_grid.space_x)
        Ny, dNy = linear_basis(Δdy, cu_grid.space_y)
        cu_mp.Ni[ix, iy]  = Nx *Ny # shape function
        cu_mp.∂Nx[ix, iy] = dNx*Ny # x-gradient shape function
        cu_mp.∂Ny[ix, iy] = dNy*Nx # y-gradient shape function
    end
    return nothing
end

function kernel_TS02!(cu_grid::    KernelGrid3D{T1, T2},
                      cu_mp  ::KernelParticle3D{T1, T2},
                             ::Val{:linear}) where {T1, T2}
    FNUM_1 = T2(1.0); INUM_1 = T1(1)
    # GPU global threads index
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x # thread x_dim index
    iy = (blockIdx().y-INUM_1)*blockDim().y+threadIdx().y # thread y_dim index
    # kernel function
    if (ix≤cu_mp.num)&&(iy≤cu_mp.NIC)
        # update momentum
        cu_mp.Ms[ix] = cu_mp.vol[ix]*cu_mp.ρs[ix]
        cu_mp.Mw[ix] = cu_mp.vol[ix]*cu_mp.ρw[ix]
        cu_mp.Mi[ix] = cu_mp.vol[ix]*((FNUM_1-cu_mp.porosity[ix])*cu_mp.ρs[ix]+
                                              cu_mp.porosity[ix] *cu_mp.ρw[ix])
        cu_mp.Ps[ix, 1] = cu_mp.Ms[ix]*cu_mp.Vs[ix, 1]
        cu_mp.Ps[ix, 2] = cu_mp.Ms[ix]*cu_mp.Vs[ix, 2]
        cu_mp.Ps[ix, 3] = cu_mp.Ms[ix]*cu_mp.Vs[ix, 3]
        cu_mp.Pw[ix, 1] = cu_mp.Mw[ix]*cu_mp.Vw[ix, 1]
        cu_mp.Pw[ix, 2] = cu_mp.Mw[ix]*cu_mp.Vw[ix, 2]
        cu_mp.Pw[ix, 3] = cu_mp.Mw[ix]*cu_mp.Vw[ix, 3]
        # compute particle to cell and particle to node index
        numxy = cu_grid.cell_num_y*cu_grid.cell_num_x
        num_y = cu_grid.cell_num_y
        cu_mp.p2c[ix] = cld(cu_mp.pos[ix, 2]-cu_grid.range_y1, cu_grid.space_y)+
                        fld(cu_mp.pos[ix, 3]-cu_grid.range_z1, cu_grid.space_z)*numxy+
                        fld(cu_mp.pos[ix, 1]-cu_grid.range_x1, cu_grid.space_x)*num_y
        cu_mp.p2n[ix, iy] = cu_grid.c2n[cu_mp.p2c[ix], iy]
        # compute the value of shape function
        Δdx = cu_mp.pos[ix, 1]-cu_grid.pos[cu_mp.p2n[ix, iy], 1]
        Δdy = cu_mp.pos[ix, 2]-cu_grid.pos[cu_mp.p2n[ix, iy], 2]
        Δdz = cu_mp.pos[ix, 3]-cu_grid.pos[cu_mp.p2n[ix, iy], 3]
        # compute basis function
        Nx, dNx = linear_basis(Δdx, cu_grid.space_x)
        Ny, dNy = linear_basis(Δdy, cu_grid.space_y)
        Nz, dNz = linear_basis(Δdz, cu_grid.space_z)
        cu_mp.Ni[ix, iy]  = Nx *Ny*Nz # shape function
        cu_mp.∂Nx[ix, iy] = dNx*Ny*Nz # x-gradient shape function
        cu_mp.∂Ny[ix, iy] = dNy*Nx*Nz # y-gradient shape function
        cu_mp.∂Nz[ix, iy] = dNz*Nx*Ny # z-gradient shape function
    end
    return nothing
end

function kernel_TS02!(cu_grid::    KernelGrid2D{T1, T2},
                      cu_mp  ::KernelParticle2D{T1, T2},
                             ::Val{:uGIMP}) where {T1, T2}
    FNUM_1 = T2(1.0); INUM_1 = T1(1); INUM_2 = T1(2)
    # GPU global threads index
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x # thread x_dim index
    iy = (blockIdx().y-INUM_1)*blockDim().y+threadIdx().y # thread y_dim index
    # kernel function
    if (ix≤cu_mp.num)&&(iy≤cu_mp.NIC)
        # update momentum
        cu_mp.Ms[ix] = cu_mp.vol[ix]*cu_mp.ρs[ix]
        cu_mp.Mw[ix] = cu_mp.vol[ix]*cu_mp.ρw[ix]
        cu_mp.Mi[ix] = cu_mp.vol[ix]*((FNUM_1-cu_mp.porosity[ix])*cu_mp.ρs[ix]+
                                              cu_mp.porosity[ix] *cu_mp.ρw[ix])
        cu_mp.Ps[ix, 1] = cu_mp.Ms[ix]*cu_mp.Vs[ix, 1]
        cu_mp.Ps[ix, 2] = cu_mp.Ms[ix]*cu_mp.Vs[ix, 2]
        cu_mp.Pw[ix, 1] = cu_mp.Mw[ix]*cu_mp.Vw[ix, 1]
        cu_mp.Pw[ix, 2] = cu_mp.Mw[ix]*cu_mp.Vw[ix, 2]
        # compute particle to cell and particle to node index
        cu_mp.p2c[ix] = cld(cu_mp.pos[ix, 2]-cu_grid.range_y1, cu_grid.space_y)+
                        fld(cu_mp.pos[ix, 1]-cu_grid.range_x1, cu_grid.space_x)*
                        cu_grid.cell_num_y
        cu_mp.p2n[ix, iy] = cu_grid.c2n[cu_mp.p2c[ix], iy]
        # compute the value of shape function
        Δdx = cu_mp.pos[ix, 1]-cu_grid.pos[cu_mp.p2n[ix, iy], 1]
        Δdy = cu_mp.pos[ix, 2]-cu_grid.pos[cu_mp.p2n[ix, iy], 2]
        # compute basis function
        Nx, dNx = uGIMP_basis(Δdx, cu_grid.space_x, cu_mp.space_x, INUM_2)
        Ny, dNy = uGIMP_basis(Δdy, cu_grid.space_y, cu_mp.space_y, INUM_2)
        cu_mp.Ni[ix, iy]  = Nx *Ny # shape function
        cu_mp.∂Nx[ix, iy] = dNx*Ny # x-gradient shape function
        cu_mp.∂Ny[ix, iy] = dNy*Nx # y-gradient shape function
    end
    return nothing
end

function kernel_TS02!(cu_grid::    KernelGrid3D{T1, T2},
                      cu_mp  ::KernelParticle3D{T1, T2},
                             ::Val{:uGIMP}) where {T1, T2}
    FNUM_1 = T2(1.0); INUM_1 = T1(1); INUM_2 = T1(2)
    # GPU global threads index
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x # thread x_dim index
    iy = (blockIdx().y-INUM_1)*blockDim().y+threadIdx().y # thread y_dim index
    # kernel function
    if (ix≤cu_mp.num)&&(iy≤cu_mp.NIC)
        # update momentum
        cu_mp.Ms[ix] = cu_mp.vol[ix]*cu_mp.ρs[ix]
        cu_mp.Mw[ix] = cu_mp.vol[ix]*cu_mp.ρw[ix]
        cu_mp.Mi[ix] = cu_mp.vol[ix]*((FNUM_1-cu_mp.porosity[ix])*cu_mp.ρs[ix]+
                                              cu_mp.porosity[ix] *cu_mp.ρw[ix])
        cu_mp.Ps[ix, 1] = cu_mp.Ms[ix]*cu_mp.Vs[ix, 1]
        cu_mp.Ps[ix, 2] = cu_mp.Ms[ix]*cu_mp.Vs[ix, 2]
        cu_mp.Ps[ix, 3] = cu_mp.Ms[ix]*cu_mp.Vs[ix, 3]
        cu_mp.Pw[ix, 1] = cu_mp.Mw[ix]*cu_mp.Vw[ix, 1]
        cu_mp.Pw[ix, 2] = cu_mp.Mw[ix]*cu_mp.Vw[ix, 2]
        cu_mp.Pw[ix, 3] = cu_mp.Mw[ix]*cu_mp.Vw[ix, 3]
        # compute particle to cell and particle to node index
        numxy = cu_grid.cell_num_y*cu_grid.cell_num_x
        num_y = cu_grid.cell_num_y
        cu_mp.p2c[ix] = cld(cu_mp.pos[ix, 2]-cu_grid.range_y1, cu_grid.space_y)+
                        fld(cu_mp.pos[ix, 3]-cu_grid.range_z1, cu_grid.space_z)*numxy+
                        fld(cu_mp.pos[ix, 1]-cu_grid.range_x1, cu_grid.space_x)*num_y
        cu_mp.p2n[ix, iy] = cu_grid.c2n[cu_mp.p2c[ix], iy]
        # compute the value of shape function
        Δdx = cu_mp.pos[ix, 1]-cu_grid.pos[cu_mp.p2n[ix, iy], 1]
        Δdy = cu_mp.pos[ix, 2]-cu_grid.pos[cu_mp.p2n[ix, iy], 2]
        Δdz = cu_mp.pos[ix, 3]-cu_grid.pos[cu_mp.p2n[ix, iy], 3]
        # compute basis function
        Nx, dNx = uGIMP_basis(Δdx, cu_grid.space_x, cu_mp.space_x, INUM_2)
        Ny, dNy = uGIMP_basis(Δdy, cu_grid.space_y, cu_mp.space_y, INUM_2)
        Nz, dNz = uGIMP_basis(Δdz, cu_grid.space_z, cu_mp.space_z, INUM_2)
        cu_mp.Ni[ix, iy]  = Nx *Ny*Nz # shape function
        cu_mp.∂Nx[ix, iy] = dNx*Ny*Nz # x-gradient shape function
        cu_mp.∂Ny[ix, iy] = dNy*Nx*Nz # y-gradient shape function
        cu_mp.∂Nz[ix, iy] = dNz*Nx*Ny # z-gradient shape function
    end
    return nothing
end

function kernel_TS03!(cu_grid::    KernelGrid2D{T1, T2},
                      cu_mp  ::KernelParticle2D{T1, T2},
                      gravity::T2) where {T1, T2}
    FNUM_1 = T2(1.0); INUM_1 = T1(1)
    # GPU global threads index
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x # thread x_dim index
    iy = (blockIdx().y-INUM_1)*blockDim().y+threadIdx().y # thread y_dim index
    # kernel function
    if (ix≤cu_mp.num)&&(iy≤cu_mp.NIC)
        Ni  = cu_mp.Ni[ix, iy]
        ∂Nx = cu_mp.∂Nx[ix, iy]
        ∂Ny = cu_mp.∂Ny[ix, iy]
        p2n = cu_mp.p2n[ix, iy]
        por = cu_mp.porosity[ix]
        # compute nodal mass
        CUDA.@atomic cu_grid.Ms[p2n] = +(cu_grid.Ms[p2n], 
            Ni*cu_mp.Ms[ix]*(FNUM_1-por))
        CUDA.@atomic cu_grid.Mi[p2n] = +(cu_grid.Mi[p2n], 
            Ni*cu_mp.Mw[ix]*(       por))   
        CUDA.@atomic cu_grid.Mw[p2n] = +(cu_grid.Mw[p2n], 
            Ni*cu_mp.Mw[ix])
        # compute nodal momentum
        CUDA.@atomic cu_grid.Ps[p2n, 1] = +(cu_grid.Ps[p2n, 1],
            Ni*cu_mp.Ps[ix, 1]*(FNUM_1-por))
        CUDA.@atomic cu_grid.Ps[p2n, 2] = +(cu_grid.Ps[p2n, 2],
            Ni*cu_mp.Ps[ix, 2]*(FNUM_1-por))
        CUDA.@atomic cu_grid.Pw[p2n, 1] = +(cu_grid.Pw[p2n, 1],
            Ni*cu_mp.Pw[ix, 1]*(       por))
        CUDA.@atomic cu_grid.Pw[p2n, 2] = +(cu_grid.Pw[p2n, 2],
            Ni*cu_mp.Pw[ix, 2]*(       por))
        # compute nodal drag force for water
        tmp = -(cu_mp.Mw[ix]*gravity*cu_mp.porosity[ix])/cu_mp.k[ix]
        CUDA.@atomic cu_grid.Fdrag[p2n, 1] = +(cu_grid.Fdrag[p2n, 1], Ni*tmp)
        CUDA.@atomic cu_grid.Fdrag[p2n, 2] = +(cu_grid.Fdrag[p2n, 2], Ni*tmp)
        # compute nodal force (fw = fw_trac+fw_grav-fw_int) for water
        CUDA.@atomic cu_grid.Fw[p2n, 1] = +(cu_grid.Fw[p2n, 1], 
            -cu_mp.vol[ix]*∂Nx*cu_mp.σw[ix])
        CUDA.@atomic cu_grid.Fw[p2n, 2] = +(cu_grid.Fw[p2n, 2],
            -cu_mp.vol[ix]*∂Ny*cu_mp.σw[ix]+Ni*cu_mp.Mw[ix]*gravity)
        # compute nodal total force for solid
        CUDA.@atomic cu_grid.Fs[p2n, 1] = +(cu_grid.Fs[p2n, 1],
            -cu_mp.vol[ix]*(∂Nx*(cu_mp.σij[ix, 1]+cu_mp.σw[ix])+
                            ∂Ny* cu_mp.σij[ix, 4]))
        CUDA.@atomic cu_grid.Fs[p2n, 2] = +(cu_grid.Fs[p2n, 2],
            -cu_mp.vol[ix]*(∂Ny*(cu_mp.σij[ix, 2]+cu_mp.σw[ix])+
                            ∂Nx* cu_mp.σij[ix, 4])+
            Ni*cu_mp.Mi[ix]*gravity)
    end
    return nothing
end

function kernel_TS03!(cu_grid::    KernelGrid3D{T1, T2},
                      cu_mp  ::KernelParticle3D{T1, T2},
                      gravity::T2) where {T1, T2}
    FNUM_1 = T2(1.0); INUM_1 = T1(1)
    # GPU global threads index
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x # thread x_dim index
    iy = (blockIdx().y-INUM_1)*blockDim().y+threadIdx().y # thread y_dim index
    # kernel function
    if (ix≤cu_mp.num)&&(iy≤cu_mp.NIC)
        Ni  = cu_mp.Ni[ix, iy]
        ∂Nx = cu_mp.∂Nx[ix, iy]
        ∂Ny = cu_mp.∂Ny[ix, iy]
        ∂Nz = cu_mp.∂Nz[ix, iy]
        p2n = cu_mp.p2n[ix, iy]
        por = cu_mp.porosity[ix]
        # compute nodal mass
        CUDA.@atomic cu_grid.Ms[p2n] = +(cu_grid.Ms[p2n], 
            Ni*cu_mp.Ms[ix]*(FNUM_1-por))
        CUDA.@atomic cu_grid.Mi[p2n] = +(cu_grid.Mi[p2n], 
            Ni*cu_mp.Mw[ix]*(       por))   
        CUDA.@atomic cu_grid.Mw[p2n] = +(cu_grid.Mw[p2n], 
            Ni*cu_mp.Mw[ix])
        # compute nodal momentum
        CUDA.@atomic cu_grid.Ps[p2n, 1] = +(cu_grid.Ps[p2n, 1],
            Ni*cu_mp.Ps[ix, 1]*(FNUM_1-por))
        CUDA.@atomic cu_grid.Ps[p2n, 2] = +(cu_grid.Ps[p2n, 2],
            Ni*cu_mp.Ps[ix, 2]*(FNUM_1-por))
        CUDA.@atomic cu_grid.Ps[p2n, 3] = +(cu_grid.Ps[p2n, 3],
            Ni*cu_mp.Ps[ix, 3]*(FNUM_1-por))
        CUDA.@atomic cu_grid.Pw[p2n, 1] = +(cu_grid.Pw[p2n, 1],
            Ni*cu_mp.Pw[ix, 1]*(       por))
        CUDA.@atomic cu_grid.Pw[p2n, 2] = +(cu_grid.Pw[p2n, 2],
            Ni*cu_mp.Pw[ix, 2]*(       por))
        CUDA.@atomic cu_grid.Pw[p2n, 3] = +(cu_grid.Pw[p2n, 3],
            Ni*cu_mp.Pw[ix, 3]*(       por))
        # compute nodal drag force for water
        tmp = -(cu_mp.Mw[ix]*gravity*cu_mp.porosity[ix])/cu_mp.k[ix]
        CUDA.@atomic cu_grid.Fdrag[p2n, 1] = +(cu_grid.Fdrag[p2n, 1], Ni*tmp)
        CUDA.@atomic cu_grid.Fdrag[p2n, 2] = +(cu_grid.Fdrag[p2n, 2], Ni*tmp)
        CUDA.@atomic cu_grid.Fdrag[p2n, 3] = +(cu_grid.Fdrag[p2n, 3], Ni*tmp)
        # compute nodal force (fw = fw_trac+fw_grav-fw_int) for water
        CUDA.@atomic cu_grid.Fw[p2n, 1] = +(cu_grid.Fw[p2n, 1], 
            -cu_mp.vol[ix]*∂Nx*cu_mp.σw[ix])
        CUDA.@atomic cu_grid.Fw[p2n, 2] = +(cu_grid.Fw[p2n, 2],
            -cu_mp.vol[ix]*∂Ny*cu_mp.σw[ix])
        CUDA.@atomic cu_grid.Fw[p2n, 3] = +(cu_grid.Fw[p2n, 3],
            -cu_mp.vol[ix]*∂Nz*cu_mp.σw[ix]+Ni*cu_mp.Mw[ix]*gravity)
        # compute nodal total force for solid
        CUDA.@atomic cu_grid.Fs[p2n, 1] = +(cu_grid.Fs[p2n, 1],
            -cu_mp.vol[ix]*(∂Nx*(cu_mp.σij[ix, 1]+cu_mp.σw[ix])+
                            ∂Ny* cu_mp.σij[ix, 4]+
                            ∂Nz* cu_mp.σij[ix, 6]))
        CUDA.@atomic cu_grid.Fs[p2n, 2] = +(cu_grid.Fs[p2n, 2],
            -cu_mp.vol[ix]*(∂Ny*(cu_mp.σij[ix, 2]+cu_mp.σw[ix])+
                            ∂Nx* cu_mp.σij[ix, 4]+
                            ∂Nz* cu_mp.σij[ix, 5]))
        CUDA.@atomic cu_grid.Fs[p2n, 3] = +(cu_grid.Fs[p2n, 3],
            -cu_mp.vol[ix]*(∂Nz*(cu_mp.σij[ix, 3]+cu_mp.σw[ix])+
                            ∂Nx* cu_mp.σij[ix, 6]+
                            ∂Ny* cu_mp.σij[ix, 5])+
            Ni*cu_mp.Mi[ix]*gravity)
    end
    return nothing
end

function kernel_TS04!(cu_grid::     KernelGrid2D{T1, T2},
                      cu_bc  ::KernelVBoundary2D{T1, T2},
                      ΔT     ::T2,
                      ζ      ::T2) where {T1, T2}
    INUM_0 = T1(0); INUM_1 = T1(1)
    INUM_2 = T1(2); FNUM_0 = T2(0.0)
    # GPU global threads index
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x # thread x_dim index
    # kernel function
    if ix≤cu_grid.node_num
        # compute nodal velocity
        cu_grid.Vs[ix, 1] = cu_grid.Ps[ix, 1]/cu_grid.Ms[ix]
        cu_grid.Vs[ix, 2] = cu_grid.Ps[ix, 2]/cu_grid.Ms[ix]
        cu_grid.Vw[ix, 1] = cu_grid.Pw[ix, 1]/cu_grid.Mi[ix]
        cu_grid.Vw[ix, 2] = cu_grid.Pw[ix, 2]/cu_grid.Mi[ix]
        isnan(cu_grid.Vs[ix, 1]) ? cu_grid.Vs[ix, 1]=FNUM_0 : nothing
        isnan(cu_grid.Vs[ix, 2]) ? cu_grid.Vs[ix, 2]=FNUM_0 : nothing
        isnan(cu_grid.Vw[ix, 1]) ? cu_grid.Vw[ix, 1]=FNUM_0 : nothing
        isnan(cu_grid.Vw[ix, 2]) ? cu_grid.Vw[ix, 2]=FNUM_0 : nothing
        # compute nodal drag force
        cu_grid.Fdrag[ix, 1] = cu_grid.Fdrag[ix, 1]*(cu_grid.Vw[ix, 1]-cu_grid.Vs[ix, 1])
        cu_grid.Fdrag[ix, 2] = cu_grid.Fdrag[ix, 2]*(cu_grid.Vw[ix, 2]-cu_grid.Vs[ix, 2])
        # damping force for water
        dampvw = -ζ*sqrt(cu_grid.Fw[ix, 1]^INUM_2+cu_grid.Fw[ix, 2]^INUM_2)
        cu_grid.Fw_damp[ix, 1] = sign(cu_grid.Vw[ix, 1])*dampvw
        cu_grid.Fw_damp[ix, 2] = sign(cu_grid.Vw[ix, 2])*dampvw
        cu_grid.Fw[ix, 1] = cu_grid.Fw[ix, 1]+cu_grid.Fw_damp[ix, 1]-cu_grid.Fdrag[ix, 1]
        cu_grid.Fw[ix, 2] = cu_grid.Fw[ix, 2]+cu_grid.Fw_damp[ix, 2]-cu_grid.Fdrag[ix, 2]
        # compute nodal accelaration for water
        cu_grid.a_w[ix, 1] = cu_grid.Fw[ix, 1]/cu_grid.Mw[ix]
        cu_grid.a_w[ix, 2] = cu_grid.Fw[ix, 2]/cu_grid.Mw[ix]
        isnan(cu_grid.a_w[ix, 1]) ? cu_grid.a_w[ix, 1]=FNUM_0 : nothing
        isnan(cu_grid.a_w[ix, 2]) ? cu_grid.a_w[ix, 2]=FNUM_0 : nothing
        # damping force for solid
        dampvs = -ζ*sqrt((cu_grid.Fs[ix, 1]-cu_grid.Fw[ix, 1])^INUM_2+
                         (cu_grid.Fs[ix, 2]-cu_grid.Fw[ix, 2])^INUM_2)
        cu_grid.Fs_damp[ix, 1] = sign(cu_grid.Vs[ix, 1])*dampvs+cu_grid.Fw_damp[ix, 1]
        cu_grid.Fs_damp[ix, 2] = sign(cu_grid.Vs[ix, 2])*dampvs+cu_grid.Fw_damp[ix, 2]
        # compute nodal total force for mixture
        cu_grid.Fs[ix, 1] = cu_grid.Fs[ix, 1]-cu_grid.Mi[ix]*cu_grid.a_w[ix, 1]+
                            cu_grid.Fs_damp[ix, 1]
        cu_grid.Fs[ix, 2] = cu_grid.Fs[ix, 2]-cu_grid.Mi[ix]*cu_grid.a_w[ix, 2]+
                            cu_grid.Fs_damp[ix, 2]
        # compute nodal accelaration for solid
        cu_grid.a_s[ix, 1] = cu_grid.Fs[ix, 1]/cu_grid.Ms[ix]
        cu_grid.a_s[ix, 2] = cu_grid.Fs[ix, 2]/cu_grid.Ms[ix]
        isnan(cu_grid.a_s[ix, 1]) ? cu_grid.a_s[ix, 1]=FNUM_0 : nothing
        isnan(cu_grid.a_s[ix, 2]) ? cu_grid.a_s[ix, 2]=FNUM_0 : nothing
        # update nodal velocity
        cu_grid.Vs_T[ix, 1] = (cu_grid.Ps[ix, 1]+cu_grid.Fs[ix, 1]*ΔT)/cu_grid.Ms[ix]
        cu_grid.Vs_T[ix, 2] = (cu_grid.Ps[ix, 2]+cu_grid.Fs[ix, 2]*ΔT)/cu_grid.Ms[ix]
        cu_grid.Vw_T[ix, 1] = (cu_grid.Pw[ix, 1]+cu_grid.Fw[ix, 1]*ΔT)/cu_grid.Mi[ix]
        cu_grid.Vw_T[ix, 2] = (cu_grid.Pw[ix, 2]+cu_grid.Fw[ix, 2]*ΔT)/cu_grid.Mi[ix]
        isnan(cu_grid.Vs_T[ix, 1]) ? cu_grid.Vs_T[ix, 1]=FNUM_0 : nothing
        isnan(cu_grid.Vs_T[ix, 2]) ? cu_grid.Vs_T[ix, 2]=FNUM_0 : nothing
        isnan(cu_grid.Vw_T[ix, 1]) ? cu_grid.Vw_T[ix, 1]=FNUM_0 : nothing
        isnan(cu_grid.Vw_T[ix, 2]) ? cu_grid.Vw_T[ix, 2]=FNUM_0 : nothing
        # boundary condition
        cu_bc.Vx_s_Idx[ix]≠INUM_0 ? cu_grid.Vs_T[ix, 1]=cu_bc.Vx_s_Val[ix] : nothing
        cu_bc.Vy_s_Idx[ix]≠INUM_0 ? cu_grid.Vs_T[ix, 2]=cu_bc.Vy_s_Val[ix] : nothing
        cu_bc.Vx_w_Idx[ix]≠INUM_0 ? cu_grid.Vw_T[ix, 1]=cu_bc.Vx_w_Val[ix] : nothing
        cu_bc.Vy_w_Idx[ix]≠INUM_0 ? cu_grid.Vw_T[ix, 2]=cu_bc.Vy_w_Val[ix] : nothing
        # reset grid momentum
        cu_grid.Ps[ix, 1] = FNUM_0
        cu_grid.Ps[ix, 2] = FNUM_0
        cu_grid.Pw[ix, 1] = FNUM_0
        cu_grid.Pw[ix, 2] = FNUM_0
    end
    return nothing
end

function kernel_TS04!(cu_grid::     KernelGrid3D{T1, T2},
                      cu_bc  ::KernelVBoundary3D{T1, T2},
                      ΔT     ::T2,
                      ζ      ::T2) where {T1, T2}
    INUM_0 = T1(0); INUM_1 = T1(1); INUM_2 = T1(2); FNUM_0 = T2(0.0)
    # GPU global threads index
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x # thread x_dim index
    # kernel function
    if ix≤cu_grid.node_num
        # compute nodal velocity
        cu_grid.Vs[ix, 1] = cu_grid.Ps[ix, 1]/cu_grid.Ms[ix]
        cu_grid.Vs[ix, 2] = cu_grid.Ps[ix, 2]/cu_grid.Ms[ix]
        cu_grid.Vs[ix, 3] = cu_grid.Ps[ix, 3]/cu_grid.Ms[ix]
        cu_grid.Vw[ix, 1] = cu_grid.Pw[ix, 1]/cu_grid.Mi[ix]
        cu_grid.Vw[ix, 2] = cu_grid.Pw[ix, 2]/cu_grid.Mi[ix]
        cu_grid.Vw[ix, 3] = cu_grid.Pw[ix, 3]/cu_grid.Mi[ix]
        isnan(cu_grid.Vs[ix, 1]) ? cu_grid.Vs[ix, 1]=FNUM_0 : nothing
        isnan(cu_grid.Vs[ix, 2]) ? cu_grid.Vs[ix, 2]=FNUM_0 : nothing
        isnan(cu_grid.Vs[ix, 3]) ? cu_grid.Vs[ix, 3]=FNUM_0 : nothing
        isnan(cu_grid.Vw[ix, 1]) ? cu_grid.Vw[ix, 1]=FNUM_0 : nothing
        isnan(cu_grid.Vw[ix, 2]) ? cu_grid.Vw[ix, 2]=FNUM_0 : nothing
        isnan(cu_grid.Vw[ix, 3]) ? cu_grid.Vw[ix, 3]=FNUM_0 : nothing
        # compute nodal drag force
        cu_grid.Fdrag[ix, 1] = cu_grid.Fdrag[ix, 1]*(cu_grid.Vw[ix, 1]-cu_grid.Vs[ix, 1])
        cu_grid.Fdrag[ix, 2] = cu_grid.Fdrag[ix, 2]*(cu_grid.Vw[ix, 2]-cu_grid.Vs[ix, 2])
        cu_grid.Fdrag[ix, 3] = cu_grid.Fdrag[ix, 3]*(cu_grid.Vw[ix, 3]-cu_grid.Vs[ix, 3])
        # damping force for water
        dampvw = -ζ*sqrt(cu_grid.Fw[ix, 1]^INUM_2+
                         cu_grid.Fw[ix, 2]^INUM_2+
                         cu_grid.Fw[ix, 3]^INUM_2)
        cu_grid.Fw_damp[ix, 1] = sign(cu_grid.Vw[ix, 1])*dampvw
        cu_grid.Fw_damp[ix, 2] = sign(cu_grid.Vw[ix, 2])*dampvw
        cu_grid.Fw_damp[ix, 3] = sign(cu_grid.Vw[ix, 3])*dampvw
        cu_grid.Fw[ix, 1] = cu_grid.Fw[ix, 1]+cu_grid.Fw_damp[ix, 1]-cu_grid.Fdrag[ix, 1]
        cu_grid.Fw[ix, 2] = cu_grid.Fw[ix, 2]+cu_grid.Fw_damp[ix, 2]-cu_grid.Fdrag[ix, 2]
        cu_grid.Fw[ix, 3] = cu_grid.Fw[ix, 3]+cu_grid.Fw_damp[ix, 3]-cu_grid.Fdrag[ix, 3]
        # compute nodal accelaration for water
        cu_grid.a_w[ix, 1] = cu_grid.Fw[ix, 1]/cu_grid.Mw[ix]
        cu_grid.a_w[ix, 2] = cu_grid.Fw[ix, 2]/cu_grid.Mw[ix]
        cu_grid.a_w[ix, 3] = cu_grid.Fw[ix, 3]/cu_grid.Mw[ix]
        isnan(cu_grid.a_w[ix, 1]) ? cu_grid.a_w[ix, 1]=FNUM_0 : nothing
        isnan(cu_grid.a_w[ix, 2]) ? cu_grid.a_w[ix, 2]=FNUM_0 : nothing
        isnan(cu_grid.a_w[ix, 3]) ? cu_grid.a_w[ix, 3]=FNUM_0 : nothing
        # damping force for solid
        dampvs = -ζ*sqrt((cu_grid.Fs[ix, 1]-cu_grid.Fw[ix, 1])^INUM_2+
                         (cu_grid.Fs[ix, 2]-cu_grid.Fw[ix, 2])^INUM_2+
                         (cu_grid.Fs[ix, 3]-cu_grid.Fw[ix, 3])^INUM_2)
        cu_grid.Fs_damp[ix, 1] = sign(cu_grid.Vs[ix, 1])*dampvs+cu_grid.Fw_damp[ix, 1]
        cu_grid.Fs_damp[ix, 2] = sign(cu_grid.Vs[ix, 2])*dampvs+cu_grid.Fw_damp[ix, 2]
        cu_grid.Fs_damp[ix, 3] = sign(cu_grid.Vs[ix, 3])*dampvs+cu_grid.Fw_damp[ix, 3]
        # compute nodal total force for mixture
        cu_grid.Fs[ix, 1] = cu_grid.Fs[ix, 1]-cu_grid.Mi[ix]*cu_grid.a_w[ix, 1]+
                            cu_grid.Fs_damp[ix, 1]
        cu_grid.Fs[ix, 2] = cu_grid.Fs[ix, 2]-cu_grid.Mi[ix]*cu_grid.a_w[ix, 2]+
                            cu_grid.Fs_damp[ix, 2]
        cu_grid.Fs[ix, 3] = cu_grid.Fs[ix, 3]-cu_grid.Mi[ix]*cu_grid.a_w[ix, 3]+
                            cu_grid.Fs_damp[ix, 3]
        # compute nodal accelaration for solid
        cu_grid.a_s[ix, 1] = cu_grid.Fs[ix, 1]/cu_grid.Ms[ix]
        cu_grid.a_s[ix, 2] = cu_grid.Fs[ix, 2]/cu_grid.Ms[ix]
        cu_grid.a_s[ix, 3] = cu_grid.Fs[ix, 3]/cu_grid.Ms[ix]
        isnan(cu_grid.a_s[ix, 1]) ? cu_grid.a_s[ix, 1]=FNUM_0 : nothing
        isnan(cu_grid.a_s[ix, 2]) ? cu_grid.a_s[ix, 2]=FNUM_0 : nothing
        isnan(cu_grid.a_s[ix, 3]) ? cu_grid.a_s[ix, 3]=FNUM_0 : nothing
        # update nodal velocity
        cu_grid.Vs_T[ix, 1] = (cu_grid.Ps[ix, 1]+cu_grid.Fs[ix, 1]*ΔT)/cu_grid.Ms[ix]
        cu_grid.Vs_T[ix, 2] = (cu_grid.Ps[ix, 2]+cu_grid.Fs[ix, 2]*ΔT)/cu_grid.Ms[ix]
        cu_grid.Vs_T[ix, 3] = (cu_grid.Ps[ix, 3]+cu_grid.Fs[ix, 3]*ΔT)/cu_grid.Ms[ix]
        cu_grid.Vw_T[ix, 1] = (cu_grid.Pw[ix, 1]+cu_grid.Fw[ix, 1]*ΔT)/cu_grid.Mi[ix]
        cu_grid.Vw_T[ix, 2] = (cu_grid.Pw[ix, 2]+cu_grid.Fw[ix, 2]*ΔT)/cu_grid.Mi[ix]
        cu_grid.Vw_T[ix, 3] = (cu_grid.Pw[ix, 3]+cu_grid.Fw[ix, 3]*ΔT)/cu_grid.Mi[ix]
        isnan(cu_grid.Vs_T[ix, 1]) ? cu_grid.Vs_T[ix, 1]=FNUM_0 : nothing
        isnan(cu_grid.Vs_T[ix, 2]) ? cu_grid.Vs_T[ix, 2]=FNUM_0 : nothing
        isnan(cu_grid.Vs_T[ix, 3]) ? cu_grid.Vs_T[ix, 3]=FNUM_0 : nothing
        isnan(cu_grid.Vw_T[ix, 1]) ? cu_grid.Vw_T[ix, 1]=FNUM_0 : nothing
        isnan(cu_grid.Vw_T[ix, 2]) ? cu_grid.Vw_T[ix, 2]=FNUM_0 : nothing
        isnan(cu_grid.Vw_T[ix, 3]) ? cu_grid.Vw_T[ix, 3]=FNUM_0 : nothing
        # boundary condition
        cu_bc.Vx_s_Idx[ix]≠INUM_0 ? cu_grid.Vs_T[ix, 1]=cu_bc.Vx_s_Val[ix] : nothing
        cu_bc.Vy_s_Idx[ix]≠INUM_0 ? cu_grid.Vs_T[ix, 2]=cu_bc.Vy_s_Val[ix] : nothing
        cu_bc.Vz_s_Idx[ix]≠INUM_0 ? cu_grid.Vs_T[ix, 3]=cu_bc.Vz_s_Val[ix] : nothing
        cu_bc.Vx_w_Idx[ix]≠INUM_0 ? cu_grid.Vw_T[ix, 1]=cu_bc.Vx_w_Val[ix] : nothing
        cu_bc.Vy_w_Idx[ix]≠INUM_0 ? cu_grid.Vw_T[ix, 2]=cu_bc.Vy_w_Val[ix] : nothing
        cu_bc.Vz_w_Idx[ix]≠INUM_0 ? cu_grid.Vw_T[ix, 3]=cu_bc.Vz_w_Val[ix] : nothing
        # reset grid momentum
        cu_grid.Ps[ix, 1] = FNUM_0
        cu_grid.Ps[ix, 2] = FNUM_0
        cu_grid.Ps[ix, 3] = FNUM_0
        cu_grid.Pw[ix, 1] = FNUM_0
        cu_grid.Pw[ix, 2] = FNUM_0
        cu_grid.Pw[ix, 3] = FNUM_0
    end
    return nothing
end

function kernel_TS05!(cu_grid::    KernelGrid2D{T1, T2},
                      cu_mp  ::KernelParticle2D{T1, T2},
                      ΔT     ::T2,
                      FLIP   ::T2,
                      PIC    ::T2) where {T1, T2}
    FNUM_0 = T2(0.0); FNUM_1 = T2(1.0); INUM_1 = T1(1)
    # GPU global threads index
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x # thread x_dim index
    # update particle position & velocity
    if ix≤cu_mp.num
        tmp_vx_s1 = tmp_vx_s2 = tmp_vy_s1 = tmp_vy_s2 = FNUM_0
        tmp_vx_w1 = tmp_vx_w2 = tmp_vy_w1 = tmp_vy_w2 = FNUM_0
        # update particle position
        for iy in INUM_1:cu_mp.NIC
            Ni  = cu_mp.Ni[ix, iy]
            p2n = cu_mp.p2n[ix, iy]
            cu_mp.pos[ix, 1] += ΔT*(Ni*cu_grid.Vs_T[p2n, 1])
            cu_mp.pos[ix, 2] += ΔT*(Ni*cu_grid.Vs_T[p2n, 2])
            tmp_vx_s1 += Ni*(cu_grid.Vs_T[p2n, 1]-cu_grid.Vs[p2n, 1])
            tmp_vx_s2 += Ni* cu_grid.Vs_T[p2n, 1]
            tmp_vy_s1 += Ni*(cu_grid.Vs_T[p2n, 2]-cu_grid.Vs[p2n, 2])
            tmp_vy_s2 += Ni* cu_grid.Vs_T[p2n, 2]
            tmp_vx_w1 += Ni*(cu_grid.Vw_T[p2n, 1]-cu_grid.Vw[p2n, 1])
            tmp_vx_w2 += Ni* cu_grid.Vw_T[p2n, 1]
            tmp_vy_w1 += Ni*(cu_grid.Vw_T[p2n, 2]-cu_grid.Vw[p2n, 2])
            tmp_vy_w2 += Ni* cu_grid.Vw_T[p2n, 2]
        end
        cu_mp.Vs[ix, 1] = FLIP*(cu_mp.Vs[ix, 1]+tmp_vx_s1)+PIC*tmp_vx_s2
        cu_mp.Vs[ix, 2] = FLIP*(cu_mp.Vs[ix, 2]+tmp_vy_s1)+PIC*tmp_vy_s2
        cu_mp.Vw[ix, 1] = FLIP*(cu_mp.Vw[ix, 1]+tmp_vx_w1)+PIC*tmp_vx_w2
        cu_mp.Vw[ix, 2] = FLIP*(cu_mp.Vw[ix, 2]+tmp_vy_w1)+PIC*tmp_vy_w2
        # update particle momentum
        cu_mp.Ps[ix, 1] = cu_mp.Ms[ix]*cu_mp.Vs[ix, 1]
        cu_mp.Ps[ix, 2] = cu_mp.Ms[ix]*cu_mp.Vs[ix, 2]
        cu_mp.Pw[ix, 1] = cu_mp.Mw[ix]*cu_mp.Vw[ix, 1]
        cu_mp.Pw[ix, 2] = cu_mp.Mw[ix]*cu_mp.Vw[ix, 2]
        # update CFL conditions
        sqr = sqrt((cu_mp.E[ix]+cu_mp.Kw[ix]*cu_mp.porosity[ix])/
                   (cu_mp.ρs[ix]*(FNUM_1-cu_mp.porosity[ix])+
                    cu_mp.ρw[ix]*        cu_mp.porosity[ix]))
        cd_sx = cu_grid.space_x/(sqr+abs(cu_mp.Vs[ix, 1]))
        cd_sy = cu_grid.space_y/(sqr+abs(cu_mp.Vs[ix, 2]))
        cd_wx = cu_grid.space_x/(sqr+abs(cu_mp.Vw[ix, 1]))
        cd_wy = cu_grid.space_y/(sqr+abs(cu_mp.Vw[ix, 2]))
        cu_mp.cfl[ix] = min(cd_sx, cd_sy, cd_wx, cd_wy)
    end
    return nothing
end

function kernel_TS05!(cu_grid::    KernelGrid3D{T1, T2},
                      cu_mp  ::KernelParticle3D{T1, T2},
                      ΔT     ::T2,
                      FLIP   ::T2,
                      PIC    ::T2) where {T1, T2}
    FNUM_0 = T2(0.0); FNUM_1 = T2(1.0); INUM_1 = T1(1)
    # GPU global threads index
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x # thread x_dim index
    # update particle position & velocity
    if ix≤cu_mp.num
        tmp_vx_s1 = tmp_vx_s2 = tmp_vy_s1 = tmp_vy_s2 = tmp_vz_s1 = tmp_vz_s2 = FNUM_0
        tmp_vx_w1 = tmp_vx_w2 = tmp_vy_w1 = tmp_vy_w2 = tmp_vz_w1 = tmp_vz_w2 = FNUM_0
        # update particle position
        for iy in INUM_1:cu_mp.NIC
            Ni  = cu_mp.Ni[ix, iy]
            p2n = cu_mp.p2n[ix, iy]
            cu_mp.pos[ix, 1] += ΔT*(Ni*cu_grid.Vs_T[p2n, 1])
            cu_mp.pos[ix, 2] += ΔT*(Ni*cu_grid.Vs_T[p2n, 2])
            cu_mp.pos[ix, 3] += ΔT*(Ni*cu_grid.Vs_T[p2n, 3])
            tmp_vx_s1 += Ni*(cu_grid.Vs_T[p2n, 1]-cu_grid.Vs[p2n, 1])
            tmp_vx_s2 += Ni* cu_grid.Vs_T[p2n, 1]
            tmp_vy_s1 += Ni*(cu_grid.Vs_T[p2n, 2]-cu_grid.Vs[p2n, 2])
            tmp_vy_s2 += Ni* cu_grid.Vs_T[p2n, 2]
            tmp_vz_s1 += Ni*(cu_grid.Vs_T[p2n, 3]-cu_grid.Vs[p2n, 3])
            tmp_vz_s2 += Ni* cu_grid.Vs_T[p2n, 3]
            tmp_vx_w1 += Ni*(cu_grid.Vw_T[p2n, 1]-cu_grid.Vw[p2n, 1])
            tmp_vx_w2 += Ni* cu_grid.Vw_T[p2n, 1]
            tmp_vy_w1 += Ni*(cu_grid.Vw_T[p2n, 2]-cu_grid.Vw[p2n, 2])
            tmp_vy_w2 += Ni* cu_grid.Vw_T[p2n, 2]
            tmp_vz_w1 += Ni*(cu_grid.Vw_T[p2n, 3]-cu_grid.Vw[p2n, 3])
            tmp_vz_w2 += Ni* cu_grid.Vw_T[p2n, 3]
        end
        cu_mp.Vs[ix, 1] = FLIP*(cu_mp.Vs[ix, 1]+tmp_vx_s1)+PIC*tmp_vx_s2
        cu_mp.Vs[ix, 2] = FLIP*(cu_mp.Vs[ix, 2]+tmp_vy_s1)+PIC*tmp_vy_s2
        cu_mp.Vs[ix, 3] = FLIP*(cu_mp.Vs[ix, 3]+tmp_vz_s1)+PIC*tmp_vz_s2
        cu_mp.Vw[ix, 1] = FLIP*(cu_mp.Vw[ix, 1]+tmp_vx_w1)+PIC*tmp_vx_w2
        cu_mp.Vw[ix, 2] = FLIP*(cu_mp.Vw[ix, 2]+tmp_vy_w1)+PIC*tmp_vy_w2
        cu_mp.Vw[ix, 3] = FLIP*(cu_mp.Vw[ix, 3]+tmp_vz_w1)+PIC*tmp_vz_w2
        # update particle momentum
        cu_mp.Ps[ix, 1] = cu_mp.Ms[ix]*cu_mp.Vs[ix, 1]
        cu_mp.Ps[ix, 2] = cu_mp.Ms[ix]*cu_mp.Vs[ix, 2]
        cu_mp.Ps[ix, 3] = cu_mp.Ms[ix]*cu_mp.Vs[ix, 3]
        cu_mp.Pw[ix, 1] = cu_mp.Mw[ix]*cu_mp.Vw[ix, 1]
        cu_mp.Pw[ix, 2] = cu_mp.Mw[ix]*cu_mp.Vw[ix, 2]
        cu_mp.Pw[ix, 3] = cu_mp.Mw[ix]*cu_mp.Vw[ix, 3]
        # update CFL conditions
        sqr = sqrt((cu_mp.E[ix]+cu_mp.Kw[ix]*cu_mp.porosity[ix])/
                   (cu_mp.ρs[ix]*(FNUM_1-cu_mp.porosity[ix])+
                    cu_mp.ρw[ix]*        cu_mp.porosity[ix]))
        cd_sx = cu_grid.space_x/(sqr+abs(cu_mp.Vs[ix, 1]))
        cd_sy = cu_grid.space_y/(sqr+abs(cu_mp.Vs[ix, 2]))
        cd_sz = cu_grid.space_z/(sqr+abs(cu_mp.Vs[ix, 3]))
        cd_wx = cu_grid.space_x/(sqr+abs(cu_mp.Vw[ix, 1]))
        cd_wy = cu_grid.space_y/(sqr+abs(cu_mp.Vw[ix, 2]))
        cd_wz = cu_grid.space_z/(sqr+abs(cu_mp.Vw[ix, 3]))
        cu_mp.cfl[ix] = min(cd_sx, cd_sy, cd_sz, cd_wx, cd_wy, cd_wz)
    end
    return nothing
end

function kernel_TS06!(cu_grid::    KernelGrid2D{T1, T2},
                      cu_mp  ::KernelParticle2D{T1, T2}) where {T1, T2}
    INUM_1 = T1(1); FNUM_1 = T2(1.0)
    # GPU global threads index
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x # thread x_dim index
    iy = (blockIdx().y-INUM_1)*blockDim().y+threadIdx().y # thread y_dim index
    # update particle position & velocity
    if (ix≤cu_mp.num)&&(iy≤cu_mp.NIC)
        Ni  = cu_mp.Ni[ix, iy]
        p2n = cu_mp.p2n[ix, iy]
        por = cu_mp.porosity[ix, iy]
        CUDA.@atomic cu_grid.Ps[p2n, 1] = +(cu_grid.Ps[p2n, 1], 
            cu_mp.Ps[ix, 1]*Ni*(FNUM_1-por))
        CUDA.@atomic cu_grid.Ps[p2n, 2] = +(cu_grid.Ps[p2n, 2], 
            cu_mp.Ps[ix, 2]*Ni*(FNUM_1-por))
        CUDA.@atomic cu_grid.Pw[p2n, 1] = +(cu_grid.Pw[p2n, 1], 
            cu_mp.Pw[ix, 1]*Ni*(       por))
        CUDA.@atomic cu_grid.Pw[p2n, 2] = +(cu_grid.Pw[p2n, 2], 
            cu_mp.Pw[ix, 2]*Ni*(       por))
    end
    return nothing
end

function kernel_TS06!(cu_grid::    KernelGrid3D{T1, T2},
                      cu_mp  ::KernelParticle3D{T1, T2}) where {T1, T2}
    INUM_1 = T1(1); FNUM_1 = T2(1.0)
    # GPU global threads index
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x # thread x_dim index
    iy = (blockIdx().y-INUM_1)*blockDim().y+threadIdx().y # thread y_dim index
    # update particle position & velocity
    if (ix≤cu_mp.num)&&(iy≤cu_mp.NIC)
        Ni  = cu_mp.Ni[ix, iy]
        p2n = cu_mp.p2n[ix, iy]
        CUDA.@atomic cu_grid.Ps[p2n, 1] = +(cu_grid.Ps[p2n, 1], 
            cu_mp.Ps[ix, 1]*Ni*(FNUM_1-cu_mp.porosity[ix]))
        CUDA.@atomic cu_grid.Ps[p2n, 2] = +(cu_grid.Ps[p2n, 2], 
            cu_mp.Ps[ix, 2]*Ni*(FNUM_1-cu_mp.porosity[ix]))
        CUDA.@atomic cu_grid.Ps[p2n, 3] = +(cu_grid.Ps[p2n, 3],
            cu_mp.Ps[ix, 3]*Ni*(FNUM_1-cu_mp.porosity[ix]))
        CUDA.@atomic cu_grid.Pw[p2n, 1] = +(cu_grid.Pw[p2n, 1], 
            cu_mp.Pw[ix, 1]*Ni*cu_mp.porosity[ix])
        CUDA.@atomic cu_grid.Pw[p2n, 2] = +(cu_grid.Pw[p2n, 2], 
            cu_mp.Pw[ix, 2]*Ni*cu_mp.porosity[ix])
        CUDA.@atomic cu_grid.Pw[p2n, 3] = +(cu_grid.Pw[p2n, 3],
            cu_mp.Pw[ix, 3]*Ni*cu_mp.porosity[ix])
    end
    return nothing
end

function kernel_TS07!(cu_grid::     KernelGrid2D{T1, T2},
                      cu_bc  ::KernelVBoundary2D{T1, T2},
                      ΔT     ::T2) where {T1, T2}
    INUM_1 = T1(1); INUM_0 = T1(0); FNUM_0 = T2(0)
    # GPU global threads index
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x # thread x_dim index
    if ix≤cu_grid.node_num
        # compute nodal velocities
        cu_grid.Vs[ix, 1] = cu_grid.Ps[ix, 1]/cu_grid.Ms[ix]
        cu_grid.Vs[ix, 2] = cu_grid.Ps[ix, 2]/cu_grid.Ms[ix]
        cu_grid.Vw[ix, 1] = cu_grid.Pw[ix, 1]/cu_grid.Mi[ix]
        cu_grid.Vw[ix, 2] = cu_grid.Pw[ix, 2]/cu_grid.Mi[ix]
        isnan(cu_grid.Vs[ix, 1]) ? cu_grid.Vs[ix, 1]=FNUM_0 : nothing
        isnan(cu_grid.Vs[ix, 2]) ? cu_grid.Vs[ix, 2]=FNUM_0 : nothing
        isnan(cu_grid.Vw[ix, 1]) ? cu_grid.Vw[ix, 1]=FNUM_0 : nothing
        isnan(cu_grid.Vw[ix, 2]) ? cu_grid.Vw[ix, 2]=FNUM_0 : nothing
        # fixed Dirichlet nodes
        cu_bc.Vx_s_Idx[ix]≠INUM_0 ? cu_grid.Vs[ix, 1]=cu_bc.Vx_s_Val[ix] : nothing
        cu_bc.Vy_s_Idx[ix]≠INUM_0 ? cu_grid.Vs[ix, 2]=cu_bc.Vy_s_Val[ix] : nothing
        cu_bc.Vx_w_Idx[ix]≠INUM_0 ? cu_grid.Vw[ix, 1]=cu_bc.Vx_w_Val[ix] : nothing
        cu_bc.Vy_w_Idx[ix]≠INUM_0 ? cu_grid.Vw[ix, 2]=cu_bc.Vy_w_Val[ix] : nothing
        # compute nodal displacement
        cu_grid.Δd_s[ix, 1] = cu_grid.Vs[ix, 1]*ΔT
        cu_grid.Δd_s[ix, 2] = cu_grid.Vs[ix, 2]*ΔT
        cu_grid.Δd_w[ix, 1] = cu_grid.Vw[ix, 1]*ΔT
        cu_grid.Δd_w[ix, 2] = cu_grid.Vw[ix, 2]*ΔT
    end
    return nothing
end

function kernel_TS07!(cu_grid::     KernelGrid3D{T1, T2},
                      cu_bc  ::KernelVBoundary3D{T1, T2},
                      ΔT     ::T2) where {T1, T2}
    INUM_1 = T1(1); INUM_0 = T1(0); FNUM_0 = T2(0)
    # GPU global threads index
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x # thread x_dim index
    if ix≤cu_grid.node_num
        # compute nodal velocities
        cu_grid.Vs[ix, 1] = cu_grid.Ps[ix, 1]/cu_grid.Ms[ix]
        cu_grid.Vs[ix, 2] = cu_grid.Ps[ix, 2]/cu_grid.Ms[ix]
        cu_grid.Vs[ix, 3] = cu_grid.Ps[ix, 3]/cu_grid.Ms[ix]
        cu_grid.Vw[ix, 1] = cu_grid.Pw[ix, 1]/cu_grid.Mi[ix]
        cu_grid.Vw[ix, 2] = cu_grid.Pw[ix, 2]/cu_grid.Mi[ix]
        cu_grid.Vw[ix, 3] = cu_grid.Pw[ix, 3]/cu_grid.Mi[ix]
        isnan(cu_grid.Vs[ix, 1]) ? cu_grid.Vs[ix, 1]=FNUM_0 : nothing
        isnan(cu_grid.Vs[ix, 2]) ? cu_grid.Vs[ix, 2]=FNUM_0 : nothing
        isnan(cu_grid.Vs[ix, 3]) ? cu_grid.Vs[ix, 3]=FNUM_0 : nothing
        isnan(cu_grid.Vw[ix, 1]) ? cu_grid.Vw[ix, 1]=FNUM_0 : nothing
        isnan(cu_grid.Vw[ix, 2]) ? cu_grid.Vw[ix, 2]=FNUM_0 : nothing
        isnan(cu_grid.Vw[ix, 3]) ? cu_grid.Vw[ix, 3]=FNUM_0 : nothing
        # fixed Dirichlet nodes
        cu_bc.Vx_s_Idx[ix]≠INUM_0 ? cu_grid.Vs[ix, 1]=cu_bc.Vx_s_Val[ix] : nothing
        cu_bc.Vy_s_Idx[ix]≠INUM_0 ? cu_grid.Vs[ix, 2]=cu_bc.Vy_s_Val[ix] : nothing
        cu_bc.Vz_s_Idx[ix]≠INUM_0 ? cu_grid.Vs[ix, 3]=cu_bc.Vz_s_Val[ix] : nothing
        cu_bc.Vx_w_Idx[ix]≠INUM_0 ? cu_grid.Vw[ix, 1]=cu_bc.Vx_w_Val[ix] : nothing
        cu_bc.Vy_w_Idx[ix]≠INUM_0 ? cu_grid.Vw[ix, 2]=cu_bc.Vy_w_Val[ix] : nothing
        cu_bc.Vz_w_Idx[ix]≠INUM_0 ? cu_grid.Vw[ix, 3]=cu_bc.Vz_w_Val[ix] : nothing
        # compute nodal displacement
        cu_grid.Δd_s[ix, 1] = cu_grid.Vs[ix, 1]*ΔT
        cu_grid.Δd_s[ix, 2] = cu_grid.Vs[ix, 2]*ΔT
        cu_grid.Δd_s[ix, 3] = cu_grid.Vs[ix, 3]*ΔT
        cu_grid.Δd_w[ix, 1] = cu_grid.Vw[ix, 1]*ΔT
        cu_grid.Δd_w[ix, 2] = cu_grid.Vw[ix, 2]*ΔT
        cu_grid.Δd_w[ix, 3] = cu_grid.Vw[ix, 3]*ΔT
    end
    return nothing
end

function kernel_TS08!(cu_grid::    KernelGrid2D{T1, T2},
                      cu_mp  ::KernelParticle2D{T1, T2}) where {T1, T2}
    INUM_1  = T1(1); FNUM_0  = T2(0.0); FNUM_1  = T2(1.0)
    # GPU global threads index
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x # thread x_dim index
    # kernel function
    if ix≤cu_mp.num
        cu_mp.∂Fs[ix, 1] = FNUM_0; cu_mp.∂Fs[ix, 2] = FNUM_0
        cu_mp.∂Fs[ix, 3] = FNUM_0; cu_mp.∂Fs[ix, 4] = FNUM_0
        cu_mp.∂Fw[ix, 1] = FNUM_0; cu_mp.∂Fw[ix, 2] = FNUM_0
        cu_mp.∂Fw[ix, 3] = FNUM_0; cu_mp.∂Fw[ix, 4] = FNUM_0
        for iy in INUM_1:cu_mp.NIC
            p2n = cu_mp.p2n[ix, iy]
            ∂Nx = cu_mp.∂Nx[ix, iy]
            ∂Ny = cu_mp.∂Ny[ix, iy]
            # compute solid incremental deformation gradient
            cu_mp.∂Fs[ix, 1] += cu_grid.Δd_s[p2n, 1]*∂Nx
            cu_mp.∂Fs[ix, 2] += cu_grid.Δd_s[p2n, 1]*∂Ny
            cu_mp.∂Fs[ix, 3] += cu_grid.Δd_s[p2n, 2]*∂Nx
            cu_mp.∂Fs[ix, 4] += cu_grid.Δd_s[p2n, 2]*∂Ny
            # compute water incremental deformation gradient
            cu_mp.∂Fw[ix, 1] += cu_grid.Δd_w[p2n, 1]*∂Nx
            cu_mp.∂Fw[ix, 2] += cu_grid.Δd_w[p2n, 1]*∂Ny
            cu_mp.∂Fw[ix, 3] += cu_grid.Δd_w[p2n, 2]*∂Nx
            cu_mp.∂Fw[ix, 4] += cu_grid.Δd_w[p2n, 2]*∂Ny
        end
        # compute strain increment 
        cu_mp.Δϵij_s[ix, 1] = cu_mp.∂Fs[ix, 1]
        cu_mp.Δϵij_s[ix, 2] = cu_mp.∂Fs[ix, 4]
        cu_mp.Δϵij_s[ix, 4] = cu_mp.∂Fs[ix, 2]+cu_mp.∂Fs[ix, 3]
        cu_mp.Δϵij_w[ix, 1] = cu_mp.∂Fw[ix, 1]             
        cu_mp.Δϵij_w[ix, 2] = cu_mp.∂Fw[ix, 4]             
        cu_mp.Δϵij_w[ix, 4] = cu_mp.∂Fw[ix, 2]+cu_mp.∂Fw[ix, 3]
        # update strain tensor
        cu_mp.ϵij_s[ix, 1] += cu_mp.Δϵij_s[ix, 1]
        cu_mp.ϵij_s[ix, 2] += cu_mp.Δϵij_s[ix, 2]
        cu_mp.ϵij_s[ix, 4] += cu_mp.Δϵij_s[ix, 4] 
        cu_mp.ϵij_w[ix, 1] += cu_mp.Δϵij_w[ix, 1]
        cu_mp.ϵij_w[ix, 2] += cu_mp.Δϵij_w[ix, 2]
        cu_mp.ϵij_w[ix, 4] += cu_mp.Δϵij_w[ix, 4]
        # update pore water pressure
        cu_mp.σw[ix] += (cu_mp.Kw[ix]/cu_mp.porosity[ix])*(
            (FNUM_1-cu_mp.porosity[ix])*(cu_mp.Δϵij_s[ix, 1]+cu_mp.Δϵij_s[ix, 2])+
            (       cu_mp.porosity[ix])*(cu_mp.Δϵij_w[ix, 1]+cu_mp.Δϵij_w[ix, 2]))
        # update jacobian matrix
        cu_mp.J[ix]   = FNUM_1+cu_mp.Δϵij_s[ix, 1]+cu_mp.Δϵij_s[ix, 2]
        cu_mp.vol[ix] = cu_mp.J[ix]*cu_mp.vol[ix]
        # update porosity
        cu_mp.porosity[ix] = FNUM_1-(FNUM_1-cu_mp.porosity[ix])/cu_mp.J[ix] 
    end
    return nothing
end

function kernel_TS08!(cu_grid::    KernelGrid3D{T1, T2},
                      cu_mp  ::KernelParticle3D{T1, T2}) where {T1, T2}
    FNUM_0 = T2(0.0); FNUM_1 = T2(1.0); INUM_1 = T1(1)
    # GPU global threads index
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x # thread x_dim index
    # kernel function
    if ix≤cu_mp.num
        cu_mp.∂Fs[ix, 1] = FNUM_0; cu_mp.∂Fs[ix, 2] = FNUM_0; cu_mp.∂Fs[ix, 3] = FNUM_0
        cu_mp.∂Fs[ix, 4] = FNUM_0; cu_mp.∂Fs[ix, 5] = FNUM_0; cu_mp.∂Fs[ix, 6] = FNUM_0
        cu_mp.∂Fs[ix, 7] = FNUM_0; cu_mp.∂Fs[ix, 8] = FNUM_0; cu_mp.∂Fs[ix, 9] = FNUM_0
        cu_mp.∂Fw[ix, 1] = FNUM_0; cu_mp.∂Fw[ix, 2] = FNUM_0; cu_mp.∂Fw[ix, 3] = FNUM_0
        cu_mp.∂Fw[ix, 4] = FNUM_0; cu_mp.∂Fw[ix, 5] = FNUM_0; cu_mp.∂Fw[ix, 6] = FNUM_0
        cu_mp.∂Fw[ix, 7] = FNUM_0; cu_mp.∂Fw[ix, 8] = FNUM_0; cu_mp.∂Fw[ix, 9] = FNUM_0
        for iy in INUM_1:cu_mp.NIC
            p2n = cu_mp.p2n[ix, iy]
            ∂Nx = cu_mp.∂Nx[ix, iy]
            ∂Ny = cu_mp.∂Ny[ix, iy]
            ∂Nz = cu_mp.∂Nz[ix, iy]
            # compute solid incremental deformation gradient
            cu_mp.∂Fs[ix, 1] += cu_grid.Δd_s[p2n, 1]*∂Nx
            cu_mp.∂Fs[ix, 2] += cu_grid.Δd_s[p2n, 1]*∂Ny
            cu_mp.∂Fs[ix, 3] += cu_grid.Δd_s[p2n, 1]*∂Nz
            cu_mp.∂Fs[ix, 4] += cu_grid.Δd_s[p2n, 2]*∂Nx
            cu_mp.∂Fs[ix, 5] += cu_grid.Δd_s[p2n, 2]*∂Ny
            cu_mp.∂Fs[ix, 6] += cu_grid.Δd_s[p2n, 2]*∂Nz
            cu_mp.∂Fs[ix, 7] += cu_grid.Δd_s[p2n, 3]*∂Nx
            cu_mp.∂Fs[ix, 8] += cu_grid.Δd_s[p2n, 3]*∂Ny
            cu_mp.∂Fs[ix, 9] += cu_grid.Δd_s[p2n, 3]*∂Nz
            # compute water incremental deformation gradient
            cu_mp.∂Fw[ix, 1] += cu_grid.Δd_w[p2n, 1]*∂Nx
            cu_mp.∂Fw[ix, 2] += cu_grid.Δd_w[p2n, 1]*∂Ny
            cu_mp.∂Fw[ix, 3] += cu_grid.Δd_w[p2n, 1]*∂Nz
            cu_mp.∂Fw[ix, 4] += cu_grid.Δd_w[p2n, 2]*∂Nx
            cu_mp.∂Fw[ix, 5] += cu_grid.Δd_w[p2n, 2]*∂Ny
            cu_mp.∂Fw[ix, 6] += cu_grid.Δd_w[p2n, 2]*∂Nz
            cu_mp.∂Fw[ix, 7] += cu_grid.Δd_w[p2n, 3]*∂Nx
            cu_mp.∂Fw[ix, 8] += cu_grid.Δd_w[p2n, 3]*∂Ny
            cu_mp.∂Fw[ix, 9] += cu_grid.Δd_w[p2n, 3]*∂Nz
        end
        # compute strain increment 
        cu_mp.Δϵij_s[ix, 1] = cu_mp.∂Fs[ix, 1]
        cu_mp.Δϵij_s[ix, 2] = cu_mp.∂Fs[ix, 5]
        cu_mp.Δϵij_s[ix, 3] = cu_mp.∂Fs[ix, 9]
        cu_mp.Δϵij_s[ix, 4] = cu_mp.∂Fs[ix, 2]+cu_mp.∂Fs[ix, 4]
        cu_mp.Δϵij_s[ix, 5] = cu_mp.∂Fs[ix, 6]+cu_mp.∂Fs[ix, 8]
        cu_mp.Δϵij_s[ix, 6] = cu_mp.∂Fs[ix, 3]+cu_mp.∂Fs[ix, 7]
        cu_mp.Δϵij_w[ix, 1] = cu_mp.∂Fw[ix, 1]             
        cu_mp.Δϵij_w[ix, 2] = cu_mp.∂Fw[ix, 5]
        cu_mp.Δϵij_w[ix, 3] = cu_mp.∂Fw[ix, 9]             
        cu_mp.Δϵij_w[ix, 4] = cu_mp.∂Fw[ix, 2]+cu_mp.∂Fw[ix, 4]
        cu_mp.Δϵij_w[ix, 5] = cu_mp.∂Fw[ix, 6]+cu_mp.∂Fw[ix, 8]
        cu_mp.Δϵij_w[ix, 6] = cu_mp.∂Fw[ix, 3]+cu_mp.∂Fw[ix, 7]
        # update strain tensor
        cu_mp.ϵij_s[ix, 1] += cu_mp.Δϵij_s[ix, 1]
        cu_mp.ϵij_s[ix, 2] += cu_mp.Δϵij_s[ix, 2]
        cu_mp.ϵij_s[ix, 3] += cu_mp.Δϵij_s[ix, 3]
        cu_mp.ϵij_s[ix, 4] += cu_mp.Δϵij_s[ix, 4] 
        cu_mp.ϵij_s[ix, 5] += cu_mp.Δϵij_s[ix, 5]
        cu_mp.ϵij_s[ix, 6] += cu_mp.Δϵij_s[ix, 6]
        cu_mp.ϵij_w[ix, 1] += cu_mp.Δϵij_w[ix, 1]
        cu_mp.ϵij_w[ix, 2] += cu_mp.Δϵij_w[ix, 2]
        cu_mp.ϵij_w[ix, 3] += cu_mp.Δϵij_w[ix, 3]
        cu_mp.ϵij_w[ix, 4] += cu_mp.Δϵij_w[ix, 4]
        cu_mp.ϵij_w[ix, 5] += cu_mp.Δϵij_w[ix, 5]
        cu_mp.ϵij_w[ix, 6] += cu_mp.Δϵij_w[ix, 6]
        # update pore water pressure
        cu_mp.σw[ix] += (cu_mp.Kw[ix]/cu_mp.porosity[ix])*(
            (FNUM_1-cu_mp.porosity[ix])*(cu_mp.Δϵij_s[ix, 1]+
                                         cu_mp.Δϵij_s[ix, 2]+
                                         cu_mp.Δϵij_s[ix, 3])+
            (       cu_mp.porosity[ix])*(cu_mp.Δϵij_w[ix, 1]+
                                         cu_mp.Δϵij_w[ix, 2]+
                                         cu_mp.Δϵij_w[ix, 3]))
        # update jacobian matrix
        cu_mp.J[ix]   = FNUM_1+cu_mp.Δϵij_s[ix, 1]+cu_mp.Δϵij_s[ix, 2]+cu_mp.Δϵij_s[ix, 3]
        cu_mp.vol[ix] = cu_mp.J[ix]*cu_mp.vol[ix]
        # update porosity
        cu_mp.porosity[ix] = FNUM_1-(FNUM_1-cu_mp.porosity[ix])/cu_mp.J[ix] 
    end
    return nothing
end

function kernel_TS09!(cu_grid::    KernelGrid2D{T1, T2},
                      cu_mp  ::KernelParticle2D{T1, T2}) where {T1, T2}
    INUM_1 = T1(1)
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x
    if ix≤cu_mp.num
        p2c = cu_mp.p2c[ix]
        CUDA.@atomic cu_grid.σm[p2c]  = +(cu_grid.σm[p2c] , cu_mp.vol[ix]*cu_mp.σm[ix])
        CUDA.@atomic cu_grid.σw[p2c]  = +(cu_grid.σw[p2c] , cu_mp.vol[ix]*cu_mp.σw[ix])
        CUDA.@atomic cu_grid.vol[p2c] = +(cu_grid.vol[p2c], cu_mp.vol[ix])
    end
    return nothing
end

function kernel_TS09!(cu_grid::    KernelGrid3D{T1, T2},
                      cu_mp  ::KernelParticle3D{T1, T2}) where {T1, T2}
    INUM_1 = T1(1)
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x
    if ix≤cu_mp.num
        p2c = cu_mp.p2c[ix]
        CUDA.@atomic cu_grid.σm[p2c]  = +(cu_grid.σm[p2c] , cu_mp.vol[ix]*cu_mp.σm[ix])
        CUDA.@atomic cu_grid.σw[p2c]  = +(cu_grid.σw[p2c] , cu_mp.vol[ix]*cu_mp.σw[ix])
        CUDA.@atomic cu_grid.vol[p2c] = +(cu_grid.vol[p2c], cu_mp.vol[ix])
    end
    return nothing
end

function kernel_TS10!(cu_grid::    KernelGrid2D{T1, T2},
                      cu_mp  ::KernelParticle2D{T1, T2}) where {T1, T2}
    INUM_1 = T1(1); FNUM_13 = T2(1/3)
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x
    if ix≤cu_mp.num
        p2c = cu_mp.p2c[ix]
        σm  = cu_grid.σm[p2c]/cu_grid.vol[p2c]
        σw  = cu_grid.σw[p2c]/cu_grid.vol[p2c]
        cu_mp.σij[ix, 1] = cu_mp.sij[ix, 1]+σm
        cu_mp.σij[ix, 2] = cu_mp.sij[ix, 2]+σm
        cu_mp.σij[ix, 3] = cu_mp.sij[ix, 3]+σm
        cu_mp.σij[ix, 4] = cu_mp.sij[ix, 4]
        cu_mp.σw[ix]     = σw
        # update mean stress tensor
        cu_mp.σm[ix] = (cu_mp.σij[ix, 1]+cu_mp.σij[ix, 2]+cu_mp.σij[ix, 3])*FNUM_13
        # update deviatoric stress tensor
        cu_mp.sij[ix, 1] = cu_mp.σij[ix, 1]-cu_mp.σm[ix]
        cu_mp.sij[ix, 2] = cu_mp.σij[ix, 2]-cu_mp.σm[ix]
        cu_mp.sij[ix, 3] = cu_mp.σij[ix, 3]-cu_mp.σm[ix]
        cu_mp.sij[ix, 4] = cu_mp.σij[ix, 4]
    end
    return nothing
end

function kernel_TS10!(cu_grid::    KernelGrid3D{T1, T2},
                      cu_mp  ::KernelParticle3D{T1, T2}) where {T1, T2}
    INUM_1  = T1(1); FNUM_13 = T2(1/3)
    ix = (blockIdx().x-INUM_1)*blockDim().x+threadIdx().x
    if ix≤cu_mp.num
        p2c = cu_mp.p2c[ix]
        σm  = cu_grid.σm[p2c]/cu_grid.vol[p2c]
        σw  = cu_grid.σw[p2c]/cu_grid.vol[p2c]
        cu_mp.σij[ix, 1] = cu_mp.sij[ix, 1]+σm
        cu_mp.σij[ix, 2] = cu_mp.sij[ix, 2]+σm
        cu_mp.σij[ix, 3] = cu_mp.sij[ix, 3]+σm
        cu_mp.σij[ix, 4] = cu_mp.sij[ix, 4]
        cu_mp.σij[ix, 5] = cu_mp.sij[ix, 5]
        cu_mp.σij[ix, 6] = cu_mp.sij[ix, 6]
        cu_mp.σw[ix]     = σw
        # update mean stress tensor
        cu_mp.σm[ix] = (cu_mp.σij[ix, 1]+cu_mp.σij[ix, 2]+cu_mp.σij[ix, 3])*FNUM_13
        # update deviatoric stress tensor
        cu_mp.sij[ix, 1] = cu_mp.σij[ix, 1]-cu_mp.σm[ix]
        cu_mp.sij[ix, 2] = cu_mp.σij[ix, 2]-cu_mp.σm[ix]
        cu_mp.sij[ix, 3] = cu_mp.σij[ix, 3]-cu_mp.σm[ix]
        cu_mp.sij[ix, 4] = cu_mp.σij[ix, 4]
        cu_mp.sij[ix, 5] = cu_mp.σij[ix, 5]
        cu_mp.sij[ix, 6] = cu_mp.σij[ix, 6]
    end
    return nothing
end

function procedure!(args   ::ARGS, 
                    cu_grid::GPUGRID, 
                    cu_mp  ::GPUPARTICLE, 
                    cu_bc  ::GPUVBC,
                    ΔT     ::T2,
                    Ti     ::T2,
                    OccAPI ::NamedTuple, 
                           ::Val{:TS}) where {T2}
    Ti<args.Te ? (G=args.gravity/args.Te*Ti) : (G=args.gravity)
    @cuda threads=OccAPI.k01_t blocks=OccAPI.k01_b kernel_TS01!(cu_grid)
    @cuda threads=OccAPI.k02_t blocks=OccAPI.k02_b kernel_TS02!(cu_grid, cu_mp, 
                                                                Val(args.basis))
    @cuda threads=OccAPI.k03_t blocks=OccAPI.k03_b kernel_TS03!(cu_grid, cu_mp, G)
    @cuda threads=OccAPI.k04_t blocks=OccAPI.k04_b kernel_TS04!(cu_grid, cu_bc, ΔT, args.ζ)
    @cuda threads=OccAPI.k05_t blocks=OccAPI.k05_b kernel_TS05!(cu_grid, cu_mp, ΔT, 
                                                                args.FLIP, args.PIC)
    @cuda threads=OccAPI.k06_t blocks=OccAPI.k06_b kernel_TS06!(cu_grid, cu_mp)
    @cuda threads=OccAPI.k07_t blocks=OccAPI.k07_b kernel_TS07!(cu_grid, cu_bc, ΔT)
    @cuda threads=OccAPI.k08_t blocks=OccAPI.k08_b kernel_TS08!(cu_grid, cu_mp)
    constitutive!(args, cu_mp, Ti, OccAPI)
    if args.vollock==true
        @cuda threads=OccAPI.k09_t blocks=OccAPI.k09_b kernel_TS09!(cu_grid, cu_mp)
        @cuda threads=OccAPI.k10_t blocks=OccAPI.k10_b kernel_TS10!(cu_grid, cu_mp)
    end
    return nothing
end