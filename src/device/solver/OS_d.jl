#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : OS_d.jl (two-phase single-point)                                           |
|  Description: Kernel functions for the computing in MPM cycle on GPU.                    |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 01. kernel_OS01!() [2D]                                                    |
|               02. kernel_OS01!() [3D]                                                    |
|               03. kernel_OS02!() [2D, linear basis]                                      |
|               04. kernel_OS02!() [3D, linear basis]                                      |
|               05. kernel_OS02!() [2D, uGIMP basis]                                       |
|               06. kernel_OS02!() [3D, uGIMP basis]                                       |
|               07. kernel_OS03!() [2D]                                                    |
|               08. kernel_OS03!() [3D]                                                    |
|               09. kernel_OS04!() [2D]                                                    |    
|               10. kernel_OS04!() [3D]                                                    |
|               11. kernel_OS05!() [2D]                                                    |
|               12. kernel_OS05!() [3D]                                                    |
|               13. kernel_OS06!() [2D]                                                    |
|               14. kernel_OS06!() [3D]                                                    |
|               15. kernel_OS07!() [2D]                                                    |
|               16. kernel_OS07!() [3D]                                                    |
|               17. kernel_OS08!() [2D]                                                    |
|               18. kernel_OS08!() [3D]                                                    |
|               19. kernel_OS09!() [2D]                                                    |
|               20. kernel_OS09!() [3D]                                                    |
|               21. kernel_OS10!() [2D]                                                    |
|               22. kernel_OS10!() [3D]                                                    |
|               23. procedure!()   [2D & 3D]                                               |
+==========================================================================================#

"""
    kernel_OS01!(cu_grid::KernelGrid2D{T1, T2}) where {T1, T2}

Description:
---
Reset some variables for the grid.

I/0 accesses:
---
- read  → 0
- write → 5*grid.node_num + 2*grid.cell_num
- total → 5*grid.node_num + 2*grid.cell_num
"""
function kernel_OS01!(cu_grid::KernelGrid2D{T1, T2}) where {T1, T2}
    FNUM_0 = T2(0.0)
    # GPU global threads index
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    # kernel function
    if ix≤cu_grid.node_num
        if ix≤cu_grid.cell_num
            cu_grid.σm[ ix] = FNUM_0
            cu_grid.vol[ix] = FNUM_0
        end
        cu_grid.Ms[ix   ] = FNUM_0
        cu_grid.Ps[ix, 1] = FNUM_0
        cu_grid.Ps[ix, 2] = FNUM_0
        cu_grid.Fs[ix, 1] = FNUM_0
        cu_grid.Fs[ix, 2] = FNUM_0
    end
    return nothing
end

"""
    kernel_OS01!(cu_grid::KernelGrid3D{T1, T2}) where {T1, T2}

Description:
---
Reset some variables for the grid.

I/0 accesses:
---
- read  → 0
- write → 7*grid.node_num + 2*grid.cell_num
- total → 7*grid.node_num + 2*grid.cell_num
"""
function kernel_OS01!(cu_grid::KernelGrid3D{T1, T2}) where {T1, T2}
    FNUM_0 = T2(0.0)
    # GPU global threads index
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    # kernel function
    if ix≤cu_grid.node_num
        if ix≤cu_grid.cell_num
            cu_grid.σm[ ix] = FNUM_0
            cu_grid.vol[ix] = FNUM_0
        end
        cu_grid.Ms[ix   ] = FNUM_0
        cu_grid.Ps[ix, 1] = FNUM_0
        cu_grid.Ps[ix, 2] = FNUM_0
        cu_grid.Ps[ix, 3] = FNUM_0
        cu_grid.Fs[ix, 1] = FNUM_0
        cu_grid.Fs[ix, 2] = FNUM_0
        cu_grid.Fs[ix, 3] = FNUM_0
    end
    return nothing
end

function kernel_OS02!(cu_grid::    KernelGrid2D{T1, T2},
                      cu_mp  ::KernelParticle2D{T1, T2},
                             ::Val{:linear}) where {T1, T2}
    FNUM_1 = T2(1.0); INUM_1 = T1(1)
    # GPU global threads index
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    # kernel function
    if ix≤cu_mp.num
        # update momentum
        cu_mp.Ms[ix   ] = cu_mp.vol[ix]*cu_mp.ρs[ix   ]
        cu_mp.Ps[ix, 1] = cu_mp.Ms[ ix]*cu_mp.Vs[ix, 1]
        cu_mp.Ps[ix, 2] = cu_mp.Ms[ ix]*cu_mp.Vs[ix, 2]
        # compute particle to cell and particle to node index
        cu_mp.p2c[ix] = cld(cu_mp.pos[ix, 2]-cu_grid.range_y1, cu_grid.space_y)+
                        fld(cu_mp.pos[ix, 1]-cu_grid.range_x1, cu_grid.space_x)*
                        cu_grid.cell_num_y
        x_denom_1 = FNUM_1/cu_grid.space_x # x component
        y_denom_1 = FNUM_1/cu_grid.space_y # y component
        for iy in INUM_1:cu_mp.NIC
            cu_mp.p2n[ix, iy] = cu_grid.c2n[cu_mp.p2c[ix], iy]
            # compute distance between particle and related nodes
            Δdx = cu_mp.pos[ix, 1]-cu_grid.pos[cu_mp.p2n[ix, iy], 1]
            Δdy = cu_mp.pos[ix, 2]-cu_grid.pos[cu_mp.p2n[ix, iy], 2]
            # compute basis function
            ## x component
            c1  = abs(Δdx)≤cu_grid.space_x
            Ni  = FNUM_1-abs(Δdx)*x_denom_1
            dN  = -sign(Δdx)*x_denom_1
            Nx  = c1*Ni
            dNx = c1*dN
            ## y component
            c1  = abs(Δdy)≤cu_grid.space_y
            Ni  = FNUM_1-abs(Δdy)*y_denom_1
            dN  = -sign(Δdy)*y_denom_1
            Ny  = c1*Ni
            dNy = c1*dN
            # compute basis function
            cu_mp.Ni[ ix, iy] =  Nx*Ny # shape function
            cu_mp.∂Nx[ix, iy] = dNx*Ny # x-gradient shape function
            cu_mp.∂Ny[ix, iy] = dNy*Nx # y-gradient shape function
        end
    end
    return nothing
end

function kernel_OS02!(cu_grid::    KernelGrid3D{T1, T2},
                      cu_mp  ::KernelParticle3D{T1, T2},
                             ::Val{:linear}) where {T1, T2}
    FNUM_1 = T2(1.0); INUM_1 = T1(1)
    # GPU global threads index
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    # kernel function
    if ix≤cu_mp.num
        # update momentum
        cu_mp.Ms[ix] = cu_mp.vol[ix]*cu_mp.ρs[ix]
        cu_mp.Ps[ix, 1] = cu_mp.Ms[ix]*cu_mp.Vs[ix, 1]
        cu_mp.Ps[ix, 2] = cu_mp.Ms[ix]*cu_mp.Vs[ix, 2]
        cu_mp.Ps[ix, 3] = cu_mp.Ms[ix]*cu_mp.Vs[ix, 3]
        # compute particle to cell and particle to node index
        numxy = cu_grid.cell_num_y*cu_grid.cell_num_x
        num_y = cu_grid.cell_num_y
        cu_mp.p2c[ix] = cld(cu_mp.pos[ix, 2]-cu_grid.range_y1, cu_grid.space_y)+
                        fld(cu_mp.pos[ix, 3]-cu_grid.range_z1, cu_grid.space_z)*numxy+
                        fld(cu_mp.pos[ix, 1]-cu_grid.range_x1, cu_grid.space_x)*num_y
        x_denom_1 = FNUM_1/cu_grid.space_x # x component
        y_denom_1 = FNUM_1/cu_grid.space_y # y component
        z_denom_1 = FNUM_1/cu_grid.space_z # z component
        for iy in INUM_1:cu_mp.NIC
            cu_mp.p2n[ix, iy] = cu_grid.c2n[cu_mp.p2c[ix], iy]
            # compute distance between particle and related nodes
            Δdx = cu_mp.pos[ix, 1]-cu_grid.pos[cu_mp.p2n[ix, iy], 1]
            Δdy = cu_mp.pos[ix, 2]-cu_grid.pos[cu_mp.p2n[ix, iy], 2]
            Δdz = cu_mp.pos[ix, 3]-cu_grid.pos[cu_mp.p2n[ix, iy], 3]
            # compute basis function
            ## x component
            c1  = abs(Δdx)≤cu_grid.space_x
            Ni  = FNUM_1-abs(Δdx)*x_denom_1
            dN  = -sign(Δdx)*x_denom_1
            Nx  = c1*Ni
            dNx = c1*dN
            ## y component
            c1  = abs(Δdy)≤cu_grid.space_y
            Ni  = FNUM_1-abs(Δdy)*y_denom_1
            dN  = -sign(Δdy)*y_denom_1
            Ny  = c1*Ni
            dNy = c1*dN
            ## z component
            c1  = abs(Δdz)≤cu_grid.space_z
            Ni  = FNUM_1-abs(Δdz)*z_denom_1
            dN  = -sign(Δdz)*z_denom_1
            Nz  = c1*Ni
            dNz = c1*dN
            ## compute basis function
            cu_mp.Ni[ ix, iy] =  Nx*Ny*Nz
            cu_mp.∂Nx[ix, iy] = dNx*Ny*Nz # x-gradient shape function
            cu_mp.∂Ny[ix, iy] = dNy*Nx*Nz # y-gradient shape function
            cu_mp.∂Nz[ix, iy] = dNz*Nx*Ny # z-gradient shape function
        end
    end
    return nothing
end

"""
    kernel_OS02!(cu_grid::KernelGrid2D{T1, T2}, cu_mp::KernelParticle2D{T1, T2},
       ::Val{:uGIMP}) where {T1, T2}

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
function kernel_OS02!(cu_grid::    KernelGrid2D{T1, T2},
                      cu_mp  ::KernelParticle2D{T1, T2},
                             ::Val{:uGIMP}) where {T1, T2}
    INUM_2 = T1(2); FNUM_h = T2(0.5); FNUM_8 = T2(8.0)
    INUM_1 = T1(1); FNUM_1 = T2(1.0); FNUM_4 = T2(4.0)
    # GPU global threads index
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    # kernel function
    if ix≤cu_mp.num
        # update momentum
        cu_mp.Ms[ix   ] = cu_mp.vol[ix]*cu_mp.ρs[ix   ]
        cu_mp.Ps[ix, 1] = cu_mp.Ms[ ix]*cu_mp.Vs[ix, 1]
        cu_mp.Ps[ix, 2] = cu_mp.Ms[ ix]*cu_mp.Vs[ix, 2]
        # compute particle to cell and particle to node index
        cu_mp.p2c[ix] = cld(cu_mp.pos[ix, 2]-cu_grid.range_y1, cu_grid.space_y)+
                        fld(cu_mp.pos[ix, 1]-cu_grid.range_x1, cu_grid.space_x)*
                        cu_grid.cell_num_y
        # x component
        x_numer_1 = FNUM_h*                        cu_mp.space_x
        x_denom_1 = FNUM_1/(FNUM_4*cu_grid.space_x*cu_mp.space_x)
        x_denom_2 = FNUM_1/        cu_grid.space_x
        x_denom_3 = FNUM_1/(       cu_grid.space_x*cu_mp.space_x)
        # y component
        y_numer_1 = FNUM_h*                        cu_mp.space_y
        y_denom_1 = FNUM_1/(FNUM_4*cu_grid.space_y*cu_mp.space_y)
        y_denom_2 = FNUM_1/        cu_grid.space_y
        y_denom_3 = FNUM_1/(       cu_grid.space_y*cu_mp.space_y)
        for iy in INUM_1:cu_mp.NIC
            cu_mp.p2n[ix, iy] = cu_grid.c2n[cu_mp.p2c[ix], iy]
            # compute distance between particle and related nodes
            p2n = cu_mp.p2n[ix, iy]
            Δdx = cu_mp.pos[ix, 1]-cu_grid.pos[p2n, 1]
            Δdy = cu_mp.pos[ix, 2]-cu_grid.pos[p2n, 2]
            # compute basis function
            ## x component
            c1  = abs(Δdx)<x_numer_1
            c2  = x_numer_1≤abs(Δdx)<(cu_grid.space_x-x_numer_1)
            c3  = (cu_grid.space_x-x_numer_1)≤abs(Δdx)<(cu_grid.space_x+x_numer_1)
            Ni1 = FNUM_1-((FNUM_4*(Δdx^INUM_2)+cu_mp.space_x^INUM_2)*x_denom_1)
            Ni2 = -x_denom_2* abs(Δdx)+FNUM_1  
            Ni3 = ((cu_grid.space_x+x_numer_1-abs(Δdx))^INUM_2)*FNUM_h*x_denom_3
            dN1 = -x_denom_1*(FNUM_8*Δdx)
            dN2 = -x_denom_2*sign(Δdx)
            dN3 = -sign(Δdx)*(cu_grid.space_x+x_numer_1-abs(Δdx))*x_denom_3
            Nx  = c1*Ni1+c2*Ni2+c3*Ni3
            dNx = c1*dN1+c2*dN2+c3*dN3
            ## y component
            c1  = abs(Δdy)<y_numer_1
            c2  = y_numer_1≤abs(Δdy)<(cu_grid.space_y-y_numer_1)
            c3  = (cu_grid.space_y-y_numer_1)≤abs(Δdy)<(cu_grid.space_y+y_numer_1)
            Ni1 = FNUM_1-((FNUM_4*(Δdy^INUM_2)+cu_mp.space_y^INUM_2)*y_denom_1)
            Ni2 = -y_denom_2* abs(Δdy)+FNUM_1  
            Ni3 = ((cu_grid.space_y+y_numer_1-abs(Δdy))^INUM_2)*FNUM_h*y_denom_3
            dN1 = -y_denom_1*(FNUM_8*Δdy)
            dN2 = -y_denom_2*sign(Δdy)
            dN3 = -sign(Δdy)*((cu_grid.space_y+y_numer_1-abs(Δdy))*y_denom_3)
            Ny  = c1*Ni1+c2*Ni2+c3*Ni3
            dNy = c1*dN1+c2*dN2+c3*dN3
            # compute basis function
            cu_mp.Ni[ ix, iy] =  Nx*Ny
            cu_mp.∂Nx[ix, iy] = dNx*Ny # x-gradient shape function
            cu_mp.∂Ny[ix, iy] = dNy*Nx # y-gradient shape function
        end
    end
    return nothing
end

"""
    kernel_OS02!(cu_grid::KernelGrid3D{T1, T2}, cu_mp::KernelParticle3D{T1, T2},
       ::Val{:uGIMP}) where {T1, T2}

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
function kernel_OS02!(cu_grid::    KernelGrid3D{T1, T2},
                      cu_mp  ::KernelParticle3D{T1, T2},
                             ::Val{:uGIMP}) where {T1, T2}
    INUM_2 = T1(2); FNUM_h = T2(0.5); FNUM_8 = T2(8.0)
    INUM_1 = T1(1); FNUM_1 = T2(1.0); FNUM_4 = T2(4.0)
    # GPU global threads index
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    # kernel function
    if ix≤cu_mp.num
        # update particle mass
        cu_mp.Ms[ix] = cu_mp.vol[ix]*cu_mp.ρs[ix]
        # get temp variables
        mp_Ms    = cu_mp.Ms[ix]
        mp_pos_1 = cu_mp.pos[ix, 1]
        mp_pos_2 = cu_mp.pos[ix, 2]
        mp_pos_3 = cu_mp.pos[ix, 3]
        # update particle momentum
        cu_mp.Ps[ix, 1] = mp_Ms*cu_mp.Vs[ix, 1]
        cu_mp.Ps[ix, 2] = mp_Ms*cu_mp.Vs[ix, 2]
        cu_mp.Ps[ix, 3] = mp_Ms*cu_mp.Vs[ix, 3]
        # compute particle to cell and particle to node index
        numxy = cu_grid.cell_num_y*cu_grid.cell_num_x
        num_y = cu_grid.cell_num_y
        cu_mp.p2c[ix] = cld(mp_pos_2-cu_grid.range_y1, cu_grid.space_y)+
                        fld(mp_pos_3-cu_grid.range_z1, cu_grid.space_z)*numxy+
                        fld(mp_pos_1-cu_grid.range_x1, cu_grid.space_x)*num_y
        # x component
        x_numer_1 = FNUM_h*                        cu_mp.space_x
        x_denom_1 = FNUM_1/(FNUM_4*cu_grid.space_x*cu_mp.space_x)
        x_denom_2 = FNUM_1/        cu_grid.space_x
        x_denom_3 = FNUM_1/(       cu_grid.space_x*cu_mp.space_x)
        # y component
        y_numer_1 = FNUM_h*                        cu_mp.space_y
        y_denom_1 = FNUM_1/(FNUM_4*cu_grid.space_y*cu_mp.space_y)
        y_denom_2 = FNUM_1/        cu_grid.space_y
        y_denom_3 = FNUM_1/(       cu_grid.space_y*cu_mp.space_y)
        # z component
        z_numer_1 = FNUM_h*                        cu_mp.space_z
        z_denom_1 = FNUM_1/(FNUM_4*cu_grid.space_z*cu_mp.space_z)
        z_denom_2 = FNUM_1/        cu_grid.space_z
        z_denom_3 = FNUM_1/(       cu_grid.space_z*cu_mp.space_z)
        for iy in INUM_1:cu_mp.NIC
            cu_mp.p2n[ix, iy] = cu_grid.c2n[cu_mp.p2c[ix], iy]
            # compute distance between particle and related nodes
            p2n = cu_mp.p2n[ix, iy]
            Δdx = mp_pos_1-cu_grid.pos[p2n, 1]
            Δdy = mp_pos_2-cu_grid.pos[p2n, 2]
            Δdz = mp_pos_3-cu_grid.pos[p2n, 3]
            # compute basis function
            ## x component
            c1  = abs(Δdx)<x_numer_1
            c2  = x_numer_1≤abs(Δdx)<(cu_grid.space_x-x_numer_1)
            c3  = (cu_grid.space_x-x_numer_1)≤abs(Δdx)<(cu_grid.space_x+x_numer_1)
            Ni1 = FNUM_1-((FNUM_4*(Δdx^INUM_2)+cu_mp.space_x^INUM_2)*x_denom_1)
            Ni2 = -x_denom_2* abs(Δdx)+FNUM_1
            Ni3 = ((cu_grid.space_x+x_numer_1-abs(Δdx))^INUM_2)*FNUM_h*x_denom_3
            dN1 = -x_denom_1*FNUM_8*Δdx
            dN2 = -x_denom_2*sign(Δdx)
            dN3 = -sign(Δdx)*(cu_grid.space_x+x_numer_1-abs(Δdx))*x_denom_3
            Nx  = c1*Ni1+c2*Ni2+c3*Ni3
            dNx = c1*dN1+c2*dN2+c3*dN3
            ## y component
            c1  = abs(Δdy)<y_numer_1
            c2  = y_numer_1≤abs(Δdy)<(cu_grid.space_y-y_numer_1)
            c3  = (cu_grid.space_y-y_numer_1)≤abs(Δdy)<(cu_grid.space_y+y_numer_1)
            Ni1 = FNUM_1-((FNUM_4*(Δdy^INUM_2)+cu_mp.space_y^INUM_2)*y_denom_1)
            Ni2 = -y_denom_2* abs(Δdy)+FNUM_1
            Ni3 = ((cu_grid.space_y+y_numer_1-abs(Δdy))^INUM_2)*FNUM_h*y_denom_3
            dN1 = -y_denom_1*FNUM_8*Δdy
            dN2 = -y_denom_2*sign(Δdy)
            dN3 = -sign(Δdy)*(cu_grid.space_y+y_numer_1-abs(Δdy))*y_denom_3
            Ny  = c1*Ni1+c2*Ni2+c3*Ni3
            dNy = c1*dN1+c2*dN2+c3*dN3
            ## z component
            c1  = abs(Δdz)<z_numer_1
            c2  = z_numer_1≤abs(Δdz)<(cu_grid.space_z-z_numer_1)
            c3  = (cu_grid.space_z-z_numer_1)≤abs(Δdz)<(cu_grid.space_z+z_numer_1)
            Ni1 = FNUM_1-((FNUM_4*(Δdz^INUM_2)+cu_mp.space_z^INUM_2)*z_denom_1)
            Ni2 = -z_denom_2* abs(Δdz)+FNUM_1
            Ni3 = ((cu_grid.space_z+z_numer_1-abs(Δdz))^INUM_2)*FNUM_h*z_denom_3
            dN1 = -z_denom_1*FNUM_8*Δdz
            dN2 = -z_denom_2*sign(Δdz)
            dN3 = -sign(Δdz)*(cu_grid.space_z+z_numer_1-abs(Δdz))*z_denom_3
            Nz  = c1*Ni1+c2*Ni2+c3*Ni3
            dNz = c1*dN1+c2*dN2+c3*dN3
            ## compute basis function 
            cu_mp.Ni[ ix, iy] =  Nx*Ny*Nz
            cu_mp.∂Nx[ix, iy] = dNx*Ny*Nz # x-gradient basis function
            cu_mp.∂Ny[ix, iy] = dNy*Nx*Nz # y-gradient basis function
            cu_mp.∂Nz[ix, iy] = dNz*Nx*Ny # z-gradient basis function
        end
    end
    return nothing
end

"""
    kernel_OS03!(cu_grid::KernelGrid2D{T1, T2}, cu_mp::KernelParticle2D{T1, T2}, 
        gravity::T2) where {T1, T2}

Description:
---
P2G procedure for scattering the mass, momentum, and forces from particles to grid.

I/0 accesses:
---
- read  → mp.num*7 + mp.num*3*mp.NIC
- write →            mp.num*5*mp.NIC
- total → mp.num*7 + mp.num*8*mp.NIC
"""
function kernel_OS03!(cu_grid::    KernelGrid2D{T1, T2},
                      cu_mp  ::KernelParticle2D{T1, T2},
                      gravity::T2) where {T1, T2}
    # GPU global threads index
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    # kernel function
    if (ix≤cu_mp.num)
        for iy in Int32(1):Int32(cu_mp.NIC)
            Ni  = cu_mp.Ni[ ix, iy]
            ∂Nx = cu_mp.∂Nx[ix, iy]
            ∂Ny = cu_mp.∂Ny[ix, iy]
            p2n = cu_mp.p2n[ix, iy]
            vol = cu_mp.vol[ix]
            NiM = Ni*cu_mp.Ms[ix]
            # compute nodal mass
            CUDA.@atomic cu_grid.Ms[p2n] = +(cu_grid.Ms[p2n], NiM)
            # compute nodal momentum
            CUDA.@atomic cu_grid.Ps[p2n, 1] = +(cu_grid.Ps[p2n, 1], Ni*cu_mp.Ps[ix, 1])
            CUDA.@atomic cu_grid.Ps[p2n, 2] = +(cu_grid.Ps[p2n, 2], Ni*cu_mp.Ps[ix, 2])
            # compute nodal total force for solid
            CUDA.@atomic cu_grid.Fs[p2n, 1] = +(cu_grid.Fs[p2n, 1],
                -vol*(∂Nx*cu_mp.σij[ix, 1]+∂Ny*cu_mp.σij[ix, 4]))
            CUDA.@atomic cu_grid.Fs[p2n, 2] = +(cu_grid.Fs[p2n, 2],
                -vol*(∂Ny*cu_mp.σij[ix, 2]+∂Nx*cu_mp.σij[ix, 4])+NiM*gravity)
        end
    end
    return nothing
end

"""
    kernel_OS03!(cu_grid::KernelGrid3D{T1, T2}, cu_mp::KernelParticle3D{T1, T2}, 
        gravity::T2) where {T1, T2}

Description:
---
P2G procedure for scattering the mass, momentum, and forces from particles to grid.

I/0 accesses:
---
- read  → mp.num*11+mp.num* 4*mp.NIC
- write →           mp.num* 7*mp.NIC
- total → mp.num*11+mp.num*11*mp.NIC
"""
function kernel_OS03!(cu_grid::    KernelGrid3D{T1, T2},
                      cu_mp  ::KernelParticle3D{T1, T2},
                      gravity::T2) where {T1, T2}
    # GPU global threads index
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    # kernel function
    if ix≤cu_mp.num
        for iy in Int32(1):Int32(cu_mp.NIC)
            Ni  = cu_mp.Ni[ ix, iy]
            ∂Nx = cu_mp.∂Nx[ix, iy]
            ∂Ny = cu_mp.∂Ny[ix, iy]
            ∂Nz = cu_mp.∂Nz[ix, iy]
            p2n = cu_mp.p2n[ix, iy]
            vol = cu_mp.vol[ix]
            NiM = Ni*cu_mp.Ms[ix]
            # compute nodal mass
            CUDA.@atomic cu_grid.Ms[p2n] = +(cu_grid.Ms[p2n], NiM)
            # compute nodal momentum
            CUDA.@atomic cu_grid.Ps[p2n, 1] = +(cu_grid.Ps[p2n, 1], Ni*cu_mp.Ps[ix, 1])
            CUDA.@atomic cu_grid.Ps[p2n, 2] = +(cu_grid.Ps[p2n, 2], Ni*cu_mp.Ps[ix, 2])
            CUDA.@atomic cu_grid.Ps[p2n, 3] = +(cu_grid.Ps[p2n, 3], Ni*cu_mp.Ps[ix, 3])
            # compute nodal total force for solid
            CUDA.@atomic cu_grid.Fs[p2n, 1] = +(cu_grid.Fs[p2n, 1],
                -vol*(∂Nx*cu_mp.σij[ix, 1]+∂Ny*cu_mp.σij[ix, 4]+∂Nz*cu_mp.σij[ix, 6]))
            CUDA.@atomic cu_grid.Fs[p2n, 2] = +(cu_grid.Fs[p2n, 2],
                -vol*(∂Ny*cu_mp.σij[ix, 2]+∂Nx*cu_mp.σij[ix, 4]+∂Nz*cu_mp.σij[ix, 5]))
            CUDA.@atomic cu_grid.Fs[p2n, 3] = +(cu_grid.Fs[p2n, 3],
                -vol*(∂Nz*cu_mp.σij[ix, 3]+∂Nx*cu_mp.σij[ix, 6]+∂Ny*cu_mp.σij[ix, 5])+
                NiM*gravity)
        end
    end
    return nothing
end


"""
    kernel_OS04!(cu_grid::KernelGrid2D{T1, T2}, cu_bc::KernelVBoundary2D{T1, T2}, ΔT::T2,
        ζ::T2) where {T1, T2}

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
function kernel_OS04!(cu_grid::     KernelGrid2D{T1, T2},
                      cu_bc  ::KernelVBoundary2D{T1, T2},
                      ΔT     ::T2,
                      ζ      ::T2) where {T1, T2}
    INUM_0 = T1(0  ); INUM_2 = T1(2)
    FNUM_0 = T2(0.0); FNUM_1 = T2(1.0)
    # GPU global threads index
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    # kernel function
    if ix≤cu_grid.node_num
        iszero(cu_grid.Ms[ix]) ? Ms_denom = FNUM_0 :  Ms_denom = FNUM_1/cu_grid.Ms[ix]
        # compute nodal velocity
        cu_grid.Vs[ix, 1] = cu_grid.Ps[ix, 1]*Ms_denom
        cu_grid.Vs[ix, 2] = cu_grid.Ps[ix, 2]*Ms_denom
        # damping force for solid
        dampvs = -ζ*sqrt(cu_grid.Fs[ix, 1]^INUM_2+cu_grid.Fs[ix, 2]^INUM_2)
        # compute nodal total force for mixture
        Fs_x = cu_grid.Fs[ix, 1]+dampvs*sign(cu_grid.Vs[ix, 1])
        Fs_y = cu_grid.Fs[ix, 2]+dampvs*sign(cu_grid.Vs[ix, 2])
        # update nodal velocity
        cu_grid.Vs_T[ix, 1] = (cu_grid.Ps[ix, 1]+Fs_x*ΔT)*Ms_denom
        cu_grid.Vs_T[ix, 2] = (cu_grid.Ps[ix, 2]+Fs_y*ΔT)*Ms_denom
        # boundary condition
        cu_bc.Vx_s_Idx[ix]≠INUM_0 ? cu_grid.Vs_T[ix, 1]=cu_bc.Vx_s_Val[ix] : nothing
        cu_bc.Vy_s_Idx[ix]≠INUM_0 ? cu_grid.Vs_T[ix, 2]=cu_bc.Vy_s_Val[ix] : nothing
        # reset grid momentum
        cu_grid.Ps[ix, 1] = FNUM_0
        cu_grid.Ps[ix, 2] = FNUM_0
    end
    return nothing
end

"""
    kernel_OS04!(cu_grid::KernelGrid3D{T1, T2}, cu_bc::KernelVBoundary3D{T1, T2}, ΔT::T2,
        ζ::T2) where {T1, T2}

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
function kernel_OS04!(cu_grid::     KernelGrid3D{T1, T2},
                      cu_bc  ::KernelVBoundary3D{T1, T2},
                      ΔT     ::T2,
                      ζ      ::T2) where {T1, T2}
    INUM_0 = T1(0); INUM_2 = T1(2); FNUM_0 = T2(0.0); FNUM_1 = T1(1.0)
    # GPU global threads index
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    # kernel function
    if ix≤cu_grid.node_num
        iszero(cu_grid.Ms[ix]) ? Ms_denom=FNUM_0 :  Ms_denom=FNUM_1/cu_grid.Ms[ix]
        # compute nodal velocity
        Ps_1 = cu_grid.Ps[ix, 1]
        Ps_2 = cu_grid.Ps[ix, 2]
        Ps_3 = cu_grid.Ps[ix, 3]
        cu_grid.Vs[ix, 1] = Ps_1*Ms_denom
        cu_grid.Vs[ix, 2] = Ps_2*Ms_denom
        cu_grid.Vs[ix, 3] = Ps_3*Ms_denom
        # damping force for solid
        dampvs = -ζ*sqrt(cu_grid.Fs[ix, 1]^INUM_2+
                         cu_grid.Fs[ix, 2]^INUM_2+
                         cu_grid.Fs[ix, 3]^INUM_2)
        # compute nodal total force for mixture
        Fs_x = cu_grid.Fs[ix, 1]+dampvs*sign(cu_grid.Vs[ix, 1])
        Fs_y = cu_grid.Fs[ix, 2]+dampvs*sign(cu_grid.Vs[ix, 2])
        Fs_z = cu_grid.Fs[ix, 3]+dampvs*sign(cu_grid.Vs[ix, 3])
        # update nodal velocity
        cu_grid.Vs_T[ix, 1] = (Ps_1+Fs_x*ΔT)*Ms_denom
        cu_grid.Vs_T[ix, 2] = (Ps_2+Fs_y*ΔT)*Ms_denom
        cu_grid.Vs_T[ix, 3] = (Ps_3+Fs_z*ΔT)*Ms_denom
        # boundary condition
        cu_bc.Vx_s_Idx[ix]≠INUM_0 ? cu_grid.Vs_T[ix, 1]=cu_bc.Vx_s_Val[ix] : nothing
        cu_bc.Vy_s_Idx[ix]≠INUM_0 ? cu_grid.Vs_T[ix, 2]=cu_bc.Vy_s_Val[ix] : nothing
        cu_bc.Vz_s_Idx[ix]≠INUM_0 ? cu_grid.Vs_T[ix, 3]=cu_bc.Vz_s_Val[ix] : nothing
        # reset grid momentum
        cu_grid.Ps[ix, 1] = FNUM_0
        cu_grid.Ps[ix, 2] = FNUM_0
        cu_grid.Ps[ix, 3] = FNUM_0
    end
    return nothing
end

"""
    kernel_OS05!(cu_grid::KernelGrid2D{T1, T2}, cu_mp::KernelParticle2D{T1, T2}, ΔT::T2,
        FLIP::T2, PIC::T2) where {T1, T2}

Description:
---
1. Solve equations on grid.
2. Compute CFL conditions.

I/0 accesses:
---
- read  → mp.num* 5+mp.num*mp.NIC*5
- write → mp.num* 6
- total → mp.num*11+mp.num*mp.NIC*5
"""
function kernel_OS05!(cu_grid::    KernelGrid2D{T1, T2},
                      cu_mp  ::KernelParticle2D{T1, T2},
                      ΔT     ::T2,
                      FLIP   ::T2,
                      PIC    ::T2) where {T1, T2}
    FNUM_0 = T2(0.0); FNUM_43 = T2(4/3); INUM_1 = T1(1)
    # GPU global threads index
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    # update particle position & velocity
    if ix≤cu_mp.num
        pid = cu_mp.layer[ix]
        Ks  = cu_mp.Ks[pid]
        G   = cu_mp.G[pid]
        tmp_vx_s1 = tmp_vx_s2 = FNUM_0
        tmp_vy_s1 = tmp_vy_s2 = FNUM_0
        tmp_pos_x = tmp_pos_y = FNUM_0
        # update particle position
        for iy in INUM_1:cu_mp.NIC
            Ni  = cu_mp.Ni[ ix, iy]
            p2n = cu_mp.p2n[ix, iy]
            tmp_pos_x += Ni* cu_grid.Vs_T[p2n, 1]
            tmp_pos_y += Ni* cu_grid.Vs_T[p2n, 2]
            tmp_vx_s1 += Ni*(cu_grid.Vs_T[p2n, 1]-cu_grid.Vs[p2n, 1])
            tmp_vx_s2 += Ni* cu_grid.Vs_T[p2n, 1]
            tmp_vy_s1 += Ni*(cu_grid.Vs_T[p2n, 2]-cu_grid.Vs[p2n, 2])
            tmp_vy_s2 += Ni* cu_grid.Vs_T[p2n, 2]
        end
        cu_mp.pos[ix, 1] += ΔT*tmp_pos_x
        cu_mp.pos[ix, 2] += ΔT*tmp_pos_y
        # update particle velocity
        cu_mp.Vs[ix, 1] = FLIP*(cu_mp.Vs[ix, 1]+tmp_vx_s1)+PIC*tmp_vx_s2
        cu_mp.Vs[ix, 2] = FLIP*(cu_mp.Vs[ix, 2]+tmp_vy_s1)+PIC*tmp_vy_s2
        # update particle momentum
        Vs_1 = cu_mp.Vs[ix, 1]
        Vs_2 = cu_mp.Vs[ix, 2]
        Ms   = cu_mp.Ms[ix]
        cu_mp.Ps[ix, 1] = Ms*Vs_1
        cu_mp.Ps[ix, 2] = Ms*Vs_2
        # update CFL conditions
        sqr = sqrt((Ks+G*FNUM_43)/cu_mp.ρs[ix])
        cd_sx = cu_grid.space_x/(sqr+abs(Vs_1))
        cd_sy = cu_grid.space_y/(sqr+abs(Vs_2))
        cu_mp.cfl[ix] = min(cd_sx, cd_sy)
    end
    return nothing
end

"""
    kernel_OS05!(cu_grid::KernelGrid3D{T1, T2}, cu_mp::KernelParticle3D{T1, T2}, ΔT::T2,
        FLIP::T2, PIC::T2) where {T1, T2}

Description:
---
1. Solve equations on grid.
2. Compute CFL conditions.

I/0 accesses:
---
- read  → mp.num* 7+mp.num*mp.NIC*7
- write → mp.num* 9
- total → mp.num*16+mp.num*mp.NIC*7
"""
function kernel_OS05!(cu_grid::    KernelGrid3D{T1, T2},
                      cu_mp  ::KernelParticle3D{T1, T2},
                      ΔT     ::T2,
                      FLIP   ::T2,
                      PIC    ::T2) where {T1, T2}
    FNUM_0 = T2(0.0); FNUM_43 = T2(4/3); INUM_1 = T1(1)
    # GPU global threads index
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    # update particle position & velocity
    if ix≤cu_mp.num
        pid = cu_mp.layer[ix]
        Ks  = cu_mp.Ks[pid]
        G   = cu_mp.G[pid]
        tmp_vx_s1 = tmp_vx_s2 = tmp_vy_s1 = FNUM_0
        tmp_vy_s2 = tmp_vz_s1 = tmp_vz_s2 = FNUM_0
        tmp_pos_x = tmp_pos_y = tmp_pos_z = FNUM_0
        # update particle position
        for iy in INUM_1:cu_mp.NIC
            Ni    = cu_mp.Ni[ ix, iy]
            p2n   = cu_mp.p2n[ix, iy]
            Vs_T1 = cu_grid.Vs_T[p2n, 1]
            Vs_T2 = cu_grid.Vs_T[p2n, 2]
            Vs_T3 = cu_grid.Vs_T[p2n, 3]
            tmp_pos_x += Ni* Vs_T1
            tmp_pos_y += Ni* Vs_T2
            tmp_pos_z += Ni* Vs_T3
            tmp_vx_s1 += Ni*(Vs_T1-cu_grid.Vs[p2n, 1])
            tmp_vx_s2 += Ni* Vs_T1
            tmp_vy_s1 += Ni*(Vs_T2-cu_grid.Vs[p2n, 2])
            tmp_vy_s2 += Ni* Vs_T2
            tmp_vz_s1 += Ni*(Vs_T3-cu_grid.Vs[p2n, 3])
            tmp_vz_s2 += Ni* Vs_T3
        end
        # update particle position
        cu_mp.pos[ix, 1] += ΔT*tmp_pos_x
        cu_mp.pos[ix, 2] += ΔT*tmp_pos_y
        cu_mp.pos[ix, 3] += ΔT*tmp_pos_z
        # update particle velocity
        cu_mp.Vs[ix, 1] = FLIP*(cu_mp.Vs[ix, 1]+tmp_vx_s1)+PIC*tmp_vx_s2
        cu_mp.Vs[ix, 2] = FLIP*(cu_mp.Vs[ix, 2]+tmp_vy_s1)+PIC*tmp_vy_s2
        cu_mp.Vs[ix, 3] = FLIP*(cu_mp.Vs[ix, 3]+tmp_vz_s1)+PIC*tmp_vz_s2
        # update particle momentum
        Vs_1 = cu_mp.Vs[ix, 1]
        Vs_2 = cu_mp.Vs[ix, 2]
        Vs_3 = cu_mp.Vs[ix, 3]
        Ms   = cu_mp.Ms[ix]
        cu_mp.Ps[ix, 1] = Ms*Vs_1
        cu_mp.Ps[ix, 2] = Ms*Vs_2
        cu_mp.Ps[ix, 3] = Ms*Vs_3
        # update CFL conditions
        sqr = sqrt((Ks+G*FNUM_43)/cu_mp.ρs[ix])
        cd_sx = cu_grid.space_x/(sqr+abs(Vs_1))
        cd_sy = cu_grid.space_y/(sqr+abs(Vs_2))
        cd_sz = cu_grid.space_z/(sqr+abs(Vs_3))
        cu_mp.cfl[ix] = min(cd_sx, cd_sy, cd_sz)
    end
    return nothing
end

"""
    kernel_OS06!(cu_grid::KernelGrid2D{T1, T2},cu_mp::KernelParticle2D{T1, T2}) where {T1, T2}

Description:
---
Scatter momentum from particles to grid.

I/0 accesses:
---
- read  → mp.num*2 + mp.num*1*mp.NIC
- write →            mp.num*2*mp.NIC
- total → mp.num*2 + mp.num*3*mp.NIC
"""
function kernel_OS06!(cu_grid::    KernelGrid2D{T1, T2},
                      cu_mp  ::KernelParticle2D{T1, T2}) where {T1, T2}
    # GPU global threads index
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    # update particle position & velocity
    if ix≤cu_mp.num
        for iy in Int32(1):Int32(cu_mp.NIC)
            Ni  = cu_mp.Ni[ ix, iy]
            p2n = cu_mp.p2n[ix, iy]
            CUDA.@atomic cu_grid.Ps[p2n, 1] = +(cu_grid.Ps[p2n, 1], cu_mp.Ps[ix, 1]*Ni)
            CUDA.@atomic cu_grid.Ps[p2n, 2] = +(cu_grid.Ps[p2n, 2], cu_mp.Ps[ix, 2]*Ni)
        end
    end
    return nothing
end

"""
    kernel_OS06!(cu_grid::KernelGrid3D{T1, T2},cu_mp::KernelParticle3D{T1, T2}) where {T1, T2}

Description:
---
Scatter momentum from particles to grid.

I/0 accesses:
---
- read  → mp.num*3 + mp.num*1*mp.NIC
- write →            mp.num*3*mp.NIC
- total → mp.num*3 + mp.num*4*mp.NIC
"""
function kernel_OS06!(cu_grid::    KernelGrid3D{T1, T2},
                      cu_mp  ::KernelParticle3D{T1, T2}) where {T1, T2}
    # GPU global threads index
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    # update particle position & velocity
    if ix≤cu_mp.num
        for iy in Int32(1):Int32(cu_mp.NIC)
            Ni  = cu_mp.Ni[ ix, iy]
            p2n = cu_mp.p2n[ix, iy]
            CUDA.@atomic cu_grid.Ps[p2n, 1] = +(cu_grid.Ps[p2n, 1], cu_mp.Ps[ix, 1]*Ni)
            CUDA.@atomic cu_grid.Ps[p2n, 2] = +(cu_grid.Ps[p2n, 2], cu_mp.Ps[ix, 2]*Ni)
            CUDA.@atomic cu_grid.Ps[p2n, 3] = +(cu_grid.Ps[p2n, 3], cu_mp.Ps[ix, 3]*Ni)
        end
    end
    return nothing
end

"""
    kernel_OS07!(cu_grid::KernelGrid2D{T1, T2}, cu_bc::KernelVBoundary2D{T1, T2}, ΔT::T2) where {T1, T2}

Description:
---
Solve equations on grid.

I/0 accesses:
---
- read  → grid.node_num*5
- write → grid.node_num*2  
- total → grid.node_num*7
"""
function kernel_OS07!(cu_grid::     KernelGrid2D{T1, T2},
                      cu_bc  ::KernelVBoundary2D{T1, T2},
                      ΔT     ::T2) where {T1, T2}
    INUM_0 = T1(0); FNUM_0 = T2(0.0); FNUM_1 = T2(1.0)
    # GPU global threads index
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    if ix≤cu_grid.node_num
        iszero(cu_grid.Ms[ix]) ? Ms_denom = FNUM_0 : Ms_denom = FNUM_1/cu_grid.Ms[ix]
        # compute nodal velocities
        cu_grid.Vs[ix, 1] = cu_grid.Ps[ix, 1]*Ms_denom
        cu_grid.Vs[ix, 2] = cu_grid.Ps[ix, 2]*Ms_denom
        # fixed Dirichlet nodes
        cu_bc.Vx_s_Idx[ix]≠INUM_0 ? cu_grid.Vs[ix, 1]=cu_bc.Vx_s_Val[ix] : nothing
        cu_bc.Vy_s_Idx[ix]≠INUM_0 ? cu_grid.Vs[ix, 2]=cu_bc.Vy_s_Val[ix] : nothing
        # compute nodal displacement
        cu_grid.Δd_s[ix, 1] = cu_grid.Vs[ix, 1]*ΔT
        cu_grid.Δd_s[ix, 2] = cu_grid.Vs[ix, 2]*ΔT
    end
    return nothing
end

"""
    kernel_OS07!(cu_grid::KernelGrid3D{T1, T2}, cu_bc::KernelVBoundary3D{T1, T2}, ΔT::T2) where {T1, T2}

Description:
---
Solve equations on grid.

I/0 accesses:
---
- read  → grid.node_num* 7
- write → grid.node_num* 3  
- total → grid.node_num*10
"""
function kernel_OS07!(cu_grid::     KernelGrid3D{T1, T2},
                      cu_bc  ::KernelVBoundary3D{T1, T2},
                      ΔT     ::T2) where {T1, T2}
    INUM_0 = T1(0); FNUM_0 = T2(0.0); FNUM_1 = T2(1.0)
    # GPU global threads index
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    if ix≤cu_grid.node_num
        iszero(cu_grid.Ms[ix]) ? Ms_denom = FNUM_0 : Ms_denom = FNUM_1/cu_grid.Ms[ix]
        # compute nodal velocities
        cu_grid.Vs[ix, 1] = cu_grid.Ps[ix, 1]*Ms_denom
        cu_grid.Vs[ix, 2] = cu_grid.Ps[ix, 2]*Ms_denom
        cu_grid.Vs[ix, 3] = cu_grid.Ps[ix, 3]*Ms_denom
        # fixed Dirichlet nodes
        cu_bc.Vx_s_Idx[ix]≠INUM_0 ? cu_grid.Vs[ix, 1]=cu_bc.Vx_s_Val[ix] : nothing
        cu_bc.Vy_s_Idx[ix]≠INUM_0 ? cu_grid.Vs[ix, 2]=cu_bc.Vy_s_Val[ix] : nothing
        cu_bc.Vz_s_Idx[ix]≠INUM_0 ? cu_grid.Vs[ix, 3]=cu_bc.Vz_s_Val[ix] : nothing
        # compute nodal displacement
        cu_grid.Δd_s[ix, 1] = cu_grid.Vs[ix, 1]*ΔT
        cu_grid.Δd_s[ix, 2] = cu_grid.Vs[ix, 2]*ΔT
        cu_grid.Δd_s[ix, 3] = cu_grid.Vs[ix, 3]*ΔT
    end
    return nothing
end

"""
    kernel_OS08!(cu_grid::KernelGrid2D{T1, T2}, cu_mp::KernelParticle2D{T1, T2}) where {T1, T2}

Description:
---
Update particle information.

I/0 accesses:
---
- read  → mp.num*10 + mp.num*4*mp.NIC
- write → mp.num*16
- total → mp.num*26 + mp.num*4*mp.NIC
"""
function kernel_OS08!(cu_grid::    KernelGrid2D{T1, T2},
                      cu_mp  ::KernelParticle2D{T1, T2}) where {T1, T2}
    INUM_1 = T1(1); FNUM_0 = T2(0.0); FNUM_1 = T2(1.0)
    # GPU global threads index
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    # kernel function
    if ix≤cu_mp.num
        dF1 = dF2 = dF3 = dF4 = FNUM_0
        for iy in INUM_1:cu_mp.NIC
            p2n = cu_mp.p2n[ix, iy]
            ∂Nx = cu_mp.∂Nx[ix, iy]
            ∂Ny = cu_mp.∂Ny[ix, iy]
            # compute solid incremental deformation gradient
            dF1 += cu_grid.Δd_s[p2n, 1]*∂Nx
            dF2 += cu_grid.Δd_s[p2n, 1]*∂Ny
            dF3 += cu_grid.Δd_s[p2n, 2]*∂Nx
            dF4 += cu_grid.Δd_s[p2n, 2]*∂Ny
        end
        cu_mp.∂Fs[ix, 1] = dF1
        cu_mp.∂Fs[ix, 2] = dF2
        cu_mp.∂Fs[ix, 3] = dF3
        cu_mp.∂Fs[ix, 4] = dF4
        # compute strain increment 
        cu_mp.Δϵij_s[ix, 1] = dF1
        cu_mp.Δϵij_s[ix, 2] = dF4
        cu_mp.Δϵij_s[ix, 4] = dF2+dF3
        # update strain tensor
        cu_mp.ϵij_s[ix, 1] += dF1
        cu_mp.ϵij_s[ix, 2] += dF4
        cu_mp.ϵij_s[ix, 4] += dF2+dF3
        # deformation gradient matrix
        F1 = cu_mp.F[ix, 1]; F2 = cu_mp.F[ix, 2]; F3 = cu_mp.F[ix, 3]; F4 = cu_mp.F[ix, 4]      
        cu_mp.F[ix, 1] = (dF1+FNUM_1)*F1+dF2*F3
        cu_mp.F[ix, 2] = (dF1+FNUM_1)*F2+dF2*F4
        cu_mp.F[ix, 3] = (dF4+FNUM_1)*F3+dF3*F1
        cu_mp.F[ix, 4] = (dF4+FNUM_1)*F4+dF3*F2
        # update jacobian value and particle volume
        cu_mp.J[  ix] = cu_mp.F[ix, 1]*cu_mp.F[ix, 4]-cu_mp.F[ix, 2]*cu_mp.F[ix, 3]
        cu_mp.vol[ix] = cu_mp.J[ix]*cu_mp.vol_init[ix]
        cu_mp.ρs[ ix] = cu_mp.ρs_init[ix]/cu_mp.J[ix]
    end
    return nothing
end

"""
    kernel_OS08!(cu_grid::KernelGrid3D{T1, T2}, cu_mp::KernelParticle3D{T1, T2}) where {T1, T2}

Description:
---
Update particle information.

I/0 accesses:
---
- read  → mp.num*20 + mp.num*6*mp.NIC
- write → mp.num*32 
- total → mp.num*52 + mp.num*6*mp.NIC
"""
function kernel_OS08!(cu_grid::    KernelGrid3D{T1, T2},
                      cu_mp  ::KernelParticle3D{T1, T2}) where {T1, T2}
    INUM_1 = T1(1); FNUM_0 = T2(0.0); FNUM_1 = T2(1.0)
    # GPU global threads index
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    # kernel function
    if ix≤cu_mp.num
        dF1 = dF2 = dF3 = dF4 = dF5 = dF6 = dF7 = dF8 = dF9 = FNUM_0
        for iy in INUM_1:cu_mp.NIC
            p2n = cu_mp.p2n[ix, iy]
            ∂Nx = cu_mp.∂Nx[ix, iy]; ds1 = cu_grid.Δd_s[p2n, 1]
            ∂Ny = cu_mp.∂Ny[ix, iy]; ds2 = cu_grid.Δd_s[p2n, 2]
            ∂Nz = cu_mp.∂Nz[ix, iy]; ds3 = cu_grid.Δd_s[p2n, 3]
            # compute solid incremental deformation gradient
            dF1 += ds1*∂Nx; dF2 += ds1*∂Ny; dF3 += ds1*∂Nz
            dF4 += ds2*∂Nx; dF5 += ds2*∂Ny; dF6 += ds2*∂Nz
            dF7 += ds3*∂Nx; dF8 += ds3*∂Ny; dF9 += ds3*∂Nz
        end
        cu_mp.∂Fs[ix, 1] = dF1; cu_mp.∂Fs[ix, 2] = dF2; cu_mp.∂Fs[ix, 3] = dF3
        cu_mp.∂Fs[ix, 4] = dF4; cu_mp.∂Fs[ix, 5] = dF5; cu_mp.∂Fs[ix, 6] = dF6
        cu_mp.∂Fs[ix, 7] = dF7; cu_mp.∂Fs[ix, 8] = dF8; cu_mp.∂Fs[ix, 9] = dF9
        # compute strain increment
        cu_mp.Δϵij_s[ix, 1] = dF1
        cu_mp.Δϵij_s[ix, 2] = dF5
        cu_mp.Δϵij_s[ix, 3] = dF9
        cu_mp.Δϵij_s[ix, 4] = dF2+dF4
        cu_mp.Δϵij_s[ix, 5] = dF6+dF8
        cu_mp.Δϵij_s[ix, 6] = dF3+dF7
        # update strain tensor
        cu_mp.ϵij_s[ix, 1] += dF1
        cu_mp.ϵij_s[ix, 2] += dF5
        cu_mp.ϵij_s[ix, 3] += dF9
        cu_mp.ϵij_s[ix, 4] += dF2+dF4
        cu_mp.ϵij_s[ix, 5] += dF6+dF8
        cu_mp.ϵij_s[ix, 6] += dF3+dF7
        # deformation gradient matrix
        F1 = cu_mp.F[ix, 1]; F2 = cu_mp.F[ix, 2]; F3 = cu_mp.F[ix, 3]
        F4 = cu_mp.F[ix, 4]; F5 = cu_mp.F[ix, 5]; F6 = cu_mp.F[ix, 6]
        F7 = cu_mp.F[ix, 7]; F8 = cu_mp.F[ix, 8]; F9 = cu_mp.F[ix, 9]        
        cu_mp.F[ix, 1] = (dF1+FNUM_1)*F1+dF2*F4+dF3*F7
        cu_mp.F[ix, 2] = (dF1+FNUM_1)*F2+dF2*F5+dF3*F8
        cu_mp.F[ix, 3] = (dF1+FNUM_1)*F3+dF2*F6+dF3*F9
        cu_mp.F[ix, 4] = (dF5+FNUM_1)*F4+dF4*F1+dF6*F7
        cu_mp.F[ix, 5] = (dF5+FNUM_1)*F5+dF4*F2+dF6*F8
        cu_mp.F[ix, 6] = (dF5+FNUM_1)*F6+dF4*F3+dF6*F9
        cu_mp.F[ix, 7] = (dF9+FNUM_1)*F7+dF8*F4+dF7*F1
        cu_mp.F[ix, 8] = (dF9+FNUM_1)*F8+dF8*F5+dF7*F2
        cu_mp.F[ix, 9] = (dF9+FNUM_1)*F9+dF8*F6+dF7*F3
        # update jacobian value and particle volume
        cu_mp.J[ix] = cu_mp.F[ix, 1]*cu_mp.F[ix, 5]*cu_mp.F[ix, 9]+
                      cu_mp.F[ix, 2]*cu_mp.F[ix, 6]*cu_mp.F[ix, 7]+
                      cu_mp.F[ix, 3]*cu_mp.F[ix, 4]*cu_mp.F[ix, 8]-
                      cu_mp.F[ix, 7]*cu_mp.F[ix, 5]*cu_mp.F[ix, 3]-
                      cu_mp.F[ix, 8]*cu_mp.F[ix, 6]*cu_mp.F[ix, 1]-
                      cu_mp.F[ix, 9]*cu_mp.F[ix, 4]*cu_mp.F[ix, 2]
        cu_mp.vol[ix] = cu_mp.J[ix]*cu_mp.vol_init[ix]
        cu_mp.ρs[ ix] = cu_mp.ρs_init[ix]/cu_mp.J[ix]
    end
    return nothing
end

"""
    kernel_OS09!(cu_grid::KernelGrid2D{T1, T2}, cu_mp::KernelParticle2D{T1, T2}) where {T1, T2}

Description:
---
Mapping mean stress and volume from particle to grid.

I/0 accesses:
---
- read  → mp.num*2
- write → mp.num*2
- total → mp.num*4
"""
function kernel_OS09!(cu_grid::    KernelGrid2D{T1, T2},
                      cu_mp  ::KernelParticle2D{T1, T2}) where {T1, T2}
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    if ix≤cu_mp.num
        p2c = cu_mp.p2c[ix]
        vol = cu_mp.vol[ix]
        CUDA.@atomic cu_grid.σm[p2c]  = +(cu_grid.σm[p2c] , vol*cu_mp.σm[ix])
        CUDA.@atomic cu_grid.vol[p2c] = +(cu_grid.vol[p2c], vol)
    end
    return nothing
end

"""
    kernel_OS09!(cu_grid::KernelGrid3D{T1, T2}, cu_mp::KernelParticle3D{T1, T2}) where {T1, T2}

Description:
---
Mapping mean stress and volume from particle to grid.

I/0 accesses:
---
- read  → mp.num*2
- write → mp.num*2
- total → mp.num*4
"""
function kernel_OS09!(cu_grid::    KernelGrid3D{T1, T2},
                      cu_mp  ::KernelParticle3D{T1, T2}) where {T1, T2}
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    if ix≤cu_mp.num
        p2c = cu_mp.p2c[ix]
        vol = cu_mp.vol[ix]
        CUDA.@atomic cu_grid.σm[p2c]  = +(cu_grid.σm[p2c] , vol*cu_mp.σm[ix])
        CUDA.@atomic cu_grid.vol[p2c] = +(cu_grid.vol[p2c], vol)
    end
    return nothing
end

"""
    kernel_OS10!(cu_grid::KernelGrid2D{T1, T2}, cu_mp::KernelParticle2D{T1, T2}) where {T1, T2}

Description:
---
Mapping back mean stress and volume from grid to particle.

I/0 accesses:
---
- read  → mp.num*10        
- write → mp.num* 9
- total → mp.num*19
"""
function kernel_OS10!(cu_grid::    KernelGrid2D{T1, T2},
                      cu_mp  ::KernelParticle2D{T1, T2}) where {T1, T2}
    FNUM_13 = T2(1/3)
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    if ix≤cu_mp.num
        p2c = cu_mp.p2c[ix]
        σm  = cu_grid.σm[p2c]/cu_grid.vol[p2c]
        cu_mp.σij[ix, 1] = cu_mp.sij[ix, 1]+σm
        cu_mp.σij[ix, 2] = cu_mp.sij[ix, 2]+σm
        cu_mp.σij[ix, 3] = cu_mp.sij[ix, 3]+σm
        cu_mp.σij[ix, 4] = cu_mp.sij[ix, 4]
        # update mean stress tensor
        σm = (cu_mp.σij[ix, 1]+cu_mp.σij[ix, 2]+cu_mp.σij[ix, 3])*FNUM_13
        cu_mp.σm[ix] = σm
        # update deviatoric stress tensor
        cu_mp.sij[ix, 1] = cu_mp.σij[ix, 1]-σm
        cu_mp.sij[ix, 2] = cu_mp.σij[ix, 2]-σm
        cu_mp.sij[ix, 3] = cu_mp.σij[ix, 3]-σm
        cu_mp.sij[ix, 4] = cu_mp.σij[ix, 4]
    end
    return nothing
end

"""
    kernel_OS10!(cu_grid::KernelGrid3D{T1, T2}, cu_mp::KernelParticle3D{T1, T2}) where {T1, T2}

Description:
---
Mapping back mean stress and volume from grid to particle.

I/0 accesses:
---
- read  → mp.num*14             
- write → mp.num*13
- total → mp.num*25
"""
function kernel_OS10!(cu_grid::    KernelGrid3D{T1, T2},
                      cu_mp  ::KernelParticle3D{T1, T2}) where {T1, T2}
    FNUM_13 = T2(1/3)
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    if ix≤cu_mp.num
        p2c = cu_mp.p2c[ix]
        σm  = cu_grid.σm[p2c]/cu_grid.vol[p2c]
        cu_mp.σij[ix, 1] = cu_mp.sij[ix, 1]+σm
        cu_mp.σij[ix, 2] = cu_mp.sij[ix, 2]+σm
        cu_mp.σij[ix, 3] = cu_mp.sij[ix, 3]+σm
        cu_mp.σij[ix, 4] = cu_mp.sij[ix, 4]
        cu_mp.σij[ix, 5] = cu_mp.sij[ix, 5]
        cu_mp.σij[ix, 6] = cu_mp.sij[ix, 6]
        # update mean stress tensor
        σm = (cu_mp.σij[ix, 1]+cu_mp.σij[ix, 2]+cu_mp.σij[ix, 3])*FNUM_13
        cu_mp.σm[ix] = σm
        # update deviatoric stress tensor
        cu_mp.sij[ix, 1] = cu_mp.σij[ix, 1]-σm
        cu_mp.sij[ix, 2] = cu_mp.σij[ix, 2]-σm
        cu_mp.sij[ix, 3] = cu_mp.σij[ix, 3]-σm
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
                           ::Val{:OS}) where {T2}
    Ti<args.Te ? (G=args.gravity/args.Te*Ti) : (G=args.gravity)
    # NVTX.@range "k01" @cuda threads=OccAPI.k01_t blocks=OccAPI.k01_b kernel_OS01!(cu_grid                                )
    # NVTX.@range "k02" @cuda threads=OccAPI.k02_t blocks=OccAPI.k02_b kernel_OS02!(cu_grid, cu_mp, Val(args.basis)        )
    # NVTX.@range "k03" @cuda threads=OccAPI.k03_t blocks=OccAPI.k03_b kernel_OS03!(cu_grid, cu_mp, G                      )
    # NVTX.@range "k04" @cuda threads=OccAPI.k04_t blocks=OccAPI.k04_b kernel_OS04!(cu_grid, cu_bc, ΔT, args.ζ             )
    # NVTX.@range "k05" @cuda threads=OccAPI.k05_t blocks=OccAPI.k05_b kernel_OS05!(cu_grid, cu_mp, ΔT, args.FLIP, args.PIC)
    # NVTX.@range "k06" @cuda threads=OccAPI.k06_t blocks=OccAPI.k06_b kernel_OS06!(cu_grid, cu_mp                         )
    # NVTX.@range "k07" @cuda threads=OccAPI.k07_t blocks=OccAPI.k07_b kernel_OS07!(cu_grid, cu_bc, ΔT                     )
    # NVTX.@range "k08" @cuda threads=OccAPI.k08_t blocks=OccAPI.k08_b kernel_OS08!(cu_grid, cu_mp                         )
    # NVTX.@range "ctm" constitutive!(args, cu_mp, Ti, OccAPI)
    # if args.vollock==true
    #     NVTX.@range "k09" @cuda threads=OccAPI.k09_t blocks=OccAPI.k09_b kernel_OS09!(cu_grid, cu_mp)
    #     NVTX.@range "k10" @cuda threads=OccAPI.k10_t blocks=OccAPI.k10_b kernel_OS10!(cu_grid, cu_mp)
    # end
    @cuda threads=OccAPI.k01_t blocks=OccAPI.k01_b kernel_OS01!(cu_grid                                )
    @cuda threads=OccAPI.k02_t blocks=OccAPI.k02_b kernel_OS02!(cu_grid, cu_mp, Val(args.basis)        )
    @cuda threads=OccAPI.k03_t blocks=OccAPI.k03_b kernel_OS03!(cu_grid, cu_mp, G                      )
    @cuda threads=OccAPI.k04_t blocks=OccAPI.k04_b kernel_OS04!(cu_grid, cu_bc, ΔT, args.ζ             )
    @cuda threads=OccAPI.k05_t blocks=OccAPI.k05_b kernel_OS05!(cu_grid, cu_mp, ΔT, args.FLIP, args.PIC)
    @cuda threads=OccAPI.k06_t blocks=OccAPI.k06_b kernel_OS06!(cu_grid, cu_mp                         )
    @cuda threads=OccAPI.k07_t blocks=OccAPI.k07_b kernel_OS07!(cu_grid, cu_bc, ΔT                     )
    @cuda threads=OccAPI.k08_t blocks=OccAPI.k08_b kernel_OS08!(cu_grid, cu_mp                         )
    constitutive!(args, cu_mp, Ti, OccAPI)
    if args.vollock==true
        @cuda threads=OccAPI.k09_t blocks=OccAPI.k09_b kernel_OS09!(cu_grid, cu_mp)
        @cuda threads=OccAPI.k10_t blocks=OccAPI.k10_b kernel_OS10!(cu_grid, cu_mp)
    end
    return nothing
end