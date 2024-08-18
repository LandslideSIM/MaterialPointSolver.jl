#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : linear.jl                                                                  |
|  Description: Linear basis function                                                      |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. basis_function!() [2D]                                                  |
|               2. basis_function!() [3D]                                                  |
|               3. linear_weighting!()                                                     |
+==========================================================================================#

"""
    basis_function!!(grid::Grid2D, mp::Particle2D, ::Val{:linear})

Description:
---
2D linear basis function
"""
@views function basis_function!(grid::    Grid2D{T1, T2},
                                mp  ::Particle2D{T1, T2},
                                    ::Val{:linear}) where {T1, T2}
    # compute particle to cell and particle to node index
    mp.p2c .= (fld.(mp.pos[:, 2].-grid.range_y1, grid.space_y).+1).+(grid.cell_num_y.*
               fld.(mp.pos[:, 1].-grid.range_x1, grid.space_x))
    mp.p2n .= grid.c2n[mp.p2c, :]
    # compute the value of shape function
    for iy in 1:mp.NIC, ix in 1:mp.num
        Δdx = mp.pos[ix, 1]-grid.pos[mp.p2n[ix, iy], 1]
        Δdy = mp.pos[ix, 2]-grid.pos[mp.p2n[ix, iy], 2]
        Nx, dNx = linear_weighting(Δdx, grid.space_x)
        Ny, dNy = linear_weighting(Δdy, grid.space_y)
        mp.Ni[ix, iy]  =  Nx*Ny # shape function
        mp.∂Nx[ix, iy] = dNx*Ny # x-gradient shape function
        mp.∂Ny[ix, iy] = dNy*Nx # y-gradient shape function
    end
    return nothing
end

"""
    basis_function!(grid::Grid2D, mp::Particle2D, ::Val{:linear})

Description:
---
3D linear basis function.  
"""
@views function basis_function!(grid::    Grid3D{T1, T2},
                                mp  ::Particle3D{T1, T2},
                                    ::Val{:linear}) where {T1, T2}
    # compute particle to cell and particle to node index
    mp.p2c .= (fld.(mp.pos[:, 3].-grid.range_z1, grid.space_z)).*(grid.cell_num_x*grid.cell_num_y).+
              (fld.(mp.pos[:, 1].-grid.range_x1, grid.space_x)).* grid.cell_num_y                 .+
               cld.(mp.pos[:, 2].-grid.range_y1, grid.space_y)
    mp.p2n .= grid.c2n[mp.p2c, :]
    # compute the value of shape function
    for iy in 1:mp.NIC, ix in 1:mp.num
        Δdx = mp.pos[ix, 1]-grid.pos[mp.p2n[ix, iy], 1]
        Δdy = mp.pos[ix, 2]-grid.pos[mp.p2n[ix, iy], 2]
        Δdz = mp.pos[ix, 3]-grid.pos[mp.p2n[ix, iy], 3]
        Nx, dNx = linear_weighting(Δdx, grid.space_x)
        Ny, dNy = linear_weighting(Δdy, grid.space_y)
        Nz, dNz = linear_weighting(Δdz, grid.space_z)
        mp.Ni[ix, iy]  =  Nx*Ny*Nz # shape function
        mp.∂Nx[ix, iy] = dNx*Ny*Nz # x-gradient shape function
        mp.∂Ny[ix, iy] = dNy*Nx*Nz # y-gradient shape function
        mp.∂Nz[ix, iy] = dNz*Nx*Ny # z-gradient shape function
    end
    return nothing
end

"""
    linear_weighting!(Δx::Array{T2, 2}, h::T2, N::Array{T2, 2}, dN::Array{T2, 2})

Description:
---
Compute 2D linear basis function.
"""
function linear_weighting(Δx::T2,           # distance between particle and node
                          h ::T2) where T2  # grid spacing
    FNUM_1 = T2(1.0); FNUM_0 = T2(0.0)
    if abs(Δx)≤h
         N = -abs(Δx)/h+FNUM_1
        dN = -sign(Δx)/h
    else
         N = FNUM_0
        dN = FNUM_0
    end
    return N, dN
end
