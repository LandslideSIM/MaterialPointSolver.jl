#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : uGIMP.jl                                                                   |
|  Description: Compute uGIMP shape function.                                              |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. basis_function!() [2D]                                                  |
|               2. basis_function!() [3D]                                                  |
|               3. uGIMP_weighting!()                                                      |
+==========================================================================================#

"""
    basis_function!(grid::Grid2D, mp::Particle2D, ::Val{:uGIMP})

Description:
---
2D uGIMP basis function.
"""
@views function basis_function!(grid::    Grid2D{T1, T2},
                                 mp ::Particle2D{T1, T2},
                                    ::Val{:uGIMP}) where {T1, T2}
    # compute particle to cell and particle to node index
    mp.p2c .= cld.(mp.pos[:, 2].-grid.range_y1, grid.space_y).+grid.cell_num_y.*
              fld.(mp.pos[:, 1].-grid.range_x1, grid.space_x)
    mp.p2n .= grid.c2n[mp.p2c, :]
    # compute the value of shape function
    for iy in 1:mp.NIC, ix in 1:mp.num
        Δdx = mp.pos[ix, 1]-grid.pos[mp.p2n[ix, iy], 1]
        Δdy = mp.pos[ix, 2]-grid.pos[mp.p2n[ix, iy], 2]
        Nx, dNx = uGIMP_weighting(Δdx, grid.space_x, mp.space_x)
        Ny, dNy = uGIMP_weighting(Δdy, grid.space_y, mp.space_y)
        mp.Ni[ix, iy]  =  Nx*Ny # shape function
        mp.∂Nx[ix, iy] = dNx*Ny # x-gradient shape function
        mp.∂Ny[ix, iy] = dNy*Nx # y-gradient shape function
    end
    return nothing
end

"""
    basis_function!(grid::Grid3D, mp::Particle3D, ::Val{:uGIMP})

Description:
---
3D uGIMP basis function.
"""
@views function basis_function!(grid::    Grid3D{T1, T2},
                                mp  ::Particle3D{T1, T2},
                                    ::Val{:uGIMP}) where {T1, T2}
    # compute particle to cell and particle to node index
    grid_cell_numxy = grid.cell_num_x*grid.cell_num_y
    mp.p2c .= (fld.(mp.pos[:, 3].-grid.range_z1, grid.space_z))*grid_cell_numxy +
              (fld.(mp.pos[:, 1].-grid.range_x1, grid.space_x))*grid.cell_num_y +
               cld.(mp.pos[:, 2].-grid.range_y1, grid.space_y)
    mp.p2n .= grid.c2n[mp.p2c, :]
    # compute the value of shape function
    for iy in 1:mp.NIC, ix in 1:mp.num
        Δdx = mp.pos[ix, 1]-grid.pos[mp.p2n[ix, iy], 1]
        Δdy = mp.pos[ix, 2]-grid.pos[mp.p2n[ix, iy], 2]
        Δdz = mp.pos[ix, 3]-grid.pos[mp.p2n[ix, iy], 3]
        Nx, dNx = uGIMP_weighting(Δdx, grid.space_x, mp.space_x)
        Ny, dNy = uGIMP_weighting(Δdy, grid.space_y, mp.space_y)
        Nz, dNz = uGIMP_weighting(Δdz, grid.space_z, mp.space_z)
        mp.Ni[ix, iy]  =  Nx*Ny*Nz # shape function
        mp.∂Nx[ix, iy] = dNx*Ny*Nz # x-gradient shape function
        mp.∂Ny[ix, iy] = dNy*Nx*Nz # y-gradient shape function
        mp.∂Nz[ix, iy] = dNz*Nx*Ny # z-gradient shape function
    end
    return nothing
end

"""
    uGIMP_weighting!(Δx::Array{T2, 2}, h::T2, lp::T2, N::Array{T2, 2}, dN::Array{T2, 2})

Description:
---
Compute uGIMP basis function.
"""
@inline function uGIMP_weighting(Δx::T2,           # distance between particle and node
                                 h ::T2,           # grid spacing
                                 lp::T2) where T2  # particle spacing
    FNUM_h = T2(0.5); FNUM_1 = T2(1.0)
    FNUM_0 = T2(0.0); FNUM_2 = T2(2.0)
    FNUM_4 = T2(4.0); FNUM_8 = T2(8.0)
    if abs(Δx)<(FNUM_h*lp)
         N = FNUM_1-((FNUM_4*(Δx^2)+lp^2)/(FNUM_4*h*lp))
        dN = -((FNUM_8*Δx)/(FNUM_4*h*lp))
    elseif (FNUM_h*lp)≤abs(Δx)<(h-FNUM_h*lp)
         N = FNUM_1-(abs(Δx)/h)
        dN = sign(Δx)*(-FNUM_1/h)
    elseif (h-FNUM_h*lp)≤abs(Δx)<(h+FNUM_h*lp)
         N = ((h+FNUM_h*lp-abs(Δx))^2)/(FNUM_2*h*lp)
        dN = -sign(Δx)*((h+FNUM_h*lp-abs(Δx))/(h*lp))
    else 
         N = FNUM_0
        dN = FNUM_0
    end
    return N, dN
end
