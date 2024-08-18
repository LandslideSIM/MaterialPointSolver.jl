#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : sharedmem.jl                                                               |
|  Description: the improvment of shared memory                                            |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
+==========================================================================================#

using MaterialPointSolver
using KernelAbstractions
using BenchmarkTools
using Printf
using CUDA

rtsdir = joinpath(homedir(), "Workbench/outputs")
include(joinpath(@__DIR__, "model.jl"))

@kernel inbounds = true function testresetmpstatus_OS!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2},
        ::Val{:uGIMP}
) where {T1, T2}
    ix = @index(Global)
    if ix <= mp.num
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
        mp.p2c[ix] = cld(mp_pos_2 - grid.range_y1, grid.space_y) +
                     fld(mp_pos_3 - grid.range_z1, grid.space_z) * 
                        grid.cell_num_y * grid.cell_num_x +
                     fld(mp_pos_1 - grid.range_x1, grid.space_x) * grid.cell_num_y |> T1
        for iy in Int32(1):Int32(mp.NIC)
            # p2n index
            p2n = getP2N_uGIMP(grid, mp.p2c[ix], iy)
            mp.p2n[ix, iy] = p2n
            # compute distance between particle and related nodes
            Δdx = mp_pos_1 - grid.pos[p2n, 1]
            Δdy = mp_pos_2 - grid.pos[p2n, 2]
            Δdz = mp_pos_3 - grid.pos[p2n, 3]
            # compute basis function
            Nx, dNx = testuGIMPbasis(Δdx, grid.space_x, mp.space_x)
            Ny, dNy = testuGIMPbasis(Δdy, grid.space_y, mp.space_y)
            Nz, dNz = testuGIMPbasis(Δdz, grid.space_z, mp.space_z)
            mp.Ni[ix, iy] = Nx * Ny * Nz
            mp.∂Nx[ix, iy] = dNx * Ny * Nz # x-gradient basis function
            mp.∂Ny[ix, iy] = dNy * Nx * Nz # y-gradient basis function
            mp.∂Nz[ix, iy] = dNz * Nx * Ny # z-gradient basis function
        end
    end
end

@inline Base.@propagate_inbounds function testuGIMPbasis(Δx::T2, h::T2, lp::T2) where T2
    T1 = T2 == Float32 ? Int32 : Int64
    if abs(Δx) < T2(0.5)*lp
        Ni = T2(1.0) - ((T2(4.0) * Δx * Δx + lp * lp) / (T2(4.0) * h * lp))
        dN = -((T2(8.0) * Δx) / (T2(4.0) * h * lp))
    elseif (T2(0.5) * lp) ≤ abs(Δx) < (h - T2(0.5) * lp)
        Ni = T2(1.0) - (abs(Δx) / h)
        dN = sign(Δx) * (T2(-1.0) / h)
    elseif (h - T2(0.5) * lp) ≤ abs(Δx) < (h + T2(0.5) * lp)
        Ni = ((h + T2(0.5) * lp - abs(Δx)) ^ T1(2)) / (T2(2.0) * h * lp)
        dN = -sign(Δx) * ((h + T2(0.5) * lp - abs(Δx)) / (h * lp))
    else
        Ni = T2(0.0)
        dN = T2(0.0)
    end
    return T2(Ni), T2(dN)
end

# benchmarks
rst1 = @belapsed begin
    resetmpstatus_OS!($dev)(ndrange=$gpu_mp.num, $gpu_grid, $gpu_mp, $Val(args.basis))
    KAsync($dev)
end

rst2 = @belapsed begin
    testresetmpstatus_OS!($dev)(ndrange=$gpu_mp.num, $gpu_grid, $gpu_mp, $Val(args.basis))
    KAsync($dev)
end

@info """\t
        shared memory: $(@sprintf("%.2e ms", rst1*1e3))
without shared memory: $(@sprintf("%.2e ms", rst2*1e3))
"""

#=
┌ Info: 
│         shared memory: 1.20e+01 ms
└ without shared memory: 1.77e+01 ms
=#