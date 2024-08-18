using MaterialPointSolver
using KernelAbstractions
using CUDA
using BenchmarkTools
using Printf

include(joinpath(@__DIR__, "model.jl"))

@kernel inbounds = true function test1!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2},
        ::Val{:uGIMP}
) where {T1, T2}
    ix = @index(Global)
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
        for iy in Int32(1):Int32(mp.NIC)
            # p2n index
            p2n = getP2N_uGIMP(grid, mp.p2c[ix], Int32(iy))
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
end

CUDA.@profile external=true test1!(CUDABackend())(ndrange=gpu_mp.num, gpu_grid, gpu_mp, Val(args.basis))

#=
rst1 = @belapsed begin
    test1!($CUDABackend())(ndrange=$gpu_mp.num, $gpu_grid, $gpu_mp, $Val(args.basis))
    KernelAbstractions.synchronize($CUDABackend())
end

dev = CUDABackend()
rst2 = @belapsed begin
    resetmpstatus_OS!($CUDABackend())(ndrange=$gpu_mp.num, $gpu_grid, $gpu_mp, $Val(args.basis))
    KAsync(dev)
end

@info """Results:
original version: $(@sprintf("%.2e ms", rst1*1e3))
resetmpstatus_OS: $(@sprintf("%.2e ms", rst2*1e3))
"""
=#