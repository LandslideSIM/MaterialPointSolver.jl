using MaterialPointSolver
using KernelAbstractions
using CUDA
using BenchmarkTools
using Printf
CUDA.device!(1)
rtsdir = joinpath(homedir(), "Workbench/outputs")
include(joinpath(@__DIR__, "funcs.jl"))
args, grid, mp, pts_attr, bc = create_model()
mp_p2n1 = CUDA.zeros(  Int64, mp.num, 64)
mp_Ni1  = CUDA.zeros(Float32, mp.num, 64)
mp_∂Nx1 = CUDA.zeros(Float32, mp.num, 64)
mp_∂Ny1 = CUDA.zeros(Float32, mp.num, 64)
mp_∂Nz1 = CUDA.zeros(Float32, mp.num, 64)
mp_p2n2 = CUDA.zeros(  Int64, mp.num, 27)
mp_Ni2  = CUDA.zeros(Float32, mp.num, 27)
mp_∂Nx2 = CUDA.zeros(Float32, mp.num, 27)
mp_∂Ny2 = CUDA.zeros(Float32, mp.num, 27)
mp_∂Nz2 = CUDA.zeros(Float32, mp.num, 27)

@kernel inbounds = true function test1!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2},
    mp_p2n, mp_Ni, mp_∂Nx, mp_∂Ny, mp_∂Nz
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
        mp.p2c[ix] = cld(mp_pos_2 - grid.range_y1, grid.space_y) +
                     fld(mp_pos_3 - grid.range_z1, grid.space_z) * 
                        grid.cell_num_y * grid.cell_num_x +
                     fld(mp_pos_1 - grid.range_x1, grid.space_x) * grid.cell_num_y |> T1
        for iy in Int32(1):Int32(mp.NIC)
            # p2n index
            p2n = getP2N_uGIMP(grid, mp.p2c[ix], iy)
            mp_p2n[ix, iy] = p2n
            # compute distance between particle and related nodes
            Δdx = mp_pos_1 - grid.pos[p2n, 1]
            Δdy = mp_pos_2 - grid.pos[p2n, 2]
            Δdz = mp_pos_3 - grid.pos[p2n, 3]
            # compute basis function
            Nx, dNx = uGIMPbasis(Δdx, grid.space_x, mp.space_x)
            Ny, dNy = uGIMPbasis(Δdy, grid.space_y, mp.space_y)
            Nz, dNz = uGIMPbasis(Δdz, grid.space_z, mp.space_z)
            mp_Ni[ix, iy] = Nx * Ny * Nz
            mp_∂Nx[ix, iy] = dNx * Ny * Nz # x-gradient basis function
            mp_∂Ny[ix, iy] = dNy * Nx * Nz # y-gradient basis function
            mp_∂Nz[ix, iy] = dNz * Nx * Ny # z-gradient basis function
        end
    end
end

@kernel inbounds = true function test2!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2},
    mp_p2n, mp_Ni, mp_∂Nx, mp_∂Ny, mp_∂Nz
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ mp.num
        for iy in Int32(1):Int32(mp.NIC)
            Ni = mp_Ni[ix, iy]
            if Ni ≠ 0
                ∂Nx = mp_∂Nx[ix, iy]
                ∂Ny = mp_∂Ny[ix, iy]
                ∂Nz = mp_∂Nz[ix, iy]
                p2n = mp_p2n[ix, iy]
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
                                                     ∂Ny * mp.σij[ix, 5]) + NiM * T2(-9.8)
            end
        end
    end
end

@kernel inbounds = true function test3!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2},
    mp_p2n, mp_Ni, mp_∂Nx, mp_∂Ny, mp_∂Nz
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
        mp.p2c[ix] = cld(mp_pos_2 - grid.range_y1, grid.space_y) +
                     fld(mp_pos_3 - grid.range_z1, grid.space_z) *
                        grid.cell_num_y * grid.cell_num_x +
                     fld(mp_pos_1 - grid.range_x1, grid.space_x) * grid.cell_num_y |> T1
        iter = Int32(1)
        for iy in Int32(1):Int32(mp.NIC)
            # p2n index
            p2n = getP2N_uGIMP(grid, mp.p2c[ix], iy)
            # compute distance between particle and related nodes
            Δdx = mp_pos_1 - grid.pos[p2n, 1]
            Δdy = mp_pos_2 - grid.pos[p2n, 2]
            Δdz = mp_pos_3 - grid.pos[p2n, 3]
            if abs(Δdx) < (grid.space_x + T2(0.5) * mp.space_x) &&
               abs(Δdy) < (grid.space_y + T2(0.5) * mp.space_y) &&
               abs(Δdz) < (grid.space_z + T2(0.5) * mp.space_z)
                mp_p2n[ix, iter] = p2n
                # compute basis function
                Nx, dNx = uGIMPbasis(Δdx, grid.space_x, mp.space_x)
                Ny, dNy = uGIMPbasis(Δdy, grid.space_y, mp.space_y)
                Nz, dNz = uGIMPbasis(Δdz, grid.space_z, mp.space_z)
                mp_Ni[ ix, iter] =  Nx * Ny * Nz
                mp_∂Nx[ix, iter] = dNx * Ny * Nz # x-gradient basis function
                mp_∂Ny[ix, iter] = dNy * Nx * Nz # y-gradient basis function
                mp_∂Nz[ix, iter] = dNz * Nx * Ny # z-gradient basis function
                iter += Int32(1)
            end
        end
    end
end

@kernel inbounds = true function test4!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2},
    mp_p2n, mp_Ni, mp_∂Nx, mp_∂Ny, mp_∂Nz
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ mp.num
        for iy in Int32(1):Int32(27)
            Ni = mp_Ni[ix, iy]
            if Ni ≠ 0
                ∂Nx = mp_∂Nx[ix, iy]
                ∂Ny = mp_∂Ny[ix, iy]
                ∂Nz = mp_∂Nz[ix, iy]
                p2n = mp_p2n[ix, iy]
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
                                                     ∂Ny * mp.σij[ix, 5]) + NiM * T2(-9.8)
            end
        end
    end
end

test1!(CUDABackend())(ndrange=mp.num, grid, mp, mp_p2n1, mp_Ni1, mp_∂Nx1, mp_∂Ny1, mp_∂Nz1)
test2!(CUDABackend())(ndrange=mp.num, grid, mp, mp_p2n1, mp_Ni1, mp_∂Nx1, mp_∂Ny1, mp_∂Nz1)
test3!(CUDABackend())(ndrange=mp.num, grid, mp, mp_p2n2, mp_Ni2, mp_∂Nx2, mp_∂Ny2, mp_∂Nz2)
test4!(CUDABackend())(ndrange=mp.num, grid, mp, mp_p2n2, mp_Ni2, mp_∂Nx2, mp_∂Ny2, mp_∂Nz2)

t1 = 1e3 * @belapsed begin 
    test1!($CUDABackend())(ndrange=$mp.num, $grid, $mp, $mp_p2n1, $mp_Ni1, $mp_∂Nx1, $mp_∂Ny1, $mp_∂Nz1)
    CUDA.synchronize()
end

t2 = 1e3 * @belapsed begin 
    test2!($CUDABackend())(ndrange=$mp.num, $grid, $mp, $mp_p2n1, $mp_Ni1, $mp_∂Nx1, $mp_∂Ny1, $mp_∂Nz1)
    CUDA.synchronize()
end

t3 = 1e3 * @belapsed begin 
    test3!($CUDABackend())(ndrange=$mp.num, $grid, $mp, $mp_p2n2, $mp_Ni2, $mp_∂Nx2, $mp_∂Ny2, $mp_∂Nz2)
    CUDA.synchronize()
end

t4 = 1e3 * @belapsed begin 
    test4!($CUDABackend())(ndrange=$mp.num, $grid, $mp, $mp_p2n2, $mp_Ni2, $mp_∂Nx2, $mp_∂Ny2, $mp_∂Nz2)
    CUDA.synchronize()
end

@info """Benchmarks
$("─"^16)
original basis_func: $(@sprintf("%.2f", t1)) ms
reduced  basis_func: $(@sprintf("%.2f", t3)) ms

original atomic: $(@sprintf("%.2f", t2)) ms
reduced  atomic: $(@sprintf("%.2f", t4)) ms
"""