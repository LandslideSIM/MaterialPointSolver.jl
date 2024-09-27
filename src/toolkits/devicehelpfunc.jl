#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : devicehelpfunc.jl                                                          |
|  Description: device helper functions                                                    |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. host2device  [2D & 3D]                                                  |
|               2. device2host! [2D & 3D]                                                  |
|               3. clean_gpu!   [2D & 3D]                                                  |
|               4. memcpy!                                                                 |
|               5. Tpeak                                                                   |
|               6. getBackend                                                              |
|               7. getArray                                                                |
+==========================================================================================#

export host2device, device2host!, clean_device!, memcpy!, Tpeak, getBackend, getArray

"""
    host2device(grid::GRID, mp::PARTICLE, pts_attr::PROPERTY, bc::BOUNDARY, 
        ::Val{:CPU})

Description:
---
Transfer data from host to device. (2D & 3D)
"""
function host2device(
    grid    ::GRID, 
    mp      ::PARTICLE, 
    pts_attr::PROPERTY, 
    bc      ::BOUNDARY, 
            ::Val{:CPU}
)
    return grid, mp, pts_attr, bc
end

"""
    device2host!(args::MODELARGS, mp::PARTICLE, dev_mp::GPUPARTICLE, ::Val{:CPU/:CUDA/...};
        verbose::Bool=false)

Description:
---
Transfer data from device to host. (2D & 3D)
"""
device2host!(args::MODELARGS, mp::PARTICLE, dev_mp::PARTICLE, ::Val{:CPU}; 
    verbose::Bool=false) = nothing

""" 
    clean_device!(args::MODELARGS, dev_grid::GPUGRID, dev_mp::GPUPARTICLE, 
        dev_pts_attr::GPUPARTICLEPROPERTY, dev_bc::GPUBOUNDARY, ::Val{:CPU/:CUDA/...})

Description:
---
Clean the device memory.
"""
function clean_device!(
    dev_grid    ::GRID, 
    dev_mp      ::PARTICLE, 
    dev_pts_attr::PROPERTY, 
    dev_bc      ::BOUNDARY,
                ::Val{:CPU}
)
    return nothing
end

function memcpy!(A, B, C, bench_backend)
    testtpeak!(bench_backend)(ndrange=size(A), A, B, C)
    KAsync((bench_backend))
    return nothing
end

@kernel inbounds=true function testtpeak!(A, B, C)
    ix, iy = @index(Global, NTuple)
    A[ix, iy] = B[ix, iy] + C[ix, iy]
end

function Tpeak(
            ::Val{:CPU}; 
    datatype::Symbol=:FP64,
    ID      ::Int=0
)
    bench_backend = CPU()
    println("\e[1;36m[ Test:\e[0m CPU → $(datatype)")
    dt = datatype==:FP64 ? Float64 : 
         datatype==:FP32 ? Float32 : error("Wrong datatype!") 
    throughputs = Float64[]

    @inbounds for pow = 5:10
        nx = ny = 32*2^pow
        A = rand(dt, nx, ny)
        B = rand(dt, nx, ny)
        C = rand(dt, nx, ny)
        # thread = (2^pow, 1)
        # block = (nx÷thread[1], ny)
        t_it = BenchmarkTools.@belapsed memcpy!($A, $B, $C, $bench_backend)
        T_tot = 3*1/1024^3*nx*ny*sizeof(dt)/t_it
        push!(throughputs, T_tot)
    end
    value = round(maximum(throughputs), digits=2)
    @info "Tpeak on :CPU [$(ID)]: $(value) GiB/s"
    return value
end

getBackend(::Val{:CPU}) = CPU()
getArray(::Val{:CPU}) = Array