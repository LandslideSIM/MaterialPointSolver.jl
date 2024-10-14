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
    host2device(grid::DeviceGrid{T1, T2}, mp::DeviceParticle{T1, T2}, 
        attr::DeviceProperty{T1, T2}, bc::DeviceVBoundary{T1, T2}, ::Val{:CPU})

Description:
---
Transfer data from host to device. (2D & 3D)
"""
function host2device(
    grid::     DeviceGrid{T1, T2}, 
    mp  :: DeviceParticle{T1, T2}, 
    attr:: DeviceProperty{T1, T2}, 
    bc  ::DeviceVBoundary{T1, T2}, 
        ::Val{:CPU}
) where {T1, T2}
    return grid, mp, attr, bc
end

"""
    device2host!(args::DeviceArgs{T1, T2}, mp::DeviceParticle{T1, T2}, 
        dev_mp::GPUDeviceParticle{T1, T2}, ::Val{:CPU/:CUDA/...}; verbose::Bool=false)

Description:
---
Transfer data from device to host. (2D & 3D)
"""
function device2host!(
    args   ::    DeviceArgs{T1, T2}, 
    mp     ::DeviceParticle{T1, T2}, 
    dev_mp ::DeviceParticle{T1, T2}, 
           ::Val{:CPU};
    verbose::Bool=false
) where {T1, T2}
    return nothing
end

""" 
    clean_device!(args::DeviceArgs{T1, T2}, dev_grid::GPUDeviceGrid{T1, T2}, 
        dev_mp::GPUDeviceParticle{T1, T2}, dev_attr::GPUDeviceParticle{T1, T2},
        DeviceProperty{T1, T2}, dev_bc::GPUDeviceVBoundary{T1, T2}, ::Val{:CPU/:CUDA/...})

Description:
---
Clean the device memory.
"""
function clean_device!(
    dev_grid::     DeviceGrid{T1, T2}, 
    dev_mp  :: DeviceParticle{T1, T2}, 
    dev_attr:: DeviceProperty{T1, T2}, 
    dev_bc  ::DeviceVBoundary{T1, T2},
            ::Val{:CPU}
) where {T1, T2}
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