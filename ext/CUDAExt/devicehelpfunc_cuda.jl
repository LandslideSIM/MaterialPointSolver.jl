#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : devicehelpfunc_cuda.jl                                                     |
|  Description: device helper functions                                                    |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. host2device  [2D & 3D]                                                  |
|               2. device2host! [2D & 3D]                                                  |
|               3. clean_gpu!   [2D & 3D]                                                  |
|               4. Tpeak                                                                   |
|               5. getBackend                                                              |
|               6. getArray                                                                |
+==========================================================================================#

function host2device(
    grid::     DeviceGrid{T1, T2}, 
    mp  :: DeviceParticle{T1, T2}, 
    attr:: DeviceProperty{T1, T2}, 
    bc  ::DeviceVBoundary{T1, T2}, 
        ::Val{:CUDA}
) where {T1, T2}
    # upload data to device
    dev_grid = user_adapt(CuArray, grid)
    dev_mp   = user_adapt(CuArray, mp)
    dev_attr = user_adapt(CuArray, attr)
    dev_bc   = user_adapt(CuArray, bc)
    # output info
    datasize = Base.summarysize(grid) + Base.summarysize(mp) +
               Base.summarysize(attr) + Base.summarysize(bc)
    outprint = @sprintf("%.1f", datasize / 1024 ^ 3)
    dev_id   = CUDA.device().handle
    content  = "uploading [≈ $(outprint) GiB] → :CUDA [$(dev_id)]"
    println("\e[1;32m[▲ I/O:\e[0m \e[0;32m$(content)\e[0m")
    return dev_grid, dev_mp, dev_attr, dev_bc
end

function device2host!(
    args   ::    DeviceArgs{T1, T2}, 
    mp     ::DeviceParticle{T1, T2}, 
    dev_mp ::DeviceParticle{T1, T2}, 
           ::Val{:CUDA}; 
    verbose::Bool=false
) where {T1, T2}
    copyto!(mp.σm  , dev_mp.σm  )
    copyto!(mp.vs  , dev_mp.vs  )
    copyto!(mp.ξ   , dev_mp.ξ   )
    copyto!(mp.Ω   , dev_mp.Ω   )
    copyto!(mp.σij , dev_mp.σij )
    copyto!(mp.ϵk  , dev_mp.ϵk  )
    copyto!(mp.ϵq  , dev_mp.ϵq  )
    copyto!(mp.ϵv  , dev_mp.ϵv  )
    copyto!(mp.ϵijs, dev_mp.ϵijs)
    copyto!(mp.ms  , dev_mp.ms  )
    args.coupling==:TS ? (
        copyto!(mp.mw  , dev_mp.mw  );
        copyto!(mp.vw  , dev_mp.vw  );
        copyto!(mp.σw  , dev_mp.σw  );
        copyto!(mp.ϵijw, dev_mp.ϵijw);
        copyto!(mp.n   , dev_mp.n   );
    ) : nothing
    if verbose == true
        dev_id = CUDA.device().handle
        content = "downloading from :CUDA [$(dev_id)] → host"
        println("\e[1;31m[▼ I/O:\e[0m \e[0;31m$(content)\e[0m")
    end
    return nothing
end

function clean_device!(
    dev_grid::     DeviceGrid{T1, T2}, 
    dev_mp  :: DeviceParticle{T1, T2}, 
    dev_attr:: DeviceProperty{T1, T2}, 
    dev_bc  ::DeviceVBoundary{T1, T2},
            ::Val{:CUDA}
) where {T1, T2}
    for i in 1:nfields(dev_grid)
        typeof(getfield(dev_grid, i)) <: AbstractArray ? 
            CUDA.unsafe_free!(getfield(dev_grid, i)) : nothing
    end
    for i in 1:nfields(dev_mp)
        typeof(getfield(dev_mp, i)) <: AbstractArray ?
            CUDA.unsafe_free!(getfield(dev_mp, i)) : nothing
    end
    for i in 1:nfields(dev_attr)
        typeof(getfield(dev_attr, i)) <: AbstractArray ?
            CUDA.unsafe_free!(getfield(dev_attr, i)) : nothing
    end
    for i in 1:nfields(dev_bc)
        typeof(getfield(dev_bc, i)) <: AbstractArray ? 
            CUDA.unsafe_free!(getfield(dev_bc, i)) : nothing
    end
    CUDA.reclaim()
    dev_id = CUDA.device().handle
    KernelAbstractions.synchronize(CUDABackend())
    content = "free device [$(dev_id)] memory"
    println("\e[1;32m[• I/O:\e[0m \e[0;32m$(content)\e[0m")
    return nothing
end

function Tpeak(
              ::Val{:CUDA};
    datatype  ::Symbol=:FP64, 
    ID        ::Int=0
)
    CUDA.device!(ID)
    bench_backend = CUDABackend()
    max_threads = CUDA.attribute(CUDA.device(), 
        CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    println("\e[1;36m[ Test:\e[0m device ($(CUDA.device().handle)) → $(datatype)")
    dt = datatype == :FP64 ? Float64 : 
         datatype == :FP32 ? Float32 : error("Wrong datatype!") 
    throughputs = Float64[]

    @inbounds for pow = Int(log2(32)):Int(log2(max_threads)) 
        nx = ny = 32 * 2^pow
        3 * nx * ny * sizeof(dt) > CUDA.available_memory() ? break : nothing
        A = CUDA.rand(dt, nx, ny)
        B = CUDA.rand(dt, nx, ny)
        C = CUDA.rand(dt, nx, ny)
        # thread = (2^pow, 1)
        # block = (nx÷thread[1], ny)
        t_it = BenchmarkTools.@belapsed memcpy!($A, $B, $C, $bench_backend)
        T_tot = 3 * 1 / 1024^3 * nx * ny * sizeof(dt) / t_it
        push!(throughputs, T_tot)
        # clean gpu memory
        CUDA.unsafe_free!(A)
        CUDA.unsafe_free!(B)
        CUDA.unsafe_free!(C)
        CUDA.reclaim()
    end
    value = round(maximum(throughputs), digits=2)
    @info "Tpeak on :CUDA [$(ID)]: $(value) GiB/s"
    return value
end

getBackend(::Val{:CUDA}) = CUDABackend()
getArray(::Val{:CUDA}) = CuArray