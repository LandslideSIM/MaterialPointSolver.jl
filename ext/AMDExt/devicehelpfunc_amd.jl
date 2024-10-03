#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : devicehelpfunc_amd.jl                                                      |
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
        ::Val{:ROCm}
) where {T1, T2}
    # upload data to device
    dev_grid = user_adapt(ROCArray, grid)
    dev_mp   = user_adapt(ROCArray, mp)
    dev_attr = user_adapt(ROCArray, attr)
    dev_bc   = user_adapt(ROCArray, bc)
    # output info
    datasize = Base.summarysize(grid) + Base.summarysize(mp) +
               Base.summarysize(attr) + Base.summarysize(bc)
    outprint = @sprintf("%.1f", datasize / 1024 ^ 3)
    dev_id   = AMDGPU.device().device_id
    content  = "uploading [≈ $(outprint) GiB] → :ROCm [$(dev_id)]"
    println("\e[1;32m[▲ I/O:\e[0m \e[0;32m$(content)\e[0m")
    return dev_grid, dev_mp, dev_attr, dev_bc
end

function device2host!(
    args   ::    DeviceArgs{T1, T2}, 
    mp     ::DeviceParticle{T1, T2}, 
    dev_mp ::DeviceParticle{T1, T2}, 
           ::Val{:ROCm}; 
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
        dev_id = AMDGPU.device().device_id
        content = "downloading from :ROCm $(dev_id) → host"
        println("\e[1;31m[▼ I/O:\e[0m \e[0;31m$(content)\e[0m")
    end
    return nothing
end

function clean_device!(
    dev_grid::     DeviceGrid{T1, T2}, 
    dev_mp  :: DeviceParticle{T1, T2}, 
    dev_attr:: DeviceProperty{T1, T2}, 
    dev_bc  ::DeviceVBoundary{T1, T2},
            ::Val{:ROCm}
) where {T1, T2}
    for i in 1:nfields(dev_grid)
        typeof(getfield(dev_grid, i)) <: AbstractArray ? 
            KernelAbstractions.unsafe_free!(getfield(dev_grid, i)) : nothing
    end
    for i in 1:nfields(dev_mp)
        typeof(getfield(dev_mp, i)) <: AbstractArray ?
            KernelAbstractions.unsafe_free!(getfield(dev_mp, i)) : nothing
    end
    for i in 1:nfields(dev_attr)
        typeof(getfield(dev_attr, i)) <: AbstractArray ?
            KernelAbstractions.unsafe_free!(getfield(dev_attr, i)) : nothing
    end
    for i in 1:nfields(dev_bc)
        typeof(getfield(dev_bc, i)) <: AbstractArray ? 
            KernelAbstractions.unsafe_free!(getfield(dev_bc, i)) : nothing
    end
    #=======================================================================================
    | !!! HERE NEED A FUNCTION LIKE CUDA.reclaim()                                         |
    =======================================================================================#
    dev_id = AMDGPU.device().device_id
    KernelAbstractions.synchronize(ROCBackend())
    content = "free device $(dev_id) memory"
    println("\e[1;32m[• I/O:\e[0m \e[0;32m$(content)\e[0m")
    return nothing
end

function Tpeak(
            ::Val{:ROCm}; 
    datatype::Symbol=:FP64, 
    ID      ::Int=0
)
    ID == 0 ? ID = 1 : nothing
    max_threads = AMDGPU.HIP.properties(AMDGPU.device_id!(ID)).maxThreadsPerBlock
    AMDGPU.device!(AMDGPU.devices()[ID])
    bench_backend = ROCBackend()
    println("\e[1;36m[ Test:\e[0m device ($(AMDGPU.device().device_id)) → $(datatype)")
    dt = datatype == :FP64 ? Float64 : 
         datatype == :FP32 ? Float32 : error("Wrong datatype!") 
    throughputs = Float64[]

    @inbounds for pow = Int(log2(32)):Int(log2(max_threads)) 
        nx = ny = 32 * 2^pow
        gmem = Int(AMDGPU.HIP.properties(AMDGPU.device_id!(ID)).totalGlobalMem)
        3 * nx * ny * sizeof(dt) > gmem ? break : nothing 
        a = rand(dt, nx, ny); A = ROCArray(a)
        b = rand(dt, nx, ny); B = ROCArray(b)
        c = rand(dt, nx, ny); C = ROCArray(c)
        # thread = (2^pow, 1)
        # block = (nx÷thread[1], ny)
        t_it = BenchmarkTools.@belapsed memcpy!($A, $B, $C, $bench_backend)
        T_tot = 3 * 1 / 1024^3 * nx * ny * sizeof(dt) / t_it
        push!(throughputs, T_tot)
        
        # clean gpu memory
        KernelAbstractions.unsafe_free!(A)
        KernelAbstractions.unsafe_free!(B)
        KernelAbstractions.unsafe_free!(C)
        #===================================================================================
        | !!! HERE NEED A FUNCTION LIKE CUDA.reclaim()                                     |
        ===================================================================================#
    end
    value = round(maximum(throughputs), digits=2)
    @info "Tpeak on :ROCm [$(ID)]: $(value) GiB/s"
    return value
end

getBackend(::Val{:ROCm}) = ROCBackend()
getArray(::Val{:ROCm}) = ROCArray