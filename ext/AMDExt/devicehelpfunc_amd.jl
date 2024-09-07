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
+==========================================================================================#

function host2device(
    grid    ::GRID, 
    mp      ::PARTICLE, 
    pts_attr::PROPERTY, 
    bc      ::BOUNDARY, 
            ::Val{:ROCm}
)
    T1 = typeof(grid).parameters[1]
    T2 = typeof(grid).parameters[2]
    T3 = ROCArray
    if typeof(grid)<:Grid2D
        dev_grid =      GPUGrid2D{T1, T2, T3, T3, T3    }(
            [getfield(grid, f) for f in fieldnames(     Grid2D)]...)
        dev_mp   =  GPUParticle2D{T1, T2, T3, T3, T3, T3}(
            [getfield(mp  , f) for f in fieldnames( Particle2D)]...)
        dev_pts_attr = GPUParticleProperty{T1, T2, T3, T3}(
            [getfield(pts_attr, f) for f in fieldnames(ParticleProperty)]...)
        dev_bc   = GPUVBoundary2D{T1, T2, T3, T3        }(
            [getfield(bc  , f) for f in fieldnames(VBoundary2D)]...)
    elseif typeof(grid)<:Grid3D
        dev_grid =      GPUGrid3D{T1, T2, T3, T3, T3    }(
            [getfield(grid, f) for f in fieldnames(     Grid3D)]...)
        dev_mp   =  GPUParticle3D{T1, T2, T3, T3, T3, T3}(
            [getfield(mp  , f) for f in fieldnames( Particle3D)]...)
        dev_pts_attr = GPUParticleProperty{T1, T2, T3, T3}(
            [getfield(pts_attr, f) for f in fieldnames(ParticleProperty)]...)
        dev_bc   = GPUVBoundary3D{T1, T2, T3, T3        }(
            [getfield(bc  , f) for f in fieldnames(VBoundary3D)]...)
    end
    datasize = Base.summarysize(grid)     + Base.summarysize(mp) +
               Base.summarysize(pts_attr) + Base.summarysize(bc)
    outprint = @sprintf("%.1f", datasize / 1024 ^ 3)
    dev_id = AMDGPU.device().device_id
    content = "uploading [≈ $(outprint) GiB] → :ROCm [$(dev_id)]"
    println("\e[1;32m[▲ I/O:\e[0m \e[0;32m$(content)\e[0m")
    return dev_grid, dev_mp, dev_pts_attr, dev_bc
end

function device2host!(
    args   ::MODELARGS, 
    mp     ::PARTICLE, 
    dev_mp ::GPUPARTICLE, 
           ::Val{:ROCm};
    verbose::Bool=false
)
    copyto!(mp.σm   , dev_mp.σm   )
    copyto!(mp.Vs   , dev_mp.Vs   )
    copyto!(mp.pos  , dev_mp.pos  )
    copyto!(mp.vol  , dev_mp.vol  )
    copyto!(mp.σij  , dev_mp.σij  )
    copyto!(mp.epK  , dev_mp.epK  )
    copyto!(mp.epII , dev_mp.epII )
    copyto!(mp.dϵ   , dev_mp.dϵ   )
    copyto!(mp.ϵij_s, dev_mp.ϵij_s)
    copyto!(mp.Ms   , dev_mp.Ms   )
    args.coupling==:TS ? (
        copyto!(mp.Mw      , dev_mp.Mw      );
        copyto!(mp.Vw      , dev_mp.Vw      );
        copyto!(mp.σw      , dev_mp.σw      );
        copyto!(mp.ϵij_w   , dev_mp.ϵij_w   );
        copyto!(mp.porosity, dev_mp.porosity);
    ) : nothing
    if verbose == true
        dev_id = AMDGPU.device().device_id
        content = "downloading from :ROCm $(dev_id) → host"
        println("\e[1;31m[▼ I/O:\e[0m \e[0;31m$(content)\e[0m")
    end
    return nothing
end

function clean_device!(
    dev_grid    ::GPUGRID, 
    dev_mp      ::GPUPARTICLE, 
    dev_pts_attr::GPUPARTICLEPROPERTY, 
    dev_bc      ::GPUBOUNDARY,
                ::Val{:ROCm}
)
    for i in 1:nfields(dev_grid)
        typeof(getfield(dev_grid, i))<:AbstractArray ? 
            KernelAbstractions.unsafe_free!(getfield(dev_grid, i)) : nothing
    end
    for i in 1:nfields(dev_mp)
        typeof(getfield(dev_mp, i))<:AbstractArray ?
            KernelAbstractions.unsafe_free!(getfield(dev_mp, i)) : nothing
    end
    for i in 1:nfields(dev_pts_attr)
        typeof(getfield(dev_pts_attr, i))<:AbstractArray ?
            KernelAbstractions.unsafe_free!(getfield(dev_pts_attr, i)) : nothing
    end
    for i in 1:nfields(dev_bc)
        typeof(getfield(dev_bc, i))<:AbstractArray ? 
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
    dev_backend = ROCBackend()
    println("\e[1;36m[ Test:\e[0m device ($(AMDGPU.device().device_id)) → $(datatype)")
    dt = datatype==:FP64 ? Float64 : 
         datatype==:FP32 ? Float32 : error("Wrong datatype!") 
    throughputs = Float64[]

    @inbounds for pow = Int(log2(32)):Int(log2(max_threads)) 
        nx = ny = 32*2^pow
        gmem = Int(AMDGPU.HIP.properties(AMDGPU.device_id!(ID)).totalGlobalMem)
        3*nx*ny*sizeof(dt)> gmem ? break : nothing 
        a = rand(dt, nx, ny); A = ROCArray(a)
        b = rand(dt, nx, ny); B = ROCArray(b)
        c = rand(dt, nx, ny); C = ROCArray(c)
        # thread = (2^pow, 1)
        # block = (nx÷thread[1], ny)
        t_it = BenchmarkTools.@belapsed memcpy!($A, $B, $C, $bench_backend)
        T_tot = 3*1/1024^3*nx*ny*sizeof(dt)/t_it
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