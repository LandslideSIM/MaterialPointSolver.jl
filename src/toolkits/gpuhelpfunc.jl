#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : gpuhelpfunc.jl                                                             |
|  Description: GPU helper functions                                                       |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. host2device  [2D & 3D]                                                  |
|               2. device2host! [2D & 3D]                                                  |
|               3. device2hostinfo!                                                        |
|               3. clean_gpu!   [2D & 3D]                                                  |
|               4. CUDAmemorycopy_KP!                                                      |
|               5. CUDAbandwidth                                                           |
|               6. getBackend                                                              |
|               7. getDArray                                                               |
+==========================================================================================#

"""
    host2device(args::MODELARGS, grid::GRID, mp::PARTICLE, pts_attr::PROPERTY, bc::BOUNDARY)

Description:
---
Transfer data from host to device.
"""
@views function host2device(args::MODELARGS, grid::GRID, mp::PARTICLE, pts_attr::PROPERTY, bc::BOUNDARY)
    T1 = typeof(grid).parameters[1]
    T2 = typeof(grid).parameters[2]
    T3 = getDArray(args)
    if typeof(grid)<:Grid2D
        gpu_grid =      GPUGrid2D{T1, T2, T3, T3, T3    }(
            [getfield(grid, f) for f in fieldnames(     Grid2D)]...)
        gpu_mp   =  GPUParticle2D{T1, T2, T3, T3, T3, T3}(
            [getfield(mp  , f) for f in fieldnames( Particle2D)]...)
        gpu_pts_attr = GPUParticleProperty{T1, T2, T3, T3}(
            [getfield(pts_attr, f) for f in fieldnames(ParticleProperty)]...)
        gpu_bc   = GPUVBoundary2D{T1, T2, T3, T3        }(
            [getfield(bc  , f) for f in fieldnames(VBoundary2D)]...)
    elseif typeof(grid)<:Grid3D
        gpu_grid =      GPUGrid3D{T1, T2, T3, T3, T3    }(
            [getfield(grid, f) for f in fieldnames(     Grid3D)]...)
        gpu_mp   =  GPUParticle3D{T1, T2, T3, T3, T3, T3}(
            [getfield(mp  , f) for f in fieldnames( Particle3D)]...)
        gpu_pts_attr = GPUParticleProperty{T1, T2, T3, T3}(
            [getfield(pts_attr, f) for f in fieldnames(ParticleProperty)]...)
        gpu_bc   = GPUVBoundary3D{T1, T2, T3, T3        }(
            [getfield(bc  , f) for f in fieldnames(VBoundary3D)]...)
    end
    datasize = Base.summarysize(grid)    +Base.summarysize(mp)+
               Base.summarysize(pts_attr)+Base.summarysize(bc)
    outprint = @sprintf("%.1f", datasize/1024^3)
    if args.device==:CUDA
        dev_id = CUDA.device().handle
    elseif args.device==:ROCm
        dev_id = AMDGPU.device().device_id
    end
    content = "host [≈ $(outprint) GiB] → device $(dev_id) [$(args.device)]"
    println("\e[1;32m[▲ I/O:\e[0m \e[0;32m$(content)\e[0m")
    return gpu_grid, gpu_mp, gpu_pts_attr, gpu_bc
end

"""
    device2host!(args::MODELARGS, mp::PARTICLE, gpu_mp::GPUPARTICLE)

Description:
---
Transfer data from device to host. (2D & 3D)
"""
function device2host!(args::MODELARGS, mp::PARTICLE, gpu_mp::GPUPARTICLE)
    copyto!(mp.σm   , gpu_mp.σm   )
    copyto!(mp.Vs   , gpu_mp.Vs   )
    copyto!(mp.pos  , gpu_mp.pos  )
    copyto!(mp.vol  , gpu_mp.vol  )
    copyto!(mp.σij  , gpu_mp.σij  )
    copyto!(mp.epK  , gpu_mp.epK  )
    copyto!(mp.epII , gpu_mp.epII )
    copyto!(mp.ϵij_s, gpu_mp.ϵij_s)
    copyto!(mp.Ms   , gpu_mp.Ms   )
    args.coupling==:TS ? (
        copyto!(mp.Mw      , gpu_mp.Mw      );
        copyto!(mp.Vw      , gpu_mp.Vw      );
        copyto!(mp.σw      , gpu_mp.σw      );
        copyto!(mp.ϵij_w   , gpu_mp.ϵij_w   );
        copyto!(mp.porosity, gpu_mp.porosity);
    ) : nothing
    return nothing
end

function device2hostinfo!(args::MODELARGS, mp::PARTICLE, gpu_mp::GPUPARTICLE)
    device2host!(args, mp, gpu_mp)
    if args.device==:CUDA
        dev_id = CUDA.device().handle
    elseif args.device==:ROCm
        dev_id = AMDGPU.device().device_id
    end
    content = "device $(dev_id) [$(args.device)] → host"
    println("\e[1;31m[▼ I/O:\e[0m \e[0;31m$(content)\e[0m")
    return nothing
end

""" 
    clean_gpu!(args::MODELARGS, gpu_grid::GPUGRID, gpu_mp::GPUPARTICLE, gpu_pts_attr::GPUPARTICLEPROPERTY, gpu_bc::GPUBOUNDARY)

Description:
---
Clean the gpu memory.
"""
function clean_gpu!(args::MODELARGS, gpu_grid::GPUGRID, gpu_mp::GPUPARTICLE, gpu_pts_attr::GPUPARTICLEPROPERTY, gpu_bc::GPUBOUNDARY)
    for i in 1:nfields(gpu_grid)
        typeof(getfield(gpu_grid, i))<:AbstractArray ? 
            KernelAbstractions.unsafe_free!(getfield(gpu_grid, i)) : nothing
    end
    for i in 1:nfields(gpu_mp)
        typeof(getfield(gpu_mp, i))<:AbstractArray ?
            KernelAbstractions.unsafe_free!(getfield(gpu_mp, i)) : nothing
    end
    for i in 1:nfields(gpu_pts_attr)
        typeof(getfield(gpu_pts_attr, i))<:AbstractArray ?
            KernelAbstractions.unsafe_free!(getfield(gpu_pts_attr, i)) : nothing
    end
    for i in 1:nfields(gpu_bc)
        typeof(getfield(gpu_bc, i))<:AbstractArray ? 
            KernelAbstractions.unsafe_free!(getfield(gpu_bc, i)) : nothing
    end
    if args.device==:CUDA
        CUDA.reclaim()
        dev_id = CUDA.device().handle
    elseif args.device==:ROCm
        dev_id = AMDGPU.device().device_id
    end
    KernelAbstractions.synchronize(getBackend(args))
    content = "free device $(dev_id) memory"
    println("\e[1;32m[• I/O:\e[0m \e[0;32m$(content)\e[0m")
    return nothing
end

@kernel inbounds=true function memcpy!(A, B, C)
    ix, iy = @index(Global, NTuple)
    A[ix, iy] = B[ix, iy] + C[ix, iy]
end

"""
    GPUbandwidth(devicetype; datatype::Symbol=:FP64, ID::Int=0)

Description:
---
Test the bandwidth of the GPU device. 
 - `devicetype` can be one of `:CUDA` or `:ROCm`.
 - `ID` is `0` by default, which is used to run the test on the specific device.
"""
function GPUbandwidth(devicetype::Symbol; datatype::Symbol=:FP64, ID::Int=0)
    if devicetype==:CUDA
        CUDA.device!(ID)
        dev_backend = CUDABackend()
        max_threads = CUDA.attribute(CUDA.device(), 
            CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
        println("\e[1;36m[ Test:\e[0m device ($(CUDA.device( ).handle)) → $(datatype)")
    elseif devicetype==:ROCm
        ID == 0 ? ID = 1 : nothing
        max_threads = AMDGPU.HIP.properties(AMDGPU.device_id!(ID)).maxThreadsPerBlock
        AMDGPU.device!(AMDGPU.devices()[ID])
        dev_backend = ROCBackend()
        println("\e[1;36m[ Test:\e[0m device ($(AMDGPU.device().device_id)) → $(datatype)")
    end
    dt = datatype==:FP64 ? Float64 : 
         datatype==:FP32 ? Float32 : error("Wrong datatype!") 
    throughputs = Float64[]
    @inbounds for pow = Int(log2(32)):Int(log2(max_threads)) 
        nx = ny = 32*2^pow
        if devicetype==:CUDA
            3*nx*ny*sizeof(dt)>CUDA.available_memory() ? break : nothing
            A = CUDA.rand(dt, nx, ny)
            B = CUDA.rand(dt, nx, ny)
            C = CUDA.rand(dt, nx, ny)
        elseif devicetype==:ROCm
            gmem = Int(AMDGPU.HIP.properties(AMDGPU.device_id!(ID)).totalGlobalMem)
            3*nx*ny*sizeof(dt)> gmem ? break : nothing 
            a = rand(dt, nx, ny); A = ROCArray(a)
            b = rand(dt, nx, ny); B = ROCArray(b)
            c = rand(dt, nx, ny); C = ROCArray(c)
        end
        # thread = (2^pow, 1)
        # block = (nx÷thread[1], ny)
        t_it = BenchmarkTools.@belapsed begin
            memcpy!($(dev_backend))(ndrange=$(size(A)), $A, $B, $C)
            KAsync($(dev_backend))
        end
        T_tot = 3*1/1024^3*nx*ny*sizeof(dt)/t_it
        push!(throughputs, T_tot)
        
        # clean gpu memory
        KernelAbstractions.unsafe_free!(A)
        KernelAbstractions.unsafe_free!(B)
        KernelAbstractions.unsafe_free!(C)
        if devicetype==:CUDA
            CUDA.reclaim()
        elseif devicetype==:ROCm
            nothing
        end
    end
    value = round(maximum(throughputs), digits=2)
    @info "MTeff peak on device ($(ID)): $(value) GiB/s"
    return value
end

function getBackend(args::MODELARGS)
    args.device==:CPU    ? backend=         CPU() :
    args.device==:CUDA   ? backend= CUDABackend() :
    args.device==:ROCm   ? backend=  ROCBackend() :
    args.device==:oneAPI ? backend=  oneBackend() :
    args.device==:Metal  ? backend=MetalBackend() : error("Wrong backend device.")
    return backend
end

function getDArray(args::MODELARGS)
    args.device==:CUDA   ? DArray= CuArray :
    args.device==:ROCm   ? DArray=ROCArray :
    args.device==:oneAPI ? DArray=oneArray :
    args.device==:Metal  ? DArray=MtlArray : error("Wrong backend device.")
    return DArray
end