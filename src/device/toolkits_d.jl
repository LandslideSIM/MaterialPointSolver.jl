#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : toolkits_d.jl                                                              |
|  Description: Some extra gpu tools for MPMSolver.jl                                      |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. host2device()  [2D]                                                     |
|               2. host2device()  [3D]                                                     |
|               3. device2host!() [2D & 3D]                                                |
|               4. clean_gpu!()   [2D & 3D]                                                |
|               5. getOccAPI!()                                                            |
|               6. memcopy_KP!()                                                           |
|               7. bandwidth()                                                             |
|               8. cal_mem()                                                               |
+==========================================================================================#

"""
    host2device(grid::GRID, mp::PARTICLE, bc::VBC)

Description:
---
Transfer data from host to device.
"""
@views function host2device(grid::GRID, mp::PARTICLE, bc::VBC)
    T1 = typeof(grid).parameters[1]
    T2 = typeof(grid).parameters[2]
    if typeof(grid)<:Grid2D
        cu_grid =      GPUGrid2D{T1, T2}([getfield(grid, f) for f in fieldnames(     Grid2D)]...)
        cu_mp   =  GPUParticle2D{T1, T2}([getfield(mp  , f) for f in fieldnames( Particle2D)]...)
        cu_bc   = GPUVBoundary2D{T1, T2}([getfield(bc  , f) for f in fieldnames(VBoundary2D)]...)
    elseif typeof(grid)<:Grid3D
        cu_grid =      GPUGrid3D{T1, T2}([getfield(grid, f) for f in fieldnames(     Grid3D)]...)
        cu_mp   =  GPUParticle3D{T1, T2}([getfield(mp  , f) for f in fieldnames( Particle3D)]...)
        cu_bc   = GPUVBoundary3D{T1, T2}([getfield(bc  , f) for f in fieldnames(VBoundary3D)]...)
    end

    datasize = @sprintf("%.1f", cal_mem(cu_grid, cu_mp, cu_bc))   
    gmemstat = Mem.info()./(1024^3)
    effratio = @sprintf("%.1f", 100*((gmemstat[2]-gmemstat[1])/gmemstat[2]))
    desc     = "\e[0;32m[▲ I/O: host [≈$(datasize) GiB] → device [$(effratio)% used]"
    println(desc)
    return cu_grid, cu_mp, cu_bc
end

"""
    device2host!(args::ARGS, mp::PARTICLE, cu_mp::GPUPARTICLE)

Description:
---
Transfer data from device to host. (2D & 3D)
"""
function device2host!(args::ARGS, mp::PARTICLE, cu_mp::GPUPARTICLE)
    copyto!(mp.σm   , cu_mp.σm   )
    copyto!(mp.Vs   , cu_mp.Vs   )
    copyto!(mp.pos  , cu_mp.pos  )
    copyto!(mp.vol  , cu_mp.vol  )
    copyto!(mp.σij  , cu_mp.σij  )
    copyto!(mp.epK  , cu_mp.epK  )
    copyto!(mp.epII , cu_mp.epII )
    copyto!(mp.ϵij_s, cu_mp.ϵij_s)
    copyto!(mp.Ms   , cu_mp.Ms   )
    args.coupling==:TS ? (
        copyto!(mp.Mw      , cu_mp.Mw      );
        copyto!(mp.Vw      , cu_mp.Vw      );
        copyto!(mp.σw      , cu_mp.σw      );
        copyto!(mp.ϵij_w   , cu_mp.ϵij_w   );
        copyto!(mp.porosity, cu_mp.porosity);
    ) : nothing
    return nothing
end

""" 
    clean_gpu!(cu_grid::GPUGRID, cu_mp::GPUPARTICLE)

Description:
---
Clean the gpu memory.
"""
function clean_gpu!(cu_grid::GPUGRID, cu_mp::GPUPARTICLE, cu_bc::GPUVBC)
    for i in 1:nfields(cu_grid)
        typeof(getfield(cu_grid, i))<:CuArray ? 
            CUDA.unsafe_free!(getfield(cu_grid, i)) : nothing
    end
    for i in 1:nfields(cu_mp)
        typeof(getfield(cu_mp, i))<:CuArray ?
            CUDA.unsafe_free!(getfield(cu_mp, i)) : nothing
    end
    for i in 1:nfields(cu_bc)
        typeof(getfield(cu_bc, i))<:CuArray ? CUDA.unsafe_free!(
            getfield(cu_bc, i)) : nothing
    end
    CUDA.reclaim()
    @info "free device memory"
    return nothing
end

function getOccAPI(args   ::       ARGS,
                   cu_grid::    GPUGRID,
                   cu_mp  ::GPUPARTICLE,
                   cu_bc  ::     GPUVBC,
                   ΔT     ::T2) where {T2}
    gravity = args.gravity
    if args.coupling==:OS
        k01_k = @cuda launch=false kernel_OS01!(cu_grid)
        k01_c = launch_configuration(k01_k.fun)
        k02_k = @cuda launch=false kernel_OS02!(cu_grid, cu_mp, Val(args.basis))
        k02_c = launch_configuration(k02_k.fun)
        k03_k = @cuda launch=false kernel_OS03!(cu_grid, cu_mp, gravity)
        k03_c = launch_configuration(k03_k.fun)
        k04_k = @cuda launch=false kernel_OS04!(cu_grid, cu_bc, ΔT, args.ζ)
        k04_c = launch_configuration(k04_k.fun) 
        k05_k = @cuda launch=false kernel_OS05!(cu_grid, cu_mp, ΔT, args.FLIP, args.PIC)
        k05_c = launch_configuration(k05_k.fun)
        k06_k = @cuda launch=false kernel_OS06!(cu_grid, cu_mp)
        k06_c = launch_configuration(k06_k.fun)
        k07_k = @cuda launch=false kernel_OS07!(cu_grid, cu_bc, ΔT)
        k07_c = launch_configuration(k07_k.fun)
        k08_k = @cuda launch=false kernel_OS08!(cu_grid, cu_mp)
        k08_c = launch_configuration(k08_k.fun)
        k09_k = @cuda launch=false kernel_OS09!(cu_grid, cu_mp)
        k09_c = launch_configuration(k09_k.fun)
        k10_k = @cuda launch=false kernel_OS10!(cu_grid, cu_mp)
        k10_c = launch_configuration(k10_k.fun)
    elseif args.coupling==:TS
        k01_k = @cuda launch=false kernel_TS01!(cu_grid)
        k01_c = launch_configuration(k01_k.fun)
        k02_k = @cuda launch=false kernel_TS02!(cu_grid, cu_mp, Val(args.basis))
        k02_c = launch_configuration(k02_k.fun)
        k03_k = @cuda launch=false kernel_TS03!(cu_grid, cu_mp, gravity)
        k03_c = launch_configuration(k03_k.fun)
        k04_k = @cuda launch=false kernel_TS04!(cu_grid, cu_bc, ΔT, args.ζ)
        k04_c = launch_configuration(k04_k.fun)
        k05_k = @cuda launch=false kernel_TS05!(cu_grid, cu_mp, ΔT, args.FLIP, args.PIC)
        k05_c = launch_configuration(k05_k.fun)
        k06_k = @cuda launch=false kernel_TS06!(cu_grid, cu_mp)
        k06_c = launch_configuration(k06_k.fun)
        k07_k = @cuda launch=false kernel_TS07!(cu_grid, cu_bc, ΔT)
        k07_c = launch_configuration(k07_k.fun)
        k08_k = @cuda launch=false kernel_TS08!(cu_grid, cu_mp)
        k08_c = launch_configuration(k08_k.fun)
        k09_k = @cuda launch=false kernel_TS09!(cu_grid, cu_mp)
        k09_c = launch_configuration(k09_k.fun)
        k10_k = @cuda launch=false kernel_TS10!(cu_grid, cu_mp)
        k10_c = launch_configuration(k10_k.fun)
    end
    liE_k = @cuda launch=false liE!(cu_mp)
    liE_c = launch_configuration(liE_k.fun)
    dpP_k = @cuda launch=false dpP!(cu_mp)
    dpP_c = launch_configuration(dpP_k.fun)
    mcP_k = @cuda launch=false mcP!(cu_mp)
    mcP_c = launch_configuration(mcP_k.fun)
    hyE_k = @cuda launch=false hyE!(cu_mp)
    hyE_c = launch_configuration(hyE_k.fun)
    twP_k = @cuda launch=false twP!(cu_mp)
    twP_c = launch_configuration(twP_k.fun)

    @info "CUDA Occupancy API estimate"
    return (k01_t=Int32(k01_c.threads),
            k01_b=Int32(cld(cu_grid.node_num, k01_c.threads)),
            k02_t=Int32(k02_c.threads),
            k02_b=Int32(cld(cu_mp.num, k02_c.threads)),
            k03_t=Int32(k03_c.threads),
            k03_b=Int32(cld(cu_mp.num, k03_c.threads)),
            k04_t=Int32(k04_c.threads),
            k04_b=Int32(cld(cu_grid.node_num, k04_c.threads)),
            k05_t=Int32(k05_c.threads),
            k05_b=Int32(cld(cu_mp.num, k05_c.threads)),
            k06_t=Int32(k06_c.threads),
            k06_b=Int32(cld(cu_mp.num, k06_c.threads)),
            k07_t=Int32(k07_c.threads),
            k07_b=Int32(cld(cu_grid.node_num, k07_c.threads)),
            k08_t=Int32(k08_c.threads),
            k08_b=Int32(cld(cu_mp.num, k08_c.threads)),
            k09_t=Int32(k09_c.threads),
            k09_b=Int32(cld(cu_mp.num, k09_c.threads)),
            k10_t=Int32(k10_c.threads),
            k10_b=Int32(cld(cu_mp.num, k10_c.threads)),
            liE_t=Int32(liE_c.threads),
            liE_b=Int32(cld(cu_mp.num, liE_c.threads)),
            dpP_t=Int32(dpP_c.threads),
            dpP_b=Int32(cld(cu_mp.num, dpP_c.threads)),
            mcP_t=Int32(mcP_c.threads),
            mcP_b=Int32(cld(cu_mp.num, mcP_c.threads)),
            hyE_t=Int32(hyE_c.threads),
            hyE_b=Int32(cld(cu_mp.num, hyE_c.threads)),
            twP_t=Int32(twP_c.threads),
            twP_b=Int32(cld(cu_mp.num, twP_c.threads)))
end

@inbounds function memcopy_KP!(A, B, C, nx, ny)
    ix = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1i32) * blockDim().y + threadIdx().y
    if ix≤nx && iy≤ny
        A[ix,iy] = B[ix,iy] + C[ix, iy]
    end
    return nothing
end

"""
    bandwidth(;deviceid::Int64=0, datatype::Symbol=:double)

Description:
---
Test the bandwidth of the GPU device.
"""
function bandwidth(;deviceid::Int64=0, datatype::Symbol=:double)
    datatype==:double ? pre = Float64 : 
    datatype==:single ? pre = Float32 : error("Wrong datatype!")
    device!(deviceid)
    max_threads  = attribute(device(),CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    throughputs  = Float64[]
    @inbounds for pow = Int(log2(32)):Int(log2(max_threads)) 
        print("\r\e[1;36m[ Test:\e[0m device($(deviceid)) → $(pre)")   
        nx = ny = 32*2^pow
        3*nx*ny*sizeof(pre)>CUDA.available_memory() ? break : nothing
        A       = CUDA.rand(pre, nx, ny)
        B       = CUDA.rand(pre, nx, ny)
        C       = CUDA.rand(pre, nx, ny)
        threads = (2^pow, 1)
        blocks  = (nx÷threads[1], ny)
        t_it    = @belapsed begin 
            @cuda blocks=$blocks threads=$threads memcopy_KP!($A, $B, $C, $nx, $ny)
            synchronize() 
        end
        T_tot = 3*1/1024^3*nx*ny*sizeof(pre)/t_it
        push!(throughputs, T_tot)
        CUDA.unsafe_free!(A)
        CUDA.unsafe_free!(B)
        CUDA.reclaim()
    end
    value = round(maximum(throughputs), digits=2)
    print("\r\e[1;36m[ Info:\e[0m Tpeak on device($(deviceid)): $(value) GB/s")
    return value
end

function cal_mem(cu_grid::GPUGRID, cu_mp::GPUPARTICLE, cu_bc::GPUVBC)
    mem = 0
    for field in fieldnames(typeof(cu_grid))
        array = getfield(cu_grid, field)
        mem += sizeof(array)
    end
    for field in fieldnames(typeof(cu_mp))
        array = getfield(cu_mp, field)
        mem += sizeof(array)
    end
    for field in fieldnames(typeof(cu_bc))
        array = getfield(cu_bc, field)
        mem += sizeof(array)
    end
    return mem/(1024^3)
end