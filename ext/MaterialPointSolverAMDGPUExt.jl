module MaterialPointSolverAMDGPUExt

using BenchmarkTools
using AMDGPU
using KernelAbstractions
using MaterialPointSolver
using Printf

import MaterialPointSolver: host2device, device2host!, clean_device!, Tpeak, getBackend,
    warmup

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
    content = "host [≈ $(outprint) GiB] → device $(dev_id) [:ROCm]"
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
        content = "device $(dev_id) [$(args.device)] → host"
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
    @info "Tpeak on device ($(ID)): $(value) GiB/s"
    return value
end

getBackend(::Val{:ROCm}) = ROCBackend()

function warmup(::Val{:ROCm}; ID::Int=0)
    ID == 0 ? ID = 1 : nothing
    AMDGPU.device!(AMDGPU.devices()[ID])
    rtsdir = joinpath(homedir(), "MaterialPointSolverTEMP_$(ID)/")
    mkpath(rtsdir)
    init_grid_space_x = 1
    init_grid_space_y = 1
    init_grid_range_x = [-5, 5]
    init_grid_range_y = [-5, 5]
    init_mp_in_space  = 2
    init_project_name = "2d_test"
    init_project_path = joinpath(rtsdir, init_project_name)
    init_constitutive = :druckerprager
    init_ζs           = 0
    init_ρs           = 2650
    init_ν            = 0.3
    init_Ks           = 7e5
    init_E            = init_Ks*(3*(1-2*init_ν))
    init_G            = init_E /(2*(1+  init_ν))
    init_T            = 20*0.5*init_grid_space_x/sqrt(init_E/init_ρs)
    init_Te           = 0
    init_ΔT           = 0.5*init_grid_space_x/sqrt(init_E/init_ρs)
    init_step         = floor(init_T/init_ΔT/200) |> Int64
    init_step<10 ? init_step=1 : nothing
    init_σt           = 0
    init_ϕ            = 19.8*π/180
    init_c            = 0
    init_ψ            = 0
    init_NIC          = 16
    init_basis        = :uGIMP
    init_phase        = 1
    init_scheme       = :MUSL
    iInt              = Int64
    iFloat            = Float64

    # parameters setup
    args = Args2D{iInt, iFloat}(
        Ttol         = init_T,
        Te           = init_Te,
        ΔT           = init_ΔT,
        time_step    = :fixed,
        FLIP         = 1.0,
        PIC          = 0.0,
        ζs           = init_ζs,
        project_name = init_project_name,
        project_path = init_project_path,
        constitutive = init_constitutive,
        animation    = false,
        hdf5         = false,
        hdf5_step    = init_step,
        MVL          = true,
        device       = :ROCm,
        coupling     = :OS,
        scheme       = init_scheme,
        basis        = init_basis
    )

    # background grid setup
    grid = Grid2D{iInt, iFloat}(
        NIC      = init_NIC,
        phase    = init_phase,
        range_x1 = init_grid_range_x[1],
        range_x2 = init_grid_range_x[2],
        range_y1 = init_grid_range_y[1],
        range_y2 = init_grid_range_y[2],
        space_x  = init_grid_space_x,
        space_y  = init_grid_space_y
    )

    # material points setup
    range_x = [-2+grid.space_x/init_mp_in_space/2, 2-grid.space_x/init_mp_in_space/2]
    range_y = [-2+grid.space_y/init_mp_in_space/2, 2-grid.space_y/init_mp_in_space/2]
    space_x = grid.space_x/init_mp_in_space
    space_y = grid.space_y/init_mp_in_space
    num_x   = length(range_x[1]:space_x:range_x[2])
    num_y   = length(range_y[1]:space_y:range_y[2])
    x_tmp   = repeat((range_x[1]:space_x:range_x[2])', num_y, 1) |> vec
    y_tmp   = repeat((range_y[1]:space_y:range_y[2]) , 1, num_x) |> vec
    mp_num  = length(x_tmp)
    mp_ρs   = ones(mp_num).*init_ρs
    mp      = Particle2D{iInt, iFloat}(space_x=space_x, space_y=space_y, pos=[x_tmp y_tmp],
        ρs=mp_ρs, NIC=init_NIC, phase=init_phase)

    # particle property setup
    mp_layer   = ones(mp_num)
    mp_ν       = [init_ν]
    mp_E       = [init_E]
    mp_G       = [init_G]
    mp_σt      = [init_σt]
    mp_ϕ       = [init_ϕ]
    mp_c       = [init_c]
    mp_ψ       = [init_ψ]
    mp_Ks      = [init_Ks]
    pts_attr = ParticleProperty{iInt, iFloat}(layer=mp_layer, ν=mp_ν, E=mp_E, G=mp_G, σt=mp_σt, 
        ϕ=mp_ϕ, c=mp_c, ψ=mp_ψ, Ks=mp_Ks)

    # boundary condition nodes index
    vx_idx = zeros(iInt, grid.node_num)
    vy_idx = zeros(iInt, grid.node_num)
    tmp_idx = findall(i->(grid.pos[i, 1]≤-2||grid.pos[i, 1]≥2||
                          grid.pos[i, 2]≤-2), 1:grid.node_num)
    tmp_idy = findall(i->(grid.pos[i, 2]≤0), 1:grid.node_num)
    vx_idx[tmp_idx] .= 1
    vy_idx[tmp_idy] .= 1
    bc = VBoundary2D{iInt, iFloat}(
        Vx_s_Idx = vx_idx,
        Vx_s_Val = zeros(grid.node_num),
        Vy_s_Idx = vy_idx,
        Vy_s_Val = zeros(grid.node_num)
    )

    # MPM solver
    @info "code warm-up, wait a moment 🔥"
    @suppress begin
        materialpointsolver!(args, grid, mp, pts_attr, bc)
    end
    rm(rtsdir, recursive=true, force=true)
    return nothing
end

end