#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : main.jl                                                                    |
|  Description: Multi-device slope model (available MPI process: 1, 2, 3, 4, 5, 6, 7, 8)   |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  References : Message passing interface and multi-GPU computing, Emmanuel Wyser, Yury    |
|               Alkhimenkov, Michel Jaboyedoff & Yury Y. Podladchikov                      |
+==========================================================================================#

using MPI
using NVTX

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
gpu_counts = MPI.Comm_size(comm)

using CUDA
using DelimitedFiles
using MaterialPointSolver
using Printf
using Random
using CUDA
using KernelAbstractions
using WriteVTK
using StatsBase
using CairoMakie

include(joinpath(@__DIR__, "func.jl"))
include(joinpath(@__DIR__, "scheme.jl"))
const rtsdir = joinpath(homedir(), "Workbench/outputs")
const ROFF = 1000 # rank tag offset

function run_mpm(comm, rank, gpu_counts, space_xyz, rst)
    # Initialize devices ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    dev_id = rank
    CUDA.device!(dev_id)
    MPI.Barrier(comm)
    @info "initialize gpu device $dev_id"
    init_mp_in_space  = 2
    init_grid_space_x = space_xyz
    init_grid_space_y = space_xyz
    init_grid_space_z = space_xyz
    init_mp_space_xyz = space_xyz / init_mp_in_space
    init_mp_offset    = init_mp_space_xyz / 2
    #↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑



    # Initialize the model ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    args, grid, mp, pts_attr, bc = init_model(;
        init_project_name   = "GPUs_part$(rank + 1)",
        init_grid_range_x   = round.([       -init_grid_space_x*3,            60+init_grid_space_x*3], digits=4),
        init_grid_range_y   = round.([       -init_grid_space_y*3, gpu_counts*12+init_grid_space_y*3], digits=4),
        init_grid_range_z   = round.([       -init_grid_space_z*3,            12+init_grid_space_z*3], digits=4),
        init_mp_range_x     = round.([        init_mp_offset     ,            60-init_mp_offset     ], digits=6),
        init_mp_range_y     = round.([rank*12+init_mp_offset     ,   (rank+1)*12-init_mp_offset     ], digits=6),
        init_mp_range_z     = round.([        init_mp_offset     ,            12-init_mp_offset     ], digits=6),
        init_mp_space_xyz   = init_mp_space_xyz,
        init_grid_space_xyz = space_xyz,
        rank_size           = gpu_counts
    )\
    gpu_grid, gpu_mp, gpu_pts_attr, gpu_bc = host2device(args, grid, mp, pts_attr, bc)
    FP = typeof(grid).parameters[2]
    KAsync(getBackend(args))
    MPI.Barrier(comm)
    #↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑



    # Start the simulation ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    rank == 0 ? tic = MPI.Wtime() : nothing # initialize the timer
    Ti  = FP(0)
    ΔT  = args.ΔT
    dev = getBackend(args)
    if gpu_counts != 1
        while Ti < args.Ttol
            G = Ti < args.Te ? (args.gravity / args.Te * Ti) : args.gravity
            NVTX.@range "Grid Reset"     testresetgridstatus_OS!(dev)(ndrange=gpu_grid.node_num, gpu_grid)
            NVTX.@range "Particle Reset" testresetmpstatus_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp, Val(args.basis))
            NVTX.@range "P2G"            testP2G_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp, G)

            NVTX.@range "Halo Preparation & Communication 1" begin
            # todo update global node mass, momentum, and force
            KAsync(getBackend(args))    
            MPI.Allreduce!(gpu_grid.Ms, MPI.SUM, comm)
            MPI.Allreduce!(gpu_grid.Ps, MPI.SUM, comm)
            MPI.Allreduce!(gpu_grid.Fs, MPI.SUM, comm)
            KAsync(getBackend(args))
            # mpi communication end
            end
            
            NVTX.@range "Solver on Grid"   testsolvegrid_OS!(dev)(ndrange=gpu_grid.node_num, gpu_grid, gpu_bc, args.ΔT, args.ζs)
            NVTX.@range "Double Mapping 1" testdoublemapping1_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp, gpu_pts_attr, 
                args.ΔT, args.FLIP, args.PIC)
            NVTX.@range "Double Mapping 2" testdoublemapping2_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp)

            NVTX.@range "Halo Preparation & Communication 2" begin
            # todo update global node momentum
            KAsync(getBackend(args))
            MPI.Allreduce!(gpu_grid.Ps, MPI.SUM, comm)
            KAsync(getBackend(args))
            # mpi communication end
            end
            
            NVTX.@range "Double Mapping 3" testdoublemapping3_OS!(dev)(ndrange=gpu_grid.node_num, gpu_grid, gpu_bc, args.ΔT)
            NVTX.@range "G2P"              testG2P_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp)
            NVTX.@range "Linear Elastic"   liE!(dev)(ndrange=gpu_mp.num, gpu_mp, gpu_pts_attr)
            if Ti≥args.Te
                NVTX.@range "Drucker-Prager" testdpP!(dev)(ndrange=gpu_mp.num, gpu_mp, gpu_pts_attr)
            end
            if args.MVL == true
                NVTX.@range "Mitigate Volume Locking 1" vollock1_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp)
                NVTX.@range "Mitigate Volume Locking 2" vollock2_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp)
            end
            Ti += args.ΔT
            args.iter_num += 1
        end
    else
        while Ti < args.Ttol
            G = Ti < args.Te ? args.gravity / args.Te * Ti : args.gravity
            dev = getBackend(args)
            testresetgridstatus_OS!(dev)(ndrange=gpu_grid.node_num, gpu_grid)
            testresetmpstatus_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp, Val(args.basis))
            testP2G_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp, G)
            testsolvegrid_OS!(dev)(ndrange=gpu_grid.node_num, gpu_grid, gpu_bc, ΔT, args.ζs)
            testdoublemapping1_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp, gpu_pts_attr, ΔT, args.FLIP, args.PIC)
            testdoublemapping2_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp)
            testdoublemapping3_OS!(dev)(ndrange=gpu_grid.node_num, gpu_grid, gpu_bc, ΔT)
            testG2P_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp)
            liE!(dev)(ndrange=gpu_mp.num, gpu_mp, gpu_pts_attr)
            if Ti >= args.Te
                testdpP!(dev)(ndrange=gpu_mp.num, gpu_mp, gpu_pts_attr)
            end
            if args.MVL == true
                vollock1_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp)
                vollock2_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp)
            end
            Ti += ΔT
            args.iter_num += 1
        end
    end

    KAsync(getBackend(args))
    MPI.Barrier(comm)
    rank == 0 ? toc = MPI.Wtime() : nothing
    #↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑



    # End simulation and post-processing ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    if rank == 0
        print_time = @sprintf("%.2f s", toc-tic)
        @info """overview
        particle number: $(mp.num*gpu_counts)
        execution time : $(print_time)
        """
        rst[1] = mp.num*gpu_counts
        rst[2] = toc-tic
    end

    testdevice2host(gpu_mp, rank, gpu_counts)
    rank == 0 ? post_process(gpu_counts) : nothing
    MPI.Barrier(comm)

    clean_gpu!(args, gpu_grid, gpu_mp, gpu_pts_attr, gpu_bc)
    KAsync(getBackend(args))
    #↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

    MPI.Barrier(comm)
    return time
end

function run_test(comm, rank, gpu_counts)
    MaterialPointSolver.warmup(rank)
    # space         pts 
    # 0.1500 13,786,880
    # 0.1875  7,047,040
    # 0.2000  5,816,520
    # 0.2500  2,982,912
    # 0.30    1,722,560
    # 0.40      727,020 
    # 0.50      370,992
    # space_list = [0.5, 0.4, 0.3, 0.25, 0.2, 0.1875, 0.15]
    space_list = [0.15]
    bench = zeros(length(space_list), 2)
    for i in eachindex(space_list)
        rst = [0.0, 0.0]
        run_mpm(comm, rank, gpu_counts, space_list[i], rst)
        MPI.Barrier(comm)
        rank == 0 ? bench[i, :] = rst : nothing
    end
    if rank == 0
        open(joinpath(@__DIR__, "assets/$(gpu_counts)_GPUs_bench.txt"), "w") do io
            writedlm(io, bench)
        end
    end
end

mkpath(joinpath(@__DIR__, "assets"))
run_test(comm, rank, gpu_counts)