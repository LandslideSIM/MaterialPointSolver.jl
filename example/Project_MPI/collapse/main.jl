#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : main.jl                                                                    |
|  Description: Multi-device 3D collapse model                                             |
|               (available MPI process: 1, 2, 3, 4, 5, 6, 7, 8)                            |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  References : Message passing interface and multi-GPU computing, Emmanuel Wyser, Yury    |
|               Alkhimenkov, Michel Jaboyedoff & Yury Y. Podladchikov                      |
+==========================================================================================#

using MPI
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
        init_grid_range_x   = round.([rank*0.05-init_grid_space_x*12, (rank+1)*0.05+init_grid_space_x*12], digits=4),
        init_grid_range_y   = round.([         -init_grid_space_y*12,          0.55+init_grid_space_y*12], digits=4),
        init_grid_range_z   = round.([         -init_grid_space_z*12,          0.12+init_grid_space_z*12], digits=4),
        init_mp_range_x     = round.([rank*0.05+init_mp_offset      , (rank+1)*0.05-init_mp_offset], digits=6),
        init_mp_range_y     = round.([     0.00+init_mp_offset      ,          0.20-init_mp_offset], digits=6),
        init_mp_range_z     = round.([     0.00+init_mp_offset      ,          0.10-init_mp_offset], digits=6),
        init_mp_space_xyz   = init_mp_space_xyz,
        init_grid_space_xyz = space_xyz,
        rank_size           = gpu_counts
    )
    gpu_grid, gpu_mp, gpu_pts_attr, gpu_bc = host2device(args, grid, mp, pts_attr, bc)
    FP = typeof(grid).parameters[2]
    if gpu_counts != 1
        if rank == 0
            halo_left  = (rank + 1) * 0.05 - space_xyz * 12
            halo_right = (rank + 1) * 0.05 + space_xyz * 12
            halo_tmp_idx = Int32.(findall(i -> halo_left ≤ grid.pos[i, 1] ≤ halo_right, 
                1:grid.node_num))
            tmp = hcat(grid.pos[halo_tmp_idx, :], collect(1:1:length(halo_tmp_idx)))
            sortid = sortslices(tmp, dims=1)[:, end] .|> Int32
            halo_idx2 = CuArray(halo_tmp_idx[sortid])
            send_buff2_Ms = CUDA.zeros(FP, length(halo_idx2))
            recv_buff2_Ms = CUDA.zeros(FP, length(halo_idx2))
            send_buff2_Ps = CUDA.zeros(FP, length(halo_idx2), 3)
            recv_buff2_Ps = CUDA.zeros(FP, length(halo_idx2), 3)
            send_buff2_Fs = CUDA.zeros(FP, length(halo_idx2), 3)
            recv_buff2_Fs = CUDA.zeros(FP, length(halo_idx2), 3)
            open(joinpath(@__DIR__, "assets/halo_dev_$(rank)_data2.txt"), "w") do io
                writedlm(io, grid.pos[halo_tmp_idx[sortid], :])
            end
        elseif rank == gpu_counts-1
            halo_left  = rank * 0.05 - space_xyz * 12
            halo_right = rank * 0.05 + space_xyz * 12
            halo_tmp_idx = Int32.(findall(i -> halo_left ≤ grid.pos[i, 1] ≤ halo_right, 
                1:grid.node_num))
            tmp = hcat(grid.pos[halo_tmp_idx, :], collect(1:1:length(halo_tmp_idx)))
            sortid = sortslices(tmp, dims=1)[:, end] .|> Int32
            halo_idx1 = CuArray(halo_tmp_idx[sortid])
            send_buff1_Ms = CUDA.zeros(FP, length(halo_idx1))
            recv_buff1_Ms = CUDA.zeros(FP, length(halo_idx1))
            send_buff1_Ps = CUDA.zeros(FP, length(halo_idx1), 3)
            recv_buff1_Ps = CUDA.zeros(FP, length(halo_idx1), 3)
            send_buff1_Fs = CUDA.zeros(FP, length(halo_idx1), 3)
            recv_buff1_Fs = CUDA.zeros(FP, length(halo_idx1), 3)
            open(joinpath(@__DIR__, "assets/halo_dev_$(rank)_data1.txt"), "w") do io
                writedlm(io, grid.pos[halo_tmp_idx[sortid], :])
            end
        else
            halo_left1  = rank * 0.05 - space_xyz * 12
            halo_right1 = rank * 0.05 + space_xyz * 12
            halo_tmp_idx1 = Int32.(findall(i -> halo_left1 ≤ grid.pos[i, 1] ≤ halo_right1, 
                1:grid.node_num))
            tmp1 = hcat(grid.pos[halo_tmp_idx1, :], collect(1:1:length(halo_tmp_idx1)))
            sortid = sortslices(tmp1, dims=1)[:, end] .|> Int32
            halo_idx1 = CuArray(halo_tmp_idx1[sortid])
            send_buff1_Ms = CUDA.zeros(FP, length(halo_idx1))
            recv_buff1_Ms = CUDA.zeros(FP, length(halo_idx1))
            send_buff1_Ps = CUDA.zeros(FP, length(halo_idx1), 3)
            recv_buff1_Ps = CUDA.zeros(FP, length(halo_idx1), 3)
            send_buff1_Fs = CUDA.zeros(FP, length(halo_idx1), 3)
            recv_buff1_Fs = CUDA.zeros(FP, length(halo_idx1), 3)
            open(joinpath(@__DIR__, "assets/halo_dev_$(rank)_data1.txt"), "w") do io
                writedlm(io, grid.pos[halo_tmp_idx1[sortid], :])
            end
            halo_left2  = (rank + 1) * 0.05 - space_xyz * 12
            halo_right2 = (rank + 1) * 0.05 + space_xyz * 12
            halo_tmp_idx2 = Int32.(findall(i -> halo_left2 ≤ grid.pos[i, 1] ≤ halo_right2, 
                1:grid.node_num))
            tmp2 = hcat(grid.pos[halo_tmp_idx2, :], collect(1:1:length(halo_tmp_idx2)))
            sortid = sortslices(tmp2, dims=1)[:, end] .|> Int32
            halo_idx2 = CuArray(halo_tmp_idx2[sortid])
            send_buff2_Ms = CUDA.zeros(FP, length(halo_idx2))
            recv_buff2_Ms = CUDA.zeros(FP, length(halo_idx2))
            send_buff2_Ps = CUDA.zeros(FP, length(halo_idx2), 3)
            recv_buff2_Ps = CUDA.zeros(FP, length(halo_idx2), 3)
            send_buff2_Fs = CUDA.zeros(FP, length(halo_idx2), 3)
            recv_buff2_Fs = CUDA.zeros(FP, length(halo_idx2), 3)
            open(joinpath(@__DIR__, "assets/halo_dev_$(rank)_data2.txt"), "w") do io
                writedlm(io, grid.pos[halo_tmp_idx2[sortid], :])
            end
        end
    end
    KAsync(getBackend(args))
    MPI.Barrier(comm)
    # check halo data
    if rank != gpu_counts - 1 && gpu_counts != 1
        d1 = rank 
        d2 = rank + 1
        halo_1 = readdlm(joinpath(@__DIR__, "assets/halo_dev_$(d1)_data2.txt"))
        halo_2 = readdlm(joinpath(@__DIR__, "assets/halo_dev_$(d2)_data1.txt"))
        if halo_1 == halo_2
            @info "halo data matched dev_$(d1) ⇄ dev_$(d2)"
        else 
            error("Wrong halo data on dev_$(d1) and dev_$(d2)")
        end
    end
    MPI.Barrier(comm)
    #↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑



    # Start the simulation ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    rank == 0 ? tic = MPI.Wtime() : nothing # initialize the timer
    Ti  = FP(0)
    ΔT  = args.ΔT
    dev = getBackend(args)
    if gpu_counts != 1
        if rank == 0
            while Ti < args.Ttol
                G = Ti < args.Te ? (args.gravity / args.Te * Ti) : args.gravity
                resetgridstatus_OS!(dev)(ndrange=gpu_grid.node_num, gpu_grid)
                resetmpstatus_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp, Val(args.basis))
                P2G_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp, G)
    
                # todo update global node mass, momentum, and force
                fill_halo1!(CUDABackend())(ndrange=length(halo_idx2),
                    halo_idx2, send_buff2_Ms, send_buff2_Ps, send_buff2_Fs, gpu_grid)
                KAsync(getBackend(args))    
                sreq1 = MPI.Isend( send_buff2_Ms, comm; dest  =rank+1, tag=(ROFF* rank   +ROFF/2)+1 |> Int)
                sreq2 = MPI.Isend( send_buff2_Ps, comm; dest  =rank+1, tag=(ROFF* rank   +ROFF/2)+2 |> Int)
                sreq3 = MPI.Isend( send_buff2_Fs, comm; dest  =rank+1, tag=(ROFF* rank   +ROFF/2)+3 |> Int)
                rreq1 = MPI.Irecv!(recv_buff2_Ms, comm; source=rank+1, tag=(ROFF*(rank+1)+ROFF/2)-1 |> Int)
                rreq2 = MPI.Irecv!(recv_buff2_Ps, comm; source=rank+1, tag=(ROFF*(rank+1)+ROFF/2)-2 |> Int)
                rreq3 = MPI.Irecv!(recv_buff2_Fs, comm; source=rank+1, tag=(ROFF*(rank+1)+ROFF/2)-3 |> Int)
                MPI.Waitall([rreq1, rreq2, rreq3, sreq1, sreq2, sreq3])
                KAsync(getBackend(args))
                update_halo1!(CUDABackend())(ndrange=length(halo_idx2),
                    halo_idx2, recv_buff2_Ms, recv_buff2_Ps, recv_buff2_Fs, gpu_grid)
                # mpi communication end
                
                solvegrid_OS!(dev)(ndrange=gpu_grid.node_num, gpu_grid, gpu_bc, args.ΔT, args.ζs)
                doublemapping1_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp, gpu_pts_attr, 
                    args.ΔT, args.FLIP, args.PIC)
                doublemapping2_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp)
    
                # todo update global node momentum
                fill_halo2!(CUDABackend())(ndrange=length(halo_idx2),
                    halo_idx2, send_buff2_Ps, gpu_grid)
                KAsync(getBackend(args))
                sreq = MPI.Isend( send_buff2_Ps, comm; dest  =1, tag=(ROFF* rank   +ROFF/2)+4 |> Int)
                rreq = MPI.Irecv!(recv_buff2_Ps, comm; source=1, tag=(ROFF*(rank+1)+ROFF/2)-4 |> Int)
                MPI.Waitall([rreq, sreq])
                KAsync(getBackend(args))
                update_halo2!(CUDABackend())(ndrange=length(halo_idx2),
                    halo_idx2, recv_buff2_Ps, gpu_grid)
                # mpi communication end
                
                doublemapping3_OS!(dev)(ndrange=gpu_grid.node_num, gpu_grid, gpu_bc, args.ΔT)
                G2P_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp)
                liE!(dev)(ndrange=gpu_mp.num, gpu_mp, gpu_pts_attr)
                if Ti≥args.Te
                    dpP!(dev)(ndrange=gpu_mp.num, gpu_mp, gpu_pts_attr)
                end
                if args.MVL == true
                    vollock1_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp)
                    vollock2_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp)
                end
                Ti += args.ΔT
                args.iter_num += 1
            end
        elseif rank == gpu_counts - 1
            while Ti < args.Ttol
                G = Ti < args.Te ? (args.gravity / args.Te * Ti) : args.gravity
                resetgridstatus_OS!(dev)(ndrange=gpu_grid.node_num, gpu_grid)
                resetmpstatus_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp, Val(args.basis))
                P2G_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp, G)
    
                # todo update global node mass, momentum, and force
                fill_halo1!(CUDABackend())(ndrange=length(halo_idx1),
                    halo_idx1, send_buff1_Ms, send_buff1_Ps, send_buff1_Fs, gpu_grid)
                KAsync(getBackend(args))    
                sreq1 = MPI.Isend( send_buff1_Ms, comm; dest  =rank-1, tag=(ROFF* rank   +ROFF/2)-1 |> Int)
                sreq2 = MPI.Isend( send_buff1_Ps, comm; dest  =rank-1, tag=(ROFF* rank   +ROFF/2)-2 |> Int)
                sreq3 = MPI.Isend( send_buff1_Fs, comm; dest  =rank-1, tag=(ROFF* rank   +ROFF/2)-3 |> Int)
                rreq1 = MPI.Irecv!(recv_buff1_Ms, comm; source=rank-1, tag=(ROFF*(rank-1)+ROFF/2)+1 |> Int)
                rreq2 = MPI.Irecv!(recv_buff1_Ps, comm; source=rank-1, tag=(ROFF*(rank-1)+ROFF/2)+2 |> Int)
                rreq3 = MPI.Irecv!(recv_buff1_Fs, comm; source=rank-1, tag=(ROFF*(rank-1)+ROFF/2)+3 |> Int)
                MPI.Waitall([rreq1, rreq2, rreq3, sreq1, sreq2, sreq3])
                KAsync(getBackend(args))
                update_halo1!(CUDABackend())(ndrange=length(halo_idx1),
                    halo_idx1, recv_buff1_Ms, recv_buff1_Ps, recv_buff1_Fs, gpu_grid)
                # mpi communication end
                
                solvegrid_OS!(dev)(ndrange=gpu_grid.node_num, gpu_grid, gpu_bc, args.ΔT, args.ζs)
                doublemapping1_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp, gpu_pts_attr, 
                    args.ΔT, args.FLIP, args.PIC)
                doublemapping2_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp)
    
                # todo update global node momentum
                fill_halo2!(CUDABackend())(ndrange=length(halo_idx1),
                    halo_idx1, send_buff1_Ps, gpu_grid)
                KAsync(getBackend(args))
                sreq = MPI.Isend( send_buff1_Ps, comm; dest  =rank-1, tag=(ROFF* rank   +ROFF/2)-4 |> Int)
                rreq = MPI.Irecv!(recv_buff1_Ps, comm; source=rank-1, tag=(ROFF*(rank-1)+ROFF/2)+4 |> Int)
                MPI.Waitall([rreq, sreq])
                KAsync(getBackend(args))
                update_halo2!(CUDABackend())(ndrange=length(halo_idx1),
                    halo_idx1, recv_buff1_Ps, gpu_grid)
                # mpi communication end
                
                doublemapping3_OS!(dev)(ndrange=gpu_grid.node_num, gpu_grid, gpu_bc, args.ΔT)
                G2P_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp)
                liE!(dev)(ndrange=gpu_mp.num, gpu_mp, gpu_pts_attr)
                if Ti≥args.Te
                    dpP!(dev)(ndrange=gpu_mp.num, gpu_mp, gpu_pts_attr)
                end
                if args.MVL == true
                    vollock1_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp)
                    vollock2_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp)
                end
                Ti += args.ΔT
                args.iter_num += 1
            end
        else
            while Ti < args.Ttol
                G = Ti < args.Te ? (args.gravity / args.Te * Ti) : args.gravity
                resetgridstatus_OS!(dev)(ndrange=gpu_grid.node_num, gpu_grid)
                resetmpstatus_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp, Val(args.basis))
                P2G_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp, G)
    
                # todo update global node mass, momentum, and force
                fill_halo1!(CUDABackend())(ndrange=length(halo_idx1),
                    halo_idx1, send_buff1_Ms, send_buff1_Ps, send_buff1_Fs, gpu_grid)
                fill_halo1!(CUDABackend())(ndrange=length(halo_idx2),
                    halo_idx2, send_buff2_Ms, send_buff2_Ps, send_buff2_Fs, gpu_grid)
                KAsync(getBackend(args))    
                sreq1 = MPI.Isend( send_buff1_Ms, comm; dest  =rank-1, tag=(ROFF* rank   +ROFF/2)-1 |> Int)
                sreq2 = MPI.Isend( send_buff1_Ps, comm; dest  =rank-1, tag=(ROFF* rank   +ROFF/2)-2 |> Int)
                sreq3 = MPI.Isend( send_buff1_Fs, comm; dest  =rank-1, tag=(ROFF* rank   +ROFF/2)-3 |> Int)
                sreq4 = MPI.Isend( send_buff2_Ms, comm; dest  =rank+1, tag=(ROFF* rank   +ROFF/2)+1 |> Int)
                sreq5 = MPI.Isend( send_buff2_Ps, comm; dest  =rank+1, tag=(ROFF* rank   +ROFF/2)+2 |> Int)
                sreq6 = MPI.Isend( send_buff2_Fs, comm; dest  =rank+1, tag=(ROFF* rank   +ROFF/2)+3 |> Int)
                rreq1 = MPI.Irecv!(recv_buff1_Ms, comm; source=rank-1, tag=(ROFF*(rank-1)+ROFF/2)+1 |> Int)
                rreq2 = MPI.Irecv!(recv_buff1_Ps, comm; source=rank-1, tag=(ROFF*(rank-1)+ROFF/2)+2 |> Int)
                rreq3 = MPI.Irecv!(recv_buff1_Fs, comm; source=rank-1, tag=(ROFF*(rank-1)+ROFF/2)+3 |> Int)
                rreq4 = MPI.Irecv!(recv_buff2_Ms, comm; source=rank+1, tag=(ROFF*(rank+1)+ROFF/2)-1 |> Int)
                rreq5 = MPI.Irecv!(recv_buff2_Ps, comm; source=rank+1, tag=(ROFF*(rank+1)+ROFF/2)-2 |> Int)
                rreq6 = MPI.Irecv!(recv_buff2_Fs, comm; source=rank+1, tag=(ROFF*(rank+1)+ROFF/2)-3 |> Int)
                MPI.Waitall([rreq1, rreq2, rreq3, rreq4, rreq5, rreq6,
                             sreq1, sreq2, sreq3, sreq4, sreq5, sreq6])
                KAsync(getBackend(args))
                update_halo1!(CUDABackend())(ndrange=length(halo_idx1),
                    halo_idx1, recv_buff1_Ms, recv_buff1_Ps, recv_buff1_Fs, gpu_grid)
                update_halo1!(CUDABackend())(ndrange=length(halo_idx2),
                    halo_idx2, recv_buff2_Ms, recv_buff2_Ps, recv_buff2_Fs, gpu_grid)
                # mpi communication end
                
                solvegrid_OS!(dev)(ndrange=grid.node_num, gpu_grid, gpu_bc, args.ΔT, args.ζs)
                doublemapping1_OS!(dev)(ndrange=mp.num, gpu_grid, gpu_mp, gpu_pts_attr, 
                    args.ΔT, args.FLIP, args.PIC)
                doublemapping2_OS!(dev)(ndrange=mp.num, gpu_grid, gpu_mp)
    
                # todo update global node momentum
                fill_halo2!(CUDABackend())(ndrange=length(halo_idx1),
                    halo_idx1, send_buff1_Ps, gpu_grid)
                fill_halo2!(CUDABackend())(ndrange=length(halo_idx2),
                    halo_idx2, send_buff2_Ps, gpu_grid)
                KAsync(getBackend(args))
                sreq1 = MPI.Isend( send_buff1_Ps, comm; dest  =rank-1, tag=(ROFF* rank   +ROFF/2)-4 |> Int)
                sreq2 = MPI.Isend( send_buff2_Ps, comm; dest  =rank+1, tag=(ROFF* rank   +ROFF/2)+4 |> Int)
                rreq1 = MPI.Irecv!(recv_buff1_Ps, comm; source=rank-1, tag=(ROFF*(rank-1)+ROFF/2)+4 |> Int)
                rreq2 = MPI.Irecv!(recv_buff2_Ps, comm; source=rank+1, tag=(ROFF*(rank+1)+ROFF/2)-4 |> Int)
                MPI.Waitall([rreq1, rreq2, sreq1, sreq2])
                KAsync(getBackend(args))
                update_halo2!(CUDABackend())(ndrange=length(halo_idx1),
                    halo_idx1, recv_buff1_Ps, gpu_grid)
                update_halo2!(CUDABackend())(ndrange=length(halo_idx2),
                    halo_idx2, recv_buff2_Ps, gpu_grid)
                # mpi communication end
                
                doublemapping3_OS!(dev)(ndrange=grid.node_num, gpu_grid, gpu_bc, args.ΔT)
                G2P_OS!(dev)(ndrange=mp.num, gpu_grid, gpu_mp)
                liE!(dev)(ndrange=mp.num, gpu_mp, gpu_pts_attr)
                if Ti≥args.Te
                    dpP!(dev)(ndrange=mp.num, gpu_mp, gpu_pts_attr)
                end
                if args.MVL == true
                    vollock1_OS!(dev)(ndrange=mp.num, gpu_grid, gpu_mp)
                    vollock2_OS!(dev)(ndrange=mp.num, gpu_grid, gpu_mp)
                end
                Ti += args.ΔT
                args.iter_num += 1
            end
        end
    else
        while Ti < args.Ttol
            G = Ti < args.Te ? args.gravity / args.Te * Ti : args.gravity
            dev = getBackend(args)
            resetgridstatus_OS!(dev)(ndrange=gpu_grid.node_num, gpu_grid)
            resetmpstatus_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp, Val(args.basis))
            P2G_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp, G)
            solvegrid_OS!(dev)(ndrange=gpu_grid.node_num, gpu_grid, gpu_bc, ΔT, args.ζs)
            doublemapping1_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp, gpu_pts_attr, ΔT, args.FLIP, args.PIC)
            doublemapping2_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp)
            doublemapping3_OS!(dev)(ndrange=gpu_grid.node_num, gpu_grid, gpu_bc, ΔT)
            G2P_OS!(dev)(ndrange=gpu_mp.num, gpu_grid, gpu_mp)
            liE!(dev)(ndrange=gpu_mp.num, gpu_mp, gpu_pts_attr)
            if Ti >= args.Te
                dpP!(dev)(ndrange=gpu_mp.num, gpu_mp, gpu_pts_attr)
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
    space_list = [0.0025]
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