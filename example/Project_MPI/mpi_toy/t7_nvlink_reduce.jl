using MPI
using CUDA
using NVTX

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    CUDA.device!(rank)
    n = 102400000
    # first time run to remove compilation time ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↑↑
    data1 = CUDA.rand(n, 3)
    data2 = CUDA.rand(n, 3)
    data3 = CUDA.rand(n)
    
    CUDA.synchronize()
    MPI.Barrier(comm)
    
    MPI.Allreduce!(data1, MPI.SUM, comm)
    MPI.Allreduce!(data2, MPI.SUM, comm)
    MPI.Allreduce!(data3, MPI.SUM, comm)

    CUDA.synchronize()
    MPI.Barrier(comm)
    # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

    NVTX.@range "prepare datasets" begin
        data1 = CUDA.rand(n, 3)
        data2 = CUDA.rand(n, 3)
        data3 = CUDA.rand(n)
        CUDA.synchronize()
        MPI.Barrier(comm)
    end

    NVTX.@range "reduce on data1" MPI.Allreduce!(data1, MPI.SUM, comm)
    NVTX.@range "reduce on data2" MPI.Allreduce!(data2, MPI.SUM, comm)
    NVTX.@range "reduce on data3" MPI.Allreduce!(data3, MPI.SUM, comm)

    CUDA.synchronize()
    MPI.Barrier(comm)
end

main()