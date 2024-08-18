using MPI
using CUDA
using Printf

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n = (10240000*5, 3)
    nbench = 5
    rst = zeros(nbench)

    CUDA.device!(rank)

    data1 = CUDA.ones(Float32, n)
    data2 = CUDA.ones(Float32, n)
    data3 = CUDA.ones(Float32, n)

    CUDA.synchronize()
    MPI.Barrier(comm)
    @info "data is ready on device $(rank)"

    for i in 1:nbench
        if rank == 0
            tic = MPI.Wtime()
        end
        MPI.Allreduce!(data1, MPI.SUM, comm)
        MPI.Allreduce!(data2, MPI.SUM, comm)
        MPI.Allreduce!(data3, MPI.SUM, comm)
        CUDA.synchronize()
        MPI.Barrier(comm)

        if rank == 0
            toc = MPI.Wtime()
            rst[i] = toc - tic
        end
    end

    @info "rank $rank: $(unique(Array(data1)))"
    MPI.Barrier(comm)

    if rank == 0
        content = @sprintf("%.2e", sum(rst[2:end])/(nbench-1))
        @info "execution time: $content s"
    end

    MPI.Barrier(comm)
    return nothing
end

main()