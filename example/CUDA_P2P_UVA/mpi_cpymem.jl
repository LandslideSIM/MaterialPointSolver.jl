using MPI
using CUDA
using Printf

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n    = 1024
    iter = 5
    rstt = zeros(iter)

    CUDA.device!(rank)

    if rank == 0
        recv_buffer = CUDA.zeros(n, n, n)
    elseif rank == 1
        send_buffer = CUDA.ones(n, n, n)
    end

    CUDA.synchronize()
    MPI.Barrier(comm)

    for i in 1:iter
        if rank == 0
            status = MPI.Irecv!(recv_buffer, comm, source=1, tag=666)
            MPI.Wait(status)
        elseif rank == 1
            tic = MPI.Wtime()
            MPI.Isend(send_buffer, comm, dest=0, tag=666)
        end

        CUDA.synchronize()
        MPI.Barrier(comm)

        if rank == 1
            toc = MPI.Wtime()
            rstt[i] = toc - tic
        end
    end

    if rank == 1
        content = @sprintf("%.2e", sum(rstt[2:end])/(iter-1))
        @info "MPI copy time: $content s" 
    end

    MPI.Barrier(comm)
end

main()