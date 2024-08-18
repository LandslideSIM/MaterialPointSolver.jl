using MPI
using Printf
using CUDA

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n = 1200
    nbench = 10
    rst = zeros(nbench)
    
    if rank == 0
        CUDA.device!(0)
        send_mesg = CUDA.rand(n, n, n)
    elseif rank == 1
        CUDA.device!(3)
        recv_mesg = CUDA.zeros(n, n, n)
    end

    CUDA.synchronize()

    for i in 1:nbench
        if rank == 0
            tic = MPI.Wtime()
            MPI.Isend(send_mesg, comm; dest=1, tag=666)
        elseif rank == 1
            rreq = MPI.Irecv!(recv_mesg, comm; source=0, tag=666)
            MPI.Wait(rreq)
        end
        CUDA.synchronize()
        if rank == 0
            toc = MPI.Wtime()
            rst[i] = toc - tic
        end
    end

    if rank == 0
        average = sum(rst[2:end]) / (nbench - 1)    
        content = @sprintf("Elapsed time: %.2es", average)
        @info content
    end

    MPI.Barrier(comm)
    return nothing
end

main()