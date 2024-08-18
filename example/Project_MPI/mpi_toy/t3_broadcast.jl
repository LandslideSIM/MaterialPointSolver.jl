using MPI

function broadcast_data(comm, rank)
    # prepare data
    n = 2
    if rank == 0
        data = rand(n, n)
    else
        data = zeros(n, n)
    end
    # MPI broadcast
    MPI.Bcast!(data, 0, comm)
    # print out the data
    @info "<$rank>: received $data"#$(size(data))"
end

function main()
    # MPI configuration
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm) # start from 0
    tic  = MPI.Wtime()
    # run the code
    broadcast_data(comm, rank)
    # finalize
    MPI.Barrier(comm)
    # print out the time
    if rank == 0 
        toc = MPI.Wtime()
        rst = round(toc-tic, digits=2)
        @info "<$rank>: total time $(rst) s"
    end
end

main()