using MPI

function reduce_data(comm, rank)
    # prepare data
    n = 2
    if rank == 0
        data = [1, 5, -2]
    else 
        data = [2, 3, 3]
    end
    # MPI reduce
    summ = MPI.Reduce(data, max, 0, comm)
    
    # print out the data
    if rank == 0
        @info "minimum value: $(summ)"
    end
end

function main()
    # MPI configuration
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm) # start from 0
    tic  = MPI.Wtime()
    # run the code
    reduce_data(comm, rank)
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