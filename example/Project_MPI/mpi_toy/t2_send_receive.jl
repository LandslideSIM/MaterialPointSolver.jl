using MPI

function fake_compute(comm, rank, size)
    # get send and receive addresses
    dst = mod(rank+1, size)
    src = mod(rank-1, size)
    # prepare data
    n         = 1024000
    send_mesg = Array{Float64}(undef, n)
    recv_mesg = Array{Float64}(undef, n)
    fill!(send_mesg, Float64(rank))
    # receive data
    rreq = MPI.Irecv!(recv_mesg, comm; source=src, tag=src+32)
    # send data
    @info "<$rank>: sending  $(unique(send_mesg)) from <$rank> → <$dst>" 
    sreq = MPI.Isend(send_mesg, comm; dest=dst, tag=rank+32)
    # wait for both send and receive to complete
    stats = MPI.Waitall([rreq, sreq])
    # finish work
    @info "<$rank>: received $(unique(recv_mesg)) from <$src> → <$rank>"
end

function main()
    # MPI configuration
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm) # start from 0
    size = MPI.Comm_size(comm) # start from 1
    tic  = MPI.Wtime()
    # run the code
    fake_compute(comm, rank, size)
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