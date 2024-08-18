using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm) # start from 0
size = MPI.Comm_size(comm) # start from 1

@info "Hello, I am $(rank) of $(size)."