using MPI
using CUDA
using Printf

function print_aligned_matrix_with_color(matrix::Array{String,2})
    @info "MPI GPU Unidirectional P2P Bandwidth Test [GB/s]"
    col_widths = [maximum(length.(matrix[:, i])) for i in 1:size(matrix, 2)]
    for row in 1:size(matrix, 1)
        for col in 1:size(matrix, 2)
            if row == 1 || col == 1
                print("\e[1;32m", lpad(matrix[row, col], col_widths[col]), "\e[0m  ")
            else
                print(lpad(matrix[row, col], col_widths[col]), "  ")
            end
        end
        println()
    end
end

# unidirectional p2p test
function main(gpu_num)
    MPI.Init()
    comm   = MPI.COMM_WORLD
    rank   = MPI.Comm_rank(comm) # [0 1] only two ranks
    n      = 16384
    nbench = 5

    if rank == 0
        result = Array{String, 2}(undef, gpu_num+1, gpu_num+1)
        result[1, :] .= ["GPU/GPU"; string.(0:1:gpu_num-1)]
        result[:, 1] .= ["GPU/GPU"; string.(0:1:gpu_num-1)]
    end

    # two ranks control two GPUs, each pair of GPUs will test `nbench` times (i.e. 5 times)
    for dev_src in 0:1:gpu_num-1, dev_dst in 0:1:gpu_num-1
        # prepare data on devices
        if rank == 0
            CUDA.device!(dev_src)
            send_mesg = CUDA.rand(Float32, n, n)
            datasize = sizeof(send_mesg)/1024^3
            trst = zeros(nbench) # save the runtime results of 5 tests
        elseif rank == 1
            CUDA.device!(dev_dst)
            recv_mesg = CUDA.zeros(Float32, n, n)
        end
        CUDA.synchronize()
        # start to test for 5 times
        for itime in 1:nbench
            if rank == 0
                tic = MPI.Wtime()
                MPI.Send(send_mesg, comm; dest=1, tag=666)
            elseif rank == 1
                rreq = MPI.Irecv!(recv_mesg, comm; source=0, tag=666)
                MPI.Wait(rreq)
            end
            CUDA.synchronize()
            MPI.Barrier(comm)
            if rank == 0
                toc = MPI.Wtime()
                trst[itime] = toc-tic
            end
        end

        # the average time will only take from the 2nd to the 5th
        # convert GiB to GB
        if rank == 0
            speed = @sprintf "%.2f" datasize/(sum(trst[2:end])/(nbench-1)) * (2^30)/(10^9)
            result[dev_src+2, dev_dst+2] = speed
        end
    end
    
    # print the results
    if rank == 0
        print_aligned_matrix_with_color(result)
    end
    return nothing
end

gpu_num = parse(Int, ARGS[1])
main(gpu_num)