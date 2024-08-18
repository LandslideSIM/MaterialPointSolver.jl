using REPL.TerminalMenus
using CUDA

function clear_terminal()
    if Sys.iswindows()
        run(`cmd /c cls`)
    else
        print("\e[1;1H\e[2J")
    end
end

function test_menu()
    clear_terminal()
    options = [
        "t1_helloworld.jl", 
        "t2_send_receive.jl",
        "t3_broadcast.jl",
        "t4_reduce.jl",
        "t5_mpi_p2p.jl",
        "t6_p2p_time.jl",
        "Exit"
    ]
    menu = RadioMenu(options)
    selected = request("Choose a test:", menu)
    filepath = joinpath(@__DIR__, options[selected])
    gpu_num = length(CUDA.devices())

    if options[selected] == "t1_helloworld.jl"
        run(`mpirun -n 8 julia -O3 --color=yes $(filepath)`)
    elseif options[selected] == "t2_send_receive.jl"
        run(`mpirun -n 8 julia -O3 --color=yes $(filepath)`)
    elseif options[selected] == "t3_broadcast.jl"
        run(`mpirun -n 8 julia -O3 --color=yes $(filepath)`)
    elseif options[selected] == "t4_reduce.jl"
        run(`mpirun -n 8 julia -O3 --color=yes $(filepath)`)
    elseif options[selected] == "t5_mpi_p2p.jl"
        run(`mpirun -n 2 julia -O3 --color=yes $(filepath) $(gpu_num)`)
    elseif options[selected] == "t6_p2p_time.jl"
        run(`mpirun -n 2 --bind-to socket julia -O3 --color=yes $(filepath)`)
    elseif options[selected] == "Exit"
        println("Exiting...")
    end
end

test_menu()