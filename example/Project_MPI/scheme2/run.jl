function testsuit()
    filepath = joinpath(@__DIR__, "main.jl")
    run(`mpirun -n 1 --bind-to socket julia -O3 --color=yes $(filepath)`)
    run(`mpirun -n 2 --bind-to socket julia -O3 --color=yes $(filepath)`)
    run(`mpirun -n 3 --bind-to socket julia -O3 --color=yes $(filepath)`)
    run(`mpirun -n 4 --bind-to socket julia -O3 --color=yes $(filepath)`)
    run(`mpirun -n 5 --bind-to socket julia -O3 --color=yes $(filepath)`)
    run(`mpirun -n 6 --bind-to socket julia -O3 --color=yes $(filepath)`)
    run(`mpirun -n 7 --bind-to socket julia -O3 --color=yes $(filepath)`)
    run(`mpirun -n 8 --bind-to socket julia -O3 --color=yes $(filepath)`)
end

testsuit()

using CairoMakie
using DelimitedFiles
using MaterialPointSolver

let
gpu_num_1 = readdlm(joinpath(@__DIR__, "assets/1_GPUs_bench.txt"))
gpu_num_2 = readdlm(joinpath(@__DIR__, "assets/2_GPUs_bench.txt"))
gpu_num_3 = readdlm(joinpath(@__DIR__, "assets/3_GPUs_bench.txt"))
gpu_num_4 = readdlm(joinpath(@__DIR__, "assets/4_GPUs_bench.txt"))
gpu_num_5 = readdlm(joinpath(@__DIR__, "assets/5_GPUs_bench.txt"))
gpu_num_6 = readdlm(joinpath(@__DIR__, "assets/6_GPUs_bench.txt"))
gpu_num_7 = readdlm(joinpath(@__DIR__, "assets/7_GPUs_bench.txt"))
gpu_num_8 = readdlm(joinpath(@__DIR__, "assets/8_GPUs_bench.txt"))

figfont = MaterialPointSolver.fontcmu
fig = Figure(size=(600, 300), fonts=(; regular=figfont, bold=figfont), fontsize=20)
ax = Axis(fig[1, 1], xlabel="Number of Particles", ylabel="Wall-clock time [s]",
    xscale=log10, xminorticksvisible=true, xminorgridvisible=true,
    xminorticks=IntervalsBetween(5))
p1 = scatterlines!(ax, gpu_num_1[:, 1], gpu_num_1[:, 2])
p2 = scatterlines!(ax, gpu_num_2[:, 1], gpu_num_2[:, 2])
#p3 = scatterlines!(ax, gpu_num_3[:, 1], gpu_num_3[:, 2])
p4 = scatterlines!(ax, gpu_num_4[:, 1], gpu_num_4[:, 2])
#p5 = scatterlines!(ax, gpu_num_5[:, 1], gpu_num_5[:, 2])
#p6 = scatterlines!(ax, gpu_num_6[:, 1], gpu_num_6[:, 2])
#p7 = scatterlines!(ax, gpu_num_7[:, 1], gpu_num_7[:, 2])
p8 = scatterlines!(ax, gpu_num_8[:, 1], gpu_num_8[:, 2])
# axislegend(ax, [p1, p2, p3, p4], 
#     ["N_GPU = 1", "N_GPU = 2", "N_GPU = 3", "N_GPU = 4"], position=:lt, labelsize=20)
# axislegend(ax, [p5, p6, p7, p8], 
#     ["N_GPU = 5", "N_GPU = 6", "N_GPU = 7", "N_GPU = 8"], position=:rb, labelsize=20)
axislegend(ax, [p1, p2, p4, p8], 
    ["N_GPU = 1", "N_GPU = 2", "N_GPU = 4", "N_GPU = 8"], position=:lt, labelsize=16)
save(joinpath(@__DIR__, "assets/V100.png"), fig)
display(fig)
end