using DelimitedFiles
using MaterialPointSolver
using CUDA

GPU_index = 0
CUDA.device!(GPU_index)
device_name = CUDA.name(CuDevice(GPU_index))
mkpath(joinpath(@__DIR__, device_name))
rtsdir = joinpath(homedir(), "Workbench/outputs")
const mtedir = joinpath(@__DIR__, device_name)
MaterialPointSolver.warmup(GPU_index)

# 0.01000:   8000 pts
# 0.00660:  27000 pts
# 0.00500:  64000 pts
# 0.00400: 125000 pts
# 0.00333: 216000 pts
# 0.00250: 512000 pts
const test_unit = [0.01, 0.01, 0.0066, 0.005, 0.004, 0.00333, 0.00250]

single_numpt = Int64[]
single_wtime = Float64[]
single_mteff = Float64[]
double_numpt = Int64[]
double_wtime = Float64[]
double_mteff = Float64[]

function testsuit1!(single_numpt, single_wtime, single_mteff)
    @inbounds for i in test_unit
        global init_grid_space_x = i
        global init_grid_space_y = i
        global init_grid_space_z = i
        global iInt              = Int32
        global iFloat            = Float32
        include(joinpath(@__DIR__, "testfile.jl"))
        push!(single_numpt, mp.num)
        push!(single_wtime, args.end_time-args.start_time)
        # effective memory throughput
        DoF = 3
        nio = ((grid.node_num+grid.node_num*DoF*5)*2+grid.node_num*DoF+grid.cell_num*mp.NIC)+
              ((8*mp.num+11*mp.num*DoF+mp.num*mp.NIC*2+mp.num*mp.NIC*DoF+2*mp.num*(DoF^2))*2+mp.num)
        args.MVL==true ? nio+=((grid.cell_num*2)*2) : nothing
        MTeff = (nio*4*args.iter_num)/((args.end_time-args.start_time)*(1024^3))
        push!(single_mteff, MTeff)
    end
    return nothing
end

function testsuit2!(double_numpt, double_wtime, double_mteff)
    @inbounds for i in test_unit
        global init_grid_space_x = i
        global init_grid_space_y = i
        global init_grid_space_z = i
        global iInt              = Int64
        global iFloat            = Float64
        include(joinpath(@__DIR__, "testfile.jl"))
        push!(double_numpt, mp.num)
        push!(double_wtime, args.end_time-args.start_time)
        # effective memory throughput
        DoF = 3
        nio = ((grid.node_num+grid.node_num*DoF*5)*2+grid.node_num*DoF+grid.cell_num*mp.NIC)+
              ((8*mp.num+11*mp.num*DoF+mp.num*mp.NIC*2+mp.num*mp.NIC*DoF+2*mp.num*(DoF^2))*2+mp.num)
        args.MVL==true ? nio+=((grid.cell_num*2)*2) : nothing
        MTeff = (nio*8*args.iter_num)/((args.end_time-args.start_time)*(1024^3))
        push!(double_mteff, MTeff)
    end
    return nothing
end

testsuit1!(single_numpt, single_wtime, single_mteff)
testsuit2!(double_numpt, double_wtime, double_mteff)

open(joinpath(mtedir, "single_precision.csv"), "w") do io
    writedlm(io, [single_numpt single_wtime single_mteff], ',')
end

open(joinpath(mtedir, "double_precision.csv"), "w") do io
    writedlm(io, [double_numpt double_wtime double_mteff], ',')
end