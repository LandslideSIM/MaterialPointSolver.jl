using CairoMakie
using CUDA
using AMDGPU
using DelimitedFiles
using MaterialPointSolver

devicetype = :CUDA
GPU_index = 0

if devicetype == :CUDA
    CUDA.device!(GPU_index)
    device_name = CUDA.name(CuDevice(GPU_index))
elseif devicetype == :ROCm
    GPU_index == 0 ? GPU_index = 1 : nothing
    AMDGPU.device!(AMDGPU.devices()[GPU_index])
    device_name = AMDGPU.HIP.name(AMDGPU.device_id!(GPU_index))
end
mkpath(joinpath(@__DIR__, device_name))
#const rtsdir = "/home/zhuo/Workbench/outputs/"#joinpath(@__DIR__, "outputs") # result path
#const rtsdir = "/users/zhuo/Workbench/outputs/" # GH 200
const rtsdir = joinpath(@__DIR__, "outputs") # AMD
const mtedir = joinpath(@__DIR__, device_name)
MaterialPointSolver.warmup(devicetype, ID=GPU_index)
single_value = GPUbandwidth(devicetype; datatype=:FP32, ID=GPU_index)
double_value = GPUbandwidth(devicetype; datatype=:FP64, ID=GPU_index)

# space = 1.00  |     40,000 pts
# space = 0.50  |    160,000 pts
# space = 0.25  |    640,000 pts
# space = 0.20  |  1,000,000 pts
# space = 0.10  |  4,000,000 pts
test_unit_2d = [1, 0.5, 0.25, 0.2, 0.1]
# space = 1.00  |     64,000 pts 
# space = 0.70  |    185,193 pts
# space = 0.50  |    512,000 pts
# space = 0.35  |  1,481,544 pts
# space = 0.25  |  4,096,000 pts 
test_unit_3d = [1, 0.7, 0.5, 0.35, 0.25]

#==========================================================================================+
| 1. 2d single precision test                                                              |
+==========================================================================================#
single_pts_2d   = Int64[]
single_mteff_2d = Float64[]
function testsuit1!(single_pts_2d, single_mteff_2d)
    @inbounds for i in test_unit_2d
        global init_grid_space_x = i
        global init_grid_space_y = i
        global iInt              = Int32
        global iFloat            = Float32
        include(joinpath(@__DIR__, "2d_uGIMP_d.jl"))
        #AMDGPU.synchronize(; stop_hostcalls=false)
        push!(single_pts_2d, mp.num)
        # effective memory throughput
        DoF = 2
        nio = ((grid.node_num + grid.node_num * DoF * 4) * 2 + grid.node_num * DoF + 
                grid.cell_num * mp.NIC) + ((8 * mp.num + 11 * mp.num * DoF + 
                mp.num * mp.NIC * 2 + mp.num * mp.NIC * DoF + 
                2 * mp.num * (DoF * DoF)) * 2 + mp.num)
        args.MVL == true ? nio += ((grid.cell_num * 2) * 2) : nothing
        MTeff = (nio * 4 * args.iter_num) / ((args.end_time - args.start_time) * (1024 ^ 3))
        push!(single_mteff_2d, MTeff)
    end
    return nothing
end

#==========================================================================================+
| 2. 2d double precision test                                                              |
+==========================================================================================#
double_pts_2d   = Int64[]
double_mteff_2d = Float64[]
function testsuit2!(double_pts_2d, double_mteff_2d)
    @inbounds for i in test_unit_2d
        global init_grid_space_x = i
        global init_grid_space_y = i
        global iInt              = Int64
        global iFloat            = Float64
        include(joinpath(@__DIR__, "2d_uGIMP_d.jl"))
        #AMDGPU.synchronize(; stop_hostcalls=false)
        push!(double_pts_2d, mp.num)
        # effective memory throughput
        DoF = 2
        nio = ((grid.node_num + grid.node_num * DoF * 4) * 2 + grid.node_num * DoF + 
                grid.cell_num * mp.NIC) + ((8 * mp.num + 11 * mp.num * DoF + 
                mp.num * mp.NIC * 2 + mp.num * mp.NIC * DoF + 
                2 * mp.num * (DoF * DoF)) * 2 + mp.num)
        args.MVL == true ? nio += ((grid.cell_num * 2) * 2) : nothing
        MTeff = (nio * 8 * args.iter_num) / ((args.end_time - args.start_time) * (1024 ^ 3))
        push!(double_mteff_2d, MTeff)
    end
    return nothing
end

#==========================================================================================+
| 3. 3d single precision test                                                              |
+==========================================================================================#
single_pts_3d   = Int64[]
single_mteff_3d = Float64[]
function testsuit3!(single_pts_3d, single_mteff_3d)
    @inbounds for i in test_unit_3d
        global init_grid_space_x = i
        global init_grid_space_y = i
        global init_grid_space_z = i
        global iInt              = Int32
        global iFloat            = Float32
        include(joinpath(@__DIR__, "3d_uGIMP_d.jl"))
        #AMDGPU.synchronize(; stop_hostcalls=false)
        push!(single_pts_3d, mp.num)
        # effective memory throughput
        DoF = 3
        nio = ((grid.node_num + grid.node_num * DoF * 4) * 2 + grid.node_num * DoF + 
                grid.cell_num * mp.NIC) + ((8 * mp.num + 11 * mp.num * DoF + 
                mp.num * mp.NIC * 2 + mp.num * mp.NIC * DoF + 
                2 * mp.num * (DoF * DoF)) * 2 + mp.num)
        args.MVL==true ? nio+=((grid.cell_num*2)*2) : nothing
        MTeff = (nio*4*args.iter_num)/((args.end_time-args.start_time)*(1024^3))
        push!(single_mteff_3d, MTeff)
    end
    return nothing
end

#==========================================================================================+
| 4. 3d double precision test                                                              |
+==========================================================================================#
double_pts_3d   = Int64[]
double_mteff_3d = Float64[]
function testsuit4!(double_pts_3d, double_mteff_3d)
    @inbounds for i in test_unit_3d
        global init_grid_space_x = i
        global init_grid_space_y = i
        global init_grid_space_z = i
        global iInt              = Int64
        global iFloat            = Float64
        include(joinpath(@__DIR__, "3d_uGIMP_d.jl"))
        #AMDGPU.synchronize(; stop_hostcalls=false)
        push!(double_pts_3d, mp.num)
        # effective memory throughput
        DoF = 3
        nio = ((grid.node_num + grid.node_num * DoF * 4) * 2 + grid.node_num * DoF + 
                grid.cell_num * mp.NIC) + ((8 * mp.num + 11 * mp.num * DoF + 
                mp.num * mp.NIC * 2 + mp.num * mp.NIC * DoF + 
                2 * mp.num * (DoF * DoF)) * 2 + mp.num)
        args.MVL==true ? nio+=((grid.cell_num*2)*2) : nothing
        MTeff = (nio*8*args.iter_num)/((args.end_time-args.start_time)*(1024^3))
        push!(double_mteff_3d, MTeff)
    end
    return nothing
end

testsuit1!(single_pts_2d, single_mteff_2d)
testsuit2!(double_pts_2d, double_mteff_2d)
testsuit3!(single_pts_3d, single_mteff_3d)
testsuit4!(double_pts_3d, double_mteff_3d)

open(joinpath(mtedir, "single_precision_2d.csv"), "w") do io
    writedlm(io, [single_pts_2d single_mteff_2d], ',')
end
open(joinpath(mtedir, "double_precision_2d.csv"), "w") do io
    writedlm(io, [double_pts_2d double_mteff_2d], ',')
end
open(joinpath(mtedir, "single_precision_3d.csv"), "w") do io
    writedlm(io, [single_pts_3d single_mteff_3d], ',')
end
open(joinpath(mtedir, "double_precision_3d.csv"), "w") do io
    writedlm(io, [double_pts_3d double_mteff_3d], ',')
end

let
    single_pts_2d   = readdlm(joinpath(mtedir, "single_precision_2d.csv"), ',', Float64)[:, 1]
    single_mteff_2d = readdlm(joinpath(mtedir, "single_precision_2d.csv"), ',', Float64)[:, 2]
    double_pts_2d   = readdlm(joinpath(mtedir, "double_precision_2d.csv"), ',', Float64)[:, 1]
    double_mteff_2d = readdlm(joinpath(mtedir, "double_precision_2d.csv"), ',', Float64)[:, 2]
    single_pts_3d   = readdlm(joinpath(mtedir, "single_precision_3d.csv"), ',', Float64)[:, 1]
    single_mteff_3d = readdlm(joinpath(mtedir, "single_precision_3d.csv"), ',', Float64)[:, 2]
    double_pts_3d   = readdlm(joinpath(mtedir, "double_precision_3d.csv"), ',', Float64)[:, 1]
    double_mteff_3d = readdlm(joinpath(mtedir, "double_precision_3d.csv"), ',', Float64)[:, 2]
    figfont = MaterialPointSolver.fonttnr
    upbound = max(single_value, double_value) * 1.1 |> trunc
    x_ticks = ["TS1", "TS2", "TS3", "TS4", "TS5"]
    fig = Figure(fonts=(; regular=figfont, bold=figfont), fontsize=16, size=(500, 600))
    ax1 = Axis(fig[1, 1], xlabel="Testsuites of varying particle numbers", ylabel="MTPeff [GiB/s]", 
        title="MTPeff on $(device_name)", xticks=(1:1:5, x_ticks), yticks=range(100, upbound, step=200))
    colors = ["#a48cf4", "#66c2a5"]
    tbl = (x=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5], height=vec([single_mteff_2d double_mteff_2d]'),
        grp=[1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    labelvalue = round.(vec([single_mteff_2d./single_value double_mteff_2d./double_value]').*100, digits=1)
    labelvalue = string.(labelvalue, "%")
    p1 = barplot!(ax1, tbl.x, tbl.height, dodge=tbl.grp, color=colors[tbl.grp], bar_labels=labelvalue,
        label_size=12, offset=10)
    labels = ["FP32", "FP64"]
    elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
    axislegend(ax1, elements, labels, labelsize=14, titlesize=14, "2D models", position=:lt)

    ax2 = Axis(fig[2, 1], xlabel="Testsuites of varying particle numbers", ylabel="MTPeff [GiB/s]", 
        title="MTPeff on $(device_name)", xticks=(1:1:5, x_ticks), yticks=range(100, upbound, step=200))
    colors = ["#a48cf4", "#66c2a5"]
    tbl = (x=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5], height=vec([single_mteff_3d double_mteff_3d]'),
        grp=[1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    labelvalue = round.(vec([single_mteff_3d./single_value double_mteff_3d./double_value]').*100, digits=1)
    labelvalue = string.(labelvalue, "%")
    p2 = barplot!(ax2, tbl.x, tbl.height, dodge=tbl.grp, color=colors[tbl.grp], bar_labels=labelvalue,
        label_size=12, offset=10)
    labels = ["FP32", "FP64"]
    elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
    axislegend(ax2, elements, labels, labelsize=14, titlesize=14, "3D models", position=:lt)

    p3 = hlines!(p1, single_value, color="#a48cf4", linestyle=:dash, linewidth=2)
    p3 = hlines!(p1, double_value, color="#66c2a5", linestyle=:dash, linewidth=2)
    p5 = hlines!(p2, single_value, color="#a48cf4", linestyle=:dash, linewidth=2)
    p6 = hlines!(p2, double_value, color="#66c2a5", linestyle=:dash, linewidth=2)

    upbound = max(single_value, double_value) * 1.1 |> trunc

    xlims!(ax1, 0.2, 6)
    xlims!(ax2, 0.2, 6)
    ylims!(ax1, 0, upbound)
    ylims!(ax2, 0, upbound)
    display(fig)
    save(joinpath(mtedir, "results.png"), fig)
end