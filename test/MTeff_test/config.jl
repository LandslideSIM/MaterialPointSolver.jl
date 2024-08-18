using CairoMakie

include(joinpath(@__DIR__, "../../src/MPMSolver.jl"))
GPU_index    = 0; device!(GPU_index)
device_name  = CUDA.name(CuDevice(GPU_index))
mkpath(joinpath(@__DIR__, device_name))
const assets = joinpath(@__DIR__, "../../assets")  # code assets
const rtsdir = "/home/zhuo/Workbench/outputs/"     # result path
const mtedir = joinpath(@__DIR__, device_name)

# space = 0.8    |    15,200 pts
# space = 0.5    |    39,160 pts
# space = 0.1    |   979,504 pts
# space = 0.07   | 2,000,571 pts
# space = 0.065  | 3,918,408 pts
test_unit_2d = [0.8, 0.5, 0.1, 0.07, 0.065]

# space = 2    |     48,800 pts
# space = 1.5  |    112,086 pts
# space = 1    |    391,200 pts
# space = 0.8  |    760,000 pts
# space = 0.5  |  3,132,800 pts
## space = 0.4  |  6,121,800 pts
test_unit_3d = [2, 1.5, 1, 0.8, 0.5]

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
        push!(single_pts_2d, mp.num)
        # effective memory throughput
        DoF = 2
        nio = ((grid.node_num+grid.node_num*DoF*5)*2+grid.node_num*DoF+grid.cell_num*mp.NIC)+
              ((8*mp.num+11*mp.num*DoF+mp.num*mp.NIC*2+mp.num*mp.NIC*DoF+2*mp.num*(DoF^2))*2+mp.num)
        args.vollock==true ? nio+=((grid.cell_num*2)*2) : nothing
        MTeff = (nio*4*args.iter_num)/((args.end_time-args.start_time)*(1024^3))
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
        push!(double_pts_2d, mp.num)
        # effective memory throughput
        DoF = 2
        nio = ((grid.node_num+grid.node_num*DoF*5)*2+grid.node_num*DoF+grid.cell_num*mp.NIC)+
              ((8*mp.num+11*mp.num*DoF+mp.num*mp.NIC*2+mp.num*mp.NIC*DoF+2*mp.num*(DoF^2))*2+mp.num)
        args.vollock==true ? nio+=((grid.cell_num*2)*2) : nothing
        MTeff = (nio*8*args.iter_num)/((args.end_time-args.start_time)*(1024^3))
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
        push!(single_pts_3d, mp.num)
        # effective memory throughput
        DoF = 3
        nio = ((grid.node_num+grid.node_num*DoF*5)*2+grid.node_num*DoF+grid.cell_num*mp.NIC)+
              ((8*mp.num+11*mp.num*DoF+mp.num*mp.NIC*2+mp.num*mp.NIC*DoF+2*mp.num*(DoF^2))*2+mp.num)
        args.vollock==true ? nio+=((grid.cell_num*2)*2) : nothing
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
        push!(double_pts_3d, mp.num)
        # effective memory throughput
        DoF = 3
        nio = ((grid.node_num+grid.node_num*DoF*5)*2+grid.node_num*DoF+grid.cell_num*mp.NIC)+
              ((8*mp.num+11*mp.num*DoF+mp.num*mp.NIC*2+mp.num*mp.NIC*DoF+2*mp.num*(DoF^2))*2+mp.num)
        args.vollock==true ? nio+=((grid.cell_num*2)*2) : nothing
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

single_value = bandwidth(deviceid=GPU_index, datatype=:single)
double_value = bandwidth(deviceid=GPU_index, datatype=:double)

let
    single_pts_2d   = readdlm(joinpath(mtedir, "single_precision_2d.csv"), ',', Float64)[:, 1]
    single_mteff_2d = readdlm(joinpath(mtedir, "single_precision_2d.csv"), ',', Float64)[:, 2]
    double_pts_2d   = readdlm(joinpath(mtedir, "double_precision_2d.csv"), ',', Float64)[:, 1]
    double_mteff_2d = readdlm(joinpath(mtedir, "double_precision_2d.csv"), ',', Float64)[:, 2]
    single_pts_3d   = readdlm(joinpath(mtedir, "single_precision_3d.csv"), ',', Float64)[:, 1]
    single_mteff_3d = readdlm(joinpath(mtedir, "single_precision_3d.csv"), ',', Float64)[:, 2]
    double_pts_3d   = readdlm(joinpath(mtedir, "double_precision_3d.csv"), ',', Float64)[:, 1]
    double_mteff_3d = readdlm(joinpath(mtedir, "double_precision_3d.csv"), ',', Float64)[:, 2]
    figfont = joinpath(assets, "fonts/tnr.ttf")
    x_ticks = ["TS1", "TS2", "TS3", "TS4", "TS5"]
    fig = Figure(fontfamily=figfont, fontsize=16, size=(500, 600))
    ax1 = Axis(fig[1, 1], xlabel="Testsuites of varying particle numbers", ylabel="MTeff [GiB/s]", 
        title="MTeff on $(device_name)", xlabelfont=figfont, xticks=(1:1:5, x_ticks),
        yticks=(100:200:900), ylabelfont=figfont, xticklabelfont=figfont, 
        yticklabelfont=figfont)
    colors = ["#a48cf4", "#66c2a5"]
    tbl = (x=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5], height=vec([single_mteff_2d double_mteff_2d]'),
        grp = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    p1 = barplot!(ax1, tbl.x, tbl.height, dodge=tbl.grp, color=colors[tbl.grp])
    labels = ["FP32", "FP64"]
    elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
    axislegend(ax1, elements, labels, labelfont=figfont, titlefont=figfont, labelsize=14, 
        titlesize=14, "2D models", position=:lt)

    ax2 = Axis(fig[2, 1], xlabel="Testsuites of varying particle numbers", ylabel="MTeff [GiB/s]", 
        title="MTeff on $(device_name)", xlabelfont=figfont, xticks=(1:1:5, x_ticks),
        yticks=(100:200:900), ylabelfont=figfont, xticklabelfont=figfont, 
        yticklabelfont=figfont)
    colors = ["#a48cf4", "#66c2a5"]
    tbl = (x=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5], height=vec([single_mteff_3d double_mteff_3d]'),
        grp = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    p2 = barplot!(ax2, tbl.x, tbl.height, dodge=tbl.grp, color=colors[tbl.grp])
    labels = ["FP32", "FP64"]
    elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
    axislegend(ax2, elements, labels, labelfont=figfont, titlefont=figfont, labelsize=14, 
        titlesize=14, "3D models", position=:lt)

    p3 = hlines!(p1, max(single_value, double_value), color=:red, linestyle=:dash, linewidth=2)
    p4 = hlines!(p2, max(single_value, double_value), color=:red, linestyle=:dash, linewidth=2)

    xlims!(ax1, 0.2, 6)
    xlims!(ax2, 0.2, 6)
    ylims!(ax1, 0, 930)
    ylims!(ax2, 0, 930)
    display(fig)
    save(joinpath(mtedir, "results.png"), fig)
end