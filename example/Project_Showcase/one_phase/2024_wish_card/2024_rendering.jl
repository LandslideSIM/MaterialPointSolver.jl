using HDF5
using ProgressMeter
using PlyIO

"""
    generate_ply(hdf_path::String, ply_path::String, splash_path::String, radius, 
        num_threads; cube_size=0.6, surface_threshold=0.6, smoothing_length=1.2)

Description:
---
radius: 粒子(原始)半径，应该是粒子直径的一半
smoothing-length: 应围绕1.2设置。值越大，等值面越平滑，但也会人为地增加流体体积
surface-threshold: 可以用来抵消由于较大粒子半径等因素导致的流体体积增加，0.6的阈值似乎效果不错
cube-size: 通常不能大于1，如果结果粗糙或者运行时间长，从0.5~0.75之间开始增大或减小
"""
function generate_ply(hdf_path   ::String, 
                      ply_path   ::String, 
                      splash_path::String, 
                      radius,
                      num_threads;
                      cube_size=0.6,
                      surface_threshold=0.6, 
                      smoothing_length=1.2)
    #-----------------------#
    # 1. Generate ply files #
    #-----------------------#
    fid = h5open(hdf_path, "r")
    itr = (read(fid["FILE_NUM"])-1) |> Int64
    rm(ply_path, recursive=true, force=true); mkpath(ply_path)
    p = Progress(length(1:1:itr)-1; 
        desc      = "\e[1;36m[ Info:\e[0m $(lpad("ply_gen", 7))",
        color     = :white,
        barlen    = 12,
        barglyphs = BarGlyphs(" ◼◼  "))
    @inbounds for i in 1:itr
        obj = Float32.(fid["group$(i)/mp_pos"] |> read)
        vertex = PlyElement("vertex",
            ArrayProperty("x", obj[:, 1]),
            ArrayProperty("y", obj[:, 2]),
            ArrayProperty("z", obj[:, 3]))
        ply = Ply(); push!(ply, vertex)
        save_ply(ply, joinpath(ply_path, "iteration_$(i).ply"))
        next!(p)
    end
    close(fid)
    #----------------------#
    # 2. Generate surfaces #
    #----------------------#
    inputs = joinpath(ply_path, "iteration_{}.ply")
    rm(splash_path, recursive=true, force=true); mkpath(splash_path)
    outputs = joinpath(splash_path, "iteration_{}.vtk")
    run(`splashsurf reconstruct $(inputs) --output-file=$(outputs)
        --particle-radius=$(radius*1.5)
        --cube-size=$(cube_size)
        --surface-threshold=$(surface_threshold)
        --smoothing-length=$(smoothing_length)
        --subdomain-grid=on
        --mesh-cleanup=on
        --mesh-smoothing-weights=on 
        --mesh-smoothing-iters=25 
        --normals=on
        --normals-smoothing-iters=10
        --mt-particles=on
        --num-threads=$(num_threads)
        --mt-files=on`
    )
    return nothing
end

#-------------------#
# Main: user inputs |
#-------------------#
hdf_path    = "/home/zhuo/Workbench/outputs/2024_wish_card/2024_wish_card.h5"
ply_path    = joinpath("/home/zhuo/Workbench/outputs/2024_wish_card", "ply_set")
splash_path = joinpath("/home/zhuo/Workbench/outputs/2024_wish_card", "splash_set")
generate_ply(hdf_path, ply_path, splash_path, 0.025, 20)