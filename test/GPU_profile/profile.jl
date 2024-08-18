#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : profile.jl                                                                 |
|  Description: Profiling for 3D collapse                                                  |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
+==========================================================================================#

using MaterialPointSolver
using KernelAbstractions
using BenchmarkTools
using Printf
using CairoMakie
using AMDGPU
using CUDA

rtsdir = joinpath(@__DIR__, "outputs")#joinpath(homedir(), "Workbench/outputs")
GPU_index = 0
devicetype = :CUDA

if devicetype == :CUDA
    CUDA.device!(GPU_index)
    device_name = CUDA.name(CuDevice(GPU_index))
elseif devicetype == :ROCm
    GPU_index == 0 ? GPU_index = 1 : nothing
    AMDGPU.device!(AMDGPU.devices()[GPU_index])
    device_name = AMDGPU.HIP.name(AMDGPU.device_id!(GPU_index))
end
mkpath(joinpath(@__DIR__, device_name))
const mtedir = joinpath(@__DIR__, device_name)
MaterialPointSolver.warmup(devicetype, ID=GPU_index)

init_grid_space_x = 0.0025
init_grid_space_y = 0.0025
init_grid_space_z = 0.0025
init_grid_range_x = [-0.02, 0.07]
init_grid_range_y = [-0.02, 0.82]
init_grid_range_z = [-0.02, 0.12]
init_mp_in_space  = 2
init_project_name = "3d_collapse"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :druckerprager
init_gravity      = -9.8
init_ζs           = 0
init_ρs           = 2700
init_ν            = 0
init_E            = 1e6
init_Ks           = init_E/(3*(1-2*init_ν))
init_G            = init_E/(2*(1+  init_ν))
init_T            = 1
init_Te           = 0
init_ΔT           = 0.5*init_grid_space_x/sqrt((init_Ks+4/3*init_G)/init_ρs)
init_step         = floor(init_T/init_ΔT/50) |> Int64
init_step<10 ? init_step=1 : nothing
init_σt           = 0
init_ϕ            = 19.8*π/180
init_c            = 0
init_ψ            = 0
init_basis        = :uGIMP
init_phase        = 1
init_NIC          = 64
iInt              = Int64
iFloat            = Float64

# parameters setup
args = Args3D{iInt, iFloat}(
    Ttol         = init_T,
    ΔT           = init_ΔT,
    Te           = init_Te,
    time_step    = :fixed,
    FLIP         = 1,
    PIC          = 0,
    ζs           = init_ζs,
    gravity      = init_gravity,
    project_name = init_project_name,
    project_path = init_project_path,
    constitutive = init_constitutive,
    animation    = false,
    hdf5         = false,
    hdf5_step    = init_step,
    MVL          = true,
    device       = devicetype,
    coupling     = :OS,
    progressbar  = true,
    basis        = init_basis
)

# background grid setup
grid = Grid3D{iInt, iFloat}(
    NIC      = init_NIC,
    range_x1 = init_grid_range_x[1],
    range_x2 = init_grid_range_x[2],
    range_y1 = init_grid_range_y[1],
    range_y2 = init_grid_range_y[2],
    range_z1 = init_grid_range_z[1],
    range_z2 = init_grid_range_z[2],
    space_x  = init_grid_space_x,
    space_y  = init_grid_space_y,
    space_z  = init_grid_space_z,
    phase    = init_phase
)

# material points setup
range_x    = [0+grid.space_x/init_mp_in_space/2, 0.05-grid.space_x/init_mp_in_space/2]
range_y    = [0+grid.space_y/init_mp_in_space/2, 0.20-grid.space_y/init_mp_in_space/2]
range_z    = [0+grid.space_z/init_mp_in_space/2, 0.10-grid.space_z/init_mp_in_space/2]
space_x    = grid.space_x/init_mp_in_space
space_y    = grid.space_y/init_mp_in_space
space_z    = grid.space_z/init_mp_in_space
vx         = range_x[1]:space_x:range_x[2] |> collect
vy         = range_y[1]:space_y:range_y[2] |> collect
vz         = range_z[1]:space_z:range_z[2] |> collect
m, n, o    = length(vy), length(vx), length(vz)
vx         = reshape(vx, 1, n, 1)
vy         = reshape(vy, m, 1, 1)
vz         = reshape(vz, 1, 1, o)
om         = ones(Int, m)
on         = ones(Int, n)
oo         = ones(Int, o)
x_tmp      = vec(vx[om, :, oo])
y_tmp      = vec(vy[:, on, oo])
z_tmp      = vec(vz[om, on, :])
mp_num     = length(x_tmp)
mp_ρs      = ones(mp_num).*init_ρs
mp         = Particle3D{iInt, iFloat}(space_x=space_x, space_y=space_y, space_z=space_z,
    pos=[x_tmp y_tmp z_tmp], ρs=mp_ρs, NIC=init_NIC, phase=init_phase
)

# particle property setup
mp_layer   = ones(mp_num)
mp_ν       = [init_ν]
mp_E       = [init_E]
mp_G       = [init_G]
mp_σt      = [init_σt]
mp_ϕ       = [init_ϕ]
mp_c       = [init_c]
mp_ψ       = [init_ψ]
mp_Ks      = [init_Ks]
pts_attr   = ParticleProperty{iInt, iFloat}(layer=mp_layer, ν=mp_ν, E=mp_E, G=mp_G, 
    σt=mp_σt, ϕ=mp_ϕ, c=mp_c, ψ=mp_ψ, Ks=mp_Ks)

# boundary condition nodes index
vx_idx  = zeros(iInt, grid.node_num)
vy_idx  = zeros(iInt, grid.node_num)
vz_idx  = zeros(iInt, grid.node_num)
tmp_idx = findall(i->grid.pos[i, 1]≤0||grid.pos[i, 1]≥0.05||
                     grid.pos[i, 3]≤0||grid.pos[i, 2]≤0, 1:grid.node_num)
tmp_idy = findall(i->grid.pos[i, 2]≤0||grid.pos[i, 3]≤0, 1:grid.node_num)
tmp_idz = findall(i->grid.pos[i, 3]≤0, 1:grid.node_num)
vx_idx[tmp_idx] .= iInt(1)
vy_idx[tmp_idy] .= iInt(1)
vz_idx[tmp_idz] .= iInt(1)
bc = VBoundary3D{iInt, iFloat}(
    Vx_s_Idx = vx_idx,
    Vx_s_Val = zeros(grid.node_num),
    Vy_s_Idx = vy_idx,
    Vy_s_Val = zeros(grid.node_num),
    Vz_s_Idx = vz_idx,
    Vz_s_Val = zeros(grid.node_num)
)

# upload data to GPU
gpu_grid, gpu_mp, gpu_pts_attr, gpu_bc = host2device(args, grid, mp, pts_attr, bc)
G = args.gravity
dev = getBackend(args)
ΔT = args.ΔT

# start benchmarks
list_name = [
    "reset grid status",
    "reset particle status",
    "P2G",
    "solver on grid",
    "double mapping 1",
    "double mapping 2",
    "double mapping 3",
    "G2P",
    "elastic model",
    "hyperelastic model",
    "drucker prager model",
    "mohr coulomb model",
    "volume lock 1",
    "volume lock 2"
]
list_time = zeros(length(list_name))

list_time[1] = @belapsed begin
    resetgridstatus_OS!($dev)(ndrange=$gpu_grid.node_num, $gpu_grid)
    KAsync(dev)
end
list_time[2] = @belapsed begin
    resetmpstatus_OS!($dev)(ndrange=$gpu_mp.num, $gpu_grid, $gpu_mp, $Val(args.basis))
    KAsync(dev)
end
list_time[3] = @belapsed begin
    P2G_OS!($dev)(ndrange=$gpu_mp.num, $gpu_grid, $gpu_mp, $G)
    KAsync(dev)
end
list_time[4] = @belapsed begin
    solvegrid_OS!($dev)(ndrange=$gpu_grid.node_num, $gpu_grid, $gpu_bc, $ΔT, $args.ζs)
    KAsync(dev)
end
list_time[5] = @belapsed begin    
    doublemapping1_OS!($dev)(ndrange=$gpu_mp.num, $gpu_grid, $gpu_mp, $gpu_pts_attr, 
        $ΔT, $args.FLIP, $args.PIC)
    KAsync(dev)
end
list_time[6] = @belapsed begin
    doublemapping2_OS!($dev)(ndrange=$gpu_mp.num, $gpu_grid, $gpu_mp)
    KAsync(dev)
end
list_time[7] = @belapsed begin
    doublemapping3_OS!($dev)(ndrange=$gpu_grid.node_num, $gpu_grid, $gpu_bc, $ΔT)
    KAsync(dev)
end    
list_time[8] = @belapsed begin
    G2P_OS!($dev)(ndrange=$gpu_mp.num, $gpu_grid, $gpu_mp)
    KAsync(dev)
end
list_time[9] = @belapsed begin
    liE!($dev)(ndrange=$gpu_mp.num, $gpu_mp, $gpu_pts_attr)
    KAsync(dev)
end
list_time[10] = @belapsed begin
    hyE!($dev)(ndrange=$gpu_mp.num, $gpu_mp, $gpu_pts_attr)
    KAsync(dev)
end
list_time[11] = @belapsed begin
    liE!($dev)(ndrange=$gpu_mp.num, $gpu_mp, $gpu_pts_attr)
    dpP!($dev)(ndrange=$gpu_mp.num, $gpu_mp, $gpu_pts_attr)
    KAsync(dev)
end    
list_time[12] = @belapsed begin
    liE!($dev)(ndrange=$gpu_mp.num, $gpu_mp, $gpu_pts_attr)
    mcP!($dev)(ndrange=$gpu_mp.num, $gpu_mp, $gpu_pts_attr)
    KAsync(dev)
end
list_time[13] = @belapsed begin
    vollock1_OS!($dev)(ndrange=$gpu_mp.num, $gpu_grid, $gpu_mp)
    KAsync(dev)
end
list_time[14] = @belapsed begin
    vollock2_OS!($dev)(ndrange=$gpu_mp.num, $gpu_grid, $gpu_mp)
    KAsync(dev)
end
clean_gpu!(args, gpu_grid, gpu_mp, gpu_pts_attr, gpu_bc)

function profile_info(list_name, list_time)
    list_time .*= 1e3
    total_time = sum(list_time)
    col1_len = max(maximum(map(length, list_name)), length("Operation Name"))
    col2_len = length("Time (ms)")
    col3_len = length("Percentage")

    strspace = "   "
    title = string(lpad("Operation Name", col1_len), strspace,
                lpad("Time (ms)", col2_len), strspace,
                lpad("Percentage", col3_len))
    bar = "─"^(col1_len + col2_len + col3_len + length(strspace)*2)

    println("3D Collapse Benchmark Result")
    println(bar)
    println(title)
    println(bar)
    # 打印每个操作的名称、时间和百分比
    for (name, time) in zip(list_name, list_time)
        percentage = @sprintf("%.2f", (time / total_time) * 100)
        time = @sprintf("%.2f", time)
        println(lpad(name, col1_len), strspace,
                lpad(time, col2_len), strspace,
                lpad(percentage, col3_len-1), "%")
    end
    # 打印总时间
    println(bar)
    println("Total Time: $(@sprintf("%.2f", total_time)) ms")

    open(joinpath(mtedir, "profile_outputs.txt"), "w") do io
        redirect_stdout(io) do
            println("3D Collapse Benchmark Result")
            println(bar)
            println(title)
            println(bar)
            # 打印每个操作的名称、时间和百分比
            for (name, time) in zip(list_name, list_time)
                percentage = @sprintf("%.2f", (time / total_time) * 100)
                time = @sprintf("%.2f", time)
                println(lpad(name, col1_len), strspace,
                        lpad(time, col2_len), strspace,
                        lpad(percentage, col3_len-1), "%")
            end
            # 打印总时间
            println(bar)
            println("Total Time: $(@sprintf("%.2f", total_time)) ms")
        end
    end
    return nothing
end
profile_info(list_name, list_time)

let 
    per_list = string.(round.(list_time./sum(list_time).*100, digits=2), "%", " ⥯ ",
                       round.(list_time, digits=2), "ms")
    fig = Figure(size=(1000, 500), fonts=(; regular=MaterialPointSolver.fonttnr, 
        bold=MaterialPointSolver.fonttnr), fontsize=18)
    ax = Axis(fig[1, 1], yticks=(1:1:length(list_name), list_name), xlabel="Time (ms)", 
        title="3D Collapse Benchmark Result\n$device_name")
    pl = barplot!(ax, 1:1:length(list_name), list_time, color=list_time, direction=:x,
        colormap=Reverse(:RdYlGn_9))
    text!(ax, list_time.+0.05, collect(1:1:length(list_name)).-0.4; text=per_list)
    xlims!(ax, -0.1, maximum(list_time).+maximum(round.(list_time, digits=2))*0.3)
    save(joinpath(mtedir, "profile_vis.png"), fig)
    display(fig)
end