using MaterialPointSolver
using KernelAbstractions

warmup()
include(joinpath(@__DIR__, "scheme.jl"))

const rtsdir = joinpath(homedir(), "Workbench/outputs")

# space         pts 
# 0.1500 13,786,880
# 0.1875  7,047,040
# 0.2000  5,816,520
# 0.2500  2,982,912
# 0.30    1,722,560
# 0.40      727,020 
# 0.50      370,992

init_grid_space_x = 0.15
init_grid_space_y = 0.15
init_grid_space_z = 0.15
init_grid_range_x = [-init_grid_space_x*3, 60+init_grid_space_x*3]
init_grid_range_y = [-init_grid_space_y*3, 12+init_grid_space_y*3]
init_grid_range_z = [-init_grid_space_z*3, 12+init_grid_space_z*3]
init_mp_in_space  = 2
init_project_name = "demo"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :druckerprager
init_gravity      = -9.8
init_ζs           = 0
init_ρs           = 2700
init_ν            = 0.3
init_E            = 1e6
init_G            = init_E / (2 * (1 +     init_ν))
init_Ks           = init_E / (3 * (1 - 2 * init_ν))
init_T            = 10 #4*0.5*init_grid_space_x/sqrt(init_E/init_ρs)
init_Te           = 7
init_ΔT           = 0.5 * init_grid_space_x / sqrt(init_E / init_ρs)
init_step         = floor(init_T / init_ΔT / 20) |> Int64
init_step < 10 ? init_step = 1 : nothing
init_ϕ1           = 20  * π / 180
init_ϕ2           = 7.5 * π / 180
init_c            = 2e4
init_NIC          = 64
init_basis        = :uGIMP
init_phase        = 1
iInt              = Int32
iFloat            = Float32

# parameters setup
args = Args3D{iInt, iFloat}(
    Ttol         = init_T,
    Te           = init_Te,
    ΔT           = init_ΔT,
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
    device       = :CUDA,
    coupling     = :OS,
    basis        = init_basis
)

# background grid setup
grid = Grid3D{iInt, iFloat}(
    range_x1 = init_grid_range_x[1],
    range_x2 = init_grid_range_x[2],
    range_y1 = init_grid_range_y[1],
    range_y2 = init_grid_range_y[2],
    range_z1 = init_grid_range_z[1],
    range_z2 = init_grid_range_z[2],
    space_x  = init_grid_space_x,
    space_y  = init_grid_space_y,
    space_z  = init_grid_space_z,
    phase    = init_phase,
    NIC      = init_NIC
)

# material points setup
space_x = grid.space_x / init_mp_in_space
space_y = grid.space_y / init_mp_in_space
space_z = grid.space_z / init_mp_in_space
vx      = 0 + space_x / 2 : space_x : 60 -space_x / 2 |> collect
vy      = 0 + space_y / 2 : space_y : 12 -space_y / 2 |> collect
vz      = 0 + space_z / 2 : space_z : 12 -space_z / 2 |> collect
m, n, o = length(vy), length(vx), length(vz)
vx      = reshape(vx, 1, n, 1)
vy      = reshape(vy, m, 1, 1)
vz      = reshape(vz, 1, 1, o)
om      = ones(Int, m)
on      = ones(Int, n)
oo      = ones(Int, o)
x_tmp   = vec(vx[om, :, oo])
y_tmp   = vec(vy[:, on, oo])
z_tmp   = vec(vz[om, on, :])
deleidx = findall(i->((27.8 ≤ x_tmp[i] ≤ 36.2) && (z_tmp[i] ≥ 39.8 - x_tmp[i])) ||
                     ((36.2 < x_tmp[i]       ) && (z_tmp[i] ≥ 3.6            )), 
                     1:length(x_tmp))
deleteat!(x_tmp, deleidx)
deleteat!(y_tmp, deleidx)
deleteat!(z_tmp, deleidx)
mp_num = length(x_tmp)
mp_ρs  = ones(mp_num) .* init_ρs
mp_pos = [x_tmp y_tmp z_tmp]
mp     = Particle3D{iInt, iFloat}(space_x=space_x, space_y=space_y, space_z=space_z, 
    pos=mp_pos, ρs=mp_ρs, NIC=init_NIC, phase=init_phase)

# particle property setup
mp_layer   = ones(mp_num)
layer2indx = findall(i -> z_tmp[i] ≤ 3.6, 1:length(x_tmp))
# mp_layer[layer2indx] .= 2
mp_ν     = [init_ν , init_ν]
mp_E     = [init_E , init_E]
mp_G     = [init_G , init_G]
mp_ϕ     = [init_ϕ1, init_ϕ2]
mp_Ks    = [init_Ks, init_Ks]
mp_c     = [init_c , init_c]
pts_attr = ParticleProperty{iInt, iFloat}(layer=mp_layer, ν=mp_ν, E=mp_E, G=mp_G, ϕ=mp_ϕ, 
    Ks=mp_Ks)

# boundary condition nodes index
vx_idx  = zeros(iInt, grid.node_num)
vy_idx  = zeros(iInt, grid.node_num)
vz_idx  = zeros(iInt, grid.node_num)
tmp_idx = findall(i -> (grid.pos[i, 1] ≤ 0) || 
                       (grid.pos[i, 1] ≥ 60), 1:grid.node_num)
tmp_idy = findall(i -> (grid.pos[i, 2] ≤ 0) || 
                       (grid.pos[i, 2] ≥ 12), 1:grid.node_num)
tmp_idz = findall(i -> (grid.pos[i, 3] ≤ 0), 1:grid.node_num)
vx_idx[tmp_idx] .= 1
vy_idx[tmp_idy] .= 1
vz_idx[tmp_idz] .= 1
bc = VBoundary3D{iInt, iFloat}(
    Vx_s_Idx = vx_idx,
    Vx_s_Val = zeros(grid.node_num),
    Vy_s_Idx = vy_idx,
    Vy_s_Val = zeros(grid.node_num),
    Vz_s_Idx = vz_idx,
    Vz_s_Val = zeros(grid.node_num),
)

materialpointsolver!(args, grid, mp, pts_attr, bc, workflow=testprocedure!)
savevtu(args, grid, mp, pts_attr)