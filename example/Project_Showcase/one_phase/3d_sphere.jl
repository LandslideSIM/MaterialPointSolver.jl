#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : 3d_sphere.jl                                                               |
|  Description: 3D sphere contact in linear elastic constitutive model                     |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
+==========================================================================================#

using MaterialPointSolver

rtsdir = joinpath(homedir(), "Workbench/outputs")

init_grid_space_x = 0.01
init_grid_space_y = 0.01
init_grid_space_z = 0.01
init_grid_range_x = [-0.6, 0.6]
init_grid_range_y = [-0.6, 0.6]
init_grid_range_z = [-0.6, 0.6]
init_mp_in_space  = 2
init_project_name = "3d_sphere"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :linearelastic
init_gravity      = 0
init_ζs           = 0
init_ρs           = 1000
init_ν            = 0.3
init_E            = 1e3
init_Ks           = init_E/(3(1-2init_ν))
init_G            = init_E/(2(1+ init_ν))
init_T            = 3.5
init_Te           = 0
init_ΔT           = 0.5*init_grid_space_x/sqrt(init_E/init_ρs)
init_step         = floor(init_T/init_ΔT/20) |> Int64
init_step<10 ? init_step=1 : nothing
init_basis        = :uGIMP
init_phase        = 1
init_NIC          = 64
iInt              = Int64
iFloat            = Float64

# parameters setup
args = Args3D{iInt, iFloat}(
    Ttol         = init_T,
    Te           = init_Te,
    ΔT           = init_ΔT,
    time_step    = :fixed,
    FLIP         = 1,
    PIC          = 0,
    gravity      = init_gravity,
    ζs           = init_ζs,
    project_name = init_project_name,
    project_path = init_project_path,
    constitutive = init_constitutive,
    animation    = true,
    hdf5         = false,
    hdf5_step    = init_step,
    coupling     = :OS,
    MVL          = true,
    device       = :CUDA,
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
range_x    = [-0.5+grid.space_x/init_mp_in_space/2, 0.5-grid.space_x/init_mp_in_space/2]
range_y    = [-0.5+grid.space_y/init_mp_in_space/2, 0.5-grid.space_y/init_mp_in_space/2]
range_z    = [-0.5+grid.space_z/init_mp_in_space/2, 0.5-grid.space_z/init_mp_in_space/2]
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
a = findall(i->((-0.5≤x_tmp[i]≤-0.1)&&(-0.5≤y_tmp[i]≤-0.1)&&
                ((z_tmp[i]+0.3)^2≤(0.04-(x_tmp[i]+0.3)^2-(y_tmp[i]+0.3)^2)) ||
                ( 0.1≤x_tmp[i]≤ 0.5)&&( 0.1≤y_tmp[i]≤ 0.5)&&
                ((z_tmp[i]-0.3)^2≤(0.04-(x_tmp[i]-0.3)^2-(y_tmp[i]-0.3)^2))),
                1:length(x_tmp))
del_id  = deleteat!(1:mp_num |> collect, a)
map(i->splice!(i, del_id), [x_tmp, y_tmp, z_tmp])
mp_num   = length(x_tmp)
mp_pos   = [x_tmp y_tmp z_tmp]
mp_ρs    = ones(mp_num).*init_ρs
mp       = Particle3D{iInt, iFloat}(space_x=space_x, space_y=space_y, space_z=space_z,
    pos=mp_pos, ρs=mp_ρs, NIC=init_NIC, phase=init_phase)
lb_id = findall(i->(-0.5≤mp.init[i, 1]≤-0.1)&&(-0.5≤mp.init[i, 2]≤-0.1)&&
                   ((mp.init[i, 3]+0.3)^2≤(0.04-(mp.init[i, 1]+0.3)^2-
                    (mp.init[i, 2]+0.3)^2)), 1:mp.num)
rt_id = findall(i->( 0.1≤mp.init[i, 1]≤ 0.5)&&( 0.1≤mp.init[i, 2]≤ 0.5)&&
                   ((mp.init[i, 3]-0.3)^2≤(0.04-(mp.init[i, 1]-0.3)^2-
                    (mp.init[i, 2]-0.3)^2)), 1:mp.num)
mp.Vs[lb_id, :] .=  0.1
mp.Vs[rt_id, :] .= -0.1

# particle property setup
mp_layer = ones(mp_num)
mp_ν     = [init_ν]
mp_E     = [init_E]
mp_G     = [init_G]
mp_Ks    = [init_Ks]
pts_attr = ParticleProperty{iInt, iFloat}(layer=mp_layer, ν=mp_ν, E=mp_E, G=mp_G, Ks=mp_Ks)

# boundary condition nodes index
vx_idx  = zeros(iInt, grid.node_num)
vy_idx  = zeros(iInt, grid.node_num)
vz_idx  = zeros(iInt, grid.node_num)
tmp_idx = findall(i->(grid.pos[i, 1]≤-1||grid.pos[i, 1]≥1), 1:grid.node_num)
tmp_idy = findall(i->(grid.pos[i, 2]≤-1||grid.pos[i, 2]≥1), 1:grid.node_num)
tmp_idz = findall(i->(grid.pos[i, 3]≤-1||grid.pos[i, 3]≥1), 1:grid.node_num)
vx_idx[tmp_idx] .= 1
vy_idx[tmp_idy] .= 1
vz_idx[tmp_idz] .= 1
bc = VBoundary3D{iInt, iFloat}(
    Vx_s_Idx = vx_idx,
    Vx_s_Val = zeros(grid.node_num),
    Vy_s_Idx = vy_idx,
    Vy_s_Val = zeros(grid.node_num),
    Vz_s_Idx = vz_idx,
    Vz_s_Val = zeros(grid.node_num)
)

# MPM solver
materialpointsolver!(args, grid, mp, pts_attr, bc)