#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : 3d_slumping.jl                                                             |
|  Description: Please run this file in VSCode with Julia ENV                              |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Test Case  : 3D slumping in Drucker-Prager constitutive model.                          |
+==========================================================================================#

init_grid_range_x = [-3, 113]
init_grid_range_y = [-3,  23]
init_grid_range_z = [-3,  38]
init_mp_in_space  = 2
init_project_name = "3d_slumping"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :druckerprager
init_gravity      = -9.8
init_ζ            = 0
init_ρs           = 2100
init_ν            = 0.3
init_E            = 70e6
init_G            = init_E/(2*(1+  init_ν))
init_Ks           = init_E/(3*(1-2*init_ν))
init_T            = 7
init_Te           = 0
init_ΔT           = 0.5*min(init_grid_space_x, init_grid_space_y)/sqrt(init_E/init_ρs)
init_step         = floor(init_T/init_ΔT/20) |> Int64
init_step<10 ? init_step=1 : nothing
init_σt           = 27.48e3
init_ϕ            = 20*π/180
init_c            = 1e4
init_ψ            = 0*π/180
init_basis        = :uGIMP
init_NIC          = 64
init_phase        = 1

# parameters setup
args = Args3D{iInt, iFloat}(
    Ttol         = init_T,
    Te           = init_Te,
    ΔT           = init_ΔT,
    time_step    = :fixed,
    FLIP         = 1,
    PIC          = 0,
    ζ            = init_ζ,
    gravity      = init_gravity,
    project_name = init_project_name,
    project_path = init_project_path,
    constitutive = init_constitutive,
    animation    = false,
    hdf5         = false,
    hdf5_step    = init_step,
    vollock      = true,
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
    NIC      = init_NIC,
    phase    = init_phase
)

# material points setup
range_x    = [0+grid.space_x/init_mp_in_space/2, 110-grid.space_x/init_mp_in_space/2]
range_y    = [0+grid.space_y/init_mp_in_space/2,  20-grid.space_y/init_mp_in_space/2]
range_z    = [0+grid.space_z/init_mp_in_space/2,  35-grid.space_z/init_mp_in_space/2]
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
deleidx    = findall(i->(   x_tmp[i]≥50&&z_tmp[i]≥15         )||
                        (30≤x_tmp[i]≤50&&z_tmp[i]≥65-x_tmp[i]), 1:length(x_tmp))
deleteat!(x_tmp, deleidx)
deleteat!(y_tmp, deleidx)
deleteat!(z_tmp, deleidx)
mp_num     = length(x_tmp)
mp_ρs    = ones(mp_num).*init_ρs
mp_layer = ones(mp_num)
mp_pos   = [x_tmp y_tmp z_tmp]
mp_ν     = [init_ν]
mp_E     = [init_E]
mp_G     = [init_G]
mp_σt    = [init_σt]
mp_ϕ     = [init_ϕ]
mp_c     = [init_c]
mp_ψ     = [init_ψ]
mp_Ks    = [init_Ks]
mp       = Particle3D{iInt, iFloat}(space_x=space_x, space_y=space_y, space_z=space_z, 
    pos=mp_pos, ν=mp_ν, E=mp_E, G=mp_G, σt=mp_σt, ϕ=mp_ϕ, c=mp_c, ψ=mp_ψ, ρs=mp_ρs, 
    Ks=mp_Ks, layer=mp_layer, NIC=init_NIC, phase=init_phase)

# boundary condition nodes index
vx_idx  = zeros(iInt, grid.node_num)
vy_idx  = zeros(iInt, grid.node_num) 
tmp_idx = findall(i->((grid.pos[i, 1]≤0)||(grid.pos[i, 1]≥110)||
                      (grid.pos[i, 3]≤0)), 1:grid.node_num)
tmp_idy = findall(i->((grid.pos[i, 2]≤0)||(grid.pos[i, 2]≥20)||
                      (grid.pos[i, 3]≤0)), 1:grid.node_num)
tmp_idz = findall(i->( grid.pos[i, 3]≤0 ), 1:grid.node_num)
vx_idx[tmp_idx] .= 1
vy_idx[tmp_idy] .= 1
vx_idx[tmp_idz] .= 1
bc = VBoundary3D{iInt, iFloat}(
    Vx_s_Idx = vx_idx,
    Vx_s_Val = zeros(grid.node_num),
    Vy_s_Idx = vy_idx,
    Vy_s_Val = zeros(grid.node_num),
    Vz_s_Idx = vx_idx,
    Vz_s_Val = zeros(grid.node_num),
)

# MPM solver
simulate!(args, grid, mp, bc)