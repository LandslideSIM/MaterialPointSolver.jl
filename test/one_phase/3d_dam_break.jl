#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : 3d_dam_break.jl                                                            |
|  Description: Please run this file in VSCode with Julia ENV                              |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Test Case  : 2D water column collapse in Tait equation.                                 |
|  Reference  : [1] Huang, Peng, Shun-li Li, Hu Guo, and Zhi-ming Hao. “Large Deformation  |
|                   Failure Analysis of the Soil Slope Based on the Material Point Method.”|
|                   Computational Geosciences 19, no. 4 (August 2015): 951–63.             |
|                   https://doi.org/10.1007/s10596-015-9512-9.                             |
|               [2] Morris, J.P., Fox, P.J., Zhu, Y.: Modeling low reynolds number         |
|                   incompressible flows using SPH. J. Comput. Phys. 136, 214–226 (1997).  |
|  Notes      : [1]: Don't use "uGIMP" basis for this case, it may break the simulation.   |
|               [2]: Remeber to comment the lines about Δϵij[:, 4]/ϵij[:, 4] (2D) and      |
|                    Δϵij[:, [4,5,6]]/ϵij[:, [4,5,6]] (3D) in OS.jl and OS_d.jl.           |
|               [3]: Comment CFL part in OS.jl and OS_d.jl.                                |
|               [4]: B and γ is useless in this case, Cs = 35m/s (from paper). Remember    |
|                    change this in taitwater.jl or taitwater_d.jl.                        |
+==========================================================================================#

init_grid_space_x = 4e-2
init_grid_space_y = 4e-2
init_grid_space_z = 4e-2
init_grid_range_x = [-0.02, 2.02]
init_grid_range_y = [-0.02, 0.12]
init_grid_range_z = [-0.02, 0.52]
init_mp_in_space  = 3
init_project_name = "3d_dam_break"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :taitwater
init_gravity      = -9.8
init_ζ            = 0
init_ρs           = 1e3
init_ν            = 0.48
init_Ks           = 2.15e9
init_E            = init_Ks*(3*(1-2*init_ν))
init_G            = init_E /(2*(1+  init_ν))
init_T            = 10
init_B            = 1.119e7
init_γ            = 7
init_Te           = 0
init_ΔT           = 0.1*min(init_grid_space_x, init_grid_space_y)/sqrt(init_E/init_ρs)
init_step         = floor(init_T/init_ΔT/100) |> Int64
init_step<10 ? init_step=1 : nothing
init_phase        = 1
init_NIC          = 8
init_basis        = :linear
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
    ζ            = init_ζ,
    gravity      = init_gravity,
    project_name = init_project_name,
    project_path = init_project_path,
    constitutive = init_constitutive,
    animation    = true,
    hdf5         = true,
    hdf5_step    = init_step,
    vollock      = false,
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
range_x  = [0+grid.space_x/init_mp_in_space/2, 1.0-grid.space_x/init_mp_in_space/2]
range_y  = [0+grid.space_y/init_mp_in_space/2, 0.1-grid.space_y/init_mp_in_space/2]
range_z  = [0+grid.space_y/init_mp_in_space/2, 0.1-grid.space_y/init_mp_in_space/2]
space_x  = grid.space_x/init_mp_in_space
space_y  = grid.space_y/init_mp_in_space
space_z  = grid.space_z/init_mp_in_space
vx       = range_x[1]:space_x:range_x[2] |> collect
vy       = range_y[1]:space_y:range_y[2] |> collect
vz       = range_z[1]:space_z:range_z[2] |> collect
m, n, o  = length(vy), length(vx), length(vz)
vx       = reshape(vx, 1, n, 1)
vy       = reshape(vy, m, 1, 1)
vz       = reshape(vz, 1, 1, o)
om       = ones(Int, m)
on       = ones(Int, n)
oo       = ones(Int, o)
x_tmp    = vec(vx[om, :, oo])
y_tmp    = vec(vy[:, on, oo])
z_tmp    = vec(vz[om, on, :])
# tmp_del  = findall(i-> x_tmp[i]≥2 && z_tmp[i]≥2, 1:length(x_tmp))
# deleteat!(x_tmp, tmp_del)
# deleteat!(y_tmp, tmp_del)
# deleteat!(z_tmp, tmp_del)
mp_num   = length(x_tmp)
mp_ρs    = ones(mp_num).*init_ρs
mp_layer = ones(mp_num)
mp_pos   = [x_tmp y_tmp z_tmp]
mp_ν     = [init_ν]
mp_E     = [init_E]
mp_G     = [init_G]
mp_Ks    = [init_Ks]
mp_B     = [init_B]
mp_γ     = [init_γ]
mp       = Particle3D{iInt, iFloat}(space_x=space_x, space_y=space_y, space_z=space_z, 
    pos=mp_pos, ν=mp_ν, E=mp_E, G=mp_G, ρs=mp_ρs, Ks=mp_Ks, layer=mp_layer, NIC=init_NIC,
    phase=init_phase, B=mp_B, γ=mp_γ)

# boundary condition nodes index
vx_idx  = zeros(iInt, grid.node_num)
vy_idx  = zeros(iInt, grid.node_num) 
vz_idx  = zeros(iInt, grid.node_num)
tmp_idx = findall(i->((grid.pos[i, 1]≤0)||(grid.pos[i, 1]≥2.0)), 1:grid.node_num)
tmp_idy = findall(i->((grid.pos[i, 2]≤0)||(grid.pos[i, 2]≥0.1)), 1:grid.node_num)
tmp_idz = findall(i->((grid.pos[i, 3]≤0)||(grid.pos[i, 3]≥1.0)), 1:grid.node_num)
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
simulate!(args, grid, mp, bc)