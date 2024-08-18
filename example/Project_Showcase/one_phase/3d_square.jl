#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : 3d_square.jl                                                               |
|  Description: 3D minimum square for testing MPM code, also used to precompile the        |
|               package                                                                    |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
+==========================================================================================#

using CairoMakie
using MaterialPointSolver

rtsdir = joinpath(tempdir(), "MPMTEST")

init_grid_space_x = 1
init_grid_space_y = 1
init_grid_space_z = 1
init_grid_range_x = [-2, 8]
init_grid_range_y = [-2, 4]
init_grid_range_z = [-2, 8]
init_mp_in_space  = 2
init_project_name = "3d_square"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :druckerprager
init_vtk_step     = 1
init_gravity      = -9.8
init_ζs           = 0
init_ρs           = 2650
init_ν            = 0.3
init_Ks           = 7e5
init_E            = init_Ks*(3*(1-2*init_ν))
init_G            = init_E /(2*(1+  init_ν))
init_T            = 5
init_Te           = 0
init_ΔT           = 0.5*init_grid_space_x/sqrt(init_E/init_ρs)
init_step         = floor(init_T/init_ΔT/200) |> Int64
init_step<10 ? init_step=1 : nothing
init_σt           = 0
init_ϕ            = 19.8*π/180
init_c            = 0
init_ψ            = 0
init_NIC          = 64
init_basis        = :uGIMP
init_phase        = 1
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
    hdf5         = true,
    hdf5_step    = 1,
    MVL          = false,
    device       = :CPU,
    coupling     = :OS,
    basis        = :uGIMP
)

# background grid setup
grid = Grid3D{iInt, iFloat}(
    NIC      = init_NIC,
    phase    = init_phase,
    range_x1 = init_grid_range_x[1],
    range_x2 = init_grid_range_x[2],
    range_y1 = init_grid_range_y[1],
    range_y2 = init_grid_range_y[2],
    range_z1 = init_grid_range_z[1],
    range_z2 = init_grid_range_z[2],
    space_x  = init_grid_space_x,
    space_y  = init_grid_space_y,
    space_z  = init_grid_space_z
)

# material points setup
range_x = [0+grid.space_x/init_mp_in_space/2, 2-grid.space_x/init_mp_in_space/2]
range_y = [0+grid.space_y/init_mp_in_space/2, 2-grid.space_y/init_mp_in_space/2]
range_z = [0+grid.space_z/init_mp_in_space/2, 2-grid.space_z/init_mp_in_space/2]
space_x = grid.space_x/init_mp_in_space
space_y = grid.space_y/init_mp_in_space
space_z = grid.space_z/init_mp_in_space
vx      = range_x[1]:space_x:range_x[2] |> collect
vy      = range_y[1]:space_y:range_y[2] |> collect
vz      = range_z[1]:space_z:range_z[2] |> collect
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
mp_num  = length(x_tmp)
mp_ρs   = ones(mp_num).*init_ρs
mp_pos  = [x_tmp y_tmp z_tmp]
mp      = Particle3D{iInt, iFloat}(space_x=space_x, space_y=space_y, space_z=space_z, 
    pos=mp_pos, ρs=mp_ρs, NIC=init_NIC, phase=init_phase)

# particle property setup
mp_layer = ones(mp_num)
mp_ν     = [init_ν]
mp_E     = [init_E]
mp_G     = [init_G]
mp_σt    = [init_σt]
mp_ϕ     = [init_ϕ]
mp_c     = [init_c]
mp_ψ     = [init_ψ]
mp_Ks    = [init_Ks]
pts_attr = ParticleProperty{iInt, iFloat}(layer=mp_layer, ν=mp_ν, E=mp_E, G=mp_G, σt=mp_σt, 
    ϕ=mp_ϕ, c=mp_c, ψ=mp_ψ, Ks=mp_Ks)

# boundary condition nodes index
vx_idx  = zeros(iInt, grid.node_num)
vy_idx  = zeros(iInt, grid.node_num)
vz_idx  = zeros(iInt, grid.node_num)
tmp_idx = findall(i->(grid.pos[i, 1]≤0||grid.pos[i, 1]≥6||
                      grid.pos[i, 2]≤0||grid.pos[i, 3]≤0), 1:grid.node_num)
tmp_idy = findall(i->(grid.pos[i, 2]≤0||grid.pos[i, 3]≤0), 1:grid.node_num)
tmp_idz = findall(i->(grid.pos[i, 3]≤0), 1:grid.node_num)
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

# visualization
@views @inbounds let
    figfont = MaterialPointSolver.fontcmu
    fig = Figure(fonts=(; regular=figfont, bold=figfont), fontsize=25)
    axis = Axis3(fig[1, 1], xlabel=L"x", ylabel=L"y", aspect=:data, xticks=(0:2:8),
        yticks=(0:2:3))
    limits!(axis, -1, 8, -1, 3, -2, 4)
    scatter!(axis, mp.pos, color=mp.σm, strokewidth=0, markersize=20)
    display(fig)
    save(joinpath(rtsdir, args.project_name, "$(args.project_name).png"), fig)
    @info "Figure saved in project path"
end