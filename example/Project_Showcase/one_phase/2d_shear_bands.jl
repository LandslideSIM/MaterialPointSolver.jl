#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : 2d_shear_bands.jl                                                          |
|  Description: 2D shear bands triggered by discontinuous materials                        |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Notes      : This case should ignore the position update of particles.                  |
+==========================================================================================#

using MaterialPointSolver

rtsdir = joinpath(homedir(), "Workbench/outputs")

function testprocedure!(args    ::MODELARGS, 
                        grid    ::GRID, 
                        mp      ::PARTICLE, 
                        pts_attr::PROPERTY,
                        bc      ::BOUNDARY,
                        ΔT      ::T2,
                        Ti      ::T2,
                                ::Val{:OS},
                                ::Val{:MUSL}) where {T2}
    Ti < args.Te ? G = args.gravity / args.Te * Ti : G = args.gravity
    dev = getBackend(args)
    resetgridstatus_OS!(dev)(ndrange=grid.node_num, grid)
    args.device == :CPU && args.basis == :uGIMP ? 
        resetmpstatus_OS!(grid, mp, Val(args.basis)) :
        resetmpstatus_OS!(dev)(ndrange=mp.num, grid, mp, Val(args.basis))
    P2G_OS!(dev)(ndrange=mp.num, grid, mp, G)
    solvegrid_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT, args.ζs)
    doublemapping1_OS!(dev)(ndrange=mp.num, grid, mp, pts_attr, ΔT, args.FLIP, args.PIC)
    doublemapping2_OS!(dev)(ndrange=mp.num, grid, mp)
    doublemapping3_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT)
    G2P_OS!(dev)(ndrange=mp.num, grid, mp)
    if args.constitutive==:hyperelastic
        hyE!(dev)(ndrange=mp.num, mp, pts_attr)
    elseif args.constitutive==:linearelastic
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
    elseif args.constitutive==:druckerprager
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
        if Ti≥args.Te
            dpP!(dev)(ndrange=mp.num, mp, pts_attr)
        end
    elseif args.constitutive==:mohrcoulomb
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
        if Ti≥args.Te
            mcP!(dev)(ndrange=mp.num, mp, pts_attr)
        end
    end
    if args.MVL == true
        vollock1_OS!(dev)(ndrange=mp.num, grid, mp)
        vollock2_OS!(dev)(ndrange=mp.num, grid, mp)
    end
    mp.pos .= mp.init
    return nothing
end

init_grid_space_x = 10
init_grid_space_y = 10
init_grid_range_x = [-1010, 1010]
init_grid_range_y = [-1010, 1010]
init_mp_in_space  = 2
init_project_name = "2d_shear_bands"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :druckerprager
init_gravity      = 0
init_ζs           = 0
init_ρs           = 2700
init_ν            = 2/7
init_Ks           = 2e10
init_G_inc        = 2.5e9
init_G_mat        = 1e10
init_E_inc        = 2*init_G_inc*(1+init_ν)
init_E_mat        = 2*init_G_mat*(1+init_ν)
init_T            = 20
init_Te           = 0
init_ΔT           = 0.1*init_grid_space_x/sqrt(init_E_mat/init_ρs)
init_step         = floor(init_T/init_ΔT/100) |> Int64
init_step<10 ? init_step=1 : nothing
init_basis        = :uGIMP
init_phase        = 1
init_NIC          = 16
init_σt           = 1e3
init_ϕ            = 50*π/180
init_c            = 3e7
init_ψ            = 0*π/180
iInt              = Int64
iFloat            = Float64

# parameters setup
args = Args2D{iInt, iFloat}(
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
    device       = :CUDA,
    coupling     = :OS,
    MVL          = true,
    basis        = init_basis
)

# background grid setup
grid = Grid2D{iInt, iFloat}(
    range_x1 = init_grid_range_x[1],
    range_x2 = init_grid_range_x[2],
    range_y1 = init_grid_range_y[1],
    range_y2 = init_grid_range_y[2],
    space_x  = init_grid_space_x,
    space_y  = init_grid_space_y,
    NIC      = init_NIC,
    phase    = init_phase
)

# material points setup
range_x = [-1000+grid.space_x/init_mp_in_space/2, 1000-grid.space_x/init_mp_in_space/2]
range_y = [-1000+grid.space_y/init_mp_in_space/2, 1000-grid.space_y/init_mp_in_space/2]
space_x = grid.space_x/init_mp_in_space
space_y = grid.space_y/init_mp_in_space
num_x   = length(range_x[1]:space_x:range_x[2])
num_y   = length(range_y[1]:space_y:range_y[2])
x_tmp   = repeat((range_x[1]:space_x:range_x[2])', num_y, 1) |> vec
y_tmp   = repeat((range_y[1]:space_y:range_y[2]) , 1, num_x) |> vec
mp_num  = length(x_tmp)
inc_id  = findall(i->(-200≤x_tmp[i]≤200)&&(y_tmp[i]^2≤4e4-x_tmp[i]^2), 1:mp_num)
mat_id  = deleteat!(1:mp_num |> collect, inc_id)
mp_ρs   = ones(mp_num)*init_ρs
mp      = Particle2D{iInt, iFloat}(space_x=space_x, space_y=space_y, pos=[x_tmp y_tmp], 
    ρs=mp_ρs, phase=init_phase)

# particle property setup
# layer1: inc | layer2: mat
mp_ν     = [init_ν    , init_ν    ]
mp_E     = [init_E_inc, init_E_mat]
mp_G     = [init_G_inc, init_G_mat]
mp_Ks    = [init_Ks   , init_Ks   ]
mp_σt    = [init_σt   , init_σt   ]
mp_ϕ     = [init_ϕ    , init_ϕ    ]
mp_c     = [init_c    , init_c    ]
mp_ψ     = [init_ψ    , init_ψ    ]
mp_layer = ones(mp_num)
mp_layer[mat_id] .= 2
mp_layer[inc_id] .= 1
pts_attr = ParticleProperty{iInt, iFloat}(layer=mp_layer, ν=mp_ν, E=mp_E, G=mp_G, Ks=mp_Ks, 
    σt=mp_σt, ϕ=mp_ϕ, c=mp_c, ψ=mp_ψ)

# boundary condition nodes index
vb     = 1e-1
l_idx  = findall(i->(grid.pos[i, 1]≤-1000), 1:grid.node_num)
l_val  = zeros(length(l_idx))
l_val .= vb
r_idx  = findall(i->(grid.pos[i, 1]≥ 1000), 1:grid.node_num)
r_val  = zeros(length(r_idx))
r_val .= -vb
vx_idx = [l_idx; r_idx]
vx_val = [l_val; r_val]
t_idx  = findall(i->(grid.pos[i, 2]≥ 1000), 1:grid.node_num)
t_val  = zeros(length(t_idx))
t_val .= vb
b_idx  = findall(i->(grid.pos[i, 2]≤-1000), 1:grid.node_num)
b_val  = zeros(length(b_idx))
b_val .= -vb
vy_idx = [t_idx; b_idx]
vy_val = [t_val; b_val]
vxidx  = zeros(iInt  , grid.node_num)
vyidx  = zeros(iInt  , grid.node_num)
vxval  = zeros(iFloat, grid.node_num)
vyval  = zeros(iFloat, grid.node_num)
vxidx[vx_idx] .= 1
vyidx[vy_idx] .= 1
vxval[vx_idx] .= vx_val
vyval[vy_idx] .= vy_val
bc = VBoundary2D{iInt, iFloat}(
    Vx_s_Idx = vxidx,
    Vx_s_Val = vxval,
    Vy_s_Idx = vyidx,
    Vy_s_Val = vyval
)

# MPM solver
materialpointsolver!(args, grid, mp, pts_attr, bc, workflow=testprocedure!)
savevtu(args, grid, mp, pts_attr)