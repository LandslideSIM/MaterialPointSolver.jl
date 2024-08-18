#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : 2d_dam_break.jl                                                            |
|  Description: 2D dam break test                                                          |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Reference  : [1] Huang, Peng, Shun-li Li, Hu Guo, and Zhi-ming Hao. “Large Deformation  |
|                   Failure Analysis of the Soil Slope Based on the Material Point Method.”|
|                   Computational Geosciences 19, no. 4 (August 2015): 951–63.             |
|                   https://doi.org/10.1007/s10596-015-9512-9.                             |
|               [2] Morris, J.P., Fox, P.J., Zhu, Y.: Modeling low reynolds number         |
|                   incompressible flows using SPH. J. Comput. Phys. 136, 214–226 (1997).  |
|  Notes      : [1] Don't use "uGIMP" basis for this case.                                 |
|               [2] Analytical solution is from shallow water theory.                      |
+==========================================================================================#

using CairoMakie
using KernelAbstractions
using MaterialPointSolver

rtsdir = joinpath(homedir(), "Workbench/outputs")

@kernel inbounds=true function testtwP!(mp::KernelParticle2D{T1, T2}) where {T1, T2}
    ix = @index(Global)
    if ix ≤ mp.num
        P = T2(1225.0)*(mp.ρs[ix]-mp.ρs_init[ix])
        mp.σij[ix, 1] = -P
        mp.σij[ix, 2] = -P
        mp.σm[ix]     =  P
    end
end

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
        resetmpstatus_OS_CPU!(grid, mp, Val(args.basis)) :
        resetmpstatus_OS!(dev)(ndrange=mp.num, grid, mp, Val(args.basis))
    P2G_OS!(dev)(ndrange=mp.num, grid, mp, G)
    solvegrid_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT, args.ζs)
    doublemapping1_OS!(dev)(ndrange=mp.num, grid, mp, pts_attr, ΔT, args.FLIP, args.PIC)
    doublemapping2_OS!(dev)(ndrange=mp.num, grid, mp)
    doublemapping3_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT)
    G2P_OS!(dev)(ndrange=mp.num, grid, mp)
    testtwP!(dev)(ndrange=mp.num, mp)
    return nothing
end

init_grid_space_x = 4e-3
init_grid_space_y = 4e-3
init_grid_range_x = [-0.008, 2.508]
init_grid_range_y = [-0.008, 0.122]
init_mp_in_space  = 2
init_project_name = "2d_dam_break"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :userdefined
init_gravity      = -9.8
init_ζs           = 0
init_ρs           = 1e3
init_ν            = 0.48
init_Ks           = 2.15e9
init_E            = init_Ks*(3*(1-2*init_ν))
init_G            = init_E /(2*(1+  init_ν))
init_T            = 0.5
init_B            = 1.119e7
init_γ            = 7
init_Te           = 0
init_ΔT           = 0.1*min(init_grid_space_x, init_grid_space_y)/sqrt(init_E/init_ρs)
init_step         = floor(init_T/init_ΔT/300) |> Int64
init_step<10 ? init_step=1 : nothing
init_phase        = 1
init_NIC          = 4
init_basis        = :linear
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
    MVL          = false,
    device       = :CUDA,
    coupling     = :OS,
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
    phase    = init_phase,
    NIC      = init_NIC
)

# material points setup
range_x = [0+grid.space_x/init_mp_in_space/2, 1.0-grid.space_x/init_mp_in_space/2]
range_y = [0+grid.space_y/init_mp_in_space/2, 0.1-grid.space_y/init_mp_in_space/2]
space_x = grid.space_x/init_mp_in_space
space_y = grid.space_y/init_mp_in_space
num_x   = length(range_x[1]:space_x:range_x[2])
num_y   = length(range_y[1]:space_y:range_y[2])
x_tmp   = repeat((range_x[1]:space_x:range_x[2])', num_y, 1) |> vec
y_tmp   = repeat((range_y[1]:space_y:range_y[2]) , 1, num_x) |> vec
mp_num  = length(x_tmp)
mp_ρs   = ones(mp_num).*init_ρs
mp_pos  = [x_tmp y_tmp]
mp      = Particle2D{iInt, iFloat}(space_x=space_x, space_y=space_y, pos=mp_pos, ρs=mp_ρs, 
    NIC=init_NIC, phase=init_phase)

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
tmp_idx = findall(i->grid.pos[i, 1]≤0, 1:grid.node_num)
tmp_idy = findall(i->grid.pos[i, 2]≤0, 1:grid.node_num)
vx_idx[tmp_idx] .= 1
vy_idx[tmp_idy] .= 1
bc = VBoundary2D{iInt, iFloat}(
    Vx_s_Idx = vx_idx,
    Vx_s_Val = zeros(grid.node_num),
    Vy_s_Idx = vy_idx,
    Vy_s_Val = zeros(grid.node_num)
)

# MPM solver
materialpointsolver!(args, grid, mp, pts_attr, bc, workflow=testprocedure!)

# post-processing
let 
    figfont = MaterialPointSolver.fonttnr
    fig = Figure(size=(850, 190), fontsize=15, fonts=(; regular=figfont, bold=figfont))
    ax1 = Axis(fig[1, 1], aspect=1.9, xticks=(0.2:0.2:1.0), yticks=(0.02:0.04:0.1), 
        xlabel=L"x\ (m)", ylabel=L"y\ (m)")
    ax2 = Axis(fig[1, 2:3], aspect=4, xticks=(0.2:0.2:2.0), yticks=(0.02:0.04:0.1), 
        xlabel=L"x\ (m)", ylabel=L"y\ (m)")
    
    pl0 = scatter!(ax1, mp.init, markersize=6, color=mp.init[:, 2], 
        colormap=Reverse(:Blues_3))
    vlines!(ax1, [1], ymax=[0.86], color=:black, linewidth=5)

    pl1 = scatter!(ax2, mp.pos, markersize=3, color=:gray)    
    y = collect(0:0.001:0.1)
    x = @. (2*sqrt(0.1*9.8)-3*sqrt(y*9.8))*0.5+1
    analytical = vcat([x y], [0 0.1])
    pl2 = lines!(ax2, analytical, linewidth=6, color=:red, alpha=0.7)
    vlines!(ax2, [0.5], color=:blue, linewidth=2, linestyle=:dash)
    
    limits!(ax1, 0, 1.08, 0, 0.122)
    limits!(ax2, 0, 2.1, 0, 0.122)
    axislegend(ax2, [pl2], ["analytical solution"], L"T=0.5s", position=:rt, labelsize=15)
    display(fig)
    #save(joinpath(args.project_path, "$(args.project_name).png"), fig)
    @info "Figure saved in project path"
end