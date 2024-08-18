#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : warmup.jl                                                                  |
|  Description: The minimal example of executing core functionality.                       |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : warmup                                                                     |
+==========================================================================================#

"""

    warmup(devicetype::Symbol; ID=0)

Description:
---
The minimal example of executing core functionality is used to reduce the first-time running
time. `devicetype` can be one of `:CPU`, `:CUDA`, or `:ROCm`. Note that on the GPU from AMD,
the device id start from `1`.`
"""
function warmup(devicetype::Symbol; ID=0)
    if devicetype==:CUDA
        CUDA.device!(ID)
    elseif devicetype==:ROCm
        ID == 0 ? ID = 1 : nothing
        AMDGPU.device!(AMDGPU.devices()[ID])
    end
    rtsdir = joinpath(homedir(), "MaterialPointSolverTEMP_$(ID)/")
    mkpath(rtsdir)
    init_grid_space_x = 1
    init_grid_space_y = 1
    init_grid_range_x = [-5, 5]
    init_grid_range_y = [-5, 5]
    init_mp_in_space  = 2
    init_project_name = "2d_test"
    init_project_path = joinpath(rtsdir, init_project_name)
    init_constitutive = :druckerprager
    init_ζs           = 0
    init_ρs           = 2650
    init_ν            = 0.3
    init_Ks           = 7e5
    init_E            = init_Ks*(3*(1-2*init_ν))
    init_G            = init_E /(2*(1+  init_ν))
    init_T            = 20*0.5*init_grid_space_x/sqrt(init_E/init_ρs)
    init_Te           = 0
    init_ΔT           = 0.5*init_grid_space_x/sqrt(init_E/init_ρs)
    init_step         = floor(init_T/init_ΔT/200) |> Int64
    init_step<10 ? init_step=1 : nothing
    init_σt           = 0
    init_ϕ            = 19.8*π/180
    init_c            = 0
    init_ψ            = 0
    init_NIC          = 16
    init_basis        = :uGIMP
    init_phase        = 1
    init_scheme       = :MUSL
    iInt              = Int64
    iFloat            = Float64

    # parameters setup
    args = Args2D{iInt, iFloat}(
        Ttol         = init_T,
        Te           = init_Te,
        ΔT           = init_ΔT,
        time_step    = :fixed,
        FLIP         = 1.0,
        PIC          = 0.0,
        ζs           = init_ζs,
        project_name = init_project_name,
        project_path = init_project_path,
        constitutive = init_constitutive,
        animation    = false,
        hdf5         = false,
        hdf5_step    = init_step,
        MVL          = true,
        device       = devicetype,
        coupling     = :OS,
        scheme       = init_scheme,
        basis        = init_basis
    )

    # background grid setup
    grid = Grid2D{iInt, iFloat}(
        NIC      = init_NIC,
        phase    = init_phase,
        range_x1 = init_grid_range_x[1],
        range_x2 = init_grid_range_x[2],
        range_y1 = init_grid_range_y[1],
        range_y2 = init_grid_range_y[2],
        space_x  = init_grid_space_x,
        space_y  = init_grid_space_y
    )

    # material points setup
    range_x = [-2+grid.space_x/init_mp_in_space/2, 2-grid.space_x/init_mp_in_space/2]
    range_y = [-2+grid.space_y/init_mp_in_space/2, 2-grid.space_y/init_mp_in_space/2]
    space_x = grid.space_x/init_mp_in_space
    space_y = grid.space_y/init_mp_in_space
    num_x   = length(range_x[1]:space_x:range_x[2])
    num_y   = length(range_y[1]:space_y:range_y[2])
    x_tmp   = repeat((range_x[1]:space_x:range_x[2])', num_y, 1) |> vec
    y_tmp   = repeat((range_y[1]:space_y:range_y[2]) , 1, num_x) |> vec
    mp_num  = length(x_tmp)
    mp_ρs   = ones(mp_num).*init_ρs
    mp      = Particle2D{iInt, iFloat}(space_x=space_x, space_y=space_y, pos=[x_tmp y_tmp],
        ρs=mp_ρs, NIC=init_NIC, phase=init_phase)

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
    pts_attr = ParticleProperty{iInt, iFloat}(layer=mp_layer, ν=mp_ν, E=mp_E, G=mp_G, σt=mp_σt, 
        ϕ=mp_ϕ, c=mp_c, ψ=mp_ψ, Ks=mp_Ks)

    # boundary condition nodes index
    vx_idx = zeros(iInt, grid.node_num)
    vy_idx = zeros(iInt, grid.node_num)
    tmp_idx = findall(i->(grid.pos[i, 1]≤-2||grid.pos[i, 1]≥2||
                          grid.pos[i, 2]≤-2), 1:grid.node_num)
    tmp_idy = findall(i->(grid.pos[i, 2]≤0), 1:grid.node_num)
    vx_idx[tmp_idx] .= 1
    vy_idx[tmp_idy] .= 1
    bc = VBoundary2D{iInt, iFloat}(
        Vx_s_Idx = vx_idx,
        Vx_s_Val = zeros(grid.node_num),
        Vy_s_Idx = vy_idx,
        Vy_s_Val = zeros(grid.node_num)
    )

    # MPM solver
    @info "code warm-up, wait a moment 🔥"
    @suppress begin
        materialpointsolver!(args, grid, mp, pts_attr, bc)
    end
    rm(rtsdir, recursive=true, force=true)
    return nothing
end