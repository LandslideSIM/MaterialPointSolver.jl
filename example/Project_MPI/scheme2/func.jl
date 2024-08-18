function init_model(;
    init_project_name::String,
    init_grid_range_x::Vector,
    init_grid_range_y::Vector,
    init_grid_range_z::Vector,
    init_mp_range_x  ::Vector,
    init_mp_range_y  ::Vector,
    init_mp_range_z  ::Vector,
    init_mp_space_xyz        ,
    init_grid_space_xyz      ,
    rank_size
)
    init_grid_space_x = init_grid_space_xyz
    init_grid_space_y = init_grid_space_xyz
    init_grid_space_z = init_grid_space_xyz
    init_mp_in_space  = 2
    init_project_path = joinpath(rtsdir, init_project_name)
    init_constitutive = :druckerprager
    init_gravity      = -9.8
    init_ζs           = 0
    init_ρs           = 2700
    init_ν            = 0.3
    init_E            = 1e6
    init_G            = init_E / (2 * (1 +     init_ν))
    init_Ks           = init_E / (3 * (1 - 2 * init_ν))
    init_T            = 4*0.5*init_grid_space_x/sqrt(init_E/init_ρs)
    init_Te           = 0
    init_ΔT           = 0.5 * init_grid_space_x / sqrt(init_E / init_ρs)
    init_step         = floor(init_T / init_ΔT / 20) |> Int64
    init_step < 10 ? init_step = 1 : nothing
    init_ϕ1           = 20  * π / 180
    init_ϕ2           = 7.5 * π / 180
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
    vx      = init_mp_range_x[1] : init_mp_space_xyz : init_mp_range_x[2] |> collect
    vy      = init_mp_range_y[1] : init_mp_space_xyz : init_mp_range_y[2] |> collect
    vz      = init_mp_range_z[1] : init_mp_space_xyz : init_mp_range_z[2] |> collect
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
    deleidx = findall(i->((27.8 ≤ x_tmp[i] ≤ 36.2) && (z_tmp[i] ≥ 39.8 - x_tmp[i]))||
                         ((36.2 < x_tmp[i]       ) && (z_tmp[i] ≥ 3.6            )), 
                         1:length(x_tmp))
    deleteat!(x_tmp, deleidx)
    deleteat!(y_tmp, deleidx)
    deleteat!(z_tmp, deleidx)
    mp_num = length(x_tmp)
    mp_ρs  = ones(mp_num) .* init_ρs
    mp_pos = [x_tmp y_tmp z_tmp]
    mp     = Particle3D{iInt, iFloat}(space_x=init_mp_space_xyz, space_y=init_mp_space_xyz, 
        space_z=init_mp_space_xyz, pos=mp_pos, ρs=mp_ρs, NIC=init_NIC, phase=init_phase)

    # particle property setup
    mp_layer   = ones(mp_num)
    layer2indx = findall(i->z_tmp[i]≤3.6, 1:length(x_tmp))
    mp_layer[layer2indx] .= 2
    mp_ν     = [init_ν , init_ν ]
    mp_E     = [init_E , init_E ]
    mp_G     = [init_G , init_G ]
    mp_ϕ     = [init_ϕ1, init_ϕ2]
    mp_Ks    = [init_Ks, init_Ks]
    pts_attr = ParticleProperty{iInt, iFloat}(layer=mp_layer, ν=mp_ν, E=mp_E, G=mp_G, 
        ϕ=mp_ϕ, Ks=mp_Ks)

    # boundary condition nodes index
    vx_idx  = zeros(iInt, grid.node_num)
    vy_idx  = zeros(iInt, grid.node_num)
    vz_idx  = zeros(iInt, grid.node_num)
    tmp_idx = findall(i -> (grid.pos[i, 1] ≤ 0) || 
                           (grid.pos[i, 1] ≥ 60), 1:grid.node_num)
    tmp_idy = findall(i -> (grid.pos[i, 2] ≤ 0) || 
                           (grid.pos[i, 2] ≥ rank_size * 12), 1:grid.node_num)
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
    return args, grid, mp, pts_attr, bc
end

@views function testhost2device(
    args    ::MODELARGS, 
    grid    ::GRID, 
    mp      ::PARTICLE, 
    pts_attr::PROPERTY, 
    bc      ::BOUNDARY
)
    T1 = typeof(grid).parameters[1]
    T2 = typeof(grid).parameters[2]
    T3 = getDArray(args)
    if typeof(grid)<:Grid2D
        gpu_grid =      GPUGrid2D{T1, T2, T3, T3, T3    }(
            [getfield(grid, f) for f in fieldnames(     Grid2D)]...)
        gpu_mp   =  GPUParticle2D{T1, T2, T3, T3, T3, T3}(
            [getfield(mp  , f) for f in fieldnames( Particle2D)]...)
        gpu_pts_attr = GPUParticleProperty{T1, T2, T3, T3}(
            [getfield(pts_attr, f) for f in fieldnames(ParticleProperty)]...)
        gpu_bc   = GPUVBoundary2D{T1, T2, T3, T3        }(
            [getfield(bc  , f) for f in fieldnames(VBoundary2D)]...)
    elseif typeof(grid)<:Grid3D
        gpu_grid =      GPUGrid3D{T1, T2, T3, T3, T3    }(
            [getfield(grid, f) for f in fieldnames(     Grid3D)]...)
        
        NIC = mp.NIC
        DoF = 3
        new_pcg = 0.1
        new_num = mp.num * (1 + new_pcg) |> trunc |> T1
        new_inc1 = mp.num * new_pcg |> T1
        new_inc2 = mp.phase == 1 ? 0 : new_inc1
        new_DoF = mp.phase == 1 ? 1 : DoF

        gpu_mp = GPUParticle3D{T1, T2, T3, T3, T3, T3}(
            new_num, mp.phase, mp.NIC, mp.space_x, mp.space_y, mp.space_z, 
            vcat(mp.p2c     , zeros(T1   , new_inc1         )), 
            vcat(mp.p2n     , zeros(Int32, new_inc1, NIC    )), 
            vcat(mp.pos     , zeros(T2   , new_inc1, DoF    )), 
            vcat(mp.σm      , zeros(T2   , new_inc1         )), 
            vcat(mp.J       , zeros(T2   , new_inc1         )), 
            vcat(mp.epII    , zeros(T2   , new_inc1         )), 
            vcat(mp.epK     , zeros(T2   , new_inc1         )), 
            vcat(mp.vol     , zeros(T2   , new_inc1         )), 
            vcat(mp.vol_init, zeros(T2   , new_inc1         )), 
            vcat(mp.Ms      , zeros(T2   , new_inc1         )), 
            vcat(mp.Mw      , zeros(T2   , new_inc2         )), 
            vcat(mp.Mi      , zeros(T2   , new_inc2         )), 
            vcat(mp.porosity, zeros(T2   , new_inc2         )), 
            vcat(mp.cfl     , zeros(T2   , new_inc1         )),
            vcat(mp.ρs      , zeros(T2   , new_inc1         )),
            vcat(mp.ρs_init , zeros(T2   , new_inc1         )),
            vcat(mp.ρw      , zeros(T2   , new_inc2         )), 
            vcat(mp.ρw_init , zeros(T2   , new_inc2         )),
            vcat(mp.init    , zeros(T2   , new_inc1, DoF    )),
            vcat(mp.σw      , zeros(T2   , new_inc2         )),
            vcat(mp.σij     , zeros(T2   , new_inc1, 6      )),
            vcat(mp.ϵij_s   , zeros(T2   , new_inc1, 6      )),
            vcat(mp.ϵij_w   , zeros(T2   , new_inc2, 6      )),
            vcat(mp.Δϵij_s  , zeros(T2   , new_inc1, 6      )),
            vcat(mp.Δϵij_w  , zeros(T2   , new_inc2, 6      )),
            vcat(mp.sij     , zeros(T2   , new_inc1, 6      )),
            vcat(mp.Vs      , zeros(T2   , new_inc1, DoF    )),
            vcat(mp.Vw      , zeros(T2   , new_inc2, new_DoF)),
            vcat(mp.Ps      , zeros(T2   , new_inc1, DoF    )),
            vcat(mp.Pw      , zeros(T2   , new_inc2, new_DoF)),
            vcat(mp.Ni      , zeros(T2   , new_inc1, NIC    )),
            vcat(mp.∂Nx     , zeros(T2   , new_inc1, NIC    )),
            vcat(mp.∂Ny     , zeros(T2   , new_inc1, NIC    )),
            vcat(mp.∂Nz     , zeros(T2   , new_inc1, NIC    )),
            vcat(mp.∂Fs     , zeros(T2   , new_inc1, 9      )),
            vcat(mp.∂Fw     , zeros(T2   , new_inc2, 9      )),
            vcat(mp.F       , zeros(T2   , new_inc1, 9      ))
        )
        gpu_pts_attr = GPUParticleProperty{T1, T2, T3, T3}(
            vcat(pts_attr.layer, zeros(T1, new_inc1)),
            pts_attr.ν   , pts_attr.E, pts_attr.G, pts_attr.Ks, pts_attr.Kw, pts_attr.k, 
            pts_attr.σt  , pts_attr.ϕ, pts_attr.ψ, pts_attr.c , pts_attr.cr, pts_attr.Hp, 
            pts_attr.tmp1, pts_attr.tmp2   
        )
        gpu_bc   = GPUVBoundary3D{T1, T2, T3, T3        }(
            [getfield(bc  , f) for f in fieldnames(VBoundary3D)]...)
    end
    datasize = Base.summarysize(grid)    +Base.summarysize(mp)+
               Base.summarysize(pts_attr)+Base.summarysize(bc)
    outprint = @sprintf("%.1f", datasize/1024^3)
    dev = CUDA.device().handle
    content = "host [≈ $(outprint) GiB] → device $(dev) [$(args.device)]"
    println("\e[1;32m[▲ I/O:\e[0m \e[0;32m$(content)\e[0m")
    return gpu_grid, gpu_mp, gpu_pts_attr, gpu_bc
end

function testdevice2host(gpu_mp, rank, gpu_counts)
    tmp1 = Array(gpu_mp.pos)
    tmp2 = Array(gpu_mp.epII)
    tmp3 = Array(gpu_mp.σm)
    vaild_idx = Int64[]

    @inbounds for i in eachindex(tmp3)
        if tmp1[i, :] != [0, 0, 0]
            push!(vaild_idx, i)
        end
    end

    pos  = tmp1[vaild_idx, :]
    epII = tmp2[vaild_idx]
    σm   = tmp3[vaild_idx]
    data = hcat(pos, epII, σm)
    open(joinpath(@__DIR__, "assets/$(gpu_counts)_GPUs_part$(rank+1).txt"), "w") do io
        writedlm(io, data)
    end
    return nothing
end

@views function post_process(gpu_counts)
    # prepare data
    tmp  = readdlm(joinpath(@__DIR__, "assets/$(gpu_counts)_GPUs_part1.txt"))
    id   = fill(0, size(tmp, 1))
    data = hcat(tmp, id)
    if gpu_counts != 1
        for i in 2:gpu_counts
            tmp = readdlm(joinpath(@__DIR__, "assets/$(gpu_counts)_GPUs_part$(i).txt"))
            id = fill((i - 1), size(tmp, 1))
            data = vcat(data, hcat(tmp, id))
        end
    end
    if size(data, 1) > 10000000
        didx = sample(1:1:size(data, 1), 10000000, replace=false)
    else
        didx = 1:size(data, 1)
    end
    data = data[didx, :]

    @info "generating VTU file"
    # write vtk file
    VTU_cls = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:size(data, 1)]
    VTU_pts = Array{Float64}(data[:, 1:3]')
    vtk_grid(joinpath(@__DIR__, "assets/$(gpu_counts)_GPUs_result"), VTU_pts, VTU_cls) do vtk
        vtk["epII"]        = data[:, 4]
        vtk["mean_stress"] = data[:, 5]
        vtk["device_id"]   = data[:, 6]
    end
    return nothing
end