#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : modelinfo.jl                                                               |
|  Description: Export or import model in JSON format                                      |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 01. json2args2D                                                            |
|               02. json2args3D                                                            |
|               03. json2grid2D                                                            |
|               04. json2grid3D                                                            |
|               05. json2particle2D                                                        |
|               06. json2particle3D                                                        |
|               07. json2particleproperty                                                  |
|               08. json2vboundary2D                                                       |
|               09. json2vboundary3D                                                       |
|               10. export_model                                                           |
|               11. import_model                                                           |
|               12. check_datasize                                                         |
+==========================================================================================#

function json2args2D(jsondata, iInt, iFloat)
    return Args2D{iInt, iFloat}(
        Ttol         = jsondata.Ttol,
        Te           = jsondata.Te,
        ΔT           = jsondata.ΔT,
        time_step    = jsondata.time_step    |> Symbol,
        FLIP         = jsondata.FLIP,
        PIC          = jsondata.PIC,
        constitutive = jsondata.constitutive |> Symbol,
        basis        = jsondata.basis        |> Symbol,
        animation    = jsondata.animation,
        hdf5         = jsondata.hdf5,
        hdf5_step    = jsondata.hdf5_step,
        MVL          = jsondata.MVL,
        device       = jsondata.device       |> Symbol,
        coupling     = jsondata.coupling     |> Symbol,
        scheme       = jsondata.scheme       |> Symbol,
        progressbar  = jsondata.progressbar,
        gravity      = jsondata.gravity,
        ζs           = jsondata.ζs,
        ζw           = jsondata.ζw,
        αT           = jsondata.αT,
        iter_num     = jsondata.iter_num,
        end_time     = jsondata.end_time,
        start_time   = jsondata.start_time,
        project_name = jsondata.project_name,
        project_path = jsondata.project_path
    )
end

function json2args3D(jsondata, iInt, iFloat)
    return Args3D{iInt, iFloat}(
        Ttol         = jsondata.Ttol,
        Te           = jsondata.Te,
        ΔT           = jsondata.ΔT,
        time_step    = jsondata.time_step    |> Symbol,
        FLIP         = jsondata.FLIP,
        PIC          = jsondata.PIC,
        constitutive = jsondata.constitutive |> Symbol,
        basis        = jsondata.basis        |> Symbol,
        animation    = jsondata.animation,
        hdf5         = jsondata.hdf5,
        hdf5_step    = jsondata.hdf5_step,
        MVL          = jsondata.MVL,
        device       = jsondata.device       |> Symbol,
        coupling     = jsondata.coupling     |> Symbol,
        scheme       = jsondata.scheme       |> Symbol,
        progressbar  = jsondata.progressbar,
        gravity      = jsondata.gravity,
        ζs           = jsondata.ζs,
        ζw           = jsondata.ζw,
        αT           = jsondata.αT,
        iter_num     = jsondata.iter_num,
        end_time     = jsondata.end_time,
        start_time   = jsondata.start_time,
        project_name = jsondata.project_name,
        project_path = jsondata.project_path
    )
end

function json2grid2D(jsondata, iInt, iFloat)
    DoF      = 2
    node_num = jsondata.node_num
    NIC      = jsondata.NIC
    if  jsondata.phase==1
        cell_new = 1
        node_new = 1
        DoF_new  = 1
    elseif jsondata.phase==2
        cell_new = jsondata.cell_num
        node_new = jsondata.node_num
        DoF_new  = 2
    end
    return Grid2D{iInt, iFloat}(
        range_x1   = jsondata.range_x1                           ,
        range_x2   = jsondata.range_x2                           ,
        range_y1   = jsondata.range_y1                           ,
        range_y2   = jsondata.range_y2                           ,
        space_x    = jsondata.space_x                            ,
        space_y    = jsondata.space_y                            ,
        phase      = jsondata.phase                              ,
        node_num_x = jsondata.node_num_x                         ,
        node_num_y = jsondata.node_num_y                         ,
        node_num   = jsondata.node_num                           ,
        NIC        = jsondata.NIC                                ,
        pos        = reshape(jsondata.pos    , node_num, DoF    ),
        cell_num_x = jsondata.cell_num_x                         ,
        cell_num_y = jsondata.cell_num_y                         ,
        cell_num   = jsondata.cell_num                           ,
        tab_p2n    = reshape(jsondata.tab_p2n, NIC     , 2      ),
        σm         = jsondata.σm                                 ,
        σw         = jsondata.σw                                 ,
        vol        = jsondata.vol                                ,
        Ms         = jsondata.Ms                                 ,
        Mw         = jsondata.Mw                                 ,
        Mi         = jsondata.Mi                                 ,
        Ps         = reshape(jsondata.Ps     , node_num, DoF    ),
        Pw         = reshape(jsondata.Pw     , node_new, DoF_new),
        Vs         = reshape(jsondata.Vs     , node_num, DoF    ),
        Vw         = reshape(jsondata.Vw     , node_new, DoF_new),
        Vs_T       = reshape(jsondata.Vs_T   , node_num, DoF    ),
        Vw_T       = reshape(jsondata.Vw_T   , node_new, DoF_new),
        Fs         = reshape(jsondata.Fs     , node_num, DoF    ),
        Fw         = reshape(jsondata.Fw     , node_new, DoF_new),
        Fdrag      = reshape(jsondata.Fdrag  , node_new, DoF_new),
        a_s        = reshape(jsondata.a_s    , node_num, DoF    ),
        a_w        = reshape(jsondata.a_w    , node_new, DoF_new),
        Δd_s       = reshape(jsondata.Δd_s   , node_num, DoF    ),
        Δd_w       = reshape(jsondata.Δd_w   , node_new, DoF_new)
    )
end

function json2grid3D(jsondata, iInt, iFloat)
    DoF      = 3
    node_num = jsondata.node_num
    NIC      = jsondata.NIC
    if  jsondata.phase==1
        cell_new = 1
        node_new = 1
        DoF_new  = 1
    elseif jsondata.phase==2
        cell_new = jsondata.cell_num
        node_new = jsondata.node_num
        DoF_new  = 3
    end
    return Grid3D{iInt, iFloat}(
        range_x1   = jsondata.range_x1                           ,
        range_x2   = jsondata.range_x2                           ,
        range_y1   = jsondata.range_y1                           ,
        range_y2   = jsondata.range_y2                           ,
        range_z1   = jsondata.range_z1                           ,
        range_z2   = jsondata.range_z2                           ,
        space_x    = jsondata.space_x                            ,
        space_y    = jsondata.space_y                            ,
        space_z    = jsondata.space_z                            ,
        phase      = jsondata.phase                              ,
        node_num_x = jsondata.node_num_x                         ,
        node_num_y = jsondata.node_num_y                         ,
        node_num_z = jsondata.node_num_z                         ,
        node_num   = jsondata.node_num                           ,
        NIC        = jsondata.NIC                                ,
        pos        = reshape(jsondata.pos    , node_num, DoF    ),
        cell_num_x = jsondata.cell_num_x                         ,
        cell_num_y = jsondata.cell_num_y                         ,
        cell_num_z = jsondata.cell_num_z                         ,
        cell_num   = jsondata.cell_num                           ,
        tab_p2n    = reshape(jsondata.tab_p2n, NIC     , 4      ),
        σm         = jsondata.σm                                 ,
        σw         = jsondata.σw                                 ,
        vol        = jsondata.vol                                ,
        Ms         = jsondata.Ms                                 ,
        Mw         = jsondata.Mw                                 ,
        Mi         = jsondata.Mi                                 ,
        Ps         = reshape(jsondata.Ps     , node_num, DoF    ),
        Pw         = reshape(jsondata.Pw     , node_new, DoF_new),
        Vs         = reshape(jsondata.Vs     , node_num, DoF    ),
        Vw         = reshape(jsondata.Vw     , node_new, DoF_new),
        Vs_T       = reshape(jsondata.Vs_T   , node_num, DoF    ),
        Vw_T       = reshape(jsondata.Vw_T   , node_new, DoF_new),
        Fs         = reshape(jsondata.Fs     , node_num, DoF    ),
        Fw         = reshape(jsondata.Fw     , node_new, DoF_new),
        Fdrag      = reshape(jsondata.Fdrag  , node_new, DoF_new),
        a_s        = reshape(jsondata.a_s    , node_num, DoF    ),
        a_w        = reshape(jsondata.a_w    , node_new, DoF_new),
        Δd_s       = reshape(jsondata.Δd_s   , node_num, DoF    ),
        Δd_w       = reshape(jsondata.Δd_w   , node_new, DoF_new)
    )
end

function json2particle2D(jsondata, iInt, iFloat)
    DoF = 2
    NIC = jsondata.NIC
    num = jsondata.num
    if jsondata.phase==1 
        num_new = 1
        DoF_new = 1
    elseif jsondata.phase==2 
        num_new = num
        DoF_new = DoF
    end
    return Particle2D{iInt, iFloat}(    
        num      = jsondata.num                              ,
        phase    = jsondata.phase                            ,
        NIC      = jsondata.NIC                              ,
        space_x  = jsondata.space_x                          ,
        space_y  = jsondata.space_y                          ,
        p2c      = jsondata.p2c                              ,
        p2n      = reshape(jsondata.p2n   , num    , NIC    ),
        pos      = reshape(jsondata.pos   , num    , DoF    ),
        σm       = jsondata.σm                               ,
        J        = jsondata.J                                ,
        epII     = jsondata.epII                             ,
        epK      = jsondata.epK                              ,
        vol      = jsondata.vol                              ,
        vol_init = jsondata.vol_init                         ,
        Ms       = jsondata.Ms                               ,
        Mw       = jsondata.Mw                               ,
        Mi       = jsondata.Mi                               ,
        porosity = jsondata.porosity                         ,
        cfl      = jsondata.cfl                              ,
        ρs       = jsondata.ρs                               ,
        ρs_init  = jsondata.ρs_init                          ,
        ρw       = jsondata.ρw                               ,
        ρw_init  = jsondata.ρw_init                          ,  
        init     = reshape(jsondata.init  , num    , DoF    ),
        σw       = jsondata.σw                               ,
        σij      = reshape(jsondata.σij   , num    , 4      ),
        ϵij_s    = reshape(jsondata.ϵij_s , num    , 4      ),
        ϵij_w    = reshape(jsondata.ϵij_w , num_new, 4      ),
        Δϵij_s   = reshape(jsondata.Δϵij_s, num    , 4      ),
        Δϵij_w   = reshape(jsondata.Δϵij_w, num_new, 4      ),
        sij      = reshape(jsondata.sij   , num    , 4      ),
        Vs       = reshape(jsondata.Vs    , num    , DoF    ),
        Vw       = reshape(jsondata.Vw    , num_new, DoF_new),
        Ps       = reshape(jsondata.Ps    , num    , DoF    ),
        Pw       = reshape(jsondata.Pw    , num_new, DoF_new),
        Ni       = reshape(jsondata.Ni    , num    , NIC    ),
        ∂Nx      = reshape(jsondata.∂Nx   , num    , NIC    ),
        ∂Ny      = reshape(jsondata.∂Ny   , num    , NIC    ),
        ΔFs      = reshape(jsondata.ΔFs   , num    , 4      ),
        ΔFw      = reshape(jsondata.ΔFw   , num_new, 4      ),
        F        = reshape(jsondata.F     , num    , 4      )
    )
end

function json2particle3D(jsondata, iInt, iFloat)
    DoF = 3
    NIC = jsondata.NIC
    num = jsondata.num
    if phase==1 
        num_new = 1
        DoF_new = 1
    elseif phase==2 
        num_new = num
        DoF_new = DoF
    end
    return Particle3D{iInt, iFloat}(
        num      = jsondata.num                              , 
        phase    = jsondata.phase                            ,
        NIC      = jsondata.NIC                              ,
        space_x  = jsondata.space_x                          ,
        space_y  = jsondata.space_y                          ,
        space_z  = jsondata.space_z                          ,
        p2c      = jsondata.p2c                              ,
        p2n      = reshape(jsondata.p2n   , num    , NIC    ),
        pos      = reshape(jsondata.pos   , num    , DoF    ),
        σm       = jsondata.σm                               ,
        J        = jsondata.J                                ,
        epII     = jsondata.epII                             ,
        epK      = jsondata.epK                              ,
        vol      = jsondata.vol                              ,
        vol_init = jsondata.vol_init                         ,
        Ms       = jsondata.Ms                               ,
        Mw       = jsondata.Mw                               ,
        Mi       = jsondata.Mi                               ,
        porosity = jsondata.porosity                         ,
        cfl      = jsondata.cfl                              ,
        ρs       = jsondata.ρs                               ,
        ρs_init  = jsondata.ρs_init                          ,
        ρw       = jsondata.ρw                               ,
        ρw_init  = jsondata.ρw_init                          ,
        init     = reshape(jsondata.init  , num    , DoF    ),
        σw       = jsondata.σw                               ,
        σij      = reshape(jsondata.σij   , num    , 6      ),
        ϵij_s    = reshape(jsondata.ϵij_s , num    , 6      ),
        ϵij_w    = reshape(jsondata.ϵij_w , num_new, 6      ),
        Δϵij_s   = reshape(jsondata.Δϵij_s, num    , 6      ),
        Δϵij_w   = reshape(jsondata.Δϵij_w, num_new, 6      ),
        sij      = reshape(jsondata.sij   , num    , 6      ),
        Vs       = reshape(jsondata.Vs    , num    , DoF    ),
        Vw       = reshape(jsondata.Vw    , num_new, DoF_new),
        Ps       = reshape(jsondata.Ps    , num    , DoF    ),
        Pw       = reshape(jsondata.Pw    , num_new, DoF_new),
        Ni       = reshape(jsondata.Ni    , num    , NIC    ),
        ∂Nx      = reshape(jsondata.∂Nx   , num    , NIC    ),
        ∂Ny      = reshape(jsondata.∂Ny   , num    , NIC    ),
        ∂Nz      = reshape(jsondata.∂Nz   , num    , NIC    ),
        ΔFs      = reshape(jsondata.ΔFs   , num    , 9      ),
        ΔFw      = reshape(jsondata.ΔFw   , num_new, 9      ),
        F        = reshape(jsondata.F     , num    , 9      )
    )
end

function json2particleproperty(jsondata, iInt, iFloat)
    return ParticleProperty{iInt, iFloat}(
        layer = jsondata.layer,
        ν     = jsondata.ν    ,
        E     = jsondata.E    ,
        G     = jsondata.G    ,
        Ks    = jsondata.Ks   ,
        Kw    = jsondata.Kw   ,
        k     = jsondata.k    ,
        σt    = jsondata.σt   ,
        ϕ     = jsondata.ϕ    ,
        ψ     = jsondata.ψ    ,     
        c     = jsondata.c    ,
        cr    = jsondata.cr   ,
        Hp    = jsondata.Hp   ,
        tmp1  = jsondata.tmp1 ,
        tmp2  = jsondata.tmp2
    )
end

function json2vboundary2D(jsondata, iInt, iFloat)
    return VBoundary2D{iInt, iFloat}(
        Vx_s_Idx = jsondata.Vx_s_Idx,
        Vx_s_Val = jsondata.Vx_s_Val,
        Vy_s_Idx = jsondata.Vy_s_Idx,
        Vy_s_Val = jsondata.Vy_s_Val,
        Vx_w_Idx = jsondata.Vx_w_Idx,
        Vx_w_Val = jsondata.Vx_w_Val,
        Vy_w_Idx = jsondata.Vy_w_Idx,
        Vy_w_Val = jsondata.Vy_w_Val,
        tmp1     = jsondata.tmp1,
        tmp2     = jsondata.tmp2
    )
end

function json2vboundary3D(jsondata, iInt, iFloat)
    return VBoundary3D{iInt, iFloat}(
        Vx_s_Idx = jsondata.Vx_s_Idx,
        Vx_s_Val = jsondata.Vx_s_Val,
        Vy_s_Idx = jsondata.Vy_s_Idx,
        Vy_s_Val = jsondata.Vy_s_Val,
        Vz_s_Idx = jsondata.Vz_s_Idx,
        Vz_s_Val = jsondata.Vz_s_Val,
        Vx_w_Idx = jsondata.Vx_w_Idx,
        Vx_w_Val = jsondata.Vx_w_Val,
        Vy_w_Idx = jsondata.Vy_w_Idx,
        Vy_w_Val = jsondata.Vy_w_Val,
        Vz_w_Idx = jsondata.Vz_w_Idx,
        Vz_w_Val = jsondata.Vz_w_Val,
        tmp1     = jsondata.tmp1,
        tmp2     = jsondata.tmp2
    )
end

function export_model(args, grid, mp, pts_attr, bc, exportdir)
    exportdir = joinpath(exportdir, args.project_name)
    rm(exportdir, recursive=true, force=true); mkpath(exportdir)
    open(joinpath(exportdir, "args.json"), "w") do io
        JSON3.pretty(io, args)
    end
    open(joinpath(exportdir, "grid.json"), "w") do io
        JSON3.pretty(io, grid)
    end
    open(joinpath(exportdir, "mp.json"), "w") do io
        JSON3.pretty(io, mp)
    end
    open(joinpath(exportdir, "pts_attr.json"), "w") do io
        JSON3.pretty(io, pts_attr)
    end
    open(joinpath(exportdir, "bc.json"), "w") do io
        JSON3.pretty(io, bc)
    end
    @info "model info (JSON) exported to project path"
end

function import_model(importdir, precision::Symbol=:FP64)
    jsonargs = JSON3.read(joinpath(importdir, "args.json"    ))
    jsongrid = JSON3.read(joinpath(importdir, "grid.json"    ))
    jsonmp   = JSON3.read(joinpath(importdir, "mp.json"      ))
    jsonpts  = JSON3.read(joinpath(importdir, "pts_attr.json"))
    jsonbc   = JSON3.read(joinpath(importdir, "bc.json"      ))
    precision==:FP64 ? (iInt=Int64; iFloat=Float64) : (iInt=Int32; iFloat=Float32)
    if haskey(jsongrid, "space_z")
        args     = json2args3D(          jsonargs, iInt, iFloat)
        grid     = json2grid3D(          jsongrid, iInt, iFloat)
        mp       = json2particle3D(      jsonmp  , iInt, iFloat)
        pts_attr = json2particleproperty(jsonpts , iInt, iFloat)
        bc       = json2vboundary3D(     jsonbc  , iInt, iFloat)
    else
        args     = json2args2D(          jsonargs, iInt, iFloat)
        grid     = json2grid2D(          jsongrid, iInt, iFloat)
        mp       = json2particle2D(      jsonmp  , iInt, iFloat)
        pts_attr = json2particleproperty(jsonpts , iInt, iFloat)
        bc       = json2vboundary2D(     jsonbc  , iInt, iFloat)
    end
    return args, grid, mp, pts_attr, bc
end

function check_datasize(args    ::MODELARGS, 
                        grid    ::GRID, 
                        mp      ::PARTICLE, 
                        pts_attr::PROPERTY, 
                        bc      ::BOUNDARY)
    args_mem     = Base.summarysize(args)
    grid_mem     = Base.summarysize(grid)
    mp_mem       = Base.summarysize(mp)
    pts_attr_mem = Base.summarysize(pts_attr)
    bc_mem       = Base.summarysize(bc)
    mem_total    = args_mem+grid_mem+mp_mem+pts_attr_mem+bc_mem

    args_size     = lpad(@sprintf("%.2f", args_mem    /1024^3), 5)
    grid_size     = lpad(@sprintf("%.2f", grid_mem    /1024^3), 5)
    mp_size       = lpad(@sprintf("%.2f", mp_mem      /1024^3), 5)
    pts_attr_size = lpad(@sprintf("%.2f", pts_attr_mem/1024^3), 5)
    bc_size       = lpad(@sprintf("%.2f", bc_mem      /1024^3), 5)
    
    argsp      = lpad(@sprintf("%.2f", args_mem    /mem_total*100), 5)
    gridp      = lpad(@sprintf("%.2f", grid_mem    /mem_total*100), 5)
    mpp        = lpad(@sprintf("%.2f", mp_mem      /mem_total*100), 5)
    pts_attr_p = lpad(@sprintf("%.2f", pts_attr_mem/mem_total*100), 5)
    bcp        = lpad(@sprintf("%.2f", bc_mem      /mem_total*100), 5)

    tbar = string("─"^19, "┬", "─"^11, "┬", "─"^8)
    bbar = string("─"^19, "┴", "─"^11, "┴", "─"^8)
    @info """model data size
    $(tbar)
    MPM model args     │ $args_size GiB │ $argsp %
    background grid    │ $grid_size GiB │ $gridp %
    material particles │ $mp_size GiB │ $mpp %
    particle properties│ $pts_attr_size GiB │ $pts_attr_p %
    boundary conditions│ $bc_size GiB │ $bcp %
    $(bbar)
    """
    return nothing
end

macro memcheck(expr)
    return quote        
        # 检查是否是CuArray并转换
        if isa($(esc(expr)), CuArray)
            $(esc(expr)) = Array($(esc(expr)))
        end
        # 计算内存大小
        data = Base.summarysize($(esc(expr))) / 1024^3
        data_str = @sprintf("%.2f GiB", data)
        # 打印信息
        @info "💾 $data_str"
    end
end



