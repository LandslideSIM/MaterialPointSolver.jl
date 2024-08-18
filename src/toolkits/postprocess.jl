#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : postprocess.jl                                                             |
|  Description: Post-process functions                                                     |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. savevtu()   [2D & 3D]                                                   |
|               3. animation() [2D & 3D]                                                   |
+==========================================================================================#

"""
    savevtu(args::Args2D, grid::Grid2D, mp::Particle2D, pts_attr::ParticleProperty)

Description:
---
Generates the final geometry and properties in `.vtu` format (2D).
"""
@views function savevtu(args::Args2D, grid::Grid2D, mp::Particle2D, pts_attr::ParticleProperty)
    mps_path = joinpath(args.project_path, args.project_name)
    nds_path = joinpath(args.project_path, "grid")
    # generate vtu files for particles
    VTU_cls = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:mp.num]
    VTU_pts = Array{Float64}(mp.pos')
    vtk_grid(mps_path, VTU_pts, VTU_cls) do vtk
        vtk["stress"    ] = mp.σij[:, [1, 2, 4]]'
        vtk["strain_s"  ] = mp.ϵij_s[:, [1, 2, 4]]'
        vtk["epII"      ] = mp.epII
        vtk["epK"       ] = mp.epK
        vtk["mass"      ] = mp.Ms
        vtk["layer"     ] = pts_attr.layer
        vtk["vol"       ] = mp.vol
        vtk["velocity_s"] = mp.Vs'
        vtk["disp"      ] = abs.(mp.pos .- mp.init)'
        args.coupling==:TS ? (
            vtk["strain_w"     ] = mp.ϵij_w[:, [1, 2, 4]]';
            vtk["pore_pressure"] = mp.σw                  ;
            vtk["velocity_w"   ] = mp.Vw'                 ;
            vtk["porosity"     ] = mp.porosity            ;
        ) : nothing
    end
    # generate vtu files for nodes
    VTU_cls = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:grid.node_num]
    VTU_pts = Array{Float64}(grid.pos')
    vtk_grid(nds_path, VTU_pts, VTU_cls) do vtk end
    @info "final vtu file is saved in project path"
    return nothing
end

"""
    savevtu(args::Args3D, grid::Grid3D, mp::Particle3D, pts_attr::ParticleProperty)

Description:
---
Generates the final geometry and properties in `.vtu` format (3D).
"""
@views function savevtu(args::Args3D, grid::Grid3D, mp::Particle3D, pts_attr::ParticleProperty)
    mps_path = joinpath(args.project_path, args.project_name)
    nds_path = joinpath(args.project_path, "grid")
    # generate vtu files for particles
    VTU_cls = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:mp.num]
    VTU_pts = Array{Float64}(mp.pos')
    vtk_grid(mps_path, VTU_pts, VTU_cls) do vtk
        vtk["stress"    ] = mp.σij'
        vtk["strain_s"  ] = mp.ϵij_s'
        vtk["epII"      ] = mp.epII
        vtk["epK"       ] = mp.epK
        vtk["mass"      ] = mp.Ms
        vtk["layer"     ] = pts_attr.layer
        vtk["vol"       ] = mp.vol
        vtk["velocity_s"] = mp.Vs'
        vtk["disp"      ] = abs.(mp.pos .- mp.init)'
        args.coupling==:TS ? (
            vtk["strain_w"     ] = mp.ϵij_w'  ;
            vtk["pore_pressure"] = mp.σw      ;
            vtk["velocity_w"   ] = mp.Vw'     ;
            vtk["porosity"     ] = mp.porosity;
        ) : nothing
    end
    # generate vtu files for nodes
    VTU_cls = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:grid.node_num]
    VTU_pts = Array{Float64}(grid.pos')
    vtk_grid(nds_path, VTU_pts, VTU_cls) do vtk end
    @info "final vtu file is saved in project path"
    return nothing
end

"""
    animation(args::Args2D{T1, T2})

Description:
---
Generates animation by using the data from HDF5 file (2D).
"""
@views function animation(args::Args2D{T1, T2}) where {T1, T2}
    fid       = h5open(joinpath(args.project_path, "$(args.project_name).h5"), "r")
    itr       = (read(fid["FILE_NUM"])-1) |> Int64
    mp_init   = fid["mp_init" ] |> read
    grid_pos  = fid["grid_pos"] |> read
    layer     = fid["layer"   ] |> read
    node_num  = size(grid_pos, 1)
    anim_path = mkpath(joinpath(args.project_path, "animation"))
    mps_path  = joinpath(args.project_path, args.project_name)
    nds_path  = joinpath(args.project_path, "grid")
    p         = Progress(length(1:1:itr)-1; 
        desc      = "\e[1;36m[ Info:\e[0m $(lpad("ani_vtu", 7))",
        color     = :white,
        barlen    = 12,
        barglyphs = BarGlyphs(" ◼◼  ")
    )
    # generate files for particles
    paraview_collection(mps_path) do pvd
        @inbounds Threads.@threads for i in 1:itr
            # read data from HDF5 file
            time   = fid["group$(i)/time"  ] |> read
            sig    = fid["group$(i)/sig"   ] |> read
            eps_s  = fid["group$(i)/eps_s" ] |> read
            epII   = fid["group$(i)/epII"  ] |> read
            epK    = fid["group$(i)/epK"   ] |> read
            v_s    = fid["group$(i)/v_s"   ] |> read
            mass   = fid["group$(i)/mass"  ] |> read
            vol    = fid["group$(i)/vol"   ] |> read
            mp_pos = fid["group$(i)/mp_pos"] |> read
            args.coupling==:TS ? (
                pp       = fid["group$(i)/pp"      ] |> read;
                eps_w    = fid["group$(i)/eps_w"   ] |> read;
                v_w      = fid["group$(i)/v_w"     ] |> read;
                porosity = fid["group$(i)/porosity"] |> read;
            ) : nothing                        
            # write data
            VTU_cls = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:size(mp_init, 1)]
            VTU_pts = Array{Float64}(mp_pos')
            let vtk = vtk_grid(joinpath(anim_path, "iter_$(i)"), VTU_pts, VTU_cls)
                vtk["stress"    ] = sig[:, [1, 2, 4]]'
                vtk["strain_s"  ] = eps_s[:, [1, 2, 4]]'
                vtk["epII"      ] = epII
                vtk["epK"       ] = epK
                vtk["mass"      ] = mass
                vtk["vol"       ] = vol
                vtk["velocity_s"] = v_s'
                vtk["layer"     ] = layer
                vtk["disp"      ] = abs.(mp_pos .- mp_init)'
                args.coupling==:TS ? (
                    vtk["strain_w"     ] = eps_w[:, [1, 2, 4]]';
                    vtk["pore_pressure"] = pp                  ;
                    vtk["velocity_w"   ] = v_w'                ;
                    vtk["porosity"     ] = porosity            ;
                ) : nothing
                pvd[time] = vtk
            end
            next!(p)
        end
    end
    # generate vtu files for nodes
    VTU_cls = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:node_num]
    VTU_pts = Array{Float64}(grid_pos')
    vtk_grid(nds_path, VTU_pts, VTU_cls) do vtk end
    close(fid)
end

"""
    animation(args::Args3D{T1, T2})

Description:
---
Generates animation by using the data from HDF5 file (3D).
"""
@views function animation(args::Args3D{T1, T2}) where {T1, T2}
    fid       = h5open(joinpath(args.project_path, "$(args.project_name).h5"), "r")
    itr       = (read(fid["FILE_NUM"])-1) |> Int64
    mp_init   = fid["mp_init" ] |> read
    grid_pos  = fid["grid_pos"] |> read
    layer     = fid["layer"   ] |> read
    mp_num    = size(mp_init, 1)
    nd_num    = size(grid_pos, 1)
    anim_path = mkpath(joinpath(args.project_path, "animation"))
    mps_path  = joinpath(args.project_path, args.project_name)
    nds_path  = joinpath(args.project_path, "grid")
    p         = Progress(length(1:1:itr)-1; 
        desc      = "\e[1;36m[ Info:\e[0m $(lpad("ani_vtu", 7))",
        color     = :white,
        barlen    = 12,
        barglyphs = BarGlyphs(" ◼◼  ")
    )
    # generate files for particles
    paraview_collection(mps_path) do pvd
        @inbounds Threads.@threads for i in 1:itr
            # read data from HDF5 file
            time   = fid["group$(i)/time"  ] |> read
            sig    = fid["group$(i)/sig"   ] |> read
            eps_s  = fid["group$(i)/eps_s" ] |> read
            epII   = fid["group$(i)/epII"  ] |> read
            epK    = fid["group$(i)/epK"   ] |> read
            v_s    = fid["group$(i)/v_s"   ] |> read
            mass   = fid["group$(i)/mass"  ] |> read
            vol    = fid["group$(i)/vol"   ] |> read
            mp_pos = fid["group$(i)/mp_pos"] |> read
            args.coupling==:TS ? (
                pp       = fid["group$(i)/pp"      ] |> read;
                eps_w    = fid["group$(i)/eps_w"   ] |> read;
                v_w      = fid["group$(i)/v_w"     ] |> read;
                porosity = fid["group$(i)/porosity"] |> read;
            ) : nothing            
            # write data
            VTU_cls = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:mp_num]
            VTU_pts = Array{Float64}(mp_pos')
            let vtk = vtk_grid(joinpath(anim_path, "iteration_$(i)"), VTU_pts, VTU_cls)
                vtk["stress"    ] = sig'
                vtk["strain_s"  ] = eps_s'
                vtk["epII"      ] = epII
                vtk["epK"       ] = epK
                vtk["mass"      ] = mass
                vtk["vol"       ] = vol
                vtk["velocity_s"] = v_s'
                vtk["layer"     ] = layer
                vtk["disp"      ] = abs.(mp_pos .- mp_init)'
                args.coupling==:TS ? (
                    vtk["strain_w"     ] = eps_w'  ;
                    vtk["pore_pressure"] = pp      ;
                    vtk["velocity_w"   ] = v_w'    ;
                    vtk["porosity"     ] = porosity;
                ) : nothing
                pvd[time] = vtk
            end
            next!(p)
        end
    end
    # generate vtu files for nodes
    VTU_cls = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:nd_num]
    VTU_pts = Array{Float64}(grid_pos')
    vtk_grid(nds_path, VTU_pts, VTU_cls) do vtk end
    close(fid)
end