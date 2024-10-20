#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : postprocess.jl                                                             |
|  Description: Post-process functions                                                     |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. fastvtu()                                                               |
|               2. savevtu()   [2D & 3D]                                                   |
|               3. animation() [2D & 3D]                                                   |
+==========================================================================================#

export fastvtu, savevtu, animation

"""
    fastvtu(coords; vtupath="output", data::T=NamedTuple())

Description:
---
Generates a `.vtu` file by passing custom fields.
"""
function fastvtu(coords; vtupath="output", data::T=NamedTuple()) where T <: NamedTuple
    pts_num = size(coords, 1)
    vtu_cls = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:pts_num]
    vtu_pts = Array{Float64}(coords')
    vtk_grid(vtupath, vtu_pts, vtu_cls) do vtk
        keys(data) ≠ () && for vtu_key in keys(data)
            vtk[string(vtu_key)] = getfield(data, vtu_key)
        end
    end
    return nothing
end

"""
    savevtu(args::DeviceArgs2D{T1, T2}, grid::DeviceGrid2D{T1, T2}, 
        mp::DeviceParticle2D{T1, T2}, attr::DeviceProperty{T1, T2})

Description:
---
Generates the final geometry and properties in `.vtu` format (2D).
"""
@views function savevtu(
    args::    DeviceArgs2D{T1, T2}, 
    grid::    DeviceGrid2D{T1, T2}, 
    mp  ::DeviceParticle2D{T1, T2}, 
    attr::  DeviceProperty{T1, T2}
) where {T1, T2}
    prj_path = joinpath(args.project_path, args.project_name)
    mps_path = joinpath(prj_path, args.project_name)
    nds_path = joinpath(prj_path, "grid")
    # generate vtu files for particles
    VTU_cls = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:mp.np]
    VTU_pts = Array{Float64}(mp.ξ')
    vtk_grid(mps_path, VTU_pts, VTU_cls) do vtk
        vtk["stress"      ] = mp.σij[:, [1, 2, 4]]'
        vtk["strain_s"    ] = mp.ϵijs[:, [1, 2, 4]]'
        vtk["eqstrain"    ] = mp.ϵq
        vtk["eqrate"      ] = mp.ϵv
        vtk["ekstrain"    ] = mp.ϵk
        vtk["mass_s"      ] = mp.ms
        vtk["nid"         ] = attr.nid
        vtk["volume"      ] = mp.Ω
        vtk["velocity_s"  ] = mp.vs'
        vtk["displacement"] = abs.(mp.ξ .- mp.ξ0)'
        args.coupling==:TS ? (
            vtk["strain_w"  ] = mp.ϵijw[:, [1, 2, 4]]';
            vtk["pressure_w"] = mp.σw                 ;
            vtk["velocity_w"] = mp.vw'                ;
            vtk["porosity"  ] = mp.n                  ;
        ) : nothing
    end
    # generate vtu files for nodes
    VTU_cls = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:grid.ni]
    VTU_pts = Array{Float64}(grid.ξ')
    vtk_grid(nds_path, VTU_pts, VTU_cls) do vtk end
    @info "final vtu file is saved in project path"
    return nothing
end

"""
    savevtu(args::DeviceArgs3D{T1, T2}, grid::DeviceGrid3D{T1, T2}, 
        mp::DeviceParticle3D{T1, T2}, attr::DeviceProperty{T1, T2})

Description:
---
Generates the final geometry and properties in `.vtu` format (3D).
"""
@views function savevtu(
    args::    DeviceArgs3D{T1, T2}, 
    grid::    DeviceGrid3D{T1, T2}, 
    mp  ::DeviceParticle3D{T1, T2}, 
    attr::  DeviceProperty{T1, T2}
) where {T1, T2}
    prj_path = joinpath(args.project_path, args.project_name)
    mps_path = joinpath(prj_path, args.project_name)
    nds_path = joinpath(prj_path, "grid")
    # generate vtu files for particles
    VTU_cls = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:mp.np]
    VTU_pts = Array{Float64}(mp.ξ')
    vtk_grid(mps_path, VTU_pts, VTU_cls) do vtk
        vtk["stress"      ] = mp.σij'
        vtk["strain_s"    ] = mp.ϵijs'
        vtk["eqstrain"    ] = mp.ϵq
        vtk["eqrate"      ] = mp.ϵv
        vtk["ekstrain"    ] = mp.ϵk
        vtk["mass_s"      ] = mp.ms
        vtk["nid"         ] = attr.nid
        vtk["volume"      ] = mp.Ω
        vtk["velocity_s"  ] = mp.vs'
        vtk["displacement"] = abs.(mp.ξ .- mp.ξ0)'
        args.coupling==:TS ? (
            vtk["strain_w"  ] = mp.ϵijw';
            vtk["pressure_w"] = mp.σw   ;
            vtk["velocity_w"] = mp.vw'  ;
            vtk["porosity"  ] = mp.n    ;
        ) : nothing
    end
    # generate vtu files for nodes
    VTU_cls = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:grid.ni]
    VTU_pts = Array{Float64}(grid.ξ')
    vtk_grid(nds_path, VTU_pts, VTU_cls) do vtk end
    @info "final vtu file is saved in project path"
    return nothing
end

"""
    animation(args::DeviceArgs2D{T1, T2})

Description:
---
Generates animation by using the data from HDF5 file (2D).
"""
@views function animation(args::DeviceArgs2D{T1, T2}) where {T1, T2}
    proj_path = joinpath(args.project_path, args.project_name)
    h5filedir = joinpath(proj_path, "$(args.project_name).h5")
    anim_path = mkpath(joinpath(proj_path, "animation"))
    nds_path  = joinpath(proj_path, "grid")
    mps_path  = joinpath(proj_path, args.project_name)
    fid       = h5open(h5filedir, "r")
    itr       = (read(fid["FILE_NUM"])-1) |> Int64
    nid       = fid["nid"        ] |> read
    ξ0        = fid["mp_coords0" ] |> read
    grid_ξ    = fid["grid_coords"] |> read
    ni        = size(grid_ξ, 1)
    p         = Progress(length(1:1:itr) - 1; 
        desc      = "\e[1;36m[ Info:\e[0m $(lpad("ani_vtu", 7))",
        color     = :white,
        barlen    = 12,
        barglyphs = BarGlyphs(" ■■  ")
    )
    # generate files for particles
    paraview_collection(mps_path) do pvd
        @inbounds for i in 1:itr
            # read data from HDF5 file
            time = fid["group$(i)/time"      ] |> read
            σij  = fid["group$(i)/stress"    ] |> read
            ϵijs = fid["group$(i)/strain_s"  ] |> read
            ϵq   = fid["group$(i)/eqstrain"  ] |> read
            ϵk   = fid["group$(i)/ekstrain"  ] |> read
            ϵv   = fid["group$(i)/eqrate"    ] |> read
            vs   = fid["group$(i)/velocity_s"] |> read
            ms   = fid["group$(i)/mass_s"    ] |> read
            Ω    = fid["group$(i)/volume"    ] |> read
            ξ    = fid["group$(i)/coords"    ] |> read
            args.coupling==:TS ? (
                σw   = fid["group$(i)/pressure_w"] |> read;
                ϵijw = fid["group$(i)/strain_w"  ] |> read;
                vw   = fid["group$(i)/velocity_w"] |> read;
                n    = fid["group$(i)/porosity"  ] |> read;
            ) : nothing
            # write data
            VTU_cls = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:size(ξ0, 1)]
            VTU_pts = Array{Float64}(ξ')
            let vtk = vtk_grid(joinpath(anim_path, "iteration_$(i)"), VTU_pts, VTU_cls)
                vtk["stress"      ] = σij[:, [1, 2, 4]]'
                vtk["strain_s"    ] = ϵijs[:, [1, 2, 4]]'
                vtk["eqstrain"    ] = ϵq
                vtk["ekstrain"    ] = ϵk
                vtk["eqrate"      ] = ϵv
                vtk["mass_s"      ] = ms
                vtk["volume"      ] = Ω
                vtk["velocity_s"  ] = vs'
                vtk["nid"         ] = nid
                vtk["displacement"] = abs.(ξ .- ξ0)'
                args.coupling==:TS ? (
                    vtk["strain_w"  ] = ϵijw[:, [1, 2, 4]]';
                    vtk["pressure_w"] = σw                 ;
                    vtk["velocity_w"] = vw'                ;
                    vtk["porosity"  ] = n                  ;
                ) : nothing
                pvd[time] = vtk
            end
            next!(p)
        end
    end
    # generate vtu files for nodes
    VTU_cls = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:ni]
    VTU_pts = Array{Float64}(grid_ξ')
    vtk_grid(nds_path, VTU_pts, VTU_cls) do vtk end
    close(fid)
end

"""
    animation(args::DeviceArgs3D{T1, T2})

Description:
---
Generates animation by using the data from HDF5 file (3D).
"""
@views function animation(args::DeviceArgs3D{T1, T2}) where {T1, T2}
    proj_path = joinpath(args.project_path, args.project_name)
    h5filedir = joinpath(proj_path, "$(args.project_name).h5")
    anim_path = mkpath(joinpath(proj_path, "animation"))
    nds_path  = joinpath(proj_path, "grid")
    mps_path  = joinpath(proj_path, args.project_name)
    fid       = h5open(h5filedir, "r")
    itr       = (read(fid["FILE_NUM"])-1) |> Int64
    nid       = fid["nid"        ] |> read
    ξ0        = fid["mp_coords0" ] |> read
    grid_ξ    = fid["grid_coords"] |> read
    ni        = size(grid_ξ, 1)
    p         = Progress(length(1:1:itr) - 1; 
        desc      = "\e[1;36m[ Info:\e[0m $(lpad("ani_vtu", 7))",
        color     = :white,
        barlen    = 12,
        barglyphs = BarGlyphs(" ■■  ")
    )
    # generate files for particles
    paraview_collection(mps_path) do pvd
        @inbounds for i in 1:itr
            # read data from HDF5 file
            time = fid["group$(i)/time"      ] |> read
            σij  = fid["group$(i)/stress"    ] |> read
            ϵijs = fid["group$(i)/strain_s"  ] |> read
            ϵq   = fid["group$(i)/eqstrain"  ] |> read
            ϵk   = fid["group$(i)/ekstrain"  ] |> read
            ϵv   = fid["group$(i)/eqrate"    ] |> read
            vs   = fid["group$(i)/velocity_s"] |> read
            ms   = fid["group$(i)/mass_s"    ] |> read
            Ω    = fid["group$(i)/volume"    ] |> read
            ξ    = fid["group$(i)/coords"    ] |> read
            args.coupling==:TS ? (
                σw   = fid["group$(i)/pressure_w"] |> read;
                ϵijw = fid["group$(i)/strain_w"  ] |> read;
                vw   = fid["group$(i)/velocity_w"] |> read;
                n    = fid["group$(i)/porosity"  ] |> read;
            ) : nothing            
            # write data
            VTU_cls = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:size(ξ0, 1)]
            VTU_pts = Array{Float64}(ξ')
            let vtk = vtk_grid(joinpath(anim_path, "iteration_$(i)"), VTU_pts, VTU_cls)
                vtk["stress"      ] = σij'
                vtk["strain_s"    ] = ϵijs'
                vtk["eqstrain"    ] = ϵq
                vtk["ekstrain"    ] = ϵk
                vtk["eqrate"      ] = ϵv
                vtk["mass_s"      ] = ms
                vtk["volume"      ] = Ω
                vtk["velocity_s"  ] = vs'
                vtk["nid"         ] = nid
                vtk["displacement"] = abs.(ξ .- ξ0)'
                args.coupling==:TS ? (
                    vtk["strain_w"  ] = ϵijw';
                    vtk["pressure_w"] = σw   ;
                    vtk["velocity_w"] = vw'  ;
                    vtk["porosity"  ] = n    ;
                ) : nothing
                pvd[time] = vtk
            end
            next!(p)
        end
    end
    # generate vtu files for nodes
    VTU_cls = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:ni]
    VTU_pts = Array{Float64}(grid_ξ')
    vtk_grid(nds_path, VTU_pts, VTU_cls) do vtk end
    close(fid)
end