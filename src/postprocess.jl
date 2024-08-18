#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : postprocess.jl                                                             |
|  Description: Post-process functions                                                     |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. savevtu()   [2D]                                                        |
|               2. savevtu()   [3D]                                                        |
|               3. animation() [2D]                                                        |
|               4. animation() [3D]                                                        |
+==========================================================================================#

"""
    savevtu(args::Args2D, mp::Particle2D)

Description:
---
Generates the final geometry and properties in `.vtu` format (2D).
"""
@views function savevtu(args::Args2D, mp::Particle2D)
    mps_path = joinpath(args.project_path, args.project_name)
    nds_path = joinpath(args.project_path, "grid")
    # generate vtu files for particles
    VTU_cls = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:mp.num]
    VTU_pts = Array{Float64}(mp.pos')
    vtk_grid(mps_path, VTU_pts, VTU_cls) do vtk
        vtk["sig_xx"  ] = mp.σij[:, 1]
        vtk["sig_yy"  ] = mp.σij[:, 2]
        vtk["sig_xy"  ] = mp.σij[:, 4]
        vtk["sig_m"   ] = (mp.σij[:, 1].+mp.σij[:, 2].+mp.σij[:, 3])./3
        vtk["eps_s_xx"] = mp.ϵij_s[:, 1]
        vtk["eps_s_yy"] = mp.ϵij_s[:, 2]
        vtk["eps_s_xy"] = mp.ϵij_s[:, 4]
        vtk["epII"    ] = mp.epII
        vtk["epK"     ] = mp.epK
        vtk["mass"    ] = mp.Ms
        vtk["vol"     ] = mp.vol
        vtk["v_s_x"   ] = mp.Vs[:, 1]
        vtk["v_s_y"   ] = mp.Vs[:, 2]
        vtk["disp_x"  ] = abs.(mp.pos[:, 1].-mp.init[:, 1])
        vtk["disp_y"  ] = abs.(mp.pos[:, 2].-mp.init[:, 2])
        vtk["disp"    ] = sqrt.((mp.pos[:, 1].-mp.init[:, 1]).^2 .+
                                (mp.pos[:, 2].-mp.init[:, 2]).^2)
        args.coupling==:TS ? (
            vtk["eps_w_xx"] = mp.ϵij_w[:, 1];
            vtk["eps_w_yy"] = mp.ϵij_w[:, 2];
            vtk["eps_w_xy"] = mp.ϵij_w[:, 4];
            vtk["pp"      ] = mp.σw         ;
            vtk["v_w_x"   ] = mp.Vw[:, 1]   ;
            vtk["v_w_y"   ] = mp.Vw[:, 2]   ;
            vtk["porosity"] = mp.porosity   ;
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
    savevtu(args::Args3D, mp::Particle3D)

Description:
---
Generates the final geometry and properties in `.vtu` format (3D).
"""
@views function savevtu(args::Args3D, mp::Particle3D)
    mps_path = joinpath(args.project_path, args.project_name)
    nds_path = joinpath(args.project_path, "grid")
    # generate vtu files for particles
    VTU_cls = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:mp.num]
    VTU_pts = Array{Float64}(mp.pos')
    vtk_grid(mps_path, VTU_pts, VTU_cls) do vtk
        vtk["sig_xx"  ] = mp.σij[:, 1]
        vtk["sig_yy"  ] = mp.σij[:, 2]
        vtk["sig_zz"  ] = mp.σij[:, 3]
        vtk["sig_xy"  ] = mp.σij[:, 4]
        vtk["sig_yz"  ] = mp.σij[:, 5]
        vtk["sig_zx"  ] = mp.σij[:, 6]
        vtk["sig_m"   ] = (mp.σij[:, 1].+mp.σij[:, 2].+mp.σij[:, 3])./3
        vtk["eps_s_xx"] = mp.ϵij_s[:, 1]
        vtk["eps_s_yy"] = mp.ϵij_s[:, 2]
        vtk["eps_s_zz"] = mp.ϵij_s[:, 3]
        vtk["eps_s_xy"] = mp.ϵij_s[:, 4]
        vtk["eps_s_yz"] = mp.ϵij_s[:, 5]
        vtk["eps_s_zx"] = mp.ϵij_s[:, 6]
        vtk["epII"    ] = mp.epII
        vtk["epK"     ] = mp.epK
        vtk["mass"    ] = mp.Ms
        vtk["vol"     ] = mp.vol
        vtk["v_s_x"   ] = mp.Vs[:, 1]
        vtk["v_s_y"   ] = mp.Vs[:, 2]
        vtk["v_s_z"   ] = mp.Vs[:, 3]
        vtk["disp_x"  ] = abs.(mp.pos[:, 1].-mp.init[:, 1])
        vtk["disp_y"  ] = abs.(mp.pos[:, 2].-mp.init[:, 2])
        vtk["disp_z"  ] = abs.(mp.pos[:, 3].-mp.init[:, 3])
        vtk["disp"    ] = sqrt.((mp.pos[:, 1].-mp.init[:, 1]).^2 .+
                                (mp.pos[:, 2].-mp.init[:, 2]).^2 .+
                                (mp.pos[:, 3].-mp.init[:, 3]).^2)
        args.coupling==:TS ? (
            vtk["eps_w_xx"] = mp.ϵij_w[:, 1];
            vtk["eps_w_yy"] = mp.ϵij_w[:, 2];
            vtk["eps_w_zz"] = mp.ϵij_w[:, 3];
            vtk["eps_w_xy"] = mp.ϵij_w[:, 4];
            vtk["eps_w_yz"] = mp.ϵij_w[:, 5];
            vtk["eps_w_zx"] = mp.ϵij_w[:, 6];
            vtk["pp"      ] = mp.σw         ;
            vtk["v_w_x"   ] = mp.Vw[:, 1]   ;
            vtk["v_w_y"   ] = mp.Vw[:, 2]   ;
            vtk["v_w_z"   ] = mp.Vw[:, 3]   ;
            vtk["porosity"] = mp.porosity   ;
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
        for i in 1:itr
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
                vtk["sig_xx"  ] = sig[:, 1]
                vtk["sig_yy"  ] = sig[:, 2]
                vtk["sig_zz"  ] = sig[:, 3]
                vtk["sig_xy"  ] = sig[:, 4]
                vtk["sig_m"   ] = (sig[:, 1].+sig[:, 2].+sig[:, 3])./3
                vtk["eps_s_xx"] = eps_s[:, 1]
                vtk["eps_s_yy"] = eps_s[:, 2]
                vtk["eps_s_xy"] = eps_s[:, 4]
                vtk["epII"    ] = epII
                vtk["epK"     ] = epK
                vtk["mass"    ] = mass
                vtk["vol"     ] = vol
                vtk["v_s_x"   ] = v_s[:, 1]
                vtk["v_s_y"   ] = v_s[:, 2]
                vtk["disp_x"  ] = abs.(mp_pos[:, 1].-mp_init[:, 1])
                vtk["disp_y"  ] = abs.(mp_pos[:, 2].-mp_init[:, 2])
                vtk["disp_Σ"  ] = sqrt.((mp_pos[:, 1].-mp_init[:, 1]).^2 .+
                                        (mp_pos[:, 2].-mp_init[:, 2]).^2)
                args.coupling==:TS ? (
                    vtk["eps_w_xx"] = eps_w[:, 1];
                    vtk["eps_w_yy"] = eps_w[:, 2];
                    vtk["eps_w_xy"] = eps_w[:, 4];
                    vtk["pp"      ] = pp         ;
                    vtk["v_w_x"   ] = v_w[:, 1]  ;
                    vtk["v_w_y"   ] = v_w[:, 2]  ;
                    vtk["porosity"] = porosity   ;
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
    mp_init   = fid[  "mp_init"] |> read
    grid_pos = fid["grid_pos"] |> read
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
        @inbounds for i in 1:itr
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
                vtk["sig_xx"  ] = sig[:, 1]
                vtk["sig_yy"  ] = sig[:, 2]
                vtk["sig_zz"  ] = sig[:, 3]
                vtk["sig_xy"  ] = sig[:, 4]
                vtk["sig_yz"  ] = sig[:, 5]
                vtk["sig_zx"  ] = sig[:, 6]
                vtk["sig_m"   ] = (sig[:, 1].+sig[:, 2].+sig[:, 3])./3
                vtk["eps_s_xx"] = eps_s[:, 1]
                vtk["eps_s_yy"] = eps_s[:, 2]
                vtk["eps_s_zz"] = eps_s[:, 3]
                vtk["eps_s_xy"] = eps_s[:, 4]
                vtk["eps_s_yz"] = eps_s[:, 5]
                vtk["eps_s_zx"] = eps_s[:, 6]
                vtk["epII"    ] = epII
                vtk["epK"     ] = epK
                vtk["mass"    ] = mass
                vtk["vol"     ] = vol
                vtk["v_s_x"   ] = v_s[:, 1]
                vtk["v_s_y"   ] = v_s[:, 2]
                vtk["v_s_z"   ] = v_s[:, 3]
                vtk["disp_x"  ] = abs.(mp_pos[:, 1].-mp_init[:, 1])
                vtk["disp_y"  ] = abs.(mp_pos[:, 2].-mp_init[:, 2])
                vtk["disp_z"  ] = abs.(mp_pos[:, 3].-mp_init[:, 3])
                vtk["disp_Σ"  ] = sqrt.((mp_pos[:, 1].-mp_init[:, 1]).^2 .+
                                        (mp_pos[:, 2].-mp_init[:, 2]).^2 .+
                                        (mp_pos[:, 3].-mp_init[:, 3]).^2)
                args.coupling==:TS ? (
                    vtk["eps_w_xx"] = eps_w[:, 1];
                    vtk["eps_w_yy"] = eps_w[:, 2];
                    vtk["eps_w_zz"] = eps_w[:, 3];
                    vtk["eps_w_xy"] = eps_w[:, 4];
                    vtk["eps_w_yz"] = eps_w[:, 5];
                    vtk["eps_w_zx"] = eps_w[:, 6];
                    vtk["pp"      ] = pp         ;
                    vtk["v_w_x"   ] = v_w[:, 1]  ;
                    vtk["v_w_y"   ] = v_w[:, 2]  ;
                    vtk["v_w_z"   ] = v_w[:, 3]  ;
                    vtk["porosity"] = porosity   ;
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

"""
    terzaghi(Tv, p0)

Description:
---
Analytical solution for Terzaghi's consolidation test.
"""
function terzaghi(Tv, p0)
    num = 100
    dat = zeros(num, 2)    
    dat[:, 1] .= collect(range(0, 1, length=num))
    
    @inbounds for i in 1:num
        p = 0.0
        for m in 1:2:1e4
            p += 4*p0/π*(1/m)*sin(m*π*dat[i, 1]/2)*exp((-m^2)*((π/2)^2)*Tv)
        end
        dat[i, 2] = p
    end
    dat[:, 2] .= dat[:, 2]./p0
    dat[:, 1] .= -dat[:, 1].+1
    return dat = dat[:, [2, 1]]
end

"""
    consolidation()

Description:
---
Analytical solution for degree of consolidation.
"""
function consolidation()
    num = 1000
    dat = zeros(num, 2)    
    dat[:, 1] .= collect(range(0, 4, length=num))

    @inbounds for i in 1:num
        tmp = 0.0
        for m in 1:2:1e4
            tmp += (8/π^2)*(1/m^2)*exp(-(m*π/2)^2*dat[i, 1]) 
        end
        dat[i, 2] = 1-tmp
    end
    return dat
end

function single_terzaghi(p0, init_porosity, init_Kw, init_E, init_k)
    num = 100
    dat = zeros(num, 2)    
    dat[:, 1] .= collect(range(0, 0.11, length=num))
    Cv = init_k/1e4*(1/init_E+init_porosity/init_Kw)
    Tv = Cv*dat[:, 1]
    
    @inbounds for i in 1:num
        p = 0.0
        for m in 1:2:1e4
            p += 4*p0/π*(1/m)*sin(m*π/2)*exp((-m^2)*((π/2)^2)*Tv[i])
        end
        dat[i, 2] = p
    end
    return dat
end
