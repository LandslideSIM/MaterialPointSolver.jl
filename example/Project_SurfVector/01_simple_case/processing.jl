using HDF5
using ProgressMeter
using WriteVTK

base_path = joinpath(homedir(), "Workbench/outputs")
hdf5_name = "3d_slumping_vector"
hdf5_path = joinpath(base_path, hdf5_name)


@views function processing_vector(hdf5_path::String, hdf5_name::String, sampling_step)
    fid       = h5open(joinpath(hdf5_path, "$(hdf5_name).h5"), "r")
    itr       = (read(fid["FILE_NUM"])-1) |> Int64
    mp_init   = fid[  "mp_init"] |> read
    grid_pos  = fid["grid_pos"] |> read
    mp_num    = size(mp_init, 1)
    nd_num    = size(grid_pos, 1)
    anim_path = mkpath(joinpath(hdf5_path, "animation"))
    mps_path  = joinpath(hdf5_path, hdf5_name)
    nds_path  = joinpath(hdf5_path, "grid")
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
            mp_num = length(mp_pos[:, 1])
            # create surface data based on HDF5 data
            #=------------------------------------------------------------------------------
                    ————————————————————————→ y
                    | 1  | 7  | 13 |.. |
                    ————————————————————
                    | 2  | 8  | .. |.. |
                    ————————————————————
                    | 3  | 9  | .. |.. |
                    ————————————————————
                    | 4  | 10 | .. |.. |
                    ————————————————————
                    | 5  | 11 | .. |.. |
                    ————————————————————
                    | 6  | 12 | 18 |.. |
                    ————————————————————
                    |
                    ↓
                    x
            ------------------------------------------------------------------------------=#
            lt = [minimum(mp_pos[:, 1]), minimum(mp_pos[:, 2])]
            rb = [maximum(mp_pos[:, 1]), maximum(mp_pos[:, 2])]
            tmpx = range(lt[1], rb[1], step=sampling_step)
            tmpy = range(lt[2], rb[2], step=sampling_step)
            gridxrange = range(tmpx[1]-2*sampling_step, tmpx[end]+2*sampling_step, step=sampling_step)
            gridyrange = range(tmpy[1]-2*sampling_step, tmpy[end]+2*sampling_step, step=sampling_step)
            cellnum = (length(gridxrange)-1)*(length(gridyrange)-1)

            p2c = fld.(mp_pos[:, 2].-gridyrange[1], sampling_step).*(length(gridxrange).-1).+
                  cld.(mp_pos[:, 1].-gridxrange[1], sampling_step) .|> Int

            cell_pts = zeros(cellnum, 5) # 1z, 2vx, 3vy, 4vz, 5mp_id
            cell_pts[:, 1] .= minimum(mp_pos[:, 3])
            for i in 1:mp_num
                if p2c[i] > 0 
                    if cell_pts[p2c[i], 1] ≤ mp_pos[i, 3]
                        cell_pts[p2c[i], 2] = v_s[i, 1]
                        cell_pts[p2c[i], 3] = v_s[i, 2]
                        cell_pts[p2c[i], 4] = v_s[i, 3]
                        cell_pts[p2c[i], 5] = i
                    end
                end
            end
            tmp = findall(all(!iszero, cell_pts[i, 4]) for i in 1:cellnum)
            cell_pts = cell_pts[tmp, :]
            surface_vx = [NaN for i = 1:mp_num]
            surface_vy = [NaN for i = 1:mp_num]
            surface_vz = [NaN for i = 1:mp_num]
            for i in 1:size(cell_pts, 1)
                surface_vx[Int(cell_pts[i, 5])] = cell_pts[i, 2]
                surface_vy[Int(cell_pts[i, 5])] = cell_pts[i, 3]
                surface_vz[Int(cell_pts[i, 5])] = cell_pts[i, 4]
            end
            # write data for block 1
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
                vtk["surface_vx"] = surface_vx
                vtk["surface_vy"] = surface_vy
                vtk["surface_vz"] = surface_vz
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

processing_vector(hdf5_path, hdf5_name, 5)