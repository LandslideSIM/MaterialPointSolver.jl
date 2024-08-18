#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : solver_d.jl                                                                |
|  Description: MPM solver implementation (CPU & GPU).                                     |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. solver!() [2D GPU]                                                      |
|               2. solver!() [3D GPU]                                                      |
+==========================================================================================#

include(joinpath(@__DIR__, "solver/OS_d.jl"))
include(joinpath(@__DIR__, "solver/TS_d.jl"))

"""
    solver!(args::Args2D{T1, T2}, grid::Grid2D{T1, T2}, mp::Particle2D{T1, T2}, 
        bc::VBoundary2D{T1, T2}, ::Val{:CUDA}) 

Description:
---
This function is the main function of the MPM solver, user has to pre-define the data of
    `args`, `grid`, `mp`, and `bc`, they are the parameters, background grid, material
    points, and boundary conditions (2D). Val{} is used to decide the device (CPU/GPU).
"""
@views function solver!(args::     Args2D{T1, T2},
                        grid::     Grid2D{T1, T2}, 
                        mp  :: Particle2D{T1, T2}, 
                        bc  ::VBoundary2D{T1, T2}, 
                            ::Val{:CUDA}) where {T1, T2}
    # specific values 
    FNUM_0 = T2(0.0)
    INUM_1 = T1(1)
    INUM_0 = T1(0)
    idc    = Ref{T1}(0)
    Ti     = FNUM_0
    pb     = progressinfo(args, "solving")
    args.time_step==:auto ? ΔT=cfl(args, grid, mp, Val(args.coupling)) : ΔT=args.ΔT
    cu_grid, cu_mp, cu_bc = host2device(grid, mp, bc)    
    OccAPI = getOccAPI(args, cu_grid, cu_mp, cu_bc, ΔT)
    if args.hdf5==true
        # write HDF5 file
        id = INUM_1 # hdf5 group index
        hdf5_path = joinpath(args.project_path, "$(args.project_name).h5")
        isfile(hdf5_path) ? rm(hdf5_path) : nothing; hdf5_switch = INUM_0
        h5open(hdf5_path, "cw") do fid
            args.start_time = time()
            while Ti < args.Ttol # start MUSL loop
                if (hdf5_switch==args.hdf5_step) || (hdf5_switch==INUM_0)
                    # writing data into HDF5 file
                    device2host!(args, mp, cu_mp)
                    g = create_group(fid, "group$(id)")
                    g["sig"   ] = mp.σij
                    g["eps_s" ] = mp.ϵij_s
                    g["epII"  ] = mp.epII
                    g["epK"   ] = mp.epK
                    g["mp_pos"] = mp.pos
                    g["v_s"   ] = mp.Vs
                    g["vol"   ] = mp.vol
                    g["mass"  ] = mp.Ms
                    g["time"  ] = Ti
                    if args.coupling==:TS
                        g["pp"      ] = mp.σw
                        g["eps_w"   ] = mp.ϵij_w
                        g["v_w"     ] = mp.Vw
                        g["porosity"] = mp.porosity
                    end
                    hdf5_switch = INUM_0; id += INUM_1
                end
                # MPM solver
                procedure!(args, cu_grid, cu_mp, cu_bc, ΔT, Ti, OccAPI, Val(args.coupling))
                # update time step, hdf5 step, iteration number, and terminal content
                args.time_step==:auto ? ΔT=args.αT*reduce(min, cu_mp.cfl) : nothing
                Ti += ΔT
                hdf5_switch += 1
                args.iter_num += 1
                updatepb!(idc, Ti, args.Ttol, pb)
            end
            args.end_time = time()
            write(fid, "FILE_NUM"  , id         )
            write(fid, "grid_pos"  , grid.pos   )
            write(fid, "mp_init"   , mp.init    )
            write(fid, "vbc_xs_idx", bc.Vx_s_Idx)
            write(fid, "vbc_xs_val", bc.Vx_s_Val)
            write(fid, "vbc_ys_idx", bc.Vy_s_Idx)
            write(fid, "vbc_ys_val", bc.Vy_s_Val)
            if args.coupling==:TS
                write(fid, "vbc_xw_idx", bc.Vx_w_Idx)
                write(fid, "vbc_xw_val", bc.Vx_w_Val)
                write(fid, "vbc_yw_idx", bc.Vy_w_Idx)
                write(fid, "vbc_yw_val", bc.Vy_w_Val)
            end
        end
    elseif args.hdf5==false
        args.start_time = time()
        while Ti < args.Ttol
            # MPM solver
            procedure!(args, cu_grid, cu_mp, cu_bc, ΔT, Ti, OccAPI, Val(args.coupling))
            # update time step, iteration number, and terminal content
            args.time_step==:auto ? ΔT=args.αT*reduce(min, cu_mp.cfl) : nothing
            Ti += ΔT
            args.iter_num += 1
            updatepb!(idc, Ti, args.Ttol, pb)
        end
        args.end_time = time()
    end
    synchronize()
    device2host!(args, mp, cu_mp); println("\e[0;31m[▼ I/O: device → host")
    clean_gpu!(cu_grid, cu_mp, cu_bc)
    return nothing
end

"""
    solver!(args::Args3D{T1, T2}, grid::Grid3D{T1, T2}, mp::Particle3D{T1, T2}, 
        bc::VBoundary3D{T1, T2}, ::Val{:CUDA}) 

Description:
---
This function is the main function of the MPM solver, user has to pre-define the data of
    `args`, `grid`, `mp`, and `bc`, they are the parameters, background grid, material
    points, and boundary conditions (3D). Val{} is used to decide the device (CPU/GPU).
"""
@views function solver!(args::     Args3D{T1, T2},
                        grid::     Grid3D{T1, T2}, 
                        mp  :: Particle3D{T1, T2}, 
                        bc  ::VBoundary3D{T1, T2}, 
                            ::Val{:CUDA}) where {T1, T2}
    # specific values 
    FNUM_0 = T2(0.0)
    INUM_1 = T1(1)
    INUM_0 = T1(0)
    idc    = Ref{T1}(0)
    Ti     = FNUM_0
    pb     = progressinfo(args, "solving")
    args.time_step==:auto ? ΔT=cfl(args, grid, mp, Val(args.coupling)) : ΔT=args.ΔT
    cu_grid, cu_mp, cu_bc = host2device(grid, mp, bc)    
    OccAPI = getOccAPI(args, cu_grid, cu_mp, cu_bc, ΔT)
    if args.hdf5==true
        # write HDF5 file
        id = INUM_1 # hdf5 group index
        hdf5_path = joinpath(args.project_path, "$(args.project_name).h5")
        isfile(hdf5_path) ? rm(hdf5_path) : nothing; hdf5_switch = INUM_0
        h5open(hdf5_path, "cw") do fid
            args.start_time = time()
            while Ti < args.Ttol # start MUSL loop
                if (hdf5_switch==args.hdf5_step) || (hdf5_switch==INUM_0)
                    # writing data into HDF5 file
                    device2host!(args, mp, cu_mp)
                    g = create_group(fid, "group$(id)")
                    g["sig"   ] = mp.σij
                    g["eps_s" ] = mp.ϵij_s
                    g["epII"  ] = mp.epII
                    g["epK"   ] = mp.epK
                    g["mp_pos"] = mp.pos
                    g["v_s"   ] = mp.Vs
                    g["vol"   ] = mp.vol
                    g["mass"  ] = mp.Ms
                    g["time"  ] = Ti
                    if args.coupling==:TS
                        g["pp"      ] = mp.σw
                        g["eps_w"   ] = mp.ϵij_w
                        g["v_w"     ] = mp.Vw
                        g["porosity"] = mp.porosity
                    end
                    hdf5_switch = INUM_0; id += INUM_1
                end
                # MPM solver
                procedure!(args, cu_grid, cu_mp, cu_bc, ΔT, Ti, OccAPI, Val(args.coupling))
                # update time step, hdf5 step, iteration number, and terminal content
                args.time_step==:auto ? ΔT=args.αT*reduce(min, cu_mp.cfl) : nothing
                Ti += ΔT
                hdf5_switch += 1
                args.iter_num += 1
                updatepb!(idc, Ti, args.Ttol, pb)
            end
            args.end_time = time()
            write(fid, "FILE_NUM"  , id         )
            write(fid, "grid_pos"  , grid.pos   )
            write(fid, "mp_init"   , mp.init    )
            write(fid, "vbc_xs_idx", bc.Vx_s_Idx)
            write(fid, "vbc_xs_val", bc.Vx_s_Val)
            write(fid, "vbc_ys_idx", bc.Vy_s_Idx)
            write(fid, "vbc_ys_val", bc.Vy_s_Val)
            write(fid, "vbc_zs_idx", bc.Vz_s_Idx)
            write(fid, "vbc_zs_val", bc.Vz_s_Val)
            if args.coupling==:TS
                write(fid, "vbc_xw_idx", bc.Vx_w_Idx)
                write(fid, "vbc_xw_val", bc.Vx_w_Val)
                write(fid, "vbc_yw_idx", bc.Vy_w_Idx)
                write(fid, "vbc_yw_val", bc.Vy_w_Val)
                write(fid, "vbc_zw_idx", bc.Vz_w_Idx)
                write(fid, "vbc_zw_val", bc.Vz_w_Val)
            end
        end
    elseif args.hdf5==false
        args.start_time = time()
        while Ti < args.Ttol
            # MPM solver
            procedure!(args, cu_grid, cu_mp, cu_bc, ΔT, Ti, OccAPI, Val(args.coupling))
            # update time step, iteration number, and terminal content
            args.time_step==:auto ? ΔT=args.αT*reduce(min, cu_mp.cfl) : nothing
            Ti += ΔT
            args.iter_num += 1
            updatepb!(idc, Ti, args.Ttol, pb)
        end
        args.end_time = time()
    end
    synchronize()
    device2host!(args, mp, cu_mp); println("\e[0;31m[▼ I/O: device → host")
    clean_gpu!(cu_grid, cu_mp, cu_bc)
    return nothing
end