#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : solver.jl                                                                  |
|  Description: MPM solver implementation                                                  |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. solver!() [2D]                                                          |
|               2. solver!() [3D]                                                          |
+==========================================================================================#

export procedure!
export dpP!, liE!, hyE!, mcP!

include(joinpath(@__DIR__, "solvers/OS.jl"     ))
include(joinpath(@__DIR__, "solvers/TS.jl"     ))
include(joinpath(@__DIR__, "solvers/OS_MUSL.jl"))
include(joinpath(@__DIR__, "solvers/OS_USL.jl" ))
include(joinpath(@__DIR__, "solvers/OS_USF.jl" ))
include(joinpath(@__DIR__, "solvers/TS_MUSL.jl"))

include(joinpath(@__DIR__, "materials/linearelastic.jl"))
include(joinpath(@__DIR__, "materials/druckerprager.jl"))
include(joinpath(@__DIR__, "materials/mohrcoulomb.jl"  ))
include(joinpath(@__DIR__, "materials/hyperelastic.jl" ))

"""
    submit_work!(args::Args2D{T1, T2}, grid::Grid2D{T1, T2}, mp::Particle2D{T1, T2}, 
        pts_attr::ParticleProperty{T1, T2}, bc::VBoundary2D{T1, T2}, workflow::Function) 

Description:
---
This function will start to run the 2D MPM solver.
"""
@views function submit_work!(args    ::          Args2D{T1, T2},
                             grid    ::          Grid2D{T1, T2}, 
                             mp      ::      Particle2D{T1, T2}, 
                             pts_attr::ParticleProperty{T1, T2},
                             bc      ::     VBoundary2D{T1, T2},
                             workflow::Function) where {T1, T2}
    # variables setup for the simulation 
    FNUM_0 = T2(0.0); INUM_1 = T1(1)
    Ti     = FNUM_0 ; INUM_0 = T1(0)
    pc     = Ref{T1}(0)
    pb     = progressinfo(args, "solving")
    args.time_step==:auto ? ΔT=cfl(args, grid, mp, pts_attr, Val(args.coupling)) : ΔT=args.ΔT
    if args.device!=:CPU 
        gpu_grid, gpu_mp, gpu_pts_attr, gpu_bc = host2device(args, grid, mp, pts_attr, bc)
    end
    # main part: HDF5 ON / OFF
    if args.hdf5==true
        hdf5_id     = INUM_1 # HDF5 group index
        hdf5_switch = INUM_0 # HDF5 step
        hdf5_path   = joinpath(args.project_path, "$(args.project_name).h5")
        isfile(hdf5_path) ? rm(hdf5_path) : nothing
        h5open(hdf5_path, "cw") do fid
            args.start_time = time()
            while Ti < args.Ttol
                if (hdf5_switch==args.hdf5_step) || (hdf5_switch==INUM_0)
                    args.device≠:CPU ? device2host!(args, mp, gpu_mp) : nothing
                    g = create_group(fid, "group$(hdf5_id)")
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
                    hdf5_switch = INUM_0; hdf5_id += INUM_1
                end
                if args.device==:CPU 
                    workflow(args, grid, mp, pts_attr, bc, ΔT, Ti, Val(args.coupling), Val(args.scheme))
                    args.time_step==:auto ? ΔT=args.αT*reduce(min, mp.cfl) : nothing
                else
                    workflow(args, gpu_grid, gpu_mp, gpu_pts_attr, gpu_bc, ΔT, Ti, Val(args.coupling), Val(args.scheme))
                    args.time_step==:auto ? ΔT=args.αT*reduce(min, gpu_mp.cfl) : nothing
                end
                Ti += ΔT
                hdf5_switch += 1
                args.iter_num += 1
                updatepb!(pc, Ti, args.Ttol, pb)
            end
            args.end_time = time()
            write(fid, "FILE_NUM"  , hdf5_id       )
            write(fid, "grid_pos"  , grid.pos      )
            write(fid, "mp_init"   , mp.init       )
            write(fid, "layer"     , pts_attr.layer)
            write(fid, "vbc_xs_idx", bc.Vx_s_Idx   )
            write(fid, "vbc_xs_val", bc.Vx_s_Val   )
            write(fid, "vbc_ys_idx", bc.Vy_s_Idx   )
            write(fid, "vbc_ys_val", bc.Vy_s_Val   )
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
            if args.device==:CPU 
                workflow(args, grid, mp, pts_attr, bc, ΔT, Ti, Val(args.coupling), Val(args.scheme))
                args.time_step==:auto ? ΔT=args.αT*reduce(min, mp.cfl) : nothing
            else
                workflow(args, gpu_grid, gpu_mp, gpu_pts_attr, gpu_bc, ΔT, Ti, Val(args.coupling), Val(args.scheme))
                args.time_step==:auto ? ΔT=args.αT*reduce(min, gpu_mp.cfl) : nothing
            end
            Ti += ΔT
            args.iter_num += 1
            updatepb!(pc, Ti, args.Ttol, pb)
        end
        args.end_time = time()
    end
    if args.device≠:CPU
        KAsync(getBackend(args))
        device2hostinfo!(args, mp, gpu_mp)
        clean_gpu!(args, gpu_grid, gpu_mp, gpu_pts_attr, gpu_bc)
    end
    return nothing
end

"""
    submit_work!(args::Args3D{T1, T2}, grid::Grid3D{T1, T2}, mp::Particle3D{T1, T2}, 
        pts_attr::ParticleProperty{T1, T2}, bc::VBoundary3D{T1, T2}, workflow::Function) 

Description:
---
This function will start to run the 3D MPM solver.
"""
@views function submit_work!(args    ::          Args3D{T1, T2},
                             grid    ::          Grid3D{T1, T2}, 
                             mp      ::      Particle3D{T1, T2}, 
                             pts_attr::ParticleProperty{T1, T2},
                             bc      ::     VBoundary3D{T1, T2},
                             workflow::Function) where {T1, T2}
    # variables setup for the simulation 
    FNUM_0 = T2(0.0); INUM_1 = T1(1)
    Ti     = FNUM_0 ; INUM_0 = T1(0)
    pc     = Ref{T1}(0)
    pb     = progressinfo(args, "solving")
    args.time_step==:auto ? ΔT=cfl(args, grid, mp, pts_attr, Val(args.coupling)) : ΔT=args.ΔT
    if args.device!=:CPU 
        gpu_grid, gpu_mp, gpu_pts_attr, gpu_bc = host2device(args, grid, mp, pts_attr, bc)
    end
    # main part: HDF5 ON / OFF
    if args.hdf5==true
        hdf5_id     = INUM_1 # HDF5 group index
        hdf5_switch = INUM_0 # HDF5 step
        hdf5_path   = joinpath(args.project_path, "$(args.project_name).h5")
        isfile(hdf5_path) ? rm(hdf5_path) : nothing
        h5open(hdf5_path, "cw") do fid
            args.start_time = time()
            while Ti < args.Ttol
                if (hdf5_switch==args.hdf5_step) || (hdf5_switch==INUM_0)
                    args.device≠:CPU ? device2host!(args, mp, gpu_mp) : nothing
                    g = create_group(fid, "group$(hdf5_id)")
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
                    hdf5_switch = INUM_0; hdf5_id += INUM_1
                end
                if args.device==:CPU 
                    workflow(args, grid, mp, pts_attr, bc, ΔT, Ti, Val(args.coupling), Val(args.scheme))
                    args.time_step==:auto ? ΔT=args.αT*reduce(min, mp.cfl) : nothing
                else
                    workflow(args, gpu_grid, gpu_mp, gpu_pts_attr, gpu_bc, ΔT, Ti, Val(args.coupling), Val(args.scheme))
                    args.time_step==:auto ? ΔT=args.αT*reduce(min, gpu_mp.cfl) : nothing
                end
                Ti += ΔT
                hdf5_switch += 1
                args.iter_num += 1
                updatepb!(pc, Ti, args.Ttol, pb)
            end
            args.end_time = time()
            write(fid, "FILE_NUM"  , hdf5_id       )
            write(fid, "grid_pos"  , grid.pos      )
            write(fid, "mp_init"   , mp.init       )
            write(fid, "layer"     , pts_attr.layer)
            write(fid, "vbc_xs_idx", bc.Vx_s_Idx   )
            write(fid, "vbc_xs_val", bc.Vx_s_Val   )
            write(fid, "vbc_ys_idx", bc.Vy_s_Idx   )
            write(fid, "vbc_ys_val", bc.Vy_s_Val   )
            write(fid, "vbc_zs_idx", bc.Vz_s_Idx   )
            write(fid, "vbc_zs_val", bc.Vz_s_Val   )
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
            if args.device==:CPU 
                workflow(args, grid, mp, pts_attr, bc, ΔT, Ti, Val(args.coupling), Val(args.scheme))
                args.time_step==:auto ? ΔT=args.αT*reduce(min, mp.cfl) : nothing
            else
                workflow(args, gpu_grid, gpu_mp, gpu_pts_attr, gpu_bc, ΔT, Ti, Val(args.coupling), Val(args.scheme))
                args.time_step==:auto ? ΔT=args.αT*reduce(min, gpu_mp.cfl) : nothing
            end
            Ti += ΔT
            args.iter_num += 1
            updatepb!(pc, Ti, args.Ttol, pb)
        end
        args.end_time = time()
    end
    if args.device≠:CPU
        KAsync(getBackend(args))
        device2hostinfo!(args, mp, gpu_mp)
        clean_gpu!(args, gpu_grid, gpu_mp, gpu_pts_attr, gpu_bc)
    end
    return nothing
end