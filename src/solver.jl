#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : solver.jl                                                                  |
|  Description: MPM solver implementation                                                  |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. solver!() [2D & 3D]                                                     |
+==========================================================================================#

include(joinpath(@__DIR__, "solvers/extras.jl"  ))
include(joinpath(@__DIR__, "solvers/utils_OS.jl"))
include(joinpath(@__DIR__, "solvers/utils_TS.jl"))
include(joinpath(@__DIR__, "solvers/OS_MUSL.jl" ))
include(joinpath(@__DIR__, "solvers/OS_USL.jl"  ))
include(joinpath(@__DIR__, "solvers/OS_USF.jl"  ))
include(joinpath(@__DIR__, "solvers/TS_MUSL.jl" ))

include(joinpath(@__DIR__, "materials/linearelastic.jl"))
include(joinpath(@__DIR__, "materials/druckerprager.jl"))
include(joinpath(@__DIR__, "materials/mohrcoulomb.jl"  ))
include(joinpath(@__DIR__, "materials/hyperelastic.jl" ))

export procedure!
export submit_work!

"""
    submit_work!(args::DeviceArgs2D{T1, T2}, grid::DeviceGrid2D{T1, T2}, 
        mp::DeviceParticle{T1, T2}, attr::DeviceProperty{T1, T2}, 
        bc::DeviceVBoundary{T1, T2}, workflow::Function) 

Description:
---
This function will start to run the MPM solver.
"""
@views function submit_work!(
    args    ::     DeviceArgs{T1, T2},
    grid    ::     DeviceGrid{T1, T2}, 
    mp      :: DeviceParticle{T1, T2}, 
    attr    :: DeviceProperty{T1, T2},
    bc      ::DeviceVBoundary{T1, T2},
    workflow::Function
) where {T1, T2}
    initmpstatus!(CPU())(ndrange=mp.np, grid, mp, Val(args.basis))
    # variables setup for the simulation 
    Ti = T2(0.0)
    pc = Ref{T1}(0)
    pb = progressinfo(args, "solving")
    ΔT = args.ΔT
    #ΔT = args.time_step==:auto ? cfl(args, grid, mp, attr, Val(args.coupling)) : args.ΔT
    dev_grid, dev_mp, dev_attr, dev_bc = host2device(grid, mp, attr, bc, Val(args.device))
    # main part: HDF5 ON / OFF
    if args.hdf5==true
        hdf5_id     = T1(1) # HDF5 group index
        hdf5_switch = T1(0) # HDF5 step
        proj_path   = joinpath(args.project_path, args.project_name)
        hdf5_path   = joinpath(proj_path, "$(args.project_name).h5")
        isfile(hdf5_path) ? rm(hdf5_path) : nothing
        h5open(hdf5_path, "cw") do fid
            args.start_time = time()
            while Ti < args.Ttol
                if (hdf5_switch == args.hdf5_step) || (hdf5_switch == T1(0))
                    device2host!(args, mp, dev_mp, Val(args.device))
                    g = create_group(fid, "group$(hdf5_id)")
                    g["stress"    ] = mp.σij
                    g["strain_s"  ] = mp.ϵijs
                    g["eqstrain"  ] = mp.ϵq
                    g["ekstrain"  ] = mp.ϵk
                    g["eqrate"    ] = mp.ϵv
                    g["coords"    ] = mp.ξ
                    g["velocity_s"] = mp.vs
                    g["volume"    ] = mp.Ω
                    g["mass_s"    ] = mp.ms
                    g["time"      ] = Ti
                    if args.coupling==:TS
                        g["pressure_w"] = mp.σw
                        g["strain_w"  ] = mp.ϵijw
                        g["velocity_w"] = mp.vw
                        g["porosity"  ] = mp.n
                    end
                    hdf5_switch = T1(0); hdf5_id += T1(1)
                end
                workflow(args, dev_grid, dev_mp, dev_attr, dev_bc, ΔT, Ti, 
                    Val(args.coupling), Val(args.scheme), Val(args.va))
                #args.time_step==:auto ? ΔT=args.αT*reduce(min, dev_mp.cfl) : nothing
                Ti += ΔT
                hdf5_switch += 1
                args.iter_num += 1
                updatepb!(pc, Ti, args.Ttol, pb)
            end
            args.end_time = time()
            write(fid, "FILE_NUM"   , hdf5_id    )
            write(fid, "grid_coords", grid.ξ     )
            write(fid, "mp_coords0" , mp.ξ0      )
            write(fid, "nid"        , attr.nid   )
            write(fid, "vbc_xs_idx" , bc.vx_s_idx)
            write(fid, "vbc_xs_val" , bc.vx_s_val)
            write(fid, "vbc_ys_idx" , bc.vy_s_idx)
            write(fid, "vbc_ys_val" , bc.vy_s_val)
            if typeof(args) <: DeviceArgs3D
                write(fid, "vbc_zs_idx", bc.vz_s_idx)
                write(fid, "vbc_zs_val", bc.vz_s_val)
            end
            if args.coupling == :TS
                write(fid, "vbc_xw_idx", bc.vx_w_idx)
                write(fid, "vbc_xw_val", bc.vx_w_val)
                write(fid, "vbc_yw_idx", bc.vy_w_idx)
                write(fid, "vbc_yw_val", bc.vy_w_val)
                if typeof(args) <: DeviceArgs3D
                    write(fid, "vbc_zw_idx", bc.vz_w_idx)
                    write(fid, "vbc_zw_val", bc.vz_w_val)
                end
            end
        end
    elseif args.hdf5==false
        args.start_time = time()
        while Ti < args.Ttol
            workflow(args, dev_grid, dev_mp, dev_attr, dev_bc, ΔT, Ti, 
                Val(args.coupling), Val(args.scheme), Val(args.va))
            #args.time_step==:auto ? ΔT=args.αT*reduce(min, dev_mp.cfl) : nothing
            Ti += ΔT
            args.iter_num += 1
            updatepb!(pc, Ti, args.Ttol, pb)
        end
        args.end_time = time()
    end
    KAsync(getBackend(Val(args.device)))
    device2host!(args, mp, dev_mp, Val(args.device), verbose=true)
    clean_device!(dev_grid, dev_mp, dev_attr, dev_bc, Val(args.device))
    return nothing
end