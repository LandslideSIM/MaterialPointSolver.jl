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

include(joinpath(@__DIR__, "solvers/utils.jl"  ))
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
@views function submit_work!(
    args    ::MODELARGSD{T1, T2},
    grid    ::     GRIDD{T1, T2}, 
    mp      :: PARTICLED{T1, T2}, 
    pts_attr:: PROPERTYD{T1, T2},
    bc      :: BOUNDARYD{T1, T2},
    workflow::Function
) where {T1, T2}
    initmpstatus!(CPU())(ndrange=mp.num, grid, mp, Val(args.basis))
    # variables setup for the simulation 
    Ti = T2(0.0)
    pc = Ref{T1}(0)
    pb = progressinfo(args, "solving")
    ΔT = args.time_step==:auto ? cfl(args, grid, mp, pts_attr, Val(args.coupling)) : args.ΔT
    dev_grid, dev_mp, dev_pts_attr, dev_bc = host2device(grid, mp, pts_attr, bc, Val(args.device))
    # main part: HDF5 ON / OFF
    if args.jld2==true
        jld2_id     = T1(1) # HDF5 group index
        jld2_switch = T1(0) # HDF5 step
        jld2_path   = joinpath(args.project_path, "$(args.project_name).jld2")
        isfile(jld2_path) ? rm(jld2_path) : nothing
        jldopen(jld2_path, "a+") do fid
            args.start_time = time()
            while Ti < args.Ttol
                if (jld2_switch==args.jld2_step) || (jld2_switch==T1(0))
                    device2host!(args, mp, dev_mp, Val(args.device))
                    g = JLD2.Group(fid, "group$(jld2_id)")
                    g["sig"        ] = mp.σij
                    g["eps_s"      ] = mp.ϵij_s
                    g["epII"       ] = mp.epII
                    g["epK"        ] = mp.epK
                    g["strain_rate"] = mp.dϵ
                    g["mp_pos"     ] = mp.pos
                    g["v_s"        ] = mp.Vs
                    g["vol"        ] = mp.vol
                    g["mass"       ] = mp.Ms
                    g["time"       ] = Ti
                    if args.coupling==:TS
                        g["pp"      ] = mp.σw
                        g["eps_w"   ] = mp.ϵij_w
                        g["v_w"     ] = mp.Vw
                        g["porosity"] = mp.porosity
                    end
                    jld2_switch = T1(0); jld2_id += T1(1)
                end
                workflow(args, dev_grid, dev_mp, dev_pts_attr, dev_bc, ΔT, Ti, 
                    Val(args.coupling), Val(args.scheme))
                args.time_step==:auto ? ΔT=args.αT*reduce(min, dev_mp.cfl) : nothing
                Ti += ΔT
                jld2_switch += 1
                args.iter_num += 1
                updatepb!(pc, Ti, args.Ttol, pb)
            end
            args.end_time = time()
            write(fid, "FILE_NUM"  , jld2_id       )
            write(fid, "grid_pos"  , grid.pos      )
            write(fid, "mp_init"   , mp.init       )
            write(fid, "layer"     , pts_attr.layer)
            write(fid, "vbc_xs_idx", bc.Vx_s_Idx   )
            write(fid, "vbc_xs_val", bc.Vx_s_Val   )
            write(fid, "vbc_ys_idx", bc.Vy_s_Idx   )
            write(fid, "vbc_ys_val", bc.Vy_s_Val   )
            if typeof(args) <: Args3D
                write(fid, "vbc_zs_idx", bc.Vz_s_Idx)
                write(fid, "vbc_zs_val", bc.Vz_s_Val)
            end
            if args.coupling==:TS
                write(fid, "vbc_xw_idx", bc.Vx_w_Idx)
                write(fid, "vbc_xw_val", bc.Vx_w_Val)
                write(fid, "vbc_yw_idx", bc.Vy_w_Idx)
                write(fid, "vbc_yw_val", bc.Vy_w_Val)
                if typeof(args) <: Args3D
                    write(fid, "vbc_zw_idx", bc.Vz_w_Idx)
                    write(fid, "vbc_zw_val", bc.Vz_w_Val)
                end
            end
        end
    elseif args.jld2==false
        args.start_time = time()
        while Ti < args.Ttol
            workflow(args, dev_grid, dev_mp, dev_pts_attr, dev_bc, ΔT, Ti, 
                Val(args.coupling), Val(args.scheme))
            args.time_step==:auto ? ΔT=args.αT*reduce(min, dev_mp.cfl) : nothing
            Ti += ΔT
            args.iter_num += 1
            updatepb!(pc, Ti, args.Ttol, pb)
        end
        args.end_time = time()
    end
    KAsync(getBackend(Val(args.device)))
    device2host!(args, mp, dev_mp, Val(args.device), verbose=true)
    clean_device!(dev_grid, dev_mp, dev_pts_attr, dev_bc, Val(args.device))
    return nothing
end