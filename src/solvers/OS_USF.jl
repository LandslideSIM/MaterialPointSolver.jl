#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : OS_USF.jl                                                                  |
|  Description: Defaule USF (update stress first) implementation of USF for one-phase      | 
|               single-point MPM                                                           |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : procedure! [2D & 3D]                                                       |
+==========================================================================================#

function procedure!(args    ::MODELARGS, 
                    grid    ::GRID, 
                    mp      ::PARTICLE,
                    pts_attr::PROPERTY, 
                    bc      ::BOUNDARY,
                    ΔT      ::T2,
                    Ti      ::T2,
                            ::Val{:OS},
                            ::Val{:USF}) where {T2}
    Ti<args.Te ? G=args.gravity/args.Te*Ti : G=args.gravity
    dev = getBackend(args)
    G2P_OS!(dev)(ndrange=mp.num, grid, mp)
    if args.constitutive==:hyperelastic
        hyE!(dev)(ndrange=mp.num, mp, pts_attr)
    elseif args.constitutive==:linearelastic
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
    elseif args.constitutive==:druckerprager
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
        if Ti≥args.Te
            dpP!(dev)(ndrange=mp.num, mp, pts_attr)
        end
    elseif args.constitutive==:mohrcoulomb
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
        if Ti≥args.Te
            mcP!(dev)(ndrange=mp.num, mp, pts_attr)
        end
    end
    resetgridstatus_OS!(dev)(ndrange=grid.node_num, grid)
    args.device == :CPU && args.basis == :uGIMP ? 
        resetmpstatus_OS_CPU!(dev)(ndrange=mp.num, grid, mp, Val(args.basis)) :
        resetmpstatus_OS!(dev)(ndrange=mp.num, grid, mp, Val(args.basis))
    P2G_OS!(dev)(ndrange=mp.num, grid, mp, G)
    solvegrid_USL_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT, args.ζs)
    doublemapping1_OS!(dev)(ndrange=mp.num, grid, mp, pts_attr, ΔT, args.FLIP, args.PIC)
    if args.MVL==true
        vollock1_OS!(dev)(ndrange=mp.num, grid, mp)
        vollock2_OS!(dev)(ndrange=mp.num, grid, mp)
    end                                  
    return nothing
end