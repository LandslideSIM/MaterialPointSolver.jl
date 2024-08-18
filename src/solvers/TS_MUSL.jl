#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : TS_MUSL.jl                                                                 |
|  Description: Defaule MUSL (modified update stress last) implementation for two-phase    |
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
                            ::Val{:TS},
                            ::Val{:MUSL}) where {T2}
    Ti<args.Te ? G=args.gravity/args.Te*Ti : G=args.gravity
    dev = getBackend(args)
    resetgridstatus_TS!(dev)(ndrange=grid.node_num, grid)
    resetmpstatus_TS!(dev)(ndrange=mp.num, grid, mp, Val(args.basis))
    P2G_TS!(dev)(ndrange=mp.num, grid, mp, pts_attr, G)
    solvegrid_TS!(dev)(ndrange=grid.node_num, grid, bc, ΔT, args.ζ)
    doublemapping1_TS!(dev)(ndrange=mp.num, grid, mp, pts_attr, ΔT, args.FLIP, args.PIC)
    doublemapping2_TS!(dev)(ndrange=mp.num, grid, mp)
    doublemapping3_TS!(dev)(ndrange=grid.node_num, grid, bc, ΔT)
    G2P_TS!(dev)(ndrange=mp.num, grid, mp, pts_attr)
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
    if args.vollock==true
        vollock1_TS!(dev)(ndrange=mp.num, grid, mp)
        vollock2_TS!(dev)(ndrange=mp.num, grid, mp)
    end                                  
    return nothing
end