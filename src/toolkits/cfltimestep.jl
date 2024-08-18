#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : cfltimestep.jl                                                             |
|  Description: CFL conditions in MaterialPointSolver.jl                                   |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. cfl [2D - one-phase]                                                    |
|               2. cfl [3D - one-phase]                                                    |
|               3. cfl [2D - two-phase]                                                    |
|               4. cfl [3D - two-phase]                                                    |
+==========================================================================================#

"""
    cfl(args::Args2D{T1, T2}, grid::Grid2D{T1, T2}, mp::Particle2D{T1, T2},  pts_attr::ParticleProperty{T1, T2}, ::Val{:OS})

Description:
---
Calculate the CFL condition for 2D simulation (one-phase single point).
"""
function cfl(args    ::          Args2D{T1, T2}, 
             grid    ::          Grid2D{T1, T2}, 
             mp      ::      Particle2D{T1, T2},
             pts_attr::ParticleProperty{T1, T2},
                     ::Val{:OS}) where {T1, T2}
    ΔT      = T2(0.0)
    FNUM_43 = T2(4/3) 
    for i in 1:mp.num
        pid       = pts_attr.layer[i]
        Ks        = pts_attr.Ks[pid]
        G         = pts_attr.G[pid]
        sqr       = sqrt((Ks+G*FNUM_43)/mp.ρs[i])
        cd_sx     = grid.space_x/(sqr+abs(mp.Vs[i, 1]))
        cd_sy     = grid.space_y/(sqr+abs(mp.Vs[i, 2]))
        val       = min(cd_sx, cd_sy)
        mp.cfl[i] = args.αT*val
    end
    return ΔT = minimum(mp.cfl)
end

"""
    cfl(args::Args3D{T1, T2}, grid::Grid3D{T1, T2}, mp::Particle3D{T1, T2}, pts_attr::ParticleProperty{T1, T2}, ::Val{:OS})

Description:
---
Calculate the CFL condition for 3D simulation (one-phase single point).
"""
function cfl(args    ::    Args3D{T1, T2}, 
             grid    ::    Grid3D{T1, T2}, 
             mp      ::Particle3D{T1, T2},
             pts_attr::ParticleProperty{T1, T2},
                     ::Val{:OS}) where {T1, T2}
    ΔT      = T2(0.0)
    FNUM_43 = T2(4/3) 
    for i in 1:mp.num
        pid       = pts_attr.layer[i]
        Ks        = pts_attr.Ks[pid]
        G         = pts_attr.G[pid]
        sqr       = sqrt((Ks+G*FNUM_43)/mp.ρs[i])
        cd_sx     = grid.space_x/(sqr+abs(mp.Vs[i, 1]))
        cd_sy     = grid.space_y/(sqr+abs(mp.Vs[i, 2]))
        cd_sz     = grid.space_z/(sqr+abs(mp.Vs[i, 3]))
        val       = min(cd_sx, cd_sy, cd_sz)
        mp.cfl[i] = args.αT*val
    end
    return ΔT = minimum(mp.cfl)
end

"""
    cfl(args::Args2D{T1, T2}, grid::Grid2D{T1, T2}, mp::Particle2D{T1, T2}, pts_attr::ParticleProperty{T1, T2}, ::Val{:TS})

Description:
---
Calculate the CFL condition for 2D simulation (two-phase single point).
"""
function cfl(args    ::    Args2D{T1, T2}, 
             grid    ::    Grid2D{T1, T2}, 
             mp      ::Particle2D{T1, T2},
             pts_attr::ParticleProperty{T1, T2},
                     ::Val{:TS}) where {T1, T2}
    FNUM_1 = T2(1.0)
    ΔT     = T2(0.0)
    for i in 1:mp.num
        pid = pts_attr.layer[i]
        sqr = sqrt((pts_attr.E[pid]+pts_attr.Kw[pid]*mp.porosity[i])/
                   (mp.ρs[i]*(FNUM_1-mp.porosity[i])+mp.ρw[i]*mp.porosity[i]))
        cd_sx = grid.space_x/(sqr+abs(mp.Vs[i, 1]))
        cd_sy = grid.space_y/(sqr+abs(mp.Vs[i, 2]))
        cd_wx = grid.space_x/(sqr+abs(mp.Vw[i, 1]))
        cd_wy = grid.space_y/(sqr+abs(mp.Vw[i, 2]))
        val   = min(cd_sx, cd_sy, cd_wx, cd_wy)
        mp.cfl[i] = args.αT*val
    end
    return ΔT = minimum(mp.cfl)
end

"""
    cfl(args::Args3D{T1, T2}, grid::Grid3D{T1, T2}, mp::Particle3D{T1, T2}, pts_attr::ParticleProperty{T1, T2}, ::Val{:TS})

Description:
---
Calculate the CFL condition for 3D simulation (two-phase single point).
"""
function cfl(args    ::    Args3D{T1, T2}, 
             grid    ::    Grid3D{T1, T2}, 
             mp      ::Particle3D{T1, T2},
             pts_attr::ParticleProperty{T1, T2},
                     ::Val{:TS}) where {T1, T2}
    FNUM_1 = T2(1.0)
    ΔT     = T2(0.0)
    for i in 1:mp.num
        pid = pts_attr.layer[i]
        sqr = sqrt((pts_attr.E[i]+pts_attr.Kw[i]*mp.porosity[i])/
                   (mp.ρs[i]*(FNUM_1-mp.porosity[i])+mp.ρw[i]*mp.porosity[i]))
        cd_sx = grid.space_x/(sqr+abs(mp.Vs[i, 1]))
        cd_sy = grid.space_y/(sqr+abs(mp.Vs[i, 2]))
        cd_sz = grid.space_z/(sqr+abs(mp.Vs[i, 3]))
        cd_wx = grid.space_x/(sqr+abs(mp.Vw[i, 1]))
        cd_wy = grid.space_y/(sqr+abs(mp.Vw[i, 2]))
        cd_wz = grid.space_z/(sqr+abs(mp.Vw[i, 3]))
        val   = min(cd_sx, cd_sy, cd_sz, cd_wx, cd_wy, cd_wz)
        mp.cfl[i] = args.αT*val
    end
    return ΔT = minimum(mp.cfl)
end
