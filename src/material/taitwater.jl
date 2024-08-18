#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : taitwater.jl                                                               |
|  Description: Tait state equation for water simulation                                   |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. taitwater! [2D]                                                         |
|               2. taitwater! [3D]                                                         |
+==========================================================================================#

"""
    taitwater!(mp::Particle2D{T1, T2})

Description:
---
Update pressure according to Tait state equation.
"""
@views function taitwater!(mp::Particle2D{T1, T2}) where {T1, T2}
    FNUM_1 = T2(1)
    for ix in 1:mp.num
        pid = mp.layer[ix]
        γ   = mp.γ[pid]
        B   = mp.B[pid]
        #P   = B*((mp.ρs[ix]/mp.ρs_init[ix])^γ-FNUM_1)
        P = T2(1225)*(mp.ρs[ix]-mp.ρs_init[ix])
        mp.σij[ix, 1] = -P
        mp.σij[ix, 2] = -P
        # update mean stress tensor
        mp.σm[ix] = P
    end
    return nothing
end

"""
    taitwater!(mp::Particle3D{T1, T2})

Description:
---
Update pressure according to Tait state equation.
"""
@views function taitwater!(mp::Particle3D{T1, T2}) where {T1, T2}
    FNUM_1 = T2(1)
    for ix in 1:mp.num
        pid = mp.layer[ix]
        γ   = mp.γ[pid]
        B   = mp.B[pid]
        #P   = B*((mp.ρs[ix]/mp.ρs_init[ix])^γ-FNUM_1)
        P = T2(1225)*(mp.ρs[ix]-mp.ρs_init[ix])
        mp.σij[ix, 1] = -P
        mp.σij[ix, 2] = -P
        mp.σij[ix, 3] = -P
        # update mean stress tensor
        mp.σm[ix] = P
    end
    return nothing
end