#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : taitwater_d.jl                                                             |
|  Description: Tait state equation for water simulation                                   |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. taitwater! [2D]                                                         |
|               2. taitwater! [3D]                                                         |
+==========================================================================================#

"""
    twP!(cu_mp::KernelParticle2D{T1, T2})

Description:
---
Update pressure according to Tait state equation.
"""
@views function twP!(cu_mp::KernelParticle2D{T1, T2}) where {T1, T2}
    FNUM_1 = T2(1)
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    if ix ≤ cu_mp.num
        pid = cu_mp.layer[ix]
        #P   = cu_mp.B[pid]*((cu_mp.ρs[ix]/cu_mp.ρs_init[ix])^cu_mp.γ[pid]-FNUM_1)
        P = T2(1225)*(cu_mp.ρs[ix]-cu_mp.ρs_init[ix])
        cu_mp.σij[ix, 1] = -P
        cu_mp.σij[ix, 2] = -P
        # update mean stress tensor
        cu_mp.σm[ix] = P
    end
    return nothing
end

"""
    twP!(mp::KernelParticle3D{T1, T2})

Description:
---
Update pressure according to Tait state equation.
"""
@views function twP!(cu_mp::KernelParticle3D{T1, T2}) where {T1, T2}
    FNUM_1 = T2(1)
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    if ix ≤ cu_mp.num
        pid = cu_mp.layer[ix]
        #P   = cu_mp.B[pid]*((cu_mp.ρs[ix]/cu_mp.ρs_init[ix])^cu_mp.γ[pid]-FNUM_1)
        P = T2(1225)*(cu_mp.ρs[ix]-cu_mp.ρs_init[ix])
        cu_mp.σij[ix, 1] = -P
        cu_mp.σij[ix, 2] = -P
        cu_mp.σij[ix, 3] = -P
        # update mean stress tensor
        cu_mp.σm[ix] = P
    end
    return nothing
end