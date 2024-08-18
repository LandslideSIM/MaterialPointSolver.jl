#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : linearelastic.jl                                                           |
|  Description: Linear elastic material's constitutive model.                              |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. liE! [2D]                                                               |
|               2. liE! [3D]                                                               |
+==========================================================================================#

"""
    liE!(mp::KernelParticle2D{T1, T2}, pts_attr::KernelParticleProperty{T1, T2})

Description:
---
GPU kernel to implement linear elastic constitutive model (2D plane strain).
"""
@kernel inbounds=true function liE!(
    mp      ::      KernelParticle2D{T1, T2},
    pts_attr::KernelParticleProperty{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix≤mp.num
        pid = pts_attr.layer[ix]
        Ks  = pts_attr.Ks[pid]
        Gt  = pts_attr.G[pid]
        # spin tensor
        # here, ωij = (vorticity tensor) × Δt, i.e.
        # ωij = (∂Ni × Vj - ∂Nj × Vi) × 0.5 × Δt
        ωxy = T2(0.5) * (mp.ΔFs[ix, 2] - mp.ΔFs[ix, 3])
        # objective stress
        # σij,new = σij,old + σij,R
        # σxx,R =  2 × σxy × ωxy
        # σyy,R = -2 × σxy × ωxy
        # σxy,R =  ωxy × (σyy - σxx)
        # where σij = σji, ωij = -ωji
        σij1 = mp.σij[ix, 1]
        σij2 = mp.σij[ix, 2]
        σij4 = mp.σij[ix, 4]
        mp.σij[ix, 1] +=  ωxy *  σij4 * T2(2.0)
        mp.σij[ix, 2] += -ωxy *  σij4 * T2(2.0)
        mp.σij[ix, 4] +=  ωxy * (σij2 -σij1)
        # linear elastic
        Dt = Ks + T2(1.333333) * Gt
        Dd = Ks - T2(0.666667) * Gt
        mp.σij[ix, 1] += Dt * mp.Δϵij_s[ix, 1] + 
                         Dd * mp.Δϵij_s[ix, 2] + 
                         Dd * mp.Δϵij_s[ix, 3]
        mp.σij[ix, 2] += Dd * mp.Δϵij_s[ix, 1] + 
                         Dt * mp.Δϵij_s[ix, 2] + 
                         Dd * mp.Δϵij_s[ix, 3]
        mp.σij[ix, 3] += Dd * mp.Δϵij_s[ix, 1] + 
                         Dd * mp.Δϵij_s[ix, 2] + 
                         Dt * mp.Δϵij_s[ix, 3]
        mp.σij[ix, 4] += Gt * mp.Δϵij_s[ix, 4]
        # update mean stress tensor
        σm = (mp.σij[ix, 1] + mp.σij[ix, 2] + mp.σij[ix, 3]) * T2(0.333333)
        mp.σm[ix] = σm
        # update deviatoric stress tensor
        mp.sij[ix, 1] = mp.σij[ix, 1] - σm
        mp.sij[ix, 2] = mp.σij[ix, 2] - σm
        mp.sij[ix, 3] = mp.σij[ix, 3] - σm
        mp.sij[ix, 4] = mp.σij[ix, 4]
    end
end

"""
    liE!(mp::KernelParticle3D{T1, T2}, pts_attr::KernelParticleProperty{T1, T2})

Description:
---
GPU kernel to implement linear elastic constitutive model (3D).
"""
@kernel inbounds=true function liE!(
    mp      ::      KernelParticle3D{T1, T2},
    pts_attr::KernelParticleProperty{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix≤mp.num
        pid = pts_attr.layer[ix]
        Ks  = pts_attr.Ks[pid]
        Gt  = pts_attr.G[pid]
        # spin tensor
        # here, ωij = (vorticity tensor) × Δt, i.e.
        # ωij = (∂Ni × Vj - ∂Nj × Vi) × 0.5 × Δt
        ωxy = T2(0.5) * (mp.ΔFs[ix, 4] - mp.ΔFs[ix, 2])
        ωyz = T2(0.5) * (mp.ΔFs[ix, 8] - mp.ΔFs[ix, 6])
        ωxz = T2(0.5) * (mp.ΔFs[ix, 7] - mp.ΔFs[ix, 3])
        # objective stress
        # σij,new = σij,old + σij,R
        # σxx,R =  2 × (σxy × ωxy + σxz × ωxz)
        # σyy,R = -2 × (σxy × ωxy - σyz × ωyz)
        # σzz,R = -2 × (σxz × ωxz + σyz × ωyz)
        # σxy,R =  ωxy × (σyy - σxx) + σyz × ωxz + σxz × ωyz
        # σyz,R =  ωyz × (σzz - σyy) - σxy × ωxz - σxz × ωxy
        # σxz,R =  ωxz × (σzz - σxx) + σyz × ωxy - σxy × ωyz
        # where σij = σji, ωij = -ωji
        σij1 = mp.σij[ix, 1]
        σij2 = mp.σij[ix, 2]
        σij3 = mp.σij[ix, 3]
        σij4 = mp.σij[ix, 4]
        σij5 = mp.σij[ix, 5]
        σij6 = mp.σij[ix, 6]
        mp.σij[ix, 1] +=  T2(2.0) * (σij4 * ωxy + σij6 * ωxz)
        mp.σij[ix, 2] += -T2(2.0) * (σij4 * ωxy - σij5 * ωyz)
        mp.σij[ix, 3] += -T2(2.0) * (σij6 * ωxz + σij5 * ωyz)
        mp.σij[ix, 4] += ωxy * (σij2 - σij1) + ωxz * σij5 + ωyz * σij6
        mp.σij[ix, 5] += ωyz * (σij3 - σij2) - ωxz * σij4 - ωxy * σij6
        mp.σij[ix, 6] += ωxz * (σij3 - σij1) + ωxy * σij5 - ωyz * σij4
        # linear elastic
        Dt = Ks + T2(1.333333) * Gt
        Dd = Ks - T2(0.666667) * Gt
        mp.σij[ix, 1] += Dt * mp.Δϵij_s[ix, 1] + 
                         Dd * mp.Δϵij_s[ix, 2] + 
                         Dd * mp.Δϵij_s[ix, 3]
        mp.σij[ix, 2] += Dd * mp.Δϵij_s[ix, 1] + 
                         Dt * mp.Δϵij_s[ix, 2] + 
                         Dd * mp.Δϵij_s[ix, 3]
        mp.σij[ix, 3] += Dd * mp.Δϵij_s[ix, 1] + 
                         Dd * mp.Δϵij_s[ix, 2] + 
                         Dt * mp.Δϵij_s[ix, 3]
        mp.σij[ix, 4] += Gt * mp.Δϵij_s[ix, 4]
        mp.σij[ix, 5] += Gt * mp.Δϵij_s[ix, 5]
        mp.σij[ix, 6] += Gt * mp.Δϵij_s[ix, 6]
        # update mean stress tensor
        σm = (mp.σij[ix, 1] + mp.σij[ix, 2] + mp.σij[ix, 3]) * T2(0.333333)
        mp.σm[ix] = σm
        # update deviatoric stress tensor
        mp.sij[ix, 1] = mp.σij[ix, 1] - σm
        mp.sij[ix, 2] = mp.σij[ix, 2] - σm
        mp.sij[ix, 3] = mp.σij[ix, 3] - σm
        mp.sij[ix, 4] = mp.σij[ix, 4]
        mp.sij[ix, 5] = mp.σij[ix, 5]
        mp.sij[ix, 6] = mp.σij[ix, 6]
    end
end