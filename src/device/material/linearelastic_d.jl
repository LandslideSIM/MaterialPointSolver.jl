#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : linearelastic_d.jl                                                         |
|  Description: Linear elastic material's constitutive model.                              |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. liE! [2D GPU kernel]                                                    |
|               2. liE! [3D GPU kernel]                                                    |
+==========================================================================================#

"""
    liE!(cu_mp::KernelParticle2D{T1, T2})

Description:
---
GPU kernel to implement linear elastic constitutive model (2D plane strain).

- read  → mp.num*13     
- write → mp.num*12
- total → mp.num*25
"""
function liE!(cu_mp::KernelParticle2D{T1, T2}) where {T1, T2}
    FNUM_12 = T2(0.5); FNUM_23 = T2(2/3); FNUM_2 = T2(2.0)
    FNUM_43 = T2(4/3); FNUM_13 = T2(1/3)
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    if ix≤cu_mp.num
        pid = cu_mp.layer[ix]
        Ks  = cu_mp.Ks[pid]
        G   = cu_mp.G[pid]
        # spin tensor
        # here, ωij = (vorticity tensor) × Δt, i.e.
        # ωij = (∂Ni × Vj - ∂Nj × Vi) × 0.5 × Δt
        ωxy = FNUM_12*(cu_mp.∂Fs[ix, 2]-cu_mp.∂Fs[ix, 3])
        # objective stress
        # σij,new = σij,old + σij,R
        # σxx,R =  2 × σxy × ωxy
        # σyy,R = -2 × σxy × ωxy
        # σxy,R =  ωxy × (σyy - σxx)
        # where σij = σji, ωij = -ωji
        σij1 = cu_mp.σij[ix, 1]
        σij2 = cu_mp.σij[ix, 2]
        σij4 = cu_mp.σij[ix, 4]
        cu_mp.σij[ix, 1] +=  ωxy*σij4*FNUM_2
        cu_mp.σij[ix, 2] += -ωxy*σij4*FNUM_2
        cu_mp.σij[ix, 4] +=  ωxy*(σij2-σij1)
        # linear elastic
        Dt = Ks+FNUM_43*G
        Dd = Ks-FNUM_23*G
        cu_mp.σij[ix, 1] += Dt*cu_mp.Δϵij_s[ix, 1]+Dd*cu_mp.Δϵij_s[ix, 2]+
                            Dd*cu_mp.Δϵij_s[ix, 3]
        cu_mp.σij[ix, 2] += Dd*cu_mp.Δϵij_s[ix, 1]+Dt*cu_mp.Δϵij_s[ix, 2]+
                            Dd*cu_mp.Δϵij_s[ix, 3]
        cu_mp.σij[ix, 3] += Dd*cu_mp.Δϵij_s[ix, 1]+Dd*cu_mp.Δϵij_s[ix, 2]+
                            Dt*cu_mp.Δϵij_s[ix, 3]
        cu_mp.σij[ix, 4] += G *cu_mp.Δϵij_s[ix, 4]
        # update mean stress tensor
        σm = (cu_mp.σij[ix, 1]+cu_mp.σij[ix, 2]+cu_mp.σij[ix, 3])*FNUM_13
        cu_mp.σm[ix] = σm
        # update deviatoric stress tensor
        cu_mp.sij[ix, 1] = cu_mp.σij[ix, 1]-σm
        cu_mp.sij[ix, 2] = cu_mp.σij[ix, 2]-σm
        cu_mp.sij[ix, 3] = cu_mp.σij[ix, 3]-σm
        cu_mp.sij[ix, 4] = cu_mp.σij[ix, 4]
    end
    return nothing
end

"""
    liE!(cu_mp::KernelParticle3D{T1, T2})

Description:
---
GPU kernel to implement linear elastic constitutive model (3D).

- read  → mp.num*24          
- write → mp.num*19
- total → mp.num*43
"""
function liE!(cu_mp::KernelParticle3D{T1, T2}) where {T1, T2}
    FNUM_12 = T2(0.5); FNUM_23 = T2(2/3); FNUM_2 = T2(2.0)
    FNUM_43 = T2(4/3); FNUM_13 = T2(1/3)
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    if ix≤cu_mp.num
        pid = cu_mp.layer[ix]
        Ks  = cu_mp.Ks[pid]
        G   = cu_mp.G[pid]
        # spin tensor
        # here, ωij = (vorticity tensor) × Δt, i.e.
        # ωij = (∂Ni × Vj - ∂Nj × Vi) × 0.5 × Δt
        ωxy = FNUM_12*(cu_mp.∂Fs[ix, 4]-cu_mp.∂Fs[ix, 2]) 
        ωyz = FNUM_12*(cu_mp.∂Fs[ix, 8]-cu_mp.∂Fs[ix, 6])
        ωxz = FNUM_12*(cu_mp.∂Fs[ix, 7]-cu_mp.∂Fs[ix, 3]) 
        # objective stress
        # σij,new = σij,old + σij,R
        # σxx,R =  2 × (σxy × ωxy + σxz × ωxz)
        # σyy,R = -2 × (σxy × ωxy - σyz × ωyz)
        # σzz,R = -2 × (σxz × ωxz + σyz × ωyz)
        # σxy,R =  ωxy × (σyy - σxx) + σyz × ωxz + σxz × ωyz
        # σyz,R =  ωyz × (σzz - σyy) - σxy × ωxz - σxz × ωxy
        # σxz,R =  ωxz × (σzz - σxx) + σyz × ωxy - σxy × ωyz
        # where σij = σji, ωij = -ωji
        σij1 = cu_mp.σij[ix, 1]
        σij2 = cu_mp.σij[ix, 2]
        σij3 = cu_mp.σij[ix, 3]
        σij4 = cu_mp.σij[ix, 4]
        σij5 = cu_mp.σij[ix, 5]
        σij6 = cu_mp.σij[ix, 6]
        cu_mp.σij[ix, 1] +=  FNUM_2*(σij4*ωxy+σij6*ωxz)
        cu_mp.σij[ix, 2] += -FNUM_2*(σij4*ωxy-σij5*ωyz)
        cu_mp.σij[ix, 3] += -FNUM_2*(σij6*ωxz+σij5*ωyz)
        cu_mp.σij[ix, 4] +=  ωxy*(σij2-σij1)+ωxz*σij5+ωyz*σij6
        cu_mp.σij[ix, 5] +=  ωyz*(σij3-σij2)-ωxz*σij4-ωxy*σij6
        cu_mp.σij[ix, 6] +=  ωxz*(σij3-σij1)+ωxy*σij5-ωyz*σij4
        # linear elastic
        Dt = Ks+FNUM_43*G
        Dd = Ks-FNUM_23*G
        cu_mp.σij[ix, 1] += Dt*cu_mp.Δϵij_s[ix, 1]+
                            Dd*cu_mp.Δϵij_s[ix, 2]+
                            Dd*cu_mp.Δϵij_s[ix, 3]
        cu_mp.σij[ix, 2] += Dd*cu_mp.Δϵij_s[ix, 1]+
                            Dt*cu_mp.Δϵij_s[ix, 2]+
                            Dd*cu_mp.Δϵij_s[ix, 3]
        cu_mp.σij[ix, 3] += Dd*cu_mp.Δϵij_s[ix, 1]+
                            Dd*cu_mp.Δϵij_s[ix, 2]+
                            Dt*cu_mp.Δϵij_s[ix, 3]
        cu_mp.σij[ix, 4] += G *cu_mp.Δϵij_s[ix, 4]
        cu_mp.σij[ix, 5] += G *cu_mp.Δϵij_s[ix, 5]
        cu_mp.σij[ix, 6] += G *cu_mp.Δϵij_s[ix, 6]
        # update mean stress tensor
        σm = (cu_mp.σij[ix, 1]+cu_mp.σij[ix, 2]+cu_mp.σij[ix, 3])*FNUM_13
        cu_mp.σm[ix] = σm
        # update deviatoric stress tensor
        cu_mp.sij[ix, 1] = cu_mp.σij[ix, 1]-σm
        cu_mp.sij[ix, 2] = cu_mp.σij[ix, 2]-σm
        cu_mp.sij[ix, 3] = cu_mp.σij[ix, 3]-σm
        cu_mp.sij[ix, 4] = cu_mp.σij[ix, 4]
        cu_mp.sij[ix, 5] = cu_mp.σij[ix, 5]
        cu_mp.sij[ix, 6] = cu_mp.σij[ix, 6]
    end
    return nothing
end