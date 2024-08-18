#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : linearelastic.jl                                                           |
|  Description: Linear elastic material's constitutive model.                              |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. linearelastic! [2D plane strain]                                        |
|               2. linearelastic! [3D]                                                     |
+==========================================================================================#

"""
    linearelastic!(mp::Particle2D{T1, T2})

Description:
---
Update strain and stress according to linear elastic constitutive model (2D plane strain).

```txt
D = | K+4/3*G, K-2/3*G, K-2/3*G, 0 | σ = | σ_xx | ϵ = | ϵ_xx |
    | K-2/3*G, K+4/3*G, K-2/3*G, 0 |     | σ_yy |     | ϵ_yy |
    | K-2/3*G, K-2/3*G, K+4/3*G, 0 |     | σ_zz |     | ϵ_zz |
    | 0      , 0      , 0      , G |     | σ_xy |     | ϵ_xy |
```
"""
@views function linearelastic!(mp::Particle2D{T1, T2}) where {T1, T2}
    FNUM_43 = T2(4/3); FNUM_13 = T2(1/3); FNUM_2  = T2(2.0)
    FNUM_23 = T2(2/3); FNUM_12 = T2(0.5)
    for ix in 1:mp.num
        pid = mp.layer[ix]
        G   = mp.G[pid]
        Ks  = mp.Ks[pid]
        # spin tensor
        # here, ωij = (vorticity tensor) × Δt, i.e.
        # ωij = (∂Ni × Vj - ∂Nj × Vi) × 0.5 × Δt
        ωxy = FNUM_12*(mp.∂Fs[ix, 2]-mp.∂Fs[ix, 3])
        # objective stress
        # σij,new = σij,old + σij,R
        # σxx,R =  2 × σxy × ωxy
        # σyy,R = -2 × σxy × ωxy
        # σxy,R =  ωxy × (σyy - σxx)
        # where σij = σji, ωij = -ωji
        σij1 = mp.σij[ix, 1]
        σij2 = mp.σij[ix, 2]
        σij4 = mp.σij[ix, 4]
        mp.σij[ix, 1] +=  ωxy*σij4*FNUM_2
        mp.σij[ix, 2] += -ωxy*σij4*FNUM_2
        mp.σij[ix, 4] +=  ωxy*(σij2-σij1)
        # linear elastic
        Dt = Ks+FNUM_43*G
        Dd = Ks-FNUM_23*G
        mp.σij[ix, 1] += Dt*mp.Δϵij_s[ix, 1]+Dd*mp.Δϵij_s[ix, 2]+Dd*mp.Δϵij_s[ix, 3]
        mp.σij[ix, 2] += Dd*mp.Δϵij_s[ix, 1]+Dt*mp.Δϵij_s[ix, 2]+Dd*mp.Δϵij_s[ix, 3]
        mp.σij[ix, 3] += Dd*mp.Δϵij_s[ix, 1]+Dd*mp.Δϵij_s[ix, 2]+Dt*mp.Δϵij_s[ix, 3]
        mp.σij[ix, 4] += G *mp.Δϵij_s[ix, 4]
        # update mean stress tensor
        mp.σm[ix] = (mp.σij[ix, 1]+mp.σij[ix, 2]+mp.σij[ix, 3])*FNUM_13
        # update deviatoric stress tensor
        mp.sij[ix, 1] = mp.σij[ix, 1]-mp.σm[ix]
        mp.sij[ix, 2] = mp.σij[ix, 2]-mp.σm[ix]
        mp.sij[ix, 3] = mp.σij[ix, 3]-mp.σm[ix]
        mp.sij[ix, 4] = mp.σij[ix, 4]
    end
    return nothing
end

"""
    linearelastic!(mp::Particle3D{T1, T2})

Description:
---
Update strain and stress according to linear elastic constitutive model (3D).

```txt
D = | K+4/3*G, K-2/3*G, Kc-2/3*G, 0, 0, 0 | σ = | σ_xx | ϵ = | ϵ_xx |
    | K-2/3*G, K+4/3*G, Kc-2/3*G, 0, 0, 0 |     | σ_yy |     | ϵ_yy |
    | K-2/3*G, K-2/3*G, Kc+4/3*G, 0, 0, 0 |     | σ_zz |     | ϵ_zz |
    | 0      , 0      , 0       , G, 0, 0 |     | σ_xy |     | ϵ_xy |  
    | 0      , 0      , 0       , 0, G, 0 |     | σ_yz |     | ϵ_yz |
    | 0      , 0      , 0       , 0, 0, G |     | σ_zx |     | ϵ_zx |
```
"""
@views function linearelastic!(mp::Particle3D{T1, T2}) where {T1, T2}
    FNUM_43 = T2(4/3); FNUM_13 = T2(1/3); FNUM_2  = T2(2.0)
    FNUM_23 = T2(2/3); FNUM_12 = T2(0.5)
    for ix in 1:mp.num
        pid = mp.layer[ix]
        G   = mp.G[pid]
        Ks  = mp.Ks[pid] 
        # spin tensor
        # here, ωij = (vorticity tensor) × Δt, i.e.
        # ωij = (∂Ni × Vj - ∂Nj × Vi) × 0.5 × Δt
        ωxy = FNUM_12*(mp.∂Fs[ix, 4]-mp.∂Fs[ix, 2]) 
        ωyz = FNUM_12*(mp.∂Fs[ix, 8]-mp.∂Fs[ix, 6])
        ωxz = FNUM_12*(mp.∂Fs[ix, 7]-mp.∂Fs[ix, 3]) 
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
        mp.σij[ix, 1] +=  FNUM_2*(σij4*ωxy+σij6*ωxz)
        mp.σij[ix, 2] += -FNUM_2*(σij4*ωxy-σij5*ωyz)
        mp.σij[ix, 3] += -FNUM_2*(σij6*ωxz+σij5*ωyz)
        mp.σij[ix, 4] +=  ωxy*(σij2-σij1)+ωxz*σij5+ωyz*σij6
        mp.σij[ix, 5] +=  ωyz*(σij3-σij2)-ωxz*σij4-ωxy*σij6
        mp.σij[ix, 6] +=  ωxz*(σij3-σij1)+ωxy*σij5-ωyz*σij4
        # linear elastic
        Dt = Ks+FNUM_43*G
        Dd = Ks-FNUM_23*G
        mp.σij[ix, 1] += Dt*mp.Δϵij_s[ix, 1]+Dd*mp.Δϵij_s[ix, 2]+Dd*mp.Δϵij_s[ix, 3]
        mp.σij[ix, 2] += Dd*mp.Δϵij_s[ix, 1]+Dt*mp.Δϵij_s[ix, 2]+Dd*mp.Δϵij_s[ix, 3]
        mp.σij[ix, 3] += Dd*mp.Δϵij_s[ix, 1]+Dd*mp.Δϵij_s[ix, 2]+Dt*mp.Δϵij_s[ix, 3]
        mp.σij[ix, 4] += G *mp.Δϵij_s[ix, 4]
        mp.σij[ix, 5] += G *mp.Δϵij_s[ix, 5]
        mp.σij[ix, 6] += G *mp.Δϵij_s[ix, 6]
        # update mean stress tensor
        mp.σm[ix] = (mp.σij[ix, 1]+mp.σij[ix, 2]+mp.σij[ix, 3])*FNUM_13
        # update deviatoric stress tensor
        mp.sij[ix, 1] = mp.σij[ix, 1]-mp.σm[ix]
        mp.sij[ix, 2] = mp.σij[ix, 2]-mp.σm[ix]
        mp.sij[ix, 3] = mp.σij[ix, 3]-mp.σm[ix]
        mp.sij[ix, 4] = mp.σij[ix, 4]
        mp.sij[ix, 5] = mp.σij[ix, 5]
        mp.sij[ix, 6] = mp.σij[ix, 6]
    end
    return nothing
end
