#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : druckerprager.jl                                                           |
|  Description: Drucker-Prager constitutive model.                                         |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. druckerprager! [2D plane strain]                                        |
|               2. druckerprager! [3D]                                                     |
+==========================================================================================#

"""
    druckerprager!(mp::Particle2D{T1, T2})

Description:
---
Update strain and stress according to Drucker-Prager constitutive model. (2D)
"""
@views function druckerprager!(mp::Particle2D{T1, T2}) where {T1, T2}
    FNUM_13 = T2(1/3); FNUM_12 = T2(0.5); FNUM_S2 = T2(sqrt(2))
    FNUM_29 = T2(2/9); FNUM_6  = T2(6.0); FNUM_S3 = T2(sqrt(3))
    FNUM_3  = T2(3.0); FNUM_1  = T2(1.0); FNUM_0  = T2(0.0); INUM_2  = T1(2)
    for ix in 1:mp.num
        pid = mp.layer[ix]
        σm  = mp.σm[ix]
        c   = mp.c[pid]
        ϕ   = mp.ϕ[pid]
        ψ   = mp.ψ[pid]
        σt  = mp.σt[pid]
        G   = mp.G[pid]
        Ks  = mp.Ks[pid]
        # drucker-prager
        τ = sqrt(FNUM_12*(mp.sij[ix, 1]^INUM_2+mp.sij[ix, 2]^INUM_2+mp.sij[ix, 3]^INUM_2)+
                          mp.sij[ix, 4]^INUM_2)
        kϕ = (FNUM_6*c*cos(ϕ))/(FNUM_S3*(FNUM_3+sin(ϕ)))
        qϕ = (FNUM_6*  sin(ϕ))/(FNUM_S3*(FNUM_3+sin(ϕ)))
        qψ = (FNUM_6*  sin(ψ))/(FNUM_S3*(FNUM_3+sin(ψ)))
        σt = min(σt, kϕ/qϕ)
        αb = sqrt(FNUM_1+qϕ^INUM_2)-qϕ
        τb = kϕ-qϕ*σt
        fs = τ+qϕ*σm-kϕ # yield function considering shear failure
        ft = σm-σt      # yield function considering tensile failure
        BF = (τ-τb)-(αb*(σm-σt)) # BF is used to classify shear failure from tensile failure
        # determination of failure criteria
        ## shear failure correction
        if ((σm<σt)&&(fs>FNUM_0))||
           ((σm≥σt)&&(BF>FNUM_0))
            Δλs  = fs/(G+Ks*qϕ*qψ)
            tmp1 = σm-Ks*qψ*Δλs
            tmp2 = (kϕ-qϕ*tmp1)/τ
            mp.σij[ix, 1] = mp.sij[ix, 1]*tmp2+tmp1
            mp.σij[ix, 2] = mp.sij[ix, 2]*tmp2+tmp1
            mp.σij[ix, 3] = mp.sij[ix, 3]*tmp2+tmp1
            mp.σij[ix, 4] = mp.sij[ix, 4]*tmp2
            mp.epII[ix]  += Δλs*sqrt(FNUM_13+FNUM_29*qψ^INUM_2)
            mp.epK[ix]   += Δλs*qψ
        end
        ## tensile failure correction
        if (σm≥σt)&&(BF≤FNUM_0)
            Δλt = ft/Ks
            mp.σij[ix, 1] = mp.sij[ix, 1]+σt
            mp.σij[ix, 2] = mp.sij[ix, 2]+σt
            mp.σij[ix, 3] = mp.sij[ix, 3]+σt
            mp.epII[ix]  += (Δλt*FNUM_13)*FNUM_S2
            mp.epK[ix]   += Δλt
        end
        # update mean stress tensor
        σm = (mp.σij[ix, 1]+mp.σij[ix, 2]+mp.σij[ix, 3])*FNUM_13
        mp.σm[ix] = σm
        # update deviatoric stress tensor
        mp.sij[ix, 1] = mp.σij[ix, 1]-σm
        mp.sij[ix, 2] = mp.σij[ix, 2]-σm
        mp.sij[ix, 3] = mp.σij[ix, 3]-σm
        mp.sij[ix, 4] = mp.σij[ix, 4]
    end
    return nothing
end

"""
    druckerprager!(mp::Particle3D{T1, T2})

Description:
---
Update strain and stress according to Drucker-Prager constitutive model. (3D)
"""
@views function druckerprager!(mp::Particle3D{T1, T2}) where {T1, T2}
    FNUM_13 = T2(1/3); FNUM_12 = T2(0.5); FNUM_S2 = T2(sqrt(2))
    FNUM_29 = T2(2/9); FNUM_6  = T2(6.0); FNUM_S3 = T2(sqrt(3))
    FNUM_3  = T2(3.0); FNUM_1  = T2(1.0); FNUM_0  = T2(0.0); INUM_2  = T1(2)
    for ix in 1:mp.num
        pid = mp.layer[ix]
        σm  = mp.σm[ix]
        c   = mp.c[pid]
        ϕ   = mp.ϕ[pid]
        ψ   = mp.ψ[pid]
        σt  = mp.σt[pid]
        G   = mp.G[pid]
        Ks  = mp.Ks[pid]
        # drucker-prager
        τ  = sqrt(FNUM_12*(mp.sij[ix, 1]^INUM_2+mp.sij[ix, 2]^INUM_2+mp.sij[ix, 3]^INUM_2)+
                           mp.sij[ix, 4]^INUM_2+mp.sij[ix, 5]^INUM_2+mp.sij[ix, 6]^INUM_2)
        kϕ = (FNUM_6*c*cos(ϕ))/(FNUM_S3*(FNUM_3+sin(ϕ)))
        qϕ = (FNUM_6  *sin(ϕ))/(FNUM_S3*(FNUM_3+sin(ϕ)))
        qψ = (FNUM_6  *sin(ψ))/(FNUM_S3*(FNUM_3+sin(ψ)))
        σt = min(σt, kϕ/qϕ)
        αb = sqrt(FNUM_1+qϕ^INUM_2)-qϕ
        τb = kϕ-qϕ*σt
        fs = τ+qϕ*σm-kϕ # yield function considering shear failure
        ft = σm-σt      # yield function considering tensile failure
        BF = (τ-τb)-(αb*(σm-σt)) # BF is used to classify shear failure from tensile failure
        # determination of failure criteria
        ## shear failure correction
        if ((σm<σt)&&(fs>FNUM_0))||
           ((σm≥σt)&&(BF>FNUM_0))
            Δλs  = fs/(G+Ks*qϕ*qψ)
            tmp1 = σm-Ks*qψ*Δλs
            tmp2 = (kϕ-qϕ*tmp1)/τ
            mp.σij[ix, 1] = mp.sij[ix, 1]*tmp2+tmp1
            mp.σij[ix, 2] = mp.sij[ix, 2]*tmp2+tmp1
            mp.σij[ix, 3] = mp.sij[ix, 3]*tmp2+tmp1
            mp.σij[ix, 4] = mp.sij[ix, 4]*tmp2
            mp.σij[ix, 5] = mp.sij[ix, 5]*tmp2
            mp.σij[ix, 6] = mp.sij[ix, 6]*tmp2
            mp.epII[ix]  += Δλs*sqrt(FNUM_13+FNUM_29*qψ^INUM_2)
            mp.epK[ix]   += Δλs*qψ
        end
        ## tensile failure correction
        if (σm≥σt)&&(BF≤FNUM_0)
            Δλt = ft/Ks
            mp.σij[ix, 1] = mp.sij[ix, 1]+σt
            mp.σij[ix, 2] = mp.sij[ix, 2]+σt
            mp.σij[ix, 3] = mp.sij[ix, 3]+σt
            mp.epII[ix]  += (Δλt*FNUM_13)*FNUM_S2
            mp.epK[ix]   += Δλt
        end
        # update mean stress tensor
        σm = (mp.σij[ix, 1]+mp.σij[ix, 2]+mp.σij[ix, 3])*FNUM_13
        mp.σm[ix] = σm
        # update deviatoric stress tensor
        mp.sij[ix, 1] = mp.σij[ix, 1]-σm
        mp.sij[ix, 2] = mp.σij[ix, 2]-σm
        mp.sij[ix, 3] = mp.σij[ix, 3]-σm
        mp.sij[ix, 4] = mp.σij[ix, 4]
        mp.sij[ix, 5] = mp.σij[ix, 5]
        mp.sij[ix, 6] = mp.σij[ix, 6]
    end
    return nothing
end
