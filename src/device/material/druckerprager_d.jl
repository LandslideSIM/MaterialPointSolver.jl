#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : druckerprager_d.jl                                                         |
|  Description: Drucker-Prager constitutive model.                                         |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. dpP! [2D GPU kernel]                                                    |
|               2. dpP! [3D GPU kernel]                                                    |
+==========================================================================================#

"""
    dpP!(cu_mp::KernelParticle2D{T1, T2})

Description:
---
GPU kernel to implement Drucker-Prager constitutive model (2D).

- read  → mp.num* 9          
- write → mp.num*11
- total → mp.num*20
"""
function dpP!(cu_mp::KernelParticle2D{T1, T2}) where {T1, T2}
    FNUM_13 = T2(1/3); FNUM_12 = T2(0.5); FNUM_S2 = T2(sqrt(2))
    FNUM_29 = T2(2/9); FNUM_6  = T2(6.0); FNUM_S3 = T2(sqrt(3))
    FNUM_3  = T2(3.0); FNUM_1  = T2(1.0); FNUM_0  = T2(0.0)
    INUM_2  = T1(2)
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    if ix≤cu_mp.num
        pid = cu_mp.layer[ix]
        σm  = cu_mp.σm[ix]
        c   = cu_mp.c[pid]
        ϕ   = cu_mp.ϕ[pid]
        ψ   = cu_mp.ψ[pid]
        σt  = cu_mp.σt[pid]
        G   = cu_mp.G[pid]
        Ks  = cu_mp.Ks[pid]
        # drucker-prager
        τ = sqrt(FNUM_12*(cu_mp.sij[ix, 1]^INUM_2 +cu_mp.sij[ix, 2]^INUM_2+
                          cu_mp.sij[ix, 3]^INUM_2)+cu_mp.sij[ix, 4]^INUM_2)
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
            cu_mp.σij[ix, 1] = cu_mp.sij[ix, 1]*tmp2+tmp1
            cu_mp.σij[ix, 2] = cu_mp.sij[ix, 2]*tmp2+tmp1
            cu_mp.σij[ix, 3] = cu_mp.sij[ix, 3]*tmp2+tmp1
            cu_mp.σij[ix, 4] = cu_mp.sij[ix, 4]*tmp2
            cu_mp.epII[ix]  += Δλs*sqrt(FNUM_13+FNUM_29*qψ^INUM_2)
            cu_mp.epK[ix]   += Δλs*qψ
        end
        ## tensile failure correction
        if (σm≥σt)&&(BF≤FNUM_0)
            Δλt = ft/Ks
            cu_mp.σij[ix, 1] = cu_mp.sij[ix, 1]+σt
            cu_mp.σij[ix, 2] = cu_mp.sij[ix, 2]+σt
            cu_mp.σij[ix, 3] = cu_mp.sij[ix, 3]+σt
            cu_mp.epII[ix]  += Δλt*FNUM_13*FNUM_S2
            cu_mp.epK[ix]   += Δλt
        end
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
    dpP!(cu_mp::KernelParticle3D{T1, T2})

Description:
---
GPU kernel to implement Drucker-Prager constitutive model (3D).

- read  → mp.num*13         
- write → mp.num*15
- total → mp.num*28
"""
function dpP!(cu_mp::KernelParticle3D{T1, T2}) where {T1, T2}
    FNUM_13 = T2(1/3); FNUM_12 = T2(0.5); FNUM_S2 = T2(sqrt(2))
    FNUM_29 = T2(2/9); FNUM_6  = T2(6.0); FNUM_S3 = T2(sqrt(3))
    FNUM_3  = T2(3.0); FNUM_1  = T2(1.0); FNUM_0  = T2(0.0)
    INUM_2  = T1(2)
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    if ix≤cu_mp.num
        pid = cu_mp.layer[ix]
        σm  = cu_mp.σm[ix]
        c   = cu_mp.c[pid]
        ϕ   = cu_mp.ϕ[pid]
        ψ   = cu_mp.ψ[pid]
        σt  = cu_mp.σt[pid]
        G   = cu_mp.G[pid]
        Ks  = cu_mp.Ks[pid]
        # drucker-prager
        τ  = sqrt(FNUM_12*(cu_mp.sij[ix, 1]^INUM_2+
                           cu_mp.sij[ix, 2]^INUM_2+
                           cu_mp.sij[ix, 3]^INUM_2)+
                           cu_mp.sij[ix, 4]^INUM_2+
                           cu_mp.sij[ix, 5]^INUM_2+
                           cu_mp.sij[ix, 6]^INUM_2)
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
            cu_mp.σij[ ix, 1]  = cu_mp.sij[ix, 1]*tmp2+tmp1
            cu_mp.σij[ ix, 2]  = cu_mp.sij[ix, 2]*tmp2+tmp1
            cu_mp.σij[ ix, 3]  = cu_mp.sij[ix, 3]*tmp2+tmp1
            cu_mp.σij[ ix, 4]  = cu_mp.sij[ix, 4]*tmp2
            cu_mp.σij[ ix, 5]  = cu_mp.sij[ix, 5]*tmp2
            cu_mp.σij[ ix, 6]  = cu_mp.sij[ix, 6]*tmp2
            cu_mp.epII[ix   ] += Δλs*sqrt(FNUM_13+FNUM_29*qψ^INUM_2)
            cu_mp.epK[ ix   ] += Δλs*qψ
        end
        ## tensile failure correction
        if (σm≥σt)&&(BF≤FNUM_0)
            Δλt = ft/Ks
            cu_mp.σij[ ix, 1]  = cu_mp.sij[ix, 1]+σt
            cu_mp.σij[ ix, 2]  = cu_mp.sij[ix, 2]+σt
            cu_mp.σij[ ix, 3]  = cu_mp.sij[ix, 3]+σt
            cu_mp.epII[ix   ] += Δλt*FNUM_13*FNUM_S2
            cu_mp.epK[ ix   ] += Δλt
        end
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