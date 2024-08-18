#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : druckerprager.jl                                                           |
|  Description: Drucker-Prager constitutive model.                                         |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. dpP! [2D]                                                               |
|               2. dpP! [3D]                                                               |
+==========================================================================================#

"""
    dpP!(mp::KernelParticle2D{T1, T2}, pts_attr::KernelParticleProperty{T1, T2})

Description:
---
Implement Drucker-Prager constitutive model (2D).
"""
@kernel inbounds=true function dpP!(
    mp      ::      KernelParticle2D{T1, T2},
    pts_attr::KernelParticleProperty{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix≤mp.num
        σm  = mp.σm[ix]
        pid = pts_attr.layer[ix]
        c   = pts_attr.c[pid]
        ϕ   = pts_attr.ϕ[pid]
        ψ   = pts_attr.ψ[pid]
        σt  = pts_attr.σt[pid]
        G   = pts_attr.G[pid]
        Ks  = pts_attr.Ks[pid]
        # drucker-prager
        τ = sqrt(T2(0.5)*(mp.sij[ix, 1]^T1(2) +mp.sij[ix, 2]^T1(2)+
                          mp.sij[ix, 3]^T1(2))+mp.sij[ix, 4]^T1(2))
        kϕ = (T2(6.0)*c*cos(ϕ))/(T2(1.732051)*(T2(3.0)+sin(ϕ)))
        qϕ = (T2(6.0)  *sin(ϕ))/(T2(1.732051)*(T2(3.0)+sin(ϕ)))
        qψ = (T2(6.0)  *sin(ψ))/(T2(1.732051)*(T2(3.0)+sin(ψ)))
        σt = min(σt, kϕ/qϕ)
        αb = sqrt(T2(1.0)+qϕ^T1(2))-qϕ
        τb = kϕ-qϕ*σt
        fs = τ+qϕ*σm-kϕ # yield function considering shear failure
        ft = σm-σt      # yield function considering tensile failure
        BF = (τ-τb)-(αb*(σm-σt)) # BF is used to classify shear failure from tensile failure
        # determination of failure criteria
        ## shear failure correction
        if ((σm<σt)&&(fs>T2(0.0)))||
           ((σm≥σt)&&(BF>T2(0.0)))
            Δλs  = fs/(G+Ks*qϕ*qψ)
            tmp1 = σm-Ks*qψ*Δλs
            tmp2 = (kϕ-qϕ*tmp1)/τ
            mp.σij[ix, 1] = mp.sij[ix, 1]*tmp2+tmp1
            mp.σij[ix, 2] = mp.sij[ix, 2]*tmp2+tmp1
            mp.σij[ix, 3] = mp.sij[ix, 3]*tmp2+tmp1
            mp.σij[ix, 4] = mp.sij[ix, 4]*tmp2
            mp.epII[ix]  += Δλs*sqrt(T2(0.333333)+T2(0.222222)*qψ^T1(2))
            mp.epK[ix]   += Δλs*qψ
        end
        ## tensile failure correction
        if (σm≥σt)&&(BF≤T2(0.0))
            Δλt = ft/Ks
            mp.σij[ix, 1] = mp.sij[ix, 1]+σt
            mp.σij[ix, 2] = mp.sij[ix, 2]+σt
            mp.σij[ix, 3] = mp.sij[ix, 3]+σt
            mp.epII[ix]  += Δλt*T2(0.333333)*T2(1.414214)
            mp.epK[ix]   += Δλt
        end
        # update mean stress tensor
        σm = (mp.σij[ix, 1]+mp.σij[ix, 2]+mp.σij[ix, 3])*T2(0.333333)
        mp.σm[ix] = σm
        # update deviatoric stress tensor
        mp.sij[ix, 1] = mp.σij[ix, 1]-σm
        mp.sij[ix, 2] = mp.σij[ix, 2]-σm
        mp.sij[ix, 3] = mp.σij[ix, 3]-σm
        mp.sij[ix, 4] = mp.σij[ix, 4]
    end
end

"""
    dpP!(mp::KernelParticle3D{T1, T2}, pts_attr::KernelParticleProperty{T1, T2})

Description:
---
Implement Drucker-Prager constitutive model (3D).
"""
@kernel inbounds=true function dpP!(
    mp      ::      KernelParticle3D{T1, T2},
    pts_attr::KernelParticleProperty{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix≤mp.num
        σm  = mp.σm[ix]
        pid = pts_attr.layer[ix]
        c   = pts_attr.c[pid]
        ϕ   = pts_attr.ϕ[pid]
        ψ   = pts_attr.ψ[pid]
        σt  = pts_attr.σt[pid]
        G   = pts_attr.G[pid]
        Ks  = pts_attr.Ks[pid]
        # drucker-prager
        τ  = sqrt(T2(0.5) * (mp.sij[ix, 1] * mp.sij[ix, 1]  + 
                             mp.sij[ix, 2] * mp.sij[ix, 2]  +
                             mp.sij[ix, 3] * mp.sij[ix, 3]) +
                             mp.sij[ix, 4] * mp.sij[ix, 4]  +
                             mp.sij[ix, 5] * mp.sij[ix, 5]  +
                             mp.sij[ix, 6] * mp.sij[ix, 6])
        kϕ = (T2(6.0) * c * cos(ϕ)) / (T2(1.732051) * (T2(3.0) + sin(ϕ)))
        qϕ = (T2(6.0) *     sin(ϕ)) / (T2(1.732051) * (T2(3.0) + sin(ϕ)))
        qψ = (T2(6.0) *     sin(ψ)) / (T2(1.732051) * (T2(3.0) + sin(ψ)))
        σt = min(σt, kϕ / qϕ)
        αb = sqrt(T2(1.0) + qϕ * qϕ) - qϕ
        τb = kϕ - qϕ * σt
        fs = τ + qϕ * σm - kϕ            # yield function considering shear failure
        ft = σm - σt                     # yield function considering tensile failure
        BF = (τ - τb) - (αb * (σm - σt)) # BF is used to classify shear failure from tensile failure
        # determination of failure criteria
        ## shear failure correction
        if ((σm < σt) && (fs > T2(0.0))) ||
           ((σm ≥ σt) && (BF > T2(0.0)))
            Δλs  = fs / (G + Ks * qϕ * qψ)
            tmp1 = σm - Ks * qψ * Δλs
            tmp2 = (kϕ - qϕ * tmp1) / τ
            mp.σij[ix, 1] = mp.sij[ix, 1] * tmp2 + tmp1
            mp.σij[ix, 2] = mp.sij[ix, 2] * tmp2 + tmp1
            mp.σij[ix, 3] = mp.sij[ix, 3] * tmp2 + tmp1
            mp.σij[ix, 4] = mp.sij[ix, 4] * tmp2
            mp.σij[ix, 5] = mp.sij[ix, 5] * tmp2
            mp.σij[ix, 6] = mp.sij[ix, 6] * tmp2
            mp.epII[ix]  += Δλs * sqrt(T2(0.333333) + T2(0.222222) * qψ * qψ)
            mp.epK[ix]   += Δλs * qψ
        end
        ## tensile failure correction
        if (σm ≥ σt) && (BF ≤ T2(0.0))
            Δλt = ft / Ks
            mp.σij[ix, 1] = mp.sij[ix, 1] + σt
            mp.σij[ix, 2] = mp.sij[ix, 2] + σt
            mp.σij[ix, 3] = mp.sij[ix, 3] + σt
            mp.epII[ix]  += Δλt * T2(0.333333) * T2(1.414214)
            mp.epK[ix]   += Δλt
        end
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