#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : mohrcoulomb_d.jl                                                           |
|  Description: Mohr-Coulomb constitutive model.                                           |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. mcP!() [2D GPU Kernel]                                                  |
|               2. mcP!() [3D GPU Kernel]                                                  |
+==========================================================================================#

"""
    mcP!(cu_mp::KernelParticle2D{T1, T2})

Description:
---
GPU kernel to implement Mohr-Coulomb constitutive model (2D plane strain).
"""
function mcP!(cu_mp::KernelParticle2D{T1, T2}) where {T1, T2}
    FNUM_0 = T2(0.0); FNUM_q = T2(0.25); FNUM_13 = T2(1/3); FNUM_1  = T2(1.0)
    FNUM_2 = T2(2.0); FNUM_h = T2(0.5) ; FNUM_43 = T2(4/3); FNUM_23 = T2(2/3)
    INUM_2 = T1(2) 
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    if ix≤cu_mp.num
        pid = cu_mp.layer[ix]
        c   = cu_mp.c[pid]
        Hp  = cu_mp.Hp[pid]
        cr  = cu_mp.cr[pid]
        ϕ   = cu_mp.ϕ[pid]
        G   = cu_mp.G[pid]
        Ks  = cu_mp.Ks[pid]

        # mohr-coulomb 
        c = max(c+Hp*cu_mp.epII[ix], cr)

        ds  = cu_mp.σij[ix, 1]-cu_mp.σij[ix, 2]
        tau = sqrt(FNUM_q*ds^INUM_2+cu_mp.σij[ix, 4]^INUM_2)
        sig = FNUM_h*(cu_mp.σij[ix, 1]+cu_mp.σij[ix, 2])
        f   = tau+sig*sin(ϕ)-c*cos(ϕ)
        sn1 = cu_mp.σij[ix, 1]
        sn2 = cu_mp.σij[ix, 2]
        sn3 = cu_mp.σij[ix, 3]
        sn4 = cu_mp.σij[ix, 4]
        
        beta  = abs(c*cos(ϕ)-sig*sin(ϕ))/tau
        dsigA = FNUM_h*beta*ds
        dsigB = c/tan(ϕ)
        if (sig≤dsigB)&&(f>FNUM_0)
            sn1 = sig+dsigA
            sn2 = sig-dsigA
            sn4 = beta*cu_mp.σij[ix, 4]
        end
        if (sig>dsigB)&&(f>FNUM_0)
            sn1 = dsigB
            sn2 = dsigB
            sn4 = FNUM_0
        end

        dsig1 = sn1-cu_mp.σij[ix, 1]
        dsig2 = sn2-cu_mp.σij[ix, 2]
        dsig3 = sn3-cu_mp.σij[ix, 3]
        dsig4 = sn4-cu_mp.σij[ix, 4]
        cu_mp.σij[ix, 1] = sn1
        cu_mp.σij[ix, 2] = sn2
        cu_mp.σij[ix, 3] = sn3
        cu_mp.σij[ix, 4] = sn4

        Dt    = Ks+FNUM_43*G
        Dd    = Ks-FNUM_23*G
        base  = FNUM_1/((Dd-Dt)*(FNUM_2*Dd+Dt))
        ep_xx = -( Dd*dsig1+Dt*dsig1-Dd*dsig2-Dd*dsig3)*base
        ep_yy = -(-Dd*dsig1+Dd*dsig2+Dt*dsig2-Dd*dsig3)*base
        ep_zz = -(-Dd*dsig1-Dd*dsig2+Dd*dsig3+Dt*dsig3)*base
        ep_xy = dsig4/G
        cu_mp.epK[ix]   = ep_xx+ep_yy+ep_zz
        cu_mp.epII[ix] += sqrt(FNUM_23*(ep_xx^INUM_2+       ep_yy^INUM_2+
                                        ep_zz^INUM_2+FNUM_2*ep_xy^INUM_2)) 
        # update mean stress tensor
        cu_mp.σm[ix] = (cu_mp.σij[ix, 1]+cu_mp.σij[ix, 2]+cu_mp.σij[ix, 3])*FNUM_13
        # update deviatoric stress tensor
        cu_mp.sij[ix, 1] = cu_mp.σij[ix, 1]-cu_mp.σm[ix]
        cu_mp.sij[ix, 2] = cu_mp.σij[ix, 2]-cu_mp.σm[ix]
        cu_mp.sij[ix, 3] = cu_mp.σij[ix, 3]-cu_mp.σm[ix]
        cu_mp.sij[ix, 4] = cu_mp.σij[ix, 4]
    end
    return nothing
end

"""
    mcP!(cu_mp::KernelParticle3D{T1, T2})

Description:
---
GPU kernel to implement Mohr-Coulomb constitutive model (3D).
"""
function mcP!(cu_mp::KernelParticle3D{T1, T2}) where {T1, T2}
    FNUM_0  = T2(0.0); FNUM_q  = T2(0.25); FNUM_h  = T2(0.5)
    FNUM_2  = T2(2.0); FNUM_13 = T2(1/3) ; FNUM_43 = T2(4/3)
    FNUM_23 = T2(2/3); INUM_2  = T1(2)
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    if ix≤cu_mp.num
        # to do
    end
    return nothing
end