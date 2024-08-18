#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : mohrcoulomb.jl                                                             |
|  Description: Mohr-Coulomb constitutive model.                                           |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. mohrcoulomb! [2D plane strain]                                          |
|               2. mohrcoulomb! [3D]                                                       |
+==========================================================================================#

"""
    mohrcoulomb!(mp::Particle2D{T1, T2}, ΔT::T2)

Description:
---
Update strain and stress according to Mohr-Coulomb constitutive model (2D plane strain).
"""
@views function mohrcoulomb!(mp::Particle2D{T1, T2}) where {T1, T2}
    FNUM_0  = T2(0.0); FNUM_q  = T2(0.25); FNUM_h  = T2(0.5)
    FNUM_2  = T2(2.0); FNUM_1  = T2(1.0 ); INUM_2  = T1(2)
    FNUM_13 = T2(1/3); FNUM_43 = T2(4/3 ); FNUM_23 = T2(2/3)
    for ix in 1:mp.num
        pid = mp.layer[ix]
        c   = mp.c[pid]
        Hp  = mp.Hp[pid]
        cr  = mp.cr[pid]
        ϕ   = mp.ϕ[pid]
        G   = mp.G[pid]
        Ks  = mp.Ks[pid]
        # mohr-coulomb 
        c = max(c+Hp*mp.epII[ix], cr)
        ds  = mp.σij[ix, 1]-mp.σij[ix, 2]
        tau = sqrt(FNUM_q*ds^INUM_2+mp.σij[ix, 4]^INUM_2)
        sig = FNUM_h*(mp.σij[ix, 1]+mp.σij[ix, 2])
        f   = tau+sig*sin(ϕ)-c*cos(ϕ)
        sn1 = mp.σij[ix, 1]
        sn2 = mp.σij[ix, 2]
        sn3 = mp.σij[ix, 3]
        sn4 = mp.σij[ix, 4]
        
        beta  = abs(c*cos(ϕ)-sig*sin(ϕ))/tau
        dsigA = FNUM_h*beta*ds
        dsigB = c/tan(ϕ)
        if (sig≤dsigB)&&(f>FNUM_0)
            sn1 = sig+dsigA
            sn2 = sig-dsigA
            sn4 = beta*mp.σij[ix, 4]
        end
        if (sig>dsigB)&&(f>FNUM_0)
            sn1 = dsigB
            sn2 = dsigB
            sn4 = FNUM_0
        end

        dsig1 = sn1-mp.σij[ix, 1]
        dsig2 = sn2-mp.σij[ix, 2]
        dsig3 = sn3-mp.σij[ix, 3]
        dsig4 = sn4-mp.σij[ix, 4]
        mp.σij[ix, 1] = sn1
        mp.σij[ix, 2] = sn2
        mp.σij[ix, 3] = sn3
        mp.σij[ix, 4] = sn4

        Dt    = Ks+FNUM_43*G
        Dd    = Ks-FNUM_23*G
        base  = FNUM_1/((Dd-Dt)*(FNUM_2*Dd+Dt))
        ep_xx = -( Dd*dsig1+Dt*dsig1-Dd*dsig2-Dd*dsig3)*base
        ep_yy = -(-Dd*dsig1+Dd*dsig2+Dt*dsig2-Dd*dsig3)*base
        ep_zz = -(-Dd*dsig1-Dd*dsig2+Dd*dsig3+Dt*dsig3)*base
        ep_xy = dsig4/G
        mp.epK[ix]   = ep_xx+ep_yy+ep_zz
        mp.epII[ix] += sqrt(FNUM_23*(ep_xx^INUM_2+       ep_yy^INUM_2+
                                    ep_zz^INUM_2+FNUM_2*ep_xy^INUM_2)) 

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
    mohrcoulomb!(mp::Particle3D{T1, T2}, ΔT::T2)

Description:
---
Update strain and stress according to Mohr-Coulomb constitutive model (3D).
"""
@views function mohrcoulomb!(mp::Particle3D{T1, T2}) where {T1, T2}
    FNUM_0  = T2(0.0)
    FNUM_q  = T2(0.25)
    FNUM_h  = T2(0.5)
    FNUM_2  = T2(2.0)
    FNUM_13 = T2(1.0/3.0)
    FNUM_43 = T2(4.0/3.0)
    FNUM_23 = T2(2.0/3.0)
    INUM_2  = T1(2)
    for ix in 1:mp.num
        # to do
    end
    return nothing
end