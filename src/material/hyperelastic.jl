#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : hyperelastic.jl                                                            |
|  Description: Neo-hookean hyperelastic constitutive model                                |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : hyperelastic!()                                                            |
+==========================================================================================#

@views function hyperelastic!(mp::Particle2D{T1, T2}) where {T1, T2}
    FNUM_h  = T2(0.5)
    FNUM_1  = T2(1.0)
    FNUM_23 = T2(2/3)
    for ix in 1:mp.num
        pid = mp.layer[ix]
        J   = mp.J[ix]
        μ   = mp.G[pid]
        λ   = mp.Ks[pid]-FNUM_23*mp.G[pid]
        # compute F*Fᵀ
        F1 = mp.F[ix, 1]; F2 = mp.F[ix, 2]; F3 = mp.F[ix, 3]; F4 = mp.F[ix, 4]
        S1 = F1*F1+F2*F2
        S2 = F1*F3+F2*F4
        S4 = F3*F3+F4*F4
        # update stress
        tmp1 =  μ/J
        tmp2 = (λ/J)*log(J)
        mp.σij[ix, 1] = tmp1*(S1-FNUM_1)+tmp2
        mp.σij[ix, 2] = tmp1*(S4-FNUM_1)+tmp2
        mp.σij[ix, 4] = tmp1* S2
        # update mean stress and deviatoric stress
        mp.σm[ix] = (mp.σij[ix, 1]+mp.σij[ix, 2])*FNUM_h
        mp.sij[ix, 1] = mp.σij[ix, 1]-mp.σm[ix]
        mp.sij[ix, 2] = mp.σij[ix, 2]-mp.σm[ix]
        mp.sij[ix, 4] = mp.σij[ix, 4]
    end
    return nothing
end

@views function hyperelastic!(mp::Particle3D{T1, T2}) where {T1, T2}
    FNUM_13 = T2(1/3)
    FNUM_23 = T2(2/3)
    FNUM_1  = T2(1.0)
    for ix in 1:mp.num
        pid = mp.layer[ix]
        J   = mp.J[ix]
        μ   = mp.G[pid]
        λ   = mp.Ks[pid]-FNUM_23*mp.G[pid]
        # compute F*Fᵀ
        F1 = mp.F[ix, 1]; F2 = mp.F[ix, 2]; F3 = mp.F[ix, 3]
        F4 = mp.F[ix, 4]; F5 = mp.F[ix, 5]; F6 = mp.F[ix, 6]
        F7 = mp.F[ix, 7]; F8 = mp.F[ix, 8]; F9 = mp.F[ix, 9]
        S1 = F1*F1+F2*F2+F3*F3
        S2 = F1*F4+F2*F5+F3*F6
        S3 = F1*F7+F2*F8+F3*F9
        S5 = F4*F4+F5*F5+F6*F6
        S6 = F4*F7+F5*F8+F6*F9
        S9 = F7*F7+F8*F8+F9*F9
        # update stress
        tmp1 =  μ/J
        tmp2 = (λ/J)*log(J)
        mp.σij[ix, 1] = tmp1*(S1-FNUM_1)+tmp2
        mp.σij[ix, 2] = tmp1*(S5-FNUM_1)+tmp2
        mp.σij[ix, 3] = tmp1*(S9-FNUM_1)+tmp2
        mp.σij[ix, 4] = tmp1* S2
        mp.σij[ix, 5] = tmp1* S6
        mp.σij[ix, 6] = tmp1* S3
        # update mean stress and deviatoric stress
        mp.σm[ix] = (mp.σij[ix, 1]+mp.σij[ix, 2]+mp.σij[ix, 3])*FNUM_13
        mp.sij[ix, 1] = mp.σij[ix, 1]-mp.σm[ix]
        mp.sij[ix, 2] = mp.σij[ix, 2]-mp.σm[ix]
        mp.sij[ix, 3] = mp.σij[ix, 3]-mp.σm[ix]
        mp.sij[ix, 4] = mp.σij[ix, 4]
        mp.sij[ix, 5] = mp.σij[ix, 5]
        mp.sij[ix, 6] = mp.σij[ix, 6]
    end
    return nothing
end