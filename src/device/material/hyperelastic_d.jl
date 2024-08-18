#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : Hyperelastic_d.jl                                                          |
|  Description: Hyper elastic material's constitutive model. (Neo-Hookean)                 |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. hyE! [2D GPU kernel]                                                    |
|               2. hyE! [3D GPU kernel]                                                    |
+==========================================================================================#

"""
    hyE!(cu_mp::KernelParticle2D{T1, T2})

Description:
---
GPU kernel to implement hyper elastic constitutive model (Neo-Hookean).
"""
function hyE!(cu_mp::KernelParticle2D{T1, T2}) where {T1, T2}
    FNUM_h  = T2(0.5)
    FNUM_1  = T2(1.0)
    FNUM_23 = T2(2/3)
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    if ix≤cu_mp.num
        pid = cu_mp.layer[ix]
        J   = cu_mp.J[ix]
        μ   = cu_mp.G[pid]
        λ   = cu_mp.Ks[pid]-FNUM_23*cu_mp.G[pid]
        # compute F*Fᵀ
        F1 = cu_mp.F[ix, 1]; F2 = cu_mp.F[ix, 2]
        F3 = cu_mp.F[ix, 3]; F4 = cu_mp.F[ix, 4]
        S1 = F1*F1+F2*F2
        S2 = F1*F3+F2*F4
        S4 = F3*F3+F4*F4
        # update stress
        tmp1 =  μ/J
        tmp2 = (λ/J)*log(J)
        cu_mp.σij[ix, 1] = tmp1*(S1-FNUM_1)+tmp2
        cu_mp.σij[ix, 2] = tmp1*(S4-FNUM_1)+tmp2
        cu_mp.σij[ix, 4] = tmp1* S2
        # update mean stress and deviatoric stress
        cu_mp.σm[ix] = (cu_mp.σij[ix, 1]+cu_mp.σij[ix, 2])*FNUM_h
        cu_mp.sij[ix, 1] = cu_mp.σij[ix, 1]-cu_mp.σm[ix]
        cu_mp.sij[ix, 2] = cu_mp.σij[ix, 2]-cu_mp.σm[ix]
        cu_mp.sij[ix, 4] = cu_mp.σij[ix, 4]
    end
    return nothing
end

"""
    hyE!(cu_mp::KernelParticle3D{T1, T2})

Description:
---
GPU kernel to implement hyper elastic constitutive model (Neo-Hookean).
"""
function hyE!(cu_mp::KernelParticle3D{T1, T2}) where {T1, T2}
    FNUM_13 = T2(1/3)
    FNUM_23 = T2(2/3)
    FNUM_1  = T2(1.0)
    ix = (blockIdx().x-1i32)*blockDim().x+threadIdx().x
    if ix≤cu_mp.num
        pid = cu_mp.layer[ix]
        J   = cu_mp.J[ix]
        μ   = cu_mp.G[pid]
        λ   = cu_mp.Ks[pid]-FNUM_23*cu_mp.G[pid]
        # compute F*Fᵀ
        F1 = cu_mp.F[ix, 1]; F2 = cu_mp.F[ix, 2]; F3 = cu_mp.F[ix, 3]
        F4 = cu_mp.F[ix, 4]; F5 = cu_mp.F[ix, 5]; F6 = cu_mp.F[ix, 6]
        F7 = cu_mp.F[ix, 7]; F8 = cu_mp.F[ix, 8]; F9 = cu_mp.F[ix, 9]
        S1 = F1*F1+F2*F2+F3*F3
        S2 = F1*F4+F2*F5+F3*F6
        S3 = F1*F7+F2*F8+F3*F9
        S5 = F4*F4+F5*F5+F6*F6
        S6 = F4*F7+F5*F8+F6*F9
        S9 = F7*F7+F8*F8+F9*F9
        # update stress
        tmp1 =  μ/J
        tmp2 = (λ/J)*log(J)
        cu_mp.σij[ix, 1] = tmp1*(S1-FNUM_1)+tmp2
        cu_mp.σij[ix, 2] = tmp1*(S5-FNUM_1)+tmp2
        cu_mp.σij[ix, 3] = tmp1*(S9-FNUM_1)+tmp2
        cu_mp.σij[ix, 4] = tmp1* S2
        cu_mp.σij[ix, 5] = tmp1* S6
        cu_mp.σij[ix, 6] = tmp1* S3
        # update mean stress and deviatoric stress
        cu_mp.σm[ix] = (cu_mp.σij[ix, 1]+cu_mp.σij[ix, 2]+cu_mp.σij[ix, 3])*FNUM_13
        cu_mp.sij[ix, 1] = cu_mp.σij[ix, 1]-cu_mp.σm[ix]
        cu_mp.sij[ix, 2] = cu_mp.σij[ix, 2]-cu_mp.σm[ix]
        cu_mp.sij[ix, 3] = cu_mp.σij[ix, 3]-cu_mp.σm[ix]
        cu_mp.sij[ix, 4] = cu_mp.σij[ix, 4]
        cu_mp.sij[ix, 5] = cu_mp.σij[ix, 5]
        cu_mp.sij[ix, 6] = cu_mp.σij[ix, 6]
    end
    return nothing
end