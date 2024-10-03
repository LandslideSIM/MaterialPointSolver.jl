#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : mohrcoulomb.jl                                                             |
|  Description: Mohr-Coulomb constitutive model.                                           |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. mcP! [2D]                                                               |
|               2. mcP! [3D]                                                               |
+==========================================================================================#

export mcP!

"""
    mcP!(mp::DeviceParticle2D{T1, T2}, attr::DeviceProperty{T1, T2})

Description:
---
Implement Mohr-Coulomb constitutive model (2D plane strain).
"""
@kernel inbounds=true function mcP!(
    mp  ::DeviceParticle2D{T1, T2},
    attr::  DeviceProperty{T1, T2}
) where {T1, T2} 
    ix = @index(Global)
    if ix ≤ mp.np
        nid = attr.nid[ix]
        c   = attr.c[nid]
        Hp  = attr.Hp[nid]
        cr  = attr.cr[nid]
        ϕ   = attr.ϕ[nid]
        Gs  = attr.Gs[nid]
        Ks  = attr.Ks[nid]
        # mohr-coulomb 
        c     = max(c + Hp * mp.ϵq[ix], cr)
        ds    = mp.σij[ix, 1] - mp.σij[ix, 2]
        tau   = sqrt(T2(0.25) * ds^T1(2) + mp.σij[ix, 4]^T1(2))
        sig   = T2(0.5) * (mp.σij[ix, 1] + mp.σij[ix, 2])
        f     = tau + sig * sin(ϕ) - c * cos(ϕ)
        sn1   = mp.σij[ix, 1]
        sn2   = mp.σij[ix, 2]
        sn3   = mp.σij[ix, 3]
        sn4   = mp.σij[ix, 4]
        beta  = abs(c * cos(ϕ) - sig * sin(ϕ)) / tau
        dsigA = T2(0.5) * beta * ds
        dsigB = c / tan(ϕ)
        if (sig ≤ dsigB) && (f > T2(0.0))
            sn1 = sig + dsigA
            sn2 = sig - dsigA
            sn4 = beta * mp.σij[ix, 4]
        end
        if (sig > dsigB) && (f > T2(0.0))
            sn1 = dsigB
            sn2 = dsigB
            sn4 = T2(0.0)
        end

        dsig1 = sn1 - mp.σij[ix, 1]
        dsig2 = sn2 - mp.σij[ix, 2]
        dsig3 = sn3 - mp.σij[ix, 3]
        dsig4 = sn4 - mp.σij[ix, 4]
        mp.σij[ix, 1] = sn1
        mp.σij[ix, 2] = sn2
        mp.σij[ix, 3] = sn3
        mp.σij[ix, 4] = sn4

        Dt        = Ks + T2(1.333333) * Gs
        Dd        = Ks - T2(0.666667) * Gs
        base      = T2(1.0) / ((Dd - Dt) * (T2(2.0) * Dd + Dt))
        ep_xx     = -(Dd * dsig1 + Dt * dsig1 - Dd * dsig2 - Dd * dsig3) * base
        ep_yy     = -(-Dd * dsig1 + Dd * dsig2 + Dt * dsig2 - Dd * dsig3) * base
        ep_zz     = -(-Dd * dsig1 - Dd * dsig2 + Dd * dsig3 + Dt * dsig3) * base
        ep_xy     = dsig4 / Gs
        mp.ϵk[ix] = ep_xx + ep_yy + ep_zz
        mp.ϵq[ix] += sqrt(T2(0.666667) * (ep_xx * ep_xx + ep_yy * ep_yy +
                                            ep_zz * ep_zz + ep_xy * ep_xy * T2(2.0)))
        # update mean stress tensor
        mp.σm[ix] = (mp.σij[ix, 1] + mp.σij[ix, 2] + mp.σij[ix, 3]) * T2(0.333333)
        # update deviatoric stress tensor
        mp.sij[ix, 1] = mp.σij[ix, 1] - mp.σm[ix]
        mp.sij[ix, 2] = mp.σij[ix, 2] - mp.σm[ix]
        mp.sij[ix, 3] = mp.σij[ix, 3] - mp.σm[ix]
        mp.sij[ix, 4] = mp.σij[ix, 4]
    end
end

"""
    mcP!(mp::DeviceParticle3D{T1, T2}, attr::DeviceProperty{T1, T2})

Description:
---
Implement Mohr-Coulomb constitutive model (3D).
"""


"""
    J₂ = 1/6 * [(σx - σy)² + (σy - σz)² + (σz - σx)²] + σxy² + σyz² + σzx²
---
    σm = (σx + σy + σz) / 3
    σ'x = σx - σm
    σ'y = σy - σm
    σ'z = σz - σm
    J₃ = σ'x * σ'y * σ'z + 2 * σxy * σyz * σzx - σ'x * σyz² - σ'y * σzx² - σ'z * σxy²
---
    Compute and return Lode angle (cos convention, between 0 and π/3)
"""
@inline Base.@propagate_inbounds function MC_paras(σij1::T2, σij2::T2, σij3::T2, σij4::T2, 
    σij5::T2, σij6::T2, σm::T2, Δ::T2) where T2
    J2 = T2(0.166667) * ((σij1 - σij2) * (σij1 - σij2)  + 
                         (σij2 - σij3) * (σij2 - σij3)  +
                         (σij3 - σij1) * (σij3 - σij1)) + 
                          σij4 * σij4  +  σij5 * σij5   + σij6 * σij6 |> T2
    J3 = (σij1 - σm) * (σij2 - σm) * (σij3 - σm     ) + 
         (σij4       *  σij5       *  σij6 * T2(2.0)) - 
         (σij1 - σm) *  σij5       *  σij5 - 
         (σij2 - σm) *  σij6       *  σij6 - 
         (σij3 - σm) *  σij4       *  σij4 |> T2
    lode_angle_val = abs(j2) > Δ ? T2(2.598076) * (J3 / J2 ^ T2(1.5)) : T2(0.0)
    if lode_angle_val > T2(1.0) 
        lode_angle_val = T2(1.0)
    elseif lode_angle_val < T2(-1.0) 
        lode_angle_val = T2(-1.0)
    end
    return T2(0.333333) * acos(lode_angle_val), J2, J3, sqrt(J2 + J2)
end

"""
    mcP!(mp::DeviceParticle3D{T1, T2}, attr::DeviceProperty{T1, T2})

Description:
---
Implement Mohr-Coulomb constitutive model (3D).

1. 获取等效塑性偏应变（pdstrain）：从状态变量中获取等效塑性偏应变。
2. 更新Mohr-Coulomb参数：如果启用了软化功能并且等效塑性偏应变超过了峰值，则根据线性软化规则更新
    Mohr-Coulomb模型的参数。
3. 弹性预测阶段：根据弹性预测法计算试验应力。
4. 计算试验应力的屈服函数值：基于试验应力计算屈服函数的值。
5. 如果处于弹性状态，则返回更新后的应力：如果试验应力处于弹性状态，则直接返回试验应力。
6. 塑性修正阶段：根据试验应力和应力输入计算塑性修正后的应力，并迭代校正应力，使其符合Mohr-Coulomb
    屈服条件。
7. 更新塑性偏应变（pdstrain）：根据塑性修正后的应力更新等效塑性偏应变。
8. 返回更新后的应力：返回迭代后的应力值。
"""
@kernel inbounds=true function mcP!(
    mp      ::      DeviceParticle3D{T1, T2},
    attr::DeviceProperty{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ mp.np
        #=
        nid = attr.nid[ix]
        coh = attr.c[nid]
        Hp  = attr.Hp[nid]
        cr  = attr.cr[nid]
        ϕ   = attr.ϕ[nid]
        ψ   = attr.ψ[nid]
        Gs  = attr.Gs[nid]
        Ks  = attr.Ks[nid]
        σt  = attr.σt[nid]
        yield_type = T1(0)
        softening_ = false
        # to do save the value: 
        pdstrain  = initial_value
        pdstrain_peak_ = 0
        pdstrain_residual_ = 0
        phi_peak_ = 0
        phi_residual_ = 0
        psi_peak_ = 0
        psi_residual_ = 0
        cohesion_peak_ = 0
        cohesion_residual_ = 0
        σm = mp.σm[ix]
        ϵq = mp.ϵq[ix]
        Δ = eps(T2)
        
        # update MC parameters using a linear softening rule
        if softening ≠ 0 && pdstrain > pdstrain_peak_
            if pdstrain < pdstrain_residual_ 
                ϕ = phi_residual_ +
                    ((phi_peak_ - phi_residual_) * (pdstrain - pdstrain_residual_) /
                     (pdstrain_peak_ - pdstrain_residual_))
                ψ = psi_residual_ + 
                    ((psi_peak_ - psi_residual_) * (pdstrain - pdstrain_residual_) /
                     (pdstrain_peak_ - pdstrain_residual_))
                c = cohesion_residual_ + 
                    ((cohesion_peak_ - cohesion_residual_) * (pdstrain - pdstrain_residual_) /
                     (pdstrain_peak_ - pdstrain_residual_))
            else
                ϕ = phi_residual_
                ψ = psi_residual_
                c = cohesion_residual_
            end
        end
        # compute stress invariants: J2, J3, and Lode angle θ, etc.
        θ, J2, J3, ρ = MC_paras(mp.σij[ix, 1], mp.σij[ix, 2], mp.σij[ix, 3], 
                                mp.σij[ix, 4], mp.σij[ix, 5], mp.σij[ix, 6], σm, Δ)
        ϵ = σm * T2(1.732051) # sqrt(3) * mean stress
        

        # compute yield function and yield_type
        ## tension
        ft = T2(0.816497) * cos(θ) * rho + epsilon / T2(1.732051) - σt 
        ## shear
        fs = T2(1.224745) * rho * ((sin(θ + T2(1.047198)) / (T2(1.732051) * cos(ϕ))) +
                                   (cos(θ + T2(1.047198)) *  T2(0.333333) * tan(ϕ))) +
            (epsilon * T2(0.57735)) * tan(ϕ) - coh
        if ft > T2(-0.1) && fs > T2(-0.1) # 0.1 is the tolerence
            ### compute tension and shear edge parameters
            n_phi = (T2(1.0) + sin(ϕ)) / (T2(1.0) - sin(ϕ))
            sigma_p = σt * n_phi - T2(2.0) * coh * sqrt(n_phi)
            alpha_p = sqrt(T2(1.0) + n_phi * n_phi) + n_phi
            ### compute the shear-tension edge
            h = ft + alpha_p * (T2(0.816497) * cos(θ - T2(4.18879)) * rho + 
                epsilon * T2(0.57735) - sigma_p)
            ### tension
            yield_type = h > eps(T2) ? T1(1) : T1(2)
        end
        yield_type = (ft < T2(-0.1)) && (fs > T2(-0.1)) ? T1(2) : nothing
        yield_type = (ft > T2(-0.1)) && (fs < T2(-0.1)) ? T1(1) : nothing

        # compute dF/dSigma and dP/dSigma
        ## compute dF / dEpsilon,  dF / dRho, dF / dTheta
        if yield_type == T1(1)
            df_depsilon = T2(0.577350)
            df_drho = T2(0.816497) * cos(θ)
            df_dtheta = -T2(0.816497) * rho * sin(θ)
        elseif yield_type == T1(2)
            df_depsilon = tan(ϕ) * T2(0.57735)
            df_drho = T2(1.224745) * ((sin(θ + T2(1.047198)) / (1.732051 * cos(ϕ))) +
                                      (cos(θ + T2(1.047198)) * tan(ϕ) * T2(0.333333)))
            df_dtheta = T2(1.224745) * rho * 
                ((cos(θ + T2(1.047198)) / (T2(1.732051) * cos(ϕ))) -
                 (sin(θ + T2(1.047198)) * tan(ϕ) * T2(0.333333)))
        end


        ## compute dEpsilon / dSigma
        dp_dsigma_0 = T2(0.333333)
        dp_dsigma_1 = T2(0.333333)
        dp_dsigma_2 = T2(0.333333)
        dp_dsigma_3 = T2(0.0)
        dp_dsigma_4 = T2(0.0)
        dp_dsigma_5 = T2(0.0)
        depsilon_dsigma_0 = dp_dsigma_0 * T2(1.732051)
        depsilon_dsigma_1 = dp_dsigma_1 * T2(1.732051)
        depsilon_dsigma_2 = dp_dsigma_2 * T2(1.732051)
        depsilon_dsigma_3 = T2(0.0)
        depsilon_dsigma_4 = T2(0.0)
        depsilon_dsigma_5 = T2(0.0)
        ## initialise dRho / dSigma
        q = sqrt(T2(3.0) * J2)
        deviatoric_stress0
        if abs(q) > eps(T2)
            dq_dsigma_0 = T2(3.0) / (q + q) * (mp.σij[ix, 1] - σm)
            dq_dsigma_1 = T2(3.0) / (q + q) * (mp.σij[ix, 2] - σm)
            dq_dsigma_2 = T2(3.0) / (q + q) * (mp.σij[ix, 3] - σm)
            dq_dsigma_3 = T2(3.0) /  q      *  mp.σij[ix, 4]
            dq_dsigma_4 = T2(3.0) /  q      *  mp.σij[ix, 5]
            dq_dsigma_5 = T2(3.0) /  q      *  mp.σij[ix, 6]
        end
        drho_dsigma_0 = dq_dsigma_0 * T2(0.816497)
        drho_dsigma_1 = dq_dsigma_1 * T2(0.816497)
        drho_dsigma_2 = dq_dsigma_2 * T2(0.816497)
        drho_dsigma_3 = dq_dsigma_3 * T2(0.816497)
        drho_dsigma_4 = dq_dsigma_4 * T2(0.816497)
        drho_dsigma_5 = dq_dsigma_5 * T2(0.816497)
        ## compute dtheta / dsigma
        dj2_dsigma_0 = (mp.σij[ix, 1] - σm)
        dj2_dsigma_1 = (mp.σij[ix, 2] - σm)
        dj2_dsigma_2 = (mp.σij[ix, 3] - σm)
        dj2_dsigma_3 =  mp.σij[ix, 4] + mp.σij[ix, 4]
        dj2_dsigma_4 =  mp.σij[ix, 5] + mp.σij[ix, 5]
        dj2_dsigma_5 =  mp.σij[ix, 6] + mp.σij[ix, 6]
        dev1_0 = (mp.σij[ix, 1] - σm)
        dev1_1 =  mp.σij[ix, 4]
        dev1_2 =  mp.σij[ix, 6]
        dev2_0 =  mp.σij[ix, 4]
        dev2_1 = (mp.σij[ix, 2] - σm)
        dev2_2 =  mp.σij[ix, 5]
        dev3_0 =  mp.σij[ix, 6]
        dev3_1 =  mp.σij[ix, 5]
        dev3_2 = (mp.σij[ix, 3] - σm)
        dj3_dsigma_0 = (dev1_0 * dev1_0 + dev1_1 * dev1_1 + dev1_2 * dev1_2) - 
                        T2(0.666667) * j2
        dj3_dsigma_1 = (dev2_0 * dev2_0 + dev2_1 * dev2_1 + dev2_2 * dev2_2) - 
                        T2(0.666667) * j2
        dj3_dsigma_2 = (dev3_0 * dev3_0 + dev3_1 * dev3_1 + dev3_2 * dev3_2) - 
                        T2(0.666667) * j2
        dj3_dsigma_3 = (dev1_0 * dev2_0 + dev1_1 * dev2_1 + dev1_2 * dev2_2) * T2(2.0) 
        dj3_dsigma_4 = (dev2_0 * dev3_0 + dev2_1 * dev3_1 + dev2_2 * dev3_2) * T2(2.0)
        dj3_dsigma_5 = (dev1_0 * dev3_0 + dev1_1 * dev3_1 + dev1_2 * dev3_2) * T2(2.0)
        dr_dj2 = T2(-3.897114) * J3
        dr_dj3 = T2(2.598076)
        dtheta_dr = T2(-0.333333)
        if abs(J2) > eps(T2)
            # declare R defined as R = cos(3 theta)
            r = j3 * T2(0.5) * (J2 * T2(0.333333)) ^ T2(-1.5)
            # update derivatives of R
            dr_dj2 *= J2 ^ T2(-2.5)
            dr_dj3 *= J2 ^ T2(-1.5)
            # update derivative of theta in terms of R, check for sqrt of zero
            factor = (abs(T2(1.0) - r * r) < eps(T2)) ? eps(T2) : abs(T2(1.0) - r * r)
            dtheta_dr = T2(-1.0) / (T2(3.0) * sqrt(factor))
        end
        dtheta_dsigma_0 = (dtheta_dr * ((dr_dj2 * dj2_dsigma_0) + (dr_dj3 * dj3_dsigma_0)))
        dtheta_dsigma_1 = (dtheta_dr * ((dr_dj2 * dj2_dsigma_1) + (dr_dj3 * dj3_dsigma_1)))
        dtheta_dsigma_2 = (dtheta_dr * ((dr_dj2 * dj2_dsigma_2) + (dr_dj3 * dj3_dsigma_2)))
        dtheta_dsigma_3 = (dtheta_dr * ((dr_dj2 * dj2_dsigma_3) + (dr_dj3 * dj3_dsigma_3)))
        dtheta_dsigma_4 = (dtheta_dr * ((dr_dj2 * dj2_dsigma_4) + (dr_dj3 * dj3_dsigma_4)))
        dtheta_dsigma_5 = (dtheta_dr * ((dr_dj2 * dj2_dsigma_5) + (dr_dj3 * dj3_dsigma_5)))
        ## compute dF/dSigma
        df_dsigma_0 = df_depsilon * depsilon_dsigma_0 +
                      df_drho     *     drho_dsigma_0 + 
                      df_dtheta   *   dtheta_dsigma_0
        df_dsigma_1 = df_depsilon * depsilon_dsigma_1 +
                      df_drho     *     drho_dsigma_1 + 
                      df_dtheta   *   dtheta_dsigma_1
        df_dsigma_2 = df_depsilon * depsilon_dsigma_2 +
                      df_drho     *     drho_dsigma_2 + 
                      df_dtheta   *   dtheta_dsigma_2
        df_dsigma_3 = df_depsilon * depsilon_dsigma_3 +
                      df_drho     *     drho_dsigma_3 + 
                      df_dtheta   *   dtheta_dsigma_3
        df_dsigma_4 = df_depsilon * depsilon_dsigma_4 +
                      df_drho     *     drho_dsigma_4 + 
                      df_dtheta   *   dtheta_dsigma_4
        df_dsigma_5 = df_depsilon * depsilon_dsigma_5 +
                      df_drho     *     drho_dsigma_5 + 
                      df_dtheta   *   dtheta_dsigma_5
        if yield_type == T1(1) ## compute dp/dsigma and dp/dj in tension yield
            ## define deviatoric eccentricity
            et_value = T2(0.6)
            ## define meridional eccentricity
            xit = T2(0.1)
            ## compute Rt
            sqpart = T2(4.0) * (T2(1.0) - et_value * et_value) * cos(θ) * cos(θ) +
                     T2(5.0) * et_value * et_value - T2(4.0) * et_value
            sqpart = sqpart < eps(T2) ? T2(0.00001) : nothing
            rt_den = T2(2.0) * (T2(1.0) - et_value * et_value) * cos(θ) +
                    (T2(2.0) * et_value - T2(1)) * sqrt(sqpart)
            rt_num = T2(4.0) * (T2(1.0) - et_value * et_value) * cos(θ) * cos(θ) +
                    (T2(2.0) * et_value - T2(1.0)) * (T2(2.0) * et_value - T2(1.0))
            rt_den = rt_den < eps(T2) ? T2(0.00001) : nothing
            rt = rt_num / (T2(3.0) * rt_den)
            ## compute dP/dRt
            dp_drt = T2(1.5) * rho * rho * rt / 
                sqrt(xit * xit * σt * σt + T2(1.5) * rt * rt * rho * rho)
            ## compute dP/dRho
            dp_drho = T2(1.5) * rho * rt * rt /
                sqrt(xit * xit * σt * σt + T2(1.5) * rt * rt * rho * rho)
            ## compute dP/dEpsilon
            dp_depsilon = T2(0.57735)
            ## compute dRt/dThera
            drtden_dtheta =
                T2(-2.0) * (T2(1.0) - et_value * et_value) * sin(θ) -
                (T2(2.0) * et_value - T2(1)) * T2(4.0) * (T2(1.0) - et_value * et_value) * 
                cos(θ) * sin(θ) / sqrt(T2(4.0) * (T2(1.0) - et_value * et_value) * 
                                       cos(θ) * cos(θ) + T2(5.0) * et_value * et_value - 
                                       T2(4.0) * et_value)
            drtnum_dtheta = T2(-8.0) * (T2(1.0) - et_value * et_value) * cos(θ) * sin(θ)
            drt_dtheta = (drtnum_dtheta * rt_den - drtden_dtheta * rt_num) /
                         (T2(3.0) * rt_den * rt_den)
            ## compute dP/dSigma
            dp_dsigma = (dp_depsilon * depsilon_dsigma) + (dp_drho * drho_dsigma) +
                        (dp_drt * drt_dtheta * dtheta_dsigma)
            ## compute dP/dJ
            dp_dq = dp_drho * T2(0.816497)
        elseif yield_type == T1(2)  ## compute dp/dsigma and dp/dj in shear yield
            ## compute Rmc
            r_mc = (T2(3.0) - sin(ϕ)) / (T2(6.0) * cos(ϕ))
            ## Compute deviatoric eccentricity
            e_val = (T2(3.0) - sin(ϕ)) / (T2(3.0) + sin(ϕ))
            if e_val ≤ T2(0.5) 
                e_val = T2(0.5000000001)
            elseif e_val > T2(1.0)
                e_val = T2(1.0)
            end
            ## compute Rmw
            sqpart = (T2(4.0) * (T2(1.0) - e_val * e_val) * (cos(θ) * cos(θ))) +
                     (T2(5.0) * e_val * e_val) - (T2(4.0) * e_val)
            sqpart = sqpart < eps(T2) ? T2(0.00001) : nothing
            m = (T2(2.0) * (T2(1.0) - e_val * e_val) * cos(θ)) +
                ((T2(2.0) * e_val - T2(1.0)) * sqrt(sqpart))
            m = m < eps(T2) ? T2(0.00001) : nothing
            l = (T2(4.0) * (T2(1.0) - e_val * e_val) * (cos(θ) * cos(θ))) +
                (T2(2.0) * e_val - T2(1.0)) * (T2(2.0) * e_val - T2(1.0))
            r_mw = (l / m) * r_mc
            ## initialise meridional eccentricity
            xi = 0.1
            omega = ((xi * coh * tan(ψ)) * (xi * coh * tan(ψ))) +
                    ((r_mw * T2(1.224745) * rho) * (r_mw * T2(1.224745) * rho))
            omega = omega < eps(T2) ? T2(0.00001) : nothing
            dl_dtheta = T2(-8.0) * (T2(1.0) - e_val * e_val) * cos(θ) * sin(θ)
            dm_dtheta = (T2(-2.0) * (T2(1.0) - e_val * e_val) * sin(θ)) +
                        (T2(0.5) * (T2(2.0) * e_val - T2(1.0)) * dl_dtheta) / sqrt(sqpart)
            drmw_dtheta = ((m * dl_dtheta) - (l * dm_dtheta)) / (m * m)
            dp_depsilon = tan(ψ) / T2(1.732051)
            dp_drho = T2(3.0) * rho * r_mw * r_mw / (T2(2.0) * sqrt(omega))
            dp_dtheta = (T2(3.0) * rho * rho * r_mw * r_mc * drmw_dtheta) / 
                        (T2(2.0) * sqrt(omega))
            ## compute the value of dp/dsigma and dp/dj in shear yield
            dp_dsigma = (dp_depsilon * depsilon_dsigma) + (dp_drho * drho_dsigma) +
                        (dp_dtheta * dtheta_dsigma);
            dp_dq = dp_drho * T2(0.816497)
        end

        # compute softening part
        dphi_dpstrain = T2(0.0)
        dc_dpstrain = T2(0.0)
        softening = T2.(0.0)
        if softening_ && (pdstrain > pdstrain_peak_) && (pdstrain < pdstrain_residual_)
            ## compute dPhi/dPstrain
            dphi_dpstrain = (phi_residual_ - phi_peak_) / 
                            (pdstrain_residual_ - pdstrain_peak_)
            ## compute dc/dPstrain
            dc_dpstrain = (cohesion_residual_ - cohesion_peak_) /
                            (pdstrain_residual_ - pdstrain_peak_)
            ## compute dF/dPstrain
            df_dphi = T2(1.224745) * rho * ((sin(ϕ) * sin(θ + T2(1.047197)) /
                        (T2(1.732051) * cos(ϕ) * cos(ϕ))) + (cos(θ + T2(1.047197)) / 
                        (T2(3.0) * cos(ϕ) * cos(ϕ)))) + (σm / (cos(ϕ) * cos(ϕ)))
            df_dc = T2(-1.0)
            softening = T2(-1.0) * ((df_dphi * dphi_dpstrain) + 
                        (df_dc * dc_dpstrain)) * dp_dq
        end

















        # update mean stress tensor
        σm = (mp.σij[ix, 1]+mp.σij[ix, 2]+mp.σij[ix, 3])*T2(0.333333)
        mp.σm[ix] = σm
        # update deviatoric stress tensor
        mp.sij[ix, 1] = mp.σij[ix, 1]-σm
        mp.sij[ix, 2] = mp.σij[ix, 2]-σm
        mp.sij[ix, 3] = mp.σij[ix, 3]-σm
        mp.sij[ix, 4] = mp.σij[ix, 4]
        mp.sij[ix, 5] = mp.σij[ix, 5]
        mp.sij[ix, 6] = mp.σij[ix, 6]
        =#
    end
end
