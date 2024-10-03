#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : mpbasisfunc.jl                                                             |
|  Description: Basis functions for MPM                                                    |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. linearBasis                                                             |
|               2. uGIMPBasis                                                              |
+==========================================================================================#

export linearBasis, uGIMPbasis
export uGIMPbasisx, uGIMPbasisy, uGIMPbasisz

"""
    linearBasis(Δx::T2, h::T2)

Description:
---
Standard `linear basis function` for MPM, where `Δx` is the distance between particle and 
    node, and `h` is the grid spacing.
"""
@inline Base.@propagate_inbounds function linearBasis(Δx::T2, h::T2) where T2
    # c1  = abs(Δx)≤h
    # N1  = T2(1.0)-abs(Δx)/h
    # dN1 = -sign(Δx)/h
    # Ni  = c1*N1
    # dN  = c1*dN1
    # if abs(Δx) ≤ h
    #     Ni = T2(1.0) - abs(Δx) / h
    #     dN = -sign(Δx) / h
    # else
    #     Ni = T2(0.0)
    #     dN = T2(0.0)
    # end
    Ni = T2(1.0) - abs(Δx) / h
    dN = -sign(Δx) / h
    return T2(Ni), T2(dN)
end

"""
    uGIMPbasis(Δx::T2, h::T2, lp::T2)

Description:
---
`Uniform Generalized Interpolation` MPM basis function for MPM, where `Δx` is the distance 
between particle and node, `h` is the grid spacing, `lp` is the particle spacing.
"""
# This version will take a longer time since too many registers are used.
@inline Base.@propagate_inbounds function uGIMPbasis(Δx::T2, h::T2, lp::T2) where T2
    T1 = T2 == Float32 ? Int32 : Int64
    # absΔx = abs(Δx); signΔx = sign(Δx)
    # c1  = absΔx<(T2(0.5)*lp)
    # c2  = (T2(0.5)*lp)≤absΔx<(h-T2(0.5)*lp)
    # c3  = (h-T2(0.5)*lp)≤absΔx<(h+T2(0.5)*lp)
    # Ni1 = T2(1.0)-((T2(4)*(Δx^T1(2))+lp^T1(2))/(T2(4)*h*lp))
    # Ni2 = T2(1.0)-(absΔx/h)
    # Ni3 = ((h+T2(0.5)*lp-absΔx)^T1(2))/(T2(2.0)*h*lp)
    # dN1 = -((T2(8.0)*Δx)/(T2(4)*h*lp))
    # dN2 = signΔx*(-T2(1.0)/h)
    # dN3 = -signΔx*((h+T2(0.5)*lp-absΔx)/(h*lp))
    # N   = c1*Ni1+c2*Ni2+c3*Ni3
    # dN  = c1*dN1+c2*dN2+c3*dN3
    if abs(Δx) < T2(0.5)*lp
        Ni = T2(1.0) - ((T2(4.0) * Δx * Δx + lp * lp) / (T2(4.0) * h * lp))
        dN = -((T2(8.0) * Δx) / (T2(4.0) * h * lp))
    elseif (T2(0.5) * lp) ≤ abs(Δx) < (h - T2(0.5) * lp)
        Ni = T2(1.0) - (abs(Δx) / h)
        dN = sign(Δx) * (T2(-1.0) / h)
    elseif (h - T2(0.5) * lp) ≤ abs(Δx) < (h + T2(0.5) * lp)
        Ni = ((h + T2(0.5) * lp - abs(Δx)) ^ T1(2)) / (T2(2.0) * h * lp)
        dN = -sign(Δx) * ((h + T2(0.5) * lp - abs(Δx)) / (h * lp))
    else
        Ni = T2(0.0)
        dN = T2(0.0)
    end
    return T2(Ni), T2(dN)
end


@inline Base.@propagate_inbounds function uGIMPbasisx(Δx::T2, smem) where T2
    # smem[1] = h
    # smem[2] = lp
    # smem[3] = T2(1.0) / (T2(4.0) * h * lp)
    # smem[4] = T2(1.0) / (h * lp)
    # smem[5] = T2(1.0) / h
    # smem[6] = T2(0.5) * lp
    absΔx = abs(Δx)
    if absΔx < smem[6]
        Ni = T2(1.0) - ((T2(4.0) * Δx * Δx + smem[2] * smem[2]) * smem[3])
        dN = -T2(8.0) * Δx * smem[3]
    elseif smem[6] ≤ absΔx < smem[1] - smem[6]
        Ni = T2(1.0) - (absΔx / smem[1])
        dN = sign(Δx) * -smem[5]
    elseif smem[1] - smem[6] ≤ absΔx < smem[1] + smem[6]
        Ni = (smem[1] + smem[6] - absΔx) * (smem[1] + smem[6] - absΔx) * T2(0.5) * smem[4]
        dN = -sign(Δx) * (smem[1] + smem[6] - absΔx) * smem[4]
    else
        Ni = T2(0.0)
        dN = T2(0.0)
    end
    return T2(Ni), T2(dN)
end

@inline Base.@propagate_inbounds function uGIMPbasisy(Δx::T2, smem) where T2
    # smem[7]  = h
    # smem[8]  = lp
    # smem[9]  = T2(1.0) / (T2(4.0) * h * lp)
    # smem[10] = T2(1.0) / (h * lp)
    # smem[11] = T2(1.0) / h
    # smem[12] = T2(0.5) * lp
    absΔx = abs(Δx)
    if absΔx < smem[12]
        Ni = T2(1.0) - ((T2(4.0) * Δx * Δx + smem[8] * smem[8]) * smem[9])
        dN = -T2(8.0) * Δx * smem[9]
    elseif smem[12] ≤ absΔx < smem[7] - smem[12]
        Ni = T2(1.0) - (absΔx / smem[7])
        dN = sign(Δx) * -smem[11]
    elseif smem[7] - smem[12] ≤ absΔx < smem[7] + smem[12]
        Ni = (smem[7] + smem[12] - absΔx) * (smem[7] + smem[12] - absΔx) * T2(0.5) * smem[10]
        dN = -sign(Δx) * (smem[7] + smem[12] - absΔx) * smem[10]
    else
        Ni = T2(0.0)
        dN = T2(0.0)
    end
    return T2(Ni), T2(dN)
end

@inline Base.@propagate_inbounds function uGIMPbasisz(Δx::T2, smem) where T2
    # smem[13] = h
    # smem[14] = lp
    # smem[15] = T2(1.0) / (T2(4.0) * h * lp)
    # smem[16] = T2(1.0) / (h * lp)
    # smem[17] = T2(1.0) / h
    # smem[18] = T2(0.5) * lp
    absΔx = abs(Δx)
    if absΔx < smem[18]
        Ni = T2(1.0) - ((T2(4.0) * Δx * Δx + smem[14] * smem[14]) * smem[15])
        dN = -T2(8.0) * Δx * smem[15]
    elseif smem[18] ≤ absΔx < smem[13] - smem[18]
        Ni = T2(1.0) - (absΔx / smem[13])
        dN = sign(Δx) * -smem[17]
    elseif smem[13] - smem[18] ≤ absΔx < smem[13] + smem[18]
        Ni = (smem[13] + smem[18] - absΔx) * (smem[13] + smem[18] - absΔx) * T2(0.5) * smem[16]
        dN = -sign(Δx) * (smem[13] + smem[18] - absΔx) * smem[16]
    else
        Ni = T2(0.0)
        dN = T2(0.0)
    end
    return T2(Ni), T2(dN)
end