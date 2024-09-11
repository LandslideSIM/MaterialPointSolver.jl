@kernel inbounds=true function testG2P_OS!(
    grid::    KernelGrid2D{T1, T2},
    mp  ::KernelParticle2D{T1, T2},
    ΔT  ::T2
) where {T1, T2}
    ix = @index(Global)
    if ix <= mp.num
        ΔT_1 = T2(1.0) / ΔT
        dF1 = dF2 = dF3 = dF4 = T2(0.0)
        for iy in Int32(1):Int32(mp.NIC)
            if mp.Ni[ix, iy] != T2(0.0)
                p2n = mp.p2n[ix, iy]
                ∂Nx = mp.∂Nx[ix, iy]
                ∂Ny = mp.∂Ny[ix, iy]
                # compute solid incremental deformation gradient
                dF1 += grid.Δd_s[p2n, 1] * ∂Nx
                dF2 += grid.Δd_s[p2n, 1] * ∂Ny
                dF3 += grid.Δd_s[p2n, 2] * ∂Nx
                dF4 += grid.Δd_s[p2n, 2] * ∂Ny
            end
        end
        mp.ΔFs[ix, 1] = dF1
        mp.ΔFs[ix, 2] = dF2
        mp.ΔFs[ix, 3] = dF3
        mp.ΔFs[ix, 4] = dF4
        # strain rate (Second Invariant of Strain Rate Tensor)
        dϵxx = dF1 * ΔT_1
        dϵyy = dF4 * ΔT_1
        dϵxy = T2(0.5) * (dF2 + dF3) * ΔT_1
        mp.dϵ[ix] = sqrt(dϵxx * dϵxx + dϵyy * dϵyy + T2(2.0) * dϵxy * dϵxy)
        # compute strain increment 
        mp.Δϵij_s[ix, 1] = dF1
        mp.Δϵij_s[ix, 2] = dF4
        mp.Δϵij_s[ix, 4] = dF2 + dF3
        # update strain tensor
        mp.ϵij_s[ix, 1] += dF1
        mp.ϵij_s[ix, 2] += dF4
        mp.ϵij_s[ix, 4] += dF2 + dF3
        # deformation gradient matrix
        F1 = mp.F[ix, 1]; F2 = mp.F[ix, 2]; F3 = mp.F[ix, 3]; F4 = mp.F[ix, 4]      
        mp.F[ix, 1] = (dF1 + T2(1.0)) * F1 + dF2 * F3
        mp.F[ix, 2] = (dF1 + T2(1.0)) * F2 + dF2 * F4
        mp.F[ix, 3] = (dF4 + T2(1.0)) * F3 + dF3 * F1
        mp.F[ix, 4] = (dF4 + T2(1.0)) * F4 + dF3 * F2
        # update jacobian value and particle volume
        mp.J[ix] = mp.F[ix, 1] * mp.F[ix, 4] - mp.F[ix, 2] * mp.F[ix, 3]
        mp.vol[ix] = mp.J[ix] * mp.vol_init[ix]
    end
end

function testprocedure!(args    ::MODELARGS, 
                        grid    ::GRID, 
                        mp      ::PARTICLE,
                        pts_attr::PROPERTY, 
                        bc      ::BOUNDARY,
                        ΔT      ::T2,
                        Ti      ::T2,
                                ::Val{:OS},
                                ::Val{:USF}) where {T2}
    Ti<args.Te ? G=args.gravity/args.Te*Ti : G=args.gravity
    dev = getBackend(Val(args.device))
    testG2P_OS!(dev)(ndrange=mp.num, grid, mp, ΔT)
    if args.constitutive == :hyperelastic
        hyE!(dev)(ndrange=mp.num, mp, pts_attr)
    elseif args.constitutive == :linearelastic
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
    elseif args.constitutive == :druckerprager
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
        if Ti ≥ args.Te
            dpP!(dev)(ndrange=mp.num, mp, pts_attr)
        end
    elseif args.constitutive == :mohrcoulomb
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
        if Ti ≥ args.Te
            mcP!(dev)(ndrange=mp.num, mp, pts_attr)
        end
    end
    resetgridstatus_OS!(dev)(ndrange=grid.node_num, grid)
    args.device == :CPU && args.basis == :uGIMP ? 
        resetmpstatus_OS_CPU!(dev)(ndrange=mp.num, grid, mp, Val(args.basis)) :
        resetmpstatus_OS!(dev)(ndrange=mp.num, grid, mp, Val(args.basis))
    P2G_OS!(dev)(ndrange=mp.num, grid, mp, G)
    solvegrid_USL_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT, args.ζs)
    doublemapping1_OS!(dev)(ndrange=mp.num, grid, mp, pts_attr, ΔT, args.FLIP, args.PIC)
    if args.MVL == true
        vollock1_OS!(dev)(ndrange=mp.num, grid, mp)
        vollock2_OS!(dev)(ndrange=mp.num, grid, mp)
    end                                  
    return nothing
end

function testprocedure!(args    ::MODELARGS, 
                        grid    ::GRID, 
                        mp      ::PARTICLE, 
                        pts_attr::PROPERTY,
                        bc      ::BOUNDARY,
                        ΔT      ::T2,
                        Ti      ::T2,
                                ::Val{:OS},
                                ::Val{:USL}) where {T2}
    Ti<args.Te ? G=args.gravity/args.Te*Ti : G=args.gravity
    dev = getBackend(Val(args.device))
    resetgridstatus_OS!(dev)(ndrange=grid.node_num, grid)
    args.device == :CPU && args.basis == :uGIMP ? 
        resetmpstatus_OS_CPU!(dev)(ndrange=mp.num, grid, mp, Val(args.basis)) :
        resetmpstatus_OS!(dev)(ndrange=mp.num, grid, mp, Val(args.basis))
    P2G_OS!(dev)(ndrange=mp.num, grid, mp, G)
    solvegrid_USL_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT, args.ζs)
    doublemapping1_OS!(dev)(ndrange=mp.num, grid, mp, pts_attr, ΔT, args.FLIP, args.PIC)
    testG2P_OS!(dev)(ndrange=mp.num, grid, mp, ΔT)
    if args.constitutive == :hyperelastic
        hyE!(dev)(ndrange=mp.num, mp, pts_attr)
    elseif args.constitutive == :linearelastic
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
    elseif args.constitutive == :druckerprager
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
        if Ti ≥ args.Te
            dpP!(dev)(ndrange=mp.num, mp, pts_attr)
        end
    elseif args.constitutive==:mohrcoulomb
        liE!(dev)(ndrange=mp.num, mp, pts_attr)
        if Ti ≥ args.Te
            mcP!(dev)(ndrange=mp.num, mp, pts_attr)
        end
    end
    if args.MVL == true
        vollock1_OS!(dev)(ndrange=mp.num, grid, mp)
        vollock2_OS!(dev)(ndrange=mp.num, grid, mp)
    end                                  
    return nothing
end