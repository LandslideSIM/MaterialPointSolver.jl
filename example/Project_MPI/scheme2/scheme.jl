@kernel inbounds = true function testresetgridstatus_OS!(
    grid::KernelGrid3D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix <= grid.node_num
        if ix <= grid.cell_num
            grid.σm[ix] = T2(0.0)
            grid.vol[ix] = T2(0.0)
        end
        grid.Ms[ix] = T2(0.0)
        grid.Ps[ix, 1] = T2(0.0)
        grid.Ps[ix, 2] = T2(0.0)
        grid.Ps[ix, 3] = T2(0.0)
        grid.Fs[ix, 1] = T2(0.0)
        grid.Fs[ix, 2] = T2(0.0)
        grid.Fs[ix, 3] = T2(0.0)
    end
end

@kernel inbounds = true function testresetmpstatus_OS!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2},
        ::Val{:linear}
) where {T1, T2}
    ix = @index(Global)
    if ix <= mp.num
        # update momentum and mass
        mp.Ms[ix] = mp.vol[ix] * mp.ρs[ix]
        mp.Ps[ix, 1] = mp.Ms[ix] * mp.Vs[ix, 1]
        mp.Ps[ix, 2] = mp.Ms[ix] * mp.Vs[ix, 2]
        mp.Ps[ix, 3] = mp.Ms[ix] * mp.Vs[ix, 3]
        # compute particle to cell and particle to node index
        mp.p2c[ix] = cld(mp.pos[ix, 2] - grid.range_y1, grid.space_y) +
                     fld(mp.pos[ix, 3] - grid.range_z1, grid.space_z) * 
                        grid.cell_num_y * grid.cell_num_x +
                     fld(mp.pos[ix, 1] - grid.range_x1, grid.space_x) * 
                        grid.cell_num_y |> T1
        for iy in Int32(1):Int32(mp.NIC)
            p2n = getP2N_linear(grid, mp.p2c[ix], iy)
            mp.p2n[ix, iy] = p2n
            # compute distance between particle and related nodes
            Δdx = mp.pos[ix, 1] - grid.pos[mp.p2n[ix, iy], 1]
            Δdy = mp.pos[ix, 2] - grid.pos[mp.p2n[ix, iy], 2]
            Δdz = mp.pos[ix, 3] - grid.pos[mp.p2n[ix, iy], 3]
            # compute basis function
            Nx, dNx = linearBasis(Δdx, grid.space_x)
            Ny, dNy = linearBasis(Δdy, grid.space_y)
            Nz, dNz = linearBasis(Δdz, grid.space_z)
            mp.Ni[ix, iy] = Nx * Ny * Nz
            mp.∂Nx[ix, iy] = dNx * Ny * Nz # x-gradient shape function
            mp.∂Ny[ix, iy] = dNy * Nx * Nz # y-gradient shape function
            mp.∂Nz[ix, iy] = dNz * Nx * Ny # z-gradient shape function
        end
    end
end

@kernel inbounds = true function testresetmpstatus_OS!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2},
        ::Val{:uGIMP}
) where {T1, T2}
    ix = @index(Global)
    if ix <= mp.num
        # update particle mass and momentum
        mp_Ms = mp.vol[ix] * mp.ρs[ix]
        mp.Ms[ix] = mp_Ms
        mp.Ps[ix, 1] = mp_Ms * mp.Vs[ix, 1]
        mp.Ps[ix, 2] = mp_Ms * mp.Vs[ix, 2]
        mp.Ps[ix, 3] = mp_Ms * mp.Vs[ix, 3]
        # get temp variables
        mp_pos_1 = mp.pos[ix, 1]
        mp_pos_2 = mp.pos[ix, 2]
        mp_pos_3 = mp.pos[ix, 3]
        # p2c index
        mp.p2c[ix] = cld(mp_pos_2 - grid.range_y1, grid.space_y) +
                     fld(mp_pos_3 - grid.range_z1, grid.space_z) * 
                        grid.cell_num_y * grid.cell_num_x +
                     fld(mp_pos_1 - grid.range_x1, grid.space_x) * grid.cell_num_y |> T1
        for iy in Int32(1):Int32(mp.NIC)
            # p2n index
            p2n = getP2N_uGIMP(grid, mp.p2c[ix], iy)
            mp.p2n[ix, iy] = p2n
            # compute distance between particle and related nodes
            Δdx = mp_pos_1 - grid.pos[p2n, 1]
            Δdy = mp_pos_2 - grid.pos[p2n, 2]
            Δdz = mp_pos_3 - grid.pos[p2n, 3]
            # compute basis function
            Nx, dNx = uGIMPbasis(Δdx, grid.space_x, mp.space_x)
            Ny, dNy = uGIMPbasis(Δdy, grid.space_y, mp.space_y)
            Nz, dNz = uGIMPbasis(Δdz, grid.space_z, mp.space_z)
            mp.Ni[ix, iy] = Nx * Ny * Nz
            mp.∂Nx[ix, iy] = dNx * Ny * Nz # x-gradient basis function
            mp.∂Ny[ix, iy] = dNy * Nx * Nz # y-gradient basis function
            mp.∂Nz[ix, iy] = dNz * Nx * Ny # z-gradient basis function
        end
    end
end

@kernel inbounds = true function testP2G_OS!(
    grid   ::    KernelGrid3D{T1, T2},
    mp     ::KernelParticle3D{T1, T2},
    gravity::T2
) where {T1, T2}
    ix = @index(Global)
    if ix <= mp.num
        for iy in Int32(1):Int32(mp.NIC)
            Ni = mp.Ni[ix, iy]
            if Ni != T2(0.0)
                ∂Nx = mp.∂Nx[ix, iy]
                ∂Ny = mp.∂Ny[ix, iy]
                ∂Nz = mp.∂Nz[ix, iy]
                p2n = mp.p2n[ix, iy]
                vol = mp.vol[ix]
                NiM = mp.Ms[ix] * Ni
                # compute nodal mass
                @KAatomic grid.Ms[p2n] += NiM
                # compute nodal momentum
                @KAatomic grid.Ps[p2n, 1] += Ni * mp.Ps[ix, 1]
                @KAatomic grid.Ps[p2n, 2] += Ni * mp.Ps[ix, 2]
                @KAatomic grid.Ps[p2n, 3] += Ni * mp.Ps[ix, 3]
                # compute nodal total force for solid
                @KAatomic grid.Fs[p2n, 1] += -vol * (∂Nx * mp.σij[ix, 1] + 
                                                     ∂Ny * mp.σij[ix, 4] + 
                                                     ∂Nz * mp.σij[ix, 6])
                @KAatomic grid.Fs[p2n, 2] += -vol * (∂Ny * mp.σij[ix, 2] + 
                                                     ∂Nx * mp.σij[ix, 4] + 
                                                     ∂Nz * mp.σij[ix, 5])
                @KAatomic grid.Fs[p2n, 3] += -vol * (∂Nz * mp.σij[ix, 3] + 
                                                     ∂Nx * mp.σij[ix, 6] + 
                                                     ∂Ny * mp.σij[ix, 5]) + NiM * gravity
            end
        end
    end
end

@kernel inbounds = true function testsolvegrid_OS!(
    grid::    KernelGrid3D{T1, T2},
    bc  ::KernelBoundary3D{T1, T2},
    ΔT  ::T2,
    ζs  ::T2
) where {T1, T2}
    ix = @index(Global)
    if ix <= grid.node_num && grid.Ms[ix] != Int32(0)
        Ms_denom = T2(1.0) / grid.Ms[ix]
        # compute nodal velocity
        Ps_1 = grid.Ps[ix, 1]
        Ps_2 = grid.Ps[ix, 2]
        Ps_3 = grid.Ps[ix, 3]
        grid.Vs[ix, 1] = Ps_1 * Ms_denom
        grid.Vs[ix, 2] = Ps_2 * Ms_denom
        grid.Vs[ix, 3] = Ps_3 * Ms_denom
        # damping force for solid
        dampvs = -ζs * sqrt(grid.Fs[ix, 1]^T1(2) +
                            grid.Fs[ix, 2]^T1(2) +
                            grid.Fs[ix, 3]^T1(2))
        # compute nodal total force for mixture
        Fs_x = grid.Fs[ix, 1] + dampvs * sign(grid.Vs[ix, 1])
        Fs_y = grid.Fs[ix, 2] + dampvs * sign(grid.Vs[ix, 2])
        Fs_z = grid.Fs[ix, 3] + dampvs * sign(grid.Vs[ix, 3])
        # update nodal velocity
        grid.Vs_T[ix, 1] = (Ps_1 + Fs_x * ΔT) * Ms_denom
        grid.Vs_T[ix, 2] = (Ps_2 + Fs_y * ΔT) * Ms_denom
        grid.Vs_T[ix, 3] = (Ps_3 + Fs_z * ΔT) * Ms_denom
        # boundary condition
        bc.Vx_s_Idx[ix] == T1(1) ? grid.Vs_T[ix, 1] = bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix] == T1(1) ? grid.Vs_T[ix, 2] = bc.Vy_s_Val[ix] : nothing
        bc.Vz_s_Idx[ix] == T1(1) ? grid.Vs_T[ix, 3] = bc.Vz_s_Val[ix] : nothing
        # reset grid momentum
        grid.Ps[ix, 1] = T2(0.0)
        grid.Ps[ix, 2] = T2(0.0)
        grid.Ps[ix, 3] = T2(0.0)
    end
end

@kernel inbounds=true function testdoublemapping1_OS!(
    grid    ::          KernelGrid3D{T1, T2},
    mp      ::      KernelParticle3D{T1, T2},
    pts_attr::KernelParticleProperty{T1, T2},
    ΔT      ::T2,
    FLIP    ::T2,
    PIC     ::T2
) where {T1, T2}
    ix = @index(Global)
    # update particle position & velocity
    if ix <= mp.num
        pid = pts_attr.layer[ix]
        Ks  = pts_attr.Ks[pid]
        G   = pts_attr.G[pid]
        tmp_vx_s1 = tmp_vx_s2 = tmp_vy_s1 = T2(0.0)
        tmp_vy_s2 = tmp_vz_s1 = tmp_vz_s2 = T2(0.0)
        tmp_pos_x = tmp_pos_y = tmp_pos_z = T2(0.0)
        # update particle position
        for iy in Int32(1):Int32(mp.NIC)
            Ni = mp.Ni[ix, iy]
            if Ni != T2(0.0)
                p2n   = mp.p2n[ix, iy]
                Vs_T1 = grid.Vs_T[p2n, 1]
                Vs_T2 = grid.Vs_T[p2n, 2]
                Vs_T3 = grid.Vs_T[p2n, 3]
                tmp_pos_x += Ni* Vs_T1
                tmp_pos_y += Ni* Vs_T2
                tmp_pos_z += Ni* Vs_T3
                tmp_vx_s1 += Ni*(Vs_T1-grid.Vs[p2n, 1])
                tmp_vx_s2 += Ni* Vs_T1
                tmp_vy_s1 += Ni*(Vs_T2-grid.Vs[p2n, 2])
                tmp_vy_s2 += Ni* Vs_T2
                tmp_vz_s1 += Ni*(Vs_T3-grid.Vs[p2n, 3])
                tmp_vz_s2 += Ni* Vs_T3
            end
        end
        # update particle position
        mp.pos[ix, 1] += ΔT*tmp_pos_x
        # mp.pos[ix, 2] += ΔT*tmp_pos_y
        mp.pos[ix, 3] += ΔT*tmp_pos_z
        # update particle velocity
        mp.Vs[ix, 1] = FLIP*(mp.Vs[ix, 1]+tmp_vx_s1)+PIC*tmp_vx_s2
        # mp.Vs[ix, 2] = FLIP*(mp.Vs[ix, 2]+tmp_vy_s1)+PIC*tmp_vy_s2
        mp.Vs[ix, 3] = FLIP*(mp.Vs[ix, 3]+tmp_vz_s1)+PIC*tmp_vz_s2
        # update particle momentum
        Vs_1 = mp.Vs[ix, 1]
        Vs_2 = mp.Vs[ix, 2]
        Vs_3 = mp.Vs[ix, 3]
        Ms   = mp.Ms[ix]
        mp.Ps[ix, 1] = Ms*Vs_1
        mp.Ps[ix, 2] = Ms*Vs_2
        mp.Ps[ix, 3] = Ms*Vs_3
        # update CFL conditions
        sqr = sqrt((Ks+G*T2(1.333333))/mp.ρs[ix])
        cd_sx = grid.space_x/(sqr+abs(Vs_1))
        cd_sy = grid.space_y/(sqr+abs(Vs_2))
        cd_sz = grid.space_z/(sqr+abs(Vs_3))
        mp.cfl[ix] = min(cd_sx, cd_sy, cd_sz)
    end
end

@kernel inbounds=true function testdoublemapping2_OS!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix <= mp.num
        # update particle position & velocity
        for iy in Int32(1):Int32(mp.NIC)
            Ni = mp.Ni[ix, iy]
            if Ni != T2(0.0)
                p2n = mp.p2n[ix, iy]
                @KAatomic grid.Ps[p2n, 1] += mp.Ps[ix, 1]*Ni
                @KAatomic grid.Ps[p2n, 2] += mp.Ps[ix, 2]*Ni
                @KAatomic grid.Ps[p2n, 3] += mp.Ps[ix, 3]*Ni
            end
        end
    end
end

@kernel inbounds=true function testdoublemapping3_OS!(
    grid::    KernelGrid3D{T1, T2},
    bc  ::KernelBoundary3D{T1, T2},
    ΔT  ::T2
) where {T1, T2}
    ix = @index(Global)
    if ix <= grid.node_num && grid.Ms[ix] != Int32(0)
        Ms_denom = T2(1.0) / grid.Ms[ix]
        # compute nodal velocities
        grid.Vs[ix, 1] = grid.Ps[ix, 1]*Ms_denom
        grid.Vs[ix, 2] = grid.Ps[ix, 2]*Ms_denom
        grid.Vs[ix, 3] = grid.Ps[ix, 3]*Ms_denom
        # fixed Dirichlet nodes
        bc.Vx_s_Idx[ix]==T1(1) ? grid.Vs[ix, 1]=bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix]==T1(1) ? grid.Vs[ix, 2]=bc.Vy_s_Val[ix] : nothing
        bc.Vz_s_Idx[ix]==T1(1) ? grid.Vs[ix, 3]=bc.Vz_s_Val[ix] : nothing
        # compute nodal displacement
        grid.Δd_s[ix, 1] = grid.Vs[ix, 1]*ΔT
        # grid.Δd_s[ix, 2] = grid.Vs[ix, 2]*ΔT
        grid.Δd_s[ix, 3] = grid.Vs[ix, 3]*ΔT
    end
end

@kernel inbounds=true function testG2P_OS!(
    grid::    KernelGrid3D{T1, T2},
    mp  ::KernelParticle3D{T1, T2}
) where {T1, T2}
    ix = @index(Global)
    if ix <= mp.num
        dF1 = dF2 = dF3 = dF4 = dF5 = dF6 = dF7 = dF8 = dF9 = T2(0.0)
        for iy in Int32(1):Int32(mp.NIC)
            if mp.Ni[ix, iy] != T2(0.0)
                p2n = mp.p2n[ix, iy]
                ∂Nx = mp.∂Nx[ix, iy]; ds1 = grid.Δd_s[p2n, 1]
                ∂Ny = mp.∂Ny[ix, iy]; ds2 = grid.Δd_s[p2n, 2]
                ∂Nz = mp.∂Nz[ix, iy]; ds3 = grid.Δd_s[p2n, 3]
                # compute solid incremental deformation gradient
                dF1 += ds1*∂Nx; dF2 += ds1*∂Ny; dF3 += ds1*∂Nz
                dF4 += ds2*∂Nx; dF5 += ds2*∂Ny; dF6 += ds2*∂Nz
                dF7 += ds3*∂Nx; dF8 += ds3*∂Ny; dF9 += ds3*∂Nz
            end
        end
        mp.∂Fs[ix, 1] = dF1; mp.∂Fs[ix, 2] = dF2; mp.∂Fs[ix, 3] = dF3
        mp.∂Fs[ix, 4] = dF4; mp.∂Fs[ix, 5] = dF5; mp.∂Fs[ix, 6] = dF6
        mp.∂Fs[ix, 7] = dF7; mp.∂Fs[ix, 8] = dF8; mp.∂Fs[ix, 9] = dF9
        # compute strain increment
        mp.Δϵij_s[ix, 1] = dF1
        mp.Δϵij_s[ix, 2] = dF5
        mp.Δϵij_s[ix, 3] = dF9
        mp.Δϵij_s[ix, 4] = dF2+dF4
        mp.Δϵij_s[ix, 5] = dF6+dF8
        mp.Δϵij_s[ix, 6] = dF3+dF7
        # update strain tensor
        mp.ϵij_s[ix, 1] += dF1
        mp.ϵij_s[ix, 2] += dF5
        mp.ϵij_s[ix, 3] += dF9
        mp.ϵij_s[ix, 4] += dF2+dF4
        mp.ϵij_s[ix, 5] += dF6+dF8
        mp.ϵij_s[ix, 6] += dF3+dF7
        # deformation gradient matrix
        F1 = mp.F[ix, 1]; F2 = mp.F[ix, 2]; F3 = mp.F[ix, 3]
        F4 = mp.F[ix, 4]; F5 = mp.F[ix, 5]; F6 = mp.F[ix, 6]
        F7 = mp.F[ix, 7]; F8 = mp.F[ix, 8]; F9 = mp.F[ix, 9]        
        mp.F[ix, 1] = (dF1+T2(1.0))*F1+dF2*F4+dF3*F7
        mp.F[ix, 2] = (dF1+T2(1.0))*F2+dF2*F5+dF3*F8
        mp.F[ix, 3] = (dF1+T2(1.0))*F3+dF2*F6+dF3*F9
        mp.F[ix, 4] = (dF5+T2(1.0))*F4+dF4*F1+dF6*F7
        mp.F[ix, 5] = (dF5+T2(1.0))*F5+dF4*F2+dF6*F8
        mp.F[ix, 6] = (dF5+T2(1.0))*F6+dF4*F3+dF6*F9
        mp.F[ix, 7] = (dF9+T2(1.0))*F7+dF8*F4+dF7*F1
        mp.F[ix, 8] = (dF9+T2(1.0))*F8+dF8*F5+dF7*F2
        mp.F[ix, 9] = (dF9+T2(1.0))*F9+dF8*F6+dF7*F3
        # update jacobian value and particle volume
        mp.J[ix] = mp.F[ix, 1]*mp.F[ix, 5]*mp.F[ix, 9]+mp.F[ix, 2]*mp.F[ix, 6]*mp.F[ix, 7]+
                   mp.F[ix, 3]*mp.F[ix, 4]*mp.F[ix, 8]-mp.F[ix, 7]*mp.F[ix, 5]*mp.F[ix, 3]-
                   mp.F[ix, 8]*mp.F[ix, 6]*mp.F[ix, 1]-mp.F[ix, 9]*mp.F[ix, 4]*mp.F[ix, 2]
        mp.vol[ix] = mp.J[ix]*mp.vol_init[ix]
        mp.ρs[ ix] = mp.ρs_init[ix]/mp.J[ix]
    end
end

@kernel inbounds = true function testdpP!(
    mp      ::      KernelParticle3D{T1, T2},
    pts_attr::KernelParticleProperty{T1, T2}
) where {T1, T2}
    FNUM_CMIN = T2(4e3)
    FNUM_CMAX = T2(2e4)
    FNUM_H    = T2(5e4)
    ix = @index(Global)
    if ix≤mp.num
        σm   = mp.σm[ix]
        pid  = pts_attr.layer[ix]
        ϕ    = pts_attr.ϕ[pid]
        ψ    = pts_attr.ψ[pid]
        σt   = pts_attr.σt[pid]
        G    = pts_attr.G[pid]
        Ks   = pts_attr.Ks[pid]
        c    = FNUM_CMAX - FNUM_H * mp.epII[ix] < FNUM_CMIN ? 
               FNUM_CMIN : 
               FNUM_CMAX - FNUM_H * mp.epII[ix]
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

@kernel inbounds = true function fill_halo1!(
    halo_idx, 
    send_buff0_Ms, 
    send_buff0_Ps, 
    send_buff0_Fs, 
    grid
)
    ix = @index(Global)
    if ix ≤ length(halo_idx)
        send_buff0_Ms[ix]    = grid.Ms[halo_idx[ix]]
        send_buff0_Ps[ix, 1] = grid.Ps[halo_idx[ix], 1]
        send_buff0_Ps[ix, 2] = grid.Ps[halo_idx[ix], 2]
        send_buff0_Ps[ix, 3] = grid.Ps[halo_idx[ix], 3]
        send_buff0_Fs[ix, 1] = grid.Fs[halo_idx[ix], 1]
        send_buff0_Fs[ix, 2] = grid.Fs[halo_idx[ix], 2]
        send_buff0_Fs[ix, 3] = grid.Fs[halo_idx[ix], 3]
    end
end

@kernel inbounds = true function fill_halo2!(
    halo_idx, 
    send_buff0_Ps, 
    grid
)
    ix = @index(Global)
    if ix ≤ length(halo_idx)
        send_buff0_Ps[ix, 1] = grid.Ps[halo_idx[ix], 1]
        send_buff0_Ps[ix, 2] = grid.Ps[halo_idx[ix], 2]
        send_buff0_Ps[ix, 3] = grid.Ps[halo_idx[ix], 3]
    end
end

@kernel inbounds = true function update_halo1!(
    halo_idx, 
    recv_buff0_Ms, 
    recv_buff0_Ps, 
    recv_buff0_Fs, 
    grid
)
    ix = @index(Global)
    if ix <= length(halo_idx)
        grid.Ms[halo_idx[ix]]    += recv_buff0_Ms[ix]
        grid.Ps[halo_idx[ix], 1] += recv_buff0_Ps[ix, 1]
        grid.Ps[halo_idx[ix], 2] += recv_buff0_Ps[ix, 2]
        grid.Ps[halo_idx[ix], 3] += recv_buff0_Ps[ix, 3]
        grid.Fs[halo_idx[ix], 1] += recv_buff0_Fs[ix, 1]
        grid.Fs[halo_idx[ix], 2] += recv_buff0_Fs[ix, 2]
        grid.Fs[halo_idx[ix], 3] += recv_buff0_Fs[ix, 3]
        recv_buff0_Ms[ix]    = 0f0
        recv_buff0_Ps[ix, 1] = 0f0
        recv_buff0_Ps[ix, 2] = 0f0
        recv_buff0_Ps[ix, 3] = 0f0
        recv_buff0_Fs[ix, 1] = 0f0
        recv_buff0_Fs[ix, 2] = 0f0
        recv_buff0_Fs[ix, 3] = 0f0
    end
end

@kernel inbounds = true function update_halo2!(
    halo_idx, 
    recv_buff0_Ps, 
    grid
)
    ix = @index(Global)
    if ix <= length(halo_idx)
        grid.Ps[halo_idx[ix], 1] += recv_buff0_Ps[ix, 1]
        grid.Ps[halo_idx[ix], 2] += recv_buff0_Ps[ix, 2]
        grid.Ps[halo_idx[ix], 3] += recv_buff0_Ps[ix, 3]
        recv_buff0_Ps[ix, 1] = 0f0
        recv_buff0_Ps[ix, 2] = 0f0
        recv_buff0_Ps[ix, 3] = 0f0
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
                                ::Val{:MUSL}) where {T2}
    Ti < args.Te ? G = args.gravity / args.Te * Ti : G = args.gravity
    dev = getBackend(args)
    testresetgridstatus_OS!(dev)(ndrange=grid.node_num, grid)
    testresetmpstatus_OS!(dev)(ndrange=mp.num, grid, mp, Val(args.basis))
    testP2G_OS!(dev)(ndrange=mp.num, grid, mp, G)
    testsolvegrid_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT, args.ζs)
    testdoublemapping1_OS!(dev)(ndrange=mp.num, grid, mp, pts_attr, ΔT, args.FLIP, args.PIC)
    testdoublemapping2_OS!(dev)(ndrange=mp.num, grid, mp)
    testdoublemapping3_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT)
    testG2P_OS!(dev)(ndrange=mp.num, grid, mp)
    liE!(dev)(ndrange=mp.num, mp, pts_attr)
    if Ti >= args.Te
        testdpP!(dev)(ndrange=mp.num, mp, pts_attr)
    end
    if args.MVL == true
        vollock1_OS!(dev)(ndrange=mp.num, grid, mp)
        vollock2_OS!(dev)(ndrange=mp.num, grid, mp)
    end
    return nothing
end