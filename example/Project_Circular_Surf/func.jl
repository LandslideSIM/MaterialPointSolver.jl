@kernel inbounds=true function testliE!(
    mp      ::      KernelParticle3D{T1, T2},
    pts_attr::KernelParticleProperty{T1, T2},
    Ks, Gt
) where {T1, T2}
    ix = @index(Global)
    if ix ≤ mp.num
        # spin tensor
        # here, ωij = (vorticity tensor) × Δt, i.e.
        # ωij = (∂Ni × Vj - ∂Nj × Vi) × 0.5 × Δt
        ωxy = T2(0.5) * (mp.∂Fs[ix, 4] - mp.∂Fs[ix, 2])
        ωyz = T2(0.5) * (mp.∂Fs[ix, 8] - mp.∂Fs[ix, 6])
        ωxz = T2(0.5) * (mp.∂Fs[ix, 7] - mp.∂Fs[ix, 3])
        # objective stress
        # σij,new = σij,old + σij,R
        # σxx,R =  2 × (σxy × ωxy + σxz × ωxz)
        # σyy,R = -2 × (σxy × ωxy - σyz × ωyz)
        # σzz,R = -2 × (σxz × ωxz + σyz × ωyz)
        # σxy,R =  ωxy × (σyy - σxx) + σyz × ωxz + σxz × ωyz
        # σyz,R =  ωyz × (σzz - σyy) - σxy × ωxz - σxz × ωxy
        # σxz,R =  ωxz × (σzz - σxx) + σyz × ωxy - σxy × ωyz
        # where σij = σji, ωij = -ωji
        σij1 = mp.σij[ix, 1]
        σij2 = mp.σij[ix, 2]
        σij3 = mp.σij[ix, 3]
        σij4 = mp.σij[ix, 4]
        σij5 = mp.σij[ix, 5]
        σij6 = mp.σij[ix, 6]
        mp.σij[ix, 1] +=  T2(2.0) * (σij4 * ωxy + σij6 * ωxz)
        mp.σij[ix, 2] += -T2(2.0) * (σij4 * ωxy - σij5 * ωyz)
        mp.σij[ix, 3] += -T2(2.0) * (σij6 * ωxz + σij5 * ωyz)
        mp.σij[ix, 4] += ωxy * (σij2 - σij1) + ωxz * σij5 + ωyz * σij6
        mp.σij[ix, 5] += ωyz * (σij3 - σij2) - ωxz * σij4 - ωxy * σij6
        mp.σij[ix, 6] += ωxz * (σij3 - σij1) + ωxy * σij5 - ωyz * σij4
        # linear elastic
        Dt = Ks[ix] + T2(1.333333) * Gt[ix]
        Dd = Ks[ix] - T2(0.666667) * Gt[ix]
        mp.σij[ix, 1] += Dt * mp.Δϵij_s[ix, 1] + 
                         Dd * mp.Δϵij_s[ix, 2] + 
                         Dd * mp.Δϵij_s[ix, 3]
        mp.σij[ix, 2] += Dd * mp.Δϵij_s[ix, 1] + 
                         Dt * mp.Δϵij_s[ix, 2] + 
                         Dd * mp.Δϵij_s[ix, 3]
        mp.σij[ix, 3] += Dd * mp.Δϵij_s[ix, 1] + 
                         Dd * mp.Δϵij_s[ix, 2] + 
                         Dt * mp.Δϵij_s[ix, 3]
        mp.σij[ix, 4] += Gt[ix] * mp.Δϵij_s[ix, 4]
        mp.σij[ix, 5] += Gt[ix] * mp.Δϵij_s[ix, 5]
        mp.σij[ix, 6] += Gt[ix] * mp.Δϵij_s[ix, 6]
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

@kernel inbounds=true function testdpP!(
    mp      ::      KernelParticle3D{T1, T2},
    pts_attr::KernelParticleProperty{T1, T2},
    cp, ϕp, ψp, σtp, Gp, Ksp
) where {T1, T2}
    ix = @index(Global)
    if ix≤mp.num
        σm  = mp.σm[ix]
        c   = cp[ix]
        ϕ   = ϕp[ix]
        ψ   = ψp[ix]
        σt  = σtp[ix]
        G   = Gp[ix]
        Ks  = Ksp[ix]
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
    if ix ≤ mp.num
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
        mp.pos[ix, 2] += ΔT*tmp_pos_y
        mp.pos[ix, 3] += ΔT*tmp_pos_z
        # update particle velocity
        mp.Vs[ix, 1] = FLIP*(mp.Vs[ix, 1]+tmp_vx_s1)+PIC*tmp_vx_s2
        mp.Vs[ix, 2] = FLIP*(mp.Vs[ix, 2]+tmp_vy_s1)+PIC*tmp_vy_s2
        mp.Vs[ix, 3] = FLIP*(mp.Vs[ix, 3]+tmp_vz_s1)+PIC*tmp_vz_s2
        # update particle momentum
        Vs_1 = mp.Vs[ix, 1]
        Vs_2 = mp.Vs[ix, 2]
        Vs_3 = mp.Vs[ix, 3]
        Ms   = mp.Ms[ix]
        mp.Ps[ix, 1] = Ms*Vs_1
        mp.Ps[ix, 2] = Ms*Vs_2
        mp.Ps[ix, 3] = Ms*Vs_3
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
                                ::Val{:MUSL},
                                cp, ϕp, ψp, σtp, Gp, Ksp) where {T2}
    Ti < args.Te ? G = args.gravity / args.Te * Ti : G = args.gravity
    dev = getBackend(args)
    resetgridstatus_OS!(dev)(ndrange=grid.node_num, grid)
    args.device == :CPU && args.basis == :uGIMP ? 
        resetmpstatus_OS!(grid, mp, Val(args.basis)) :
        resetmpstatus_OS!(dev)(ndrange=mp.num, grid, mp, Val(args.basis))
    P2G_OS!(dev)(ndrange=mp.num, grid, mp, G)
    solvegrid_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT, args.ζs)
    testdoublemapping1_OS!(dev)(ndrange=mp.num, grid, mp, pts_attr, ΔT, args.FLIP, args.PIC)
    doublemapping2_OS!(dev)(ndrange=mp.num, grid, mp)
    doublemapping3_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT)
    G2P_OS!(dev)(ndrange=mp.num, grid, mp)
    testliE!(dev)(ndrange=mp.num, mp, pts_attr, Ksp, Gp)
    if Ti≥args.Te
        testdpP!(dev)(ndrange=mp.num, mp, pts_attr, cp, ϕp, ψp, σtp, Gp, Ksp)
    end
    if args.MVL == true
        vollock1_OS!(dev)(ndrange=mp.num, grid, mp)
        vollock2_OS!(dev)(ndrange=mp.num, grid, mp)
    end
    return nothing
end

@views function testsubmit_work!(args    ::          Args3D{T1, T2},
                             grid    ::          Grid3D{T1, T2}, 
                             mp      ::      Particle3D{T1, T2}, 
                             pts_attr::ParticleProperty{T1, T2},
                             bc      ::     VBoundary3D{T1, T2},
                             workflow::Function,
                             cp, ϕp, ψp, σtp, Gp, Ksp) where {T1, T2}
    # variables setup for the simulation 
    FNUM_0 = T2(0.0); INUM_1 = T1(1)
    Ti     = FNUM_0 ; INUM_0 = T1(0)
    pc     = Ref{T1}(0)
    pb     = progressinfo(args, "solving")
    args.time_step==:auto ? ΔT=cfl(args, grid, mp, pts_attr, Val(args.coupling)) : ΔT=args.ΔT
    if args.device!=:CPU 
        gpu_grid, gpu_mp, gpu_pts_attr, gpu_bc = host2device(args, grid, mp, pts_attr, bc)
        cp  = CuArray(cp)
        ϕp  = CuArray(ϕp)
        ψp  = CuArray(ψp)
        σtp = CuArray(σtp)
        Gp  = CuArray(Gp)
        Ksp = CuArray(Ksp)
    end
    # main part: HDF5 ON / OFF
    if args.hdf5==true
        hdf5_id     = INUM_1 # HDF5 group index
        hdf5_switch = INUM_0 # HDF5 step
        hdf5_path   = joinpath(args.project_path, "$(args.project_name).h5")
        isfile(hdf5_path) ? rm(hdf5_path) : nothing
        h5open(hdf5_path, "cw") do fid
            args.start_time = time()
            while Ti < args.Ttol
                if (hdf5_switch==args.hdf5_step) || (hdf5_switch==INUM_0)
                    args.device≠:CPU ? device2host!(args, mp, gpu_mp) : nothing
                    g = create_group(fid, "group$(hdf5_id)")
                    g["sig"   ] = mp.σij
                    g["eps_s" ] = mp.ϵij_s
                    g["epII"  ] = mp.epII
                    g["epK"   ] = mp.epK
                    g["mp_pos"] = mp.pos
                    g["v_s"   ] = mp.Vs
                    g["vol"   ] = mp.vol
                    g["mass"  ] = mp.Ms
                    g["time"  ] = Ti
                    if args.coupling==:TS
                        g["pp"      ] = mp.σw
                        g["eps_w"   ] = mp.ϵij_w
                        g["v_w"     ] = mp.Vw
                        g["porosity"] = mp.porosity
                    end
                    hdf5_switch = INUM_0; hdf5_id += INUM_1
                end
                if args.device==:CPU 
                    workflow(args, grid, mp, pts_attr, bc, ΔT, Ti, Val(args.coupling), Val(args.scheme),
                        cp, ϕp, ψp, σtp, Gp, Ksp)
                    args.time_step==:auto ? ΔT=args.αT*reduce(min, mp.cfl) : nothing
                else
                    workflow(args, gpu_grid, gpu_mp, gpu_pts_attr, gpu_bc, ΔT, Ti, Val(args.coupling), Val(args.scheme),
                        cp, ϕp, ψp, σtp, Gp, Ksp)
                    args.time_step==:auto ? ΔT=args.αT*reduce(min, gpu_mp.cfl) : nothing
                end
                Ti += ΔT
                hdf5_switch += 1
                args.iter_num += 1
                updatepb!(pc, Ti, args.Ttol, pb)
            end
            args.end_time = time()
            write(fid, "FILE_NUM"  , hdf5_id       )
            write(fid, "grid_pos"  , grid.pos      )
            write(fid, "mp_init"   , mp.init       )
            write(fid, "layer"     , pts_attr.layer)
            write(fid, "vbc_xs_idx", bc.Vx_s_Idx   )
            write(fid, "vbc_xs_val", bc.Vx_s_Val   )
            write(fid, "vbc_ys_idx", bc.Vy_s_Idx   )
            write(fid, "vbc_ys_val", bc.Vy_s_Val   )
            write(fid, "vbc_zs_idx", bc.Vz_s_Idx   )
            write(fid, "vbc_zs_val", bc.Vz_s_Val   )
            if args.coupling==:TS
                write(fid, "vbc_xw_idx", bc.Vx_w_Idx)
                write(fid, "vbc_xw_val", bc.Vx_w_Val)
                write(fid, "vbc_yw_idx", bc.Vy_w_Idx)
                write(fid, "vbc_yw_val", bc.Vy_w_Val)
                write(fid, "vbc_zw_idx", bc.Vz_w_Idx)
                write(fid, "vbc_zw_val", bc.Vz_w_Val)
            end
        end
    elseif args.hdf5==false
        args.start_time = time()
        while Ti < args.Ttol
            if args.device==:CPU 
                workflow(args, grid, mp, pts_attr, bc, ΔT, Ti, Val(args.coupling), Val(args.scheme), 
                    cp, ϕp, ψp, σtp, Gp, Ksp)
                args.time_step==:auto ? ΔT=args.αT*reduce(min, mp.cfl) : nothing
            else
                workflow(args, gpu_grid, gpu_mp, gpu_pts_attr, gpu_bc, ΔT, Ti, Val(args.coupling), Val(args.scheme),
                    cp, ϕp, ψp, σtp, Gp, Ksp)
                args.time_step==:auto ? ΔT=args.αT*reduce(min, gpu_mp.cfl) : nothing
            end
            Ti += ΔT
            args.iter_num += 1
            updatepb!(pc, Ti, args.Ttol, pb)
        end
        args.end_time = time()
    end
    if args.device≠:CPU
        KAsync(getBackend(args))
        device2hostinfo!(args, mp, gpu_mp)
        clean_gpu!(args, gpu_grid, gpu_mp, gpu_pts_attr, gpu_bc)
    end
    return nothing
end


function testmaterialpointsolver!(args    ::MODELARGS, 
                              grid    ::GRID, 
                              mp      ::PARTICLE, 
                              pts_attr::PROPERTY,
                              bc      ::BOUNDARY,
                              cp, ϕp, ψp, σtp, Gp, Ksp; 
                              workflow::Function=procedure!)
    info_print(args, grid, mp) # terminal info
    testsubmit_work!(args, grid, mp, pts_attr, bc, workflow,
        cp, ϕp, ψp, σtp, Gp, Ksp) # MPM solver
    perf(args, grid, mp) # performance summary
    # generate animation files (ParaView)
    args.animation==true ? animation(args) : nothing 
    return nothing
end