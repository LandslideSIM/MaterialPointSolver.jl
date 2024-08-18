#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : OS.jl (one-phase single-point)                                             |
|  Description: Functions for the computing in MPM cycle                                   |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : procedure!() [2D]                                                          |
|               procedure!() [3D]                                                          |
+==========================================================================================#

"""
    procedure!(args::Args2D{T1, T2}, grid::Grid2D{T1, T2}, mp::Particle2D{T1, T2}, 
        bc::VBoundary2D{T1, T2}, Ti::T2, ΔT, ::Val{:OS})

Description:
---
MPM update procedure with one-phase single-point model. (2D)
"""
@views function procedure!(args::     Args2D{T1, T2},
                           grid::     Grid2D{T1, T2},
                           mp  :: Particle2D{T1, T2},
                           bc  ::VBoundary2D{T1, T2}, 
                           Ti  ::T2,
                           ΔT  ::T2,
                               ::Val{:OS}) where {T1, T2}
    # specific values
    FNUM_43 = T2(4/3); FNUM_0 = T2(0.0); INUM_2 = T1(2)
    FNUM_13 = T2(1/3); FNUM_1 = T2(1.0); INUM_1 = T1(1)
    # compute particle info
    mp.Ms .= mp.ρs.*mp.vol
    mp.Ps .= mp.Ms.*mp.Vs
    # reset particle values
    fill!(mp.∂Fs, FNUM_0)
    # reset grid values
    fill!(grid.Ms , FNUM_0)
    fill!(grid.Ps , FNUM_0)
    fill!(grid.σm , FNUM_0)
    fill!(grid.vol, FNUM_0)
    fill!(grid.Fs , FNUM_0)
    # compute p2c, p2n, and shape function values
    basis_function!(grid, mp, Val(args.basis))
    # get current gravity (elastic loading)
    Ti<args.Te ? (gravity=args.gravity/args.Te*Ti) : (gravity=args.gravity)
    # particle to node mapping procedure
    for iy in 1:mp.NIC, ix in 1:mp.num
        p2n   = mp.p2n[  ix, iy]
        ∂Nx   = mp.∂Nx[  ix, iy]
        ∂Ny   = mp.∂Ny[  ix, iy]
        Ni    = mp.Ni[   ix, iy]
        Ni_Ms = Ni*mp.Ms[ix    ]
        ## scattering mass from particle to node
        grid.Ms[p2n] += Ni_Ms
        ## scattering momentum from particle to node
        grid.Ps[p2n, 1] += Ni*mp.Ps[ix, 1]
        grid.Ps[p2n, 2] += Ni*mp.Ps[ix, 2]
        ## compute nodal total force for solid
        grid.Fs[p2n, 1] += -mp.vol[ix]*(∂Nx*mp.σij[ix, 1]+∂Ny*mp.σij[ix, 4])
        grid.Fs[p2n, 2] += -mp.vol[ix]*(∂Nx*mp.σij[ix, 4]+∂Ny*mp.σij[ix, 2])+Ni_Ms*gravity
    end
    # solving equations on the grid
    for ix in 1:grid.node_num
        grid.Ms[ix]==T2(0) ? dMs=FNUM_0 : dMs=FNUM_1/grid.Ms[ix] 
        ## compute nodal velocity
        grid.Vs[ix, 1] = grid.Ps[ix, 1]*dMs
        grid.Vs[ix, 2] = grid.Ps[ix, 2]*dMs
        ## damping force for solid
        Fdamp_c = args.ζ*sqrt(grid.Fs[ix, 1]^INUM_2+grid.Fs[ix, 2]^INUM_2) 
        Fdamp_x = Fdamp_c*sign(grid.Ps[ix, 1])
        Fdamp_y = Fdamp_c*sign(grid.Ps[ix, 2])
        grid.Fs[ix, 1] -= Fdamp_x
        grid.Fs[ix, 2] -= Fdamp_y
        ## update TMP nodal velocity
        grid.Vs_T[ix, 1] = (grid.Ps[ix, 1]+grid.Fs[ix, 1]*ΔT)*dMs
        grid.Vs_T[ix, 2] = (grid.Ps[ix, 2]+grid.Fs[ix, 2]*ΔT)*dMs
        ## apply boundary conditions
        bc.Vx_s_Idx[ix]==INUM_1 ? grid.Vs_T[ix, 1]=bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix]==INUM_1 ? grid.Vs_T[ix, 2]=bc.Vy_s_Val[ix] : nothing
    end
    # gathering from grid to particle
    for ix in 1:mp.num
        pid = mp.layer[ix]
        Ks  = mp.Ks[pid]
        G   = mp.G[ pid]
        tmp_vx_s1 = tmp_vx_s2 = FNUM_0
        tmp_vy_s1 = tmp_vy_s2 = FNUM_0
        tmp_pos_x = tmp_pos_y = FNUM_0
        for iy in 1:mp.NIC
            Ni  = mp.Ni[ ix, iy]
            p2n = mp.p2n[ix, iy]
            tmp_pos_x += Ni* grid.Vs_T[p2n, 1]
            tmp_pos_y += Ni* grid.Vs_T[p2n, 2]
            tmp_vx_s1 += Ni*(grid.Vs_T[p2n, 1]-grid.Vs[p2n, 1])
            tmp_vx_s2 += Ni* grid.Vs_T[p2n, 1]
            tmp_vy_s1 += Ni*(grid.Vs_T[p2n, 2]-grid.Vs[p2n, 2])
            tmp_vy_s2 += Ni* grid.Vs_T[p2n, 2]
        end
        ## update particle position & velocity
        mp.pos[ix, 1] += ΔT*tmp_pos_x
        mp.pos[ix, 2] += ΔT*tmp_pos_y
        ## update particle velocity
        mp.Vs[ix, 1] = args.FLIP*(mp.Vs[ix, 1]+tmp_vx_s1)+args.PIC*tmp_vx_s2
        mp.Vs[ix, 2] = args.FLIP*(mp.Vs[ix, 2]+tmp_vy_s1)+args.PIC*tmp_vy_s2
        ## update CFL conditions
        sqr = sqrt((Ks+G*FNUM_43)/mp.ρs[ix])
        cd_sx = grid.space_x/(sqr+abs(mp.Vs[ix, 1]))
        cd_sy = grid.space_y/(sqr+abs(mp.Vs[ix, 2]))
        val   = min(cd_sx, cd_sy)
        mp.cfl[ix] = args.αT*val
        ## update particle momentum
        mp.Ps[ix, 1] = mp.Ms[ix]*mp.Vs[ix, 1]
        mp.Ps[ix, 2] = mp.Ms[ix]*mp.Vs[ix, 2]
    end
    # update nodal momentum
    fill!(grid.Ps, FNUM_0)
    for iy in 1:mp.NIC, ix in 1:mp.num
        p2n = mp.p2n[ix, iy]
        Ni  = mp.Ni[ ix, iy]
        grid.Ps[p2n, 1] += Ni*mp.Ps[ix, 1]
        grid.Ps[p2n, 2] += Ni*mp.Ps[ix, 2]
    end
    # solving equations on the grid
    for ix in 1:grid.node_num
        grid.Ms[ix]==T2(0) ? dMs=FNUM_0 : dMs=FNUM_1/grid.Ms[ix]
        ## compute nodal velocity
        grid.Vs[ix, 1] = grid.Ps[ix, 1]*dMs
        grid.Vs[ix, 2] = grid.Ps[ix, 2]*dMs
        ## apply boundary conditions
        bc.Vx_s_Idx[ix]==INUM_1 ? grid.Vs[ix, 1]=bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix]==INUM_1 ? grid.Vs[ix, 2]=bc.Vy_s_Val[ix] : nothing
        ## compute nodal displacement
        grid.Δd_s[ix, 1] = grid.Vs[ix, 1]*ΔT
        grid.Δd_s[ix, 2] = grid.Vs[ix, 2]*ΔT
    end
    # compute incremental deformation gradient
    for iy in 1:mp.NIC, ix in 1:mp.num
        p2n = mp.p2n[ix, iy]
        ∂Nx = mp.∂Nx[ix, iy]
        ∂Ny = mp.∂Ny[ix, iy]
        mp.∂Fs[ix, 1] += grid.Δd_s[p2n, 1]*∂Nx
        mp.∂Fs[ix, 2] += grid.Δd_s[p2n, 1]*∂Ny
        mp.∂Fs[ix, 3] += grid.Δd_s[p2n, 2]*∂Nx
        mp.∂Fs[ix, 4] += grid.Δd_s[p2n, 2]*∂Ny
    end; mp.∂Fs[:, [1, 4]] .+= FNUM_1
    # update particle properties
    for ix in 1:mp.num
        ## compute strain increment 
        ## Δϵij = B × Δu
        mp.Δϵij_s[ix, 1] = mp.∂Fs[ix, 1]-FNUM_1
        mp.Δϵij_s[ix, 2] = mp.∂Fs[ix, 4]-FNUM_1
        mp.Δϵij_s[ix, 4] = mp.∂Fs[ix, 2]+mp.∂Fs[ix, 3]
        ## update strain tensor
        mp.ϵij_s[ix, 1] += mp.Δϵij_s[ix, 1]
        mp.ϵij_s[ix, 2] += mp.Δϵij_s[ix, 2]
        mp.ϵij_s[ix, 4] += mp.Δϵij_s[ix, 4]
        ## deformation gradient matrix
        F1 = mp.F[ix, 1]; F2 = mp.F[ix, 2]; F3 = mp.F[ix, 3]; F4 = mp.F[ix, 4]
        mp.F[ix, 1] = mp.∂Fs[ix, 1]*F1+mp.∂Fs[ix, 2]*F3
        mp.F[ix, 2] = mp.∂Fs[ix, 1]*F2+mp.∂Fs[ix, 2]*F4
        mp.F[ix, 3] = mp.∂Fs[ix, 4]*F3+mp.∂Fs[ix, 3]*F1
        mp.F[ix, 4] = mp.∂Fs[ix, 4]*F4+mp.∂Fs[ix, 3]*F2
        ## update jacobian value and particle volume
        mp.J[  ix] = mp.F[ix, 1]*mp.F[ix, 4]-mp.F[ix, 2]*mp.F[ix, 3]
        mp.vol[ix] = mp.J[ix]*mp.vol_init[ix]
        mp.ρs[ ix] = mp.ρs_init[ix]/mp.J[ix]
    end
    # update by the constitutive model
    constitutive!(args, mp, Ti)
    # volumn locking
    if args.vollock == true
        for ix in 1:mp.num
            p2c = mp.p2c[ix]
            grid.σm[p2c]  += mp.vol[ix]*mp.σm[ix]
            grid.vol[p2c] += mp.vol[ix]
        end
        for ix in 1:mp.num
            p2c = mp.p2c[ix]
            σm  = grid.σm[p2c]/grid.vol[p2c]
            mp.σij[ix, 1] = mp.sij[ix, 1]+σm
            mp.σij[ix, 2] = mp.sij[ix, 2]+σm
            mp.σij[ix, 3] = mp.sij[ix, 3]+σm
            mp.σij[ix, 4] = mp.sij[ix, 4]
            # update mean stress tensor
            mp.σm[ix] = (mp.σij[ix, 1]+mp.σij[ix, 2]+mp.σij[ix, 3])*FNUM_13
            # update deviatoric stress tensor
            mp.sij[ix, 1] = mp.σij[ix, 1]-mp.σm[ix]
            mp.sij[ix, 2] = mp.σij[ix, 2]-mp.σm[ix]
            mp.sij[ix, 3] = mp.σij[ix, 3]-mp.σm[ix]
            mp.sij[ix, 4] = mp.σij[ix, 4]
        end
    end
    return nothing
end

"""
    procedure!(args::Args3D{T1, T2}, grid::Grid3D{T1, T2}, mp::Particle3D{T1, T2}, 
        bc::VBoundary3D{T1, T2}, Ti::T2, ΔT, ::Val{:OS})

Description:
---
MPM update procedure with one-phase single-point model. (3D)
"""
@views function procedure!(args::     Args3D{T1, T2},
                           grid::     Grid3D{T1, T2},
                           mp  :: Particle3D{T1, T2},
                           bc  ::VBoundary3D{T1, T2}, 
                           Ti  ::T2,
                           ΔT  ::T2,
                               ::Val{:OS}) where {T1, T2}
    # specific values
    FNUM_43 = T2(4/3); FNUM_0 = T2(0.0); INUM_2 = T1(2)
    FNUM_13 = T2(1/3); FNUM_1 = T2(1.0); INUM_1 = T1(1)
    # compute particle info
    mp.Ms .= mp.ρs.*mp.vol
    mp.Ps .= mp.Ms.*mp.Vs
    # reset particle values
    fill!(mp.∂Fs, FNUM_0)
    # reset grid values
    fill!(grid.Ms , FNUM_0)
    fill!(grid.Ps , FNUM_0)
    fill!(grid.σm , FNUM_0)
    fill!(grid.vol, FNUM_0)
    fill!(grid.Fs , FNUM_0)
    # compute p2c, p2n, and shape function values
    basis_function!(grid, mp, Val(args.basis))
    # get current gravity (elastic loading)
    Ti<args.Te ? (gravity=args.gravity/args.Te*Ti) : (gravity=args.gravity)
    # particle to node mapping procedure
    for iy in 1:mp.NIC, ix in 1:mp.num
        p2n   = mp.p2n[ix, iy]
        ∂Nx   = mp.∂Nx[ix, iy]
        ∂Ny   = mp.∂Ny[ix, iy]
        ∂Nz   = mp.∂Nz[ix, iy]
        vol   = mp.vol[ix]
        Ni    = mp.Ni[ix, iy]
        Ni_Ms = Ni*mp.Ms[ix]
        ## scattering mass from particle to node
        grid.Ms[p2n] += Ni_Ms
        ## scattering momentum from particle to node
        grid.Ps[p2n, 1] += Ni*mp.Ps[ix, 1]
        grid.Ps[p2n, 2] += Ni*mp.Ps[ix, 2]
        grid.Ps[p2n, 3] += Ni*mp.Ps[ix, 3]
        ## compute nodal total force for solid
        grid.Fs[p2n, 1] += -vol*(∂Nx*mp.σij[ix, 1]+
                                 ∂Ny*mp.σij[ix, 4]+
                                 ∂Nz*mp.σij[ix, 6])
        grid.Fs[p2n, 2] += -vol*(∂Nx*mp.σij[ix, 4]+
                                 ∂Ny*mp.σij[ix, 2]+
                                 ∂Nz*mp.σij[ix, 5])
        grid.Fs[p2n, 3] += -vol*(∂Nx*mp.σij[ix, 6]+
                                 ∂Ny*mp.σij[ix, 5]+
                                 ∂Nz*mp.σij[ix, 3])+Ni_Ms*gravity
    end
    # solving equations on the grid
    for ix in 1:grid.node_num
        grid.Ms[ix]==T2(0) ? dMs=FNUM_0 : dMs=FNUM_1/grid.Ms[ix] 
        ## compute nodal velocity
        grid.Vs[ix, 1] = grid.Ps[ix, 1]*dMs
        grid.Vs[ix, 2] = grid.Ps[ix, 2]*dMs
        grid.Vs[ix, 3] = grid.Ps[ix, 3]*dMs
        ## damping force for solid
        Fdamp_c = args.ζ*sqrt(grid.Fs[ix, 1]^INUM_2+
                              grid.Fs[ix, 2]^INUM_2+
                              grid.Fs[ix, 3]^INUM_2) 
        Fdamp_x = Fdamp_c*sign(grid.Ps[ix, 1])
        Fdamp_y = Fdamp_c*sign(grid.Ps[ix, 2])
        Fdamp_z = Fdamp_c*sign(grid.Ps[ix, 3])
        grid.Fs[ix, 1] -= Fdamp_x
        grid.Fs[ix, 2] -= Fdamp_y
        grid.Fs[ix, 3] -= Fdamp_z
        ## update TMP nodal velocity
        grid.Vs_T[ix, 1] = (grid.Ps[ix, 1]+grid.Fs[ix, 1]*ΔT)*dMs
        grid.Vs_T[ix, 2] = (grid.Ps[ix, 2]+grid.Fs[ix, 2]*ΔT)*dMs
        grid.Vs_T[ix, 3] = (grid.Ps[ix, 3]+grid.Fs[ix, 3]*ΔT)*dMs
        ## apply boundary conditions
        bc.Vx_s_Idx[ix]==INUM_1 ? grid.Vs_T[ix, 1]=bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix]==INUM_1 ? grid.Vs_T[ix, 2]=bc.Vy_s_Val[ix] : nothing
        bc.Vz_s_Idx[ix]==INUM_1 ? grid.Vs_T[ix, 3]=bc.Vz_s_Val[ix] : nothing
    end
    # gathering from grid to particle
    for ix in 1:mp.num
        pid = mp.layer[ix]
        Ks  = mp.Ks[pid]
        G   = mp.G[pid]
        tmp_vx_s1 = tmp_vy_s1 = tmp_vz_s1 = FNUM_0
        tmp_vx_s2 = tmp_vy_s2 = tmp_vz_s2 = FNUM_0
        tmp_pos_x = tmp_pos_y = tmp_pos_z = FNUM_0
        for iy in 1:mp.NIC
            Ni  = mp.Ni[ix, iy]
            p2n = mp.p2n[ix, iy]
            tmp_pos_x += Ni* grid.Vs_T[p2n, 1]
            tmp_pos_y += Ni* grid.Vs_T[p2n, 2]
            tmp_pos_z += Ni* grid.Vs_T[p2n, 3]
            tmp_vx_s1 += Ni*(grid.Vs_T[p2n, 1]-grid.Vs[p2n, 1])
            tmp_vx_s2 += Ni* grid.Vs_T[p2n, 1]
            tmp_vy_s1 += Ni*(grid.Vs_T[p2n, 2]-grid.Vs[p2n, 2])
            tmp_vy_s2 += Ni* grid.Vs_T[p2n, 2]
            tmp_vz_s1 += Ni*(grid.Vs_T[p2n, 3]-grid.Vs[p2n, 3])
            tmp_vz_s2 += Ni* grid.Vs_T[p2n, 3]
        end
        ## update particle position & velocity
        mp.pos[ix, 1] += ΔT*tmp_pos_x
        mp.pos[ix, 2] += ΔT*tmp_pos_y
        mp.pos[ix, 3] += ΔT*tmp_pos_z
        ## update particle velocity
        mp.Vs[ix, 1] = args.FLIP*(mp.Vs[ix, 1]+tmp_vx_s1)+args.PIC*tmp_vx_s2
        mp.Vs[ix, 2] = args.FLIP*(mp.Vs[ix, 2]+tmp_vy_s1)+args.PIC*tmp_vy_s2
        mp.Vs[ix, 3] = args.FLIP*(mp.Vs[ix, 3]+tmp_vz_s1)+args.PIC*tmp_vz_s2
        ## update CFL conditions
        sqr = sqrt((Ks+G*FNUM_43)/mp.ρs[ix])
        cd_sx = grid.space_x/(sqr+abs(mp.Vs[ix, 1]))
        cd_sy = grid.space_y/(sqr+abs(mp.Vs[ix, 2]))
        cd_sz = grid.space_z/(sqr+abs(mp.Vs[ix, 3]))
        val   = min(cd_sx, cd_sy, cd_sz)
        mp.cfl[ix] = args.αT*val
        # update particle momentum
        mp.Ps[ix, 1] = mp.Ms[ix]*mp.Vs[ix, 1]
        mp.Ps[ix, 2] = mp.Ms[ix]*mp.Vs[ix, 2]
        mp.Ps[ix, 3] = mp.Ms[ix]*mp.Vs[ix, 3]
    end
    # update nodal momentum
    fill!(grid.Ps, FNUM_0)
    for iy in 1:mp.NIC, ix in 1:mp.num
        p2n = mp.p2n[ix, iy]
        Ni  = mp.Ni[ix, iy]
        grid.Ps[p2n, 1] += Ni*mp.Ps[ix, 1]
        grid.Ps[p2n, 2] += Ni*mp.Ps[ix, 2]
        grid.Ps[p2n, 3] += Ni*mp.Ps[ix, 3]
    end
    # solving equations on the grid
    for ix in 1:grid.node_num
        grid.Ms[ix]==T2(0) ? dMs=FNUM_0 : dMs=FNUM_1/grid.Ms[ix]
        ## compute nodal velocity
        grid.Vs[ix, 1] = grid.Ps[ix, 1]*dMs
        grid.Vs[ix, 2] = grid.Ps[ix, 2]*dMs
        grid.Vs[ix, 3] = grid.Ps[ix, 3]*dMs
        ## apply boundary conditions
        bc.Vx_s_Idx[ix]==INUM_1 ? grid.Vs[ix, 1]=bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix]==INUM_1 ? grid.Vs[ix, 2]=bc.Vy_s_Val[ix] : nothing
        bc.Vz_s_Idx[ix]==INUM_1 ? grid.Vs[ix, 3]=bc.Vz_s_Val[ix] : nothing
        ## compute nodal displacement
        grid.Δd_s[ix, 1] = grid.Vs[ix, 1]*ΔT
        grid.Δd_s[ix, 2] = grid.Vs[ix, 2]*ΔT
        grid.Δd_s[ix, 3] = grid.Vs[ix, 3]*ΔT
    end
    # compute incremental deformation gradient
    for iy in 1:mp.NIC, ix in 1:mp.num
        p2n = mp.p2n[ix, iy]
        ∂Nx = mp.∂Nx[ix, iy]
        ∂Ny = mp.∂Ny[ix, iy]
        ∂Nz = mp.∂Nz[ix, iy]
        mp.∂Fs[ix, 1] += grid.Δd_s[p2n, 1]*∂Nx
        mp.∂Fs[ix, 2] += grid.Δd_s[p2n, 1]*∂Ny
        mp.∂Fs[ix, 3] += grid.Δd_s[p2n, 1]*∂Nz
        mp.∂Fs[ix, 4] += grid.Δd_s[p2n, 2]*∂Nx
        mp.∂Fs[ix, 5] += grid.Δd_s[p2n, 2]*∂Ny
        mp.∂Fs[ix, 6] += grid.Δd_s[p2n, 2]*∂Nz
        mp.∂Fs[ix, 7] += grid.Δd_s[p2n, 3]*∂Nx
        mp.∂Fs[ix, 8] += grid.Δd_s[p2n, 3]*∂Ny
        mp.∂Fs[ix, 9] += grid.Δd_s[p2n, 3]*∂Nz
    end; mp.∂Fs[:, [1, 5, 9]] .+= FNUM_1
    for ix in 1:mp.num
        # compute strain increment 
        ## Δϵij,p = 0.5*(∂Nj * Vi,I + ∂Ni * Vj,I) * Δt
        mp.Δϵij_s[ix, 1] = mp.∂Fs[ix, 1]-FNUM_1
        mp.Δϵij_s[ix, 2] = mp.∂Fs[ix, 5]-FNUM_1
        mp.Δϵij_s[ix, 3] = mp.∂Fs[ix, 9]-FNUM_1
        mp.Δϵij_s[ix, 4] = mp.∂Fs[ix, 2]+mp.∂Fs[ix, 4]
        mp.Δϵij_s[ix, 5] = mp.∂Fs[ix, 6]+mp.∂Fs[ix, 8]
        mp.Δϵij_s[ix, 6] = mp.∂Fs[ix, 3]+mp.∂Fs[ix, 7]
        # update strain tensor
        mp.ϵij_s[ix, 1] += mp.Δϵij_s[ix, 1]
        mp.ϵij_s[ix, 2] += mp.Δϵij_s[ix, 2]
        mp.ϵij_s[ix, 3] += mp.Δϵij_s[ix, 3]
        mp.ϵij_s[ix, 4] += mp.Δϵij_s[ix, 4]
        mp.ϵij_s[ix, 5] += mp.Δϵij_s[ix, 5]
        mp.ϵij_s[ix, 6] += mp.Δϵij_s[ix, 6] 
        # deformation gradient matrix
        F1 = mp.F[ix, 1]; F2 = mp.F[ix, 2]; F3 = mp.F[ix, 3]
        F4 = mp.F[ix, 4]; F5 = mp.F[ix, 5]; F6 = mp.F[ix, 6]
        F7 = mp.F[ix, 7]; F8 = mp.F[ix, 8]; F9 = mp.F[ix, 9]        
        mp.F[ix, 1] = mp.∂Fs[ix, 1]*F1+mp.∂Fs[ix, 2]*F4+mp.∂Fs[ix, 3]*F7
        mp.F[ix, 2] = mp.∂Fs[ix, 1]*F2+mp.∂Fs[ix, 2]*F5+mp.∂Fs[ix, 3]*F8
        mp.F[ix, 3] = mp.∂Fs[ix, 1]*F3+mp.∂Fs[ix, 2]*F6+mp.∂Fs[ix, 3]*F9
        mp.F[ix, 4] = mp.∂Fs[ix, 5]*F4+mp.∂Fs[ix, 4]*F1+mp.∂Fs[ix, 6]*F7
        mp.F[ix, 5] = mp.∂Fs[ix, 5]*F5+mp.∂Fs[ix, 4]*F2+mp.∂Fs[ix, 6]*F8
        mp.F[ix, 6] = mp.∂Fs[ix, 5]*F6+mp.∂Fs[ix, 4]*F3+mp.∂Fs[ix, 6]*F9
        mp.F[ix, 7] = mp.∂Fs[ix, 9]*F7+mp.∂Fs[ix, 8]*F4+mp.∂Fs[ix, 7]*F1
        mp.F[ix, 8] = mp.∂Fs[ix, 9]*F8+mp.∂Fs[ix, 8]*F5+mp.∂Fs[ix, 7]*F2
        mp.F[ix, 9] = mp.∂Fs[ix, 9]*F9+mp.∂Fs[ix, 8]*F6+mp.∂Fs[ix, 7]*F3
        # update jacobian value and particle volume
        mp.J[ix] = mp.F[ix, 1]*mp.F[ix, 5]*mp.F[ix, 9]+mp.F[ix, 2]*mp.F[ix, 6]*mp.F[ix, 7]+
                   mp.F[ix, 3]*mp.F[ix, 4]*mp.F[ix, 8]-mp.F[ix, 7]*mp.F[ix, 5]*mp.F[ix, 3]-
                   mp.F[ix, 8]*mp.F[ix, 6]*mp.F[ix, 1]-mp.F[ix, 9]*mp.F[ix, 4]*mp.F[ix, 2]
        mp.vol[ix] = mp.J[ix]*mp.vol_init[ix]
        mp.ρs[ ix] = mp.ρs_init[ix]/mp.J[ix]
    end
    # update by the constitutive model
    constitutive!(args, mp, Ti)
    # volumn locking
    if args.vollock == true
        for ix in 1:mp.num
            p2c = mp.p2c[ix]
            grid.σm[p2c]  += mp.vol[ix]*mp.σm[ix]
            grid.vol[p2c] += mp.vol[ix]
        end
        for ix in 1:mp.num
            p2c = mp.p2c[ix]
            σm  = grid.σm[p2c]/grid.vol[p2c]
            mp.σij[ix, 1] = mp.sij[ix, 1]+σm
            mp.σij[ix, 2] = mp.sij[ix, 2]+σm
            mp.σij[ix, 3] = mp.sij[ix, 3]+σm
            mp.σij[ix, 4] = mp.sij[ix, 4]
            mp.σij[ix, 5] = mp.sij[ix, 5]
            mp.σij[ix, 6] = mp.sij[ix, 6]
            # update mean stress tensor
            mp.σm[ix] = (mp.σij[ix, 1]+mp.σij[ix, 2]+mp.σij[ix, 3])*FNUM_13
            # update deviatoric stress tensor
            mp.sij[ix, 1] = mp.σij[ix, 1]-mp.σm[ix]
            mp.sij[ix, 2] = mp.σij[ix, 2]-mp.σm[ix]
            mp.sij[ix, 3] = mp.σij[ix, 3]-mp.σm[ix]
            mp.sij[ix, 4] = mp.σij[ix, 4]
            mp.sij[ix, 5] = mp.σij[ix, 5]
            mp.sij[ix, 6] = mp.σij[ix, 6]
        end
    end
    return nothing
end