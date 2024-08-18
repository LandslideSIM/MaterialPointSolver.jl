#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : TS.jl (two-phase single-point)                                             |
|  Description: Functions for the computing in MPM cycle on CPU.                           |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : procedure!() [2D]                                                          |
|               procedure!() [3D]                                                          |
+==========================================================================================#

"""
    procedure!(args::Args2D{T1, T2}, grid::Grid2D{T1, T2}, mp::Particle2D{T1, T2}, 
        bc::VBoundary2D{T1, T2}, Ti::T2, ΔT, ::Val{:TS})

Description:
---
MPM update procedure with two-phase single-point model. (2D)
"""
@views function procedure!(args::     Args2D{T1, T2},
                           grid::     Grid2D{T1, T2},
                           mp  :: Particle2D{T1, T2},
                           bc  ::VBoundary2D{T1, T2}, 
                           Ti  ::T2,
                           ΔT  ::T2,
                               ::Val{:TS}) where {T1, T2}
    # specific values
    FNUM_0 = T2(0.0); FNUM_43 = T2(4/3); INUM_2 = T1(2)
    FNUM_1 = T2(1.0); FNUM_13 = T2(1/3); INUM_1 = T1(1)
    # compute particle info
    @. mp.Ms = mp.vol*mp.ρs                                           # solid mass
    @. mp.Mw = mp.vol*mp.ρw                                           # water mass
    @. mp.Mi = mp.vol*((FNUM_1-mp.porosity)*mp.ρs+mp.porosity*mp.ρw)  # total mass
    @. mp.Ps = mp.Ms*mp.Vs*(FNUM_1-mp.porosity)
    @. mp.Pw = mp.Mw*mp.Vw*        mp.porosity
    # reset particle values
    fill!(mp.∂Fs, FNUM_0)
    fill!(mp.∂Fw, FNUM_0)
    # reset grid values
    fill!(grid.Ms   , FNUM_0)
    fill!(grid.Mw   , FNUM_0)
    fill!(grid.Mi   , FNUM_0)
    fill!(grid.Ps   , FNUM_0)
    fill!(grid.Pw   , FNUM_0)
    fill!(grid.σm   , FNUM_0)
    fill!(grid.vol  , FNUM_0)
    fill!(grid.Fs   , FNUM_0)
    fill!(grid.Fw   , FNUM_0)
    fill!(grid.Fdrag, FNUM_0)
    # compute p2c, p2n, and shape function values
    basis_function!(grid, mp, Val(args.basis))
    # get current gravity (elastic loading)
    Ti<args.Te ? (gravity=args.gravity/args.Te*Ti) : (gravity=args.gravity)
    # particle to node mapping procedure
    for iy in 1:mp.NIC, ix in 1:mp.num
        Ni  = mp.Ni[ix, iy]
        p2n = mp.p2n[ix, iy]
        ∂Nx = mp.∂Nx[ix, iy]
        ∂Ny = mp.∂Ny[ix, iy]
        ## scattering mass from particle to node
        grid.Ms[p2n] += Ni*mp.Ms[ix]*(FNUM_1-mp.porosity[ix])
        grid.Mw[p2n] += Ni*mp.Mw[ix]
        grid.Mi[p2n] += Ni*mp.Mw[ix]*        mp.porosity[ix]
        ## scattering momentum from particle to node
        grid.Ps[p2n, 1] += Ni*mp.Ps[ix, 1]
        grid.Ps[p2n, 2] += Ni*mp.Ps[ix, 2]
        grid.Pw[p2n, 1] += Ni*mp.Pw[ix, 1]
        grid.Pw[p2n, 2] += Ni*mp.Pw[ix, 2]
        ## compute nodal total force
        grid.Fs[p2n, 1] += -mp.vol[ix]*(∂Nx*(mp.σij[ix, 1]+mp.σw[ix])+∂Ny*mp.σij[ix, 4])
        grid.Fs[p2n, 2] += -mp.vol[ix]*(∂Ny*(mp.σij[ix, 2]+mp.σw[ix])+∂Nx*mp.σij[ix, 4])+
                            Ni*mp.Mi[ix]*gravity
        grid.Fw[p2n, 1] += -mp.vol[ix]*∂Nx*mp.σw[ix]
        grid.Fw[p2n, 2] += -mp.vol[ix]*∂Ny*mp.σw[ix]+Ni*mp.Mw[ix]*gravity
    end
    #grid.Fs[trac_idx, 2] .+= trac_nds 
    # solving equations on the grid
    for ix in 1:grid.node_num
        grid.Ms[ix]==T2(0) ? dMs=FNUM_0 : dMs=FNUM_1/grid.Ms[ix]
        grid.Mw[ix]==T2(0) ? dMw=FNUM_0 : dMw=FNUM_1/grid.Mw[ix]
        grid.a_w[ix, 1] = dMw*grid.Fw[ix, 1]
        grid.a_w[ix, 2] = dMw*grid.Fw[ix, 2]
        grid.a_s[ix, 1] = dMs*grid.Fs[ix, 1]-grid.Mi[ix]*grid.a_w[ix, 1]
        grid.a_s[ix, 2] = dMs*grid.Fs[ix, 2]-grid.Mi[ix]*grid.a_w[ix, 2]
        ## apply boundary conditions
        bc.Vx_s_Idx[ix]==INUM_1 ? grid.a_s[ix, 1]=bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix]==INUM_1 ? grid.a_s[ix, 2]=bc.Vy_s_Val[ix] : nothing
        bc.Vx_w_Idx[ix]==INUM_1 ? grid.a_w[ix, 1]=bc.Vx_w_Val[ix] : nothing
        bc.Vy_w_Idx[ix]==INUM_1 ? grid.a_w[ix, 2]=bc.Vy_w_Val[ix] : nothing
    end
    # gathering from grid to particle
    for ix in 1:mp.num
        pid = mp.layer[ix]
        Ks  = mp.Ks[pid]
        G   = mp.G[pid]
        tmp_vx_s1 = FNUM_0
        tmp_vy_s1 = FNUM_0
        tmp_vx_w1 = FNUM_0
        tmp_vy_w1 = FNUM_0
        for iy in 1:mp.NIC
            Ni  = mp.Ni[ix, iy]
            p2n = mp.p2n[ix, iy]
            tmp_vx_s1 += Ni*grid.a_s[p2n, 1]
            tmp_vy_s1 += Ni*grid.a_s[p2n, 2]
            tmp_vx_w1 += Ni*grid.a_w[p2n, 1]
            tmp_vy_w1 += Ni*grid.a_w[p2n, 2]
        end
        ## update particle velocity
        mp.Vs[ix, 1] += ΔT*tmp_vx_s1
        mp.Vs[ix, 2] += ΔT*tmp_vy_s1
        mp.Vw[ix, 1] += ΔT*tmp_vx_w1
        mp.Vw[ix, 2] += ΔT*tmp_vy_w1
        ## update CFL conditions
        sqr = sqrt((Ks+G*FNUM_43)/mp.ρs[ix])
        cd_sx = grid.space_x/(sqr+abs(mp.Vs[ix, 1]))
        cd_sy = grid.space_y/(sqr+abs(mp.Vs[ix, 2]))
        val   = min(cd_sx, cd_sy)
        mp.cfl[ix] = args.αT*val
        ## update particle momentum
        mp.Ps[ix, 1] = mp.Ms[ix]*mp.Vs[ix, 1]*(FNUM_1-mp.porosity[ix])
        mp.Ps[ix, 2] = mp.Ms[ix]*mp.Vs[ix, 2]*(FNUM_1-mp.porosity[ix])
        mp.Pw[ix, 1] = mp.Mw[ix]*mp.Vw[ix, 1]*        mp.porosity[ix]
        mp.Pw[ix, 2] = mp.Mw[ix]*mp.Vw[ix, 2]*        mp.porosity[ix]
    end
    # update nodal momentum
    fill!(grid.Ps, FNUM_0)
    for iy in 1:mp.NIC, ix in 1:mp.num
        p2n = mp.p2n[ix, iy]
        Ni  = mp.Ni[ix, iy]
        grid.Ps[p2n, 1] += Ni*mp.Ps[ix, 1]
        grid.Ps[p2n, 2] += Ni*mp.Ps[ix, 2]
        grid.Pw[p2n, 1] += Ni*mp.Pw[ix, 1]
        grid.Pw[p2n, 2] += Ni*mp.Pw[ix, 2]
    end
    # solving equations on the grid
    for ix in 1:grid.node_num
        grid.Ms[ix]==T2(0) ? dMs=FNUM_0 : dMs=FNUM_1/grid.Ms[ix]
        grid.Mw[ix]==T2(0) ? dMw=FNUM_0 : dMw=FNUM_1/grid.Mw[ix]
        ## compute nodal velocity
        grid.Vs[ix, 1] = grid.Ps[ix, 1]*dMs
        grid.Vs[ix, 2] = grid.Ps[ix, 2]*dMs
        grid.Vw[ix, 1] = grid.Pw[ix, 1]*dMw
        grid.Vw[ix, 2] = grid.Pw[ix, 2]*dMw
        ## apply boundary conditions
        bc.Vx_s_Idx[ix]==INUM_1 ? grid.Vs[ix, 1]=bc.Vx_s_Val[ix] : nothing
        bc.Vy_s_Idx[ix]==INUM_1 ? grid.Vs[ix, 2]=bc.Vy_s_Val[ix] : nothing
        bc.Vx_w_Idx[ix]==INUM_1 ? grid.Vw[ix, 1]=bc.Vx_w_Val[ix] : nothing
        bc.Vy_w_Idx[ix]==INUM_1 ? grid.Vw[ix, 2]=bc.Vy_w_Val[ix] : nothing
        ## compute nodal displacement
        grid.Δd_s[ix, 1] = grid.Vs[ix, 1]*ΔT
        grid.Δd_s[ix, 2] = grid.Vs[ix, 2]*ΔT
        grid.Δd_w[ix, 1] = grid.Vw[ix, 1]*ΔT
        grid.Δd_w[ix, 2] = grid.Vw[ix, 2]*ΔT
    end
    # compute incremental deformation gradient
    for iy in 1:mp.NIC, ix in 1:mp.num
        Ni  = mp.Ni[ix, iy]
        p2n = mp.p2n[ix, iy]
        ∂Nx = mp.∂Nx[ix, iy]
        ∂Ny = mp.∂Ny[ix, iy]
        mp.∂Fs[ix, 1] += grid.Δd_s[p2n, 1]*∂Nx
        mp.∂Fs[ix, 2] += grid.Δd_s[p2n, 1]*∂Ny
        mp.∂Fs[ix, 3] += grid.Δd_s[p2n, 2]*∂Nx
        mp.∂Fs[ix, 4] += grid.Δd_s[p2n, 2]*∂Ny
        mp.∂Fw[ix, 1] += grid.Δd_w[p2n, 1]*∂Nx
        mp.∂Fw[ix, 2] += grid.Δd_w[p2n, 1]*∂Ny
        mp.∂Fw[ix, 3] += grid.Δd_w[p2n, 2]*∂Nx
        mp.∂Fw[ix, 4] += grid.Δd_w[p2n, 2]*∂Ny
        mp.pos[ix, 1] += grid.Δd_s[p2n, 1]*Ni
        mp.pos[ix, 2] += grid.Δd_s[p2n, 2]*Ni
    end
    mp.∂Fs[:, [1, 4]] .+= FNUM_1
    mp.∂Fw[:, [1, 4]] .+= FNUM_1
    # update particle properties
    for ix in 1:mp.num
        Kw = mp.Kw[mp.layer[ix]]
        ## compute strain increment 
        mp.Δϵij_s[ix, 1] = mp.∂Fs[ix, 1]-FNUM_1
        mp.Δϵij_s[ix, 2] = mp.∂Fs[ix, 4]-FNUM_1
        mp.Δϵij_s[ix, 4] = mp.∂Fs[ix, 2]+mp.∂Fs[ix, 3]
        mp.Δϵij_w[ix, 1] = mp.∂Fw[ix, 1]-FNUM_1
        mp.Δϵij_w[ix, 2] = mp.∂Fw[ix, 4]-FNUM_1
        mp.Δϵij_w[ix, 4] = mp.∂Fw[ix, 2]+mp.∂Fw[ix, 3]
        ## update strain tensor
        mp.ϵij_s[ix, 1] += mp.Δϵij_s[ix, 1]
        mp.ϵij_s[ix, 2] += mp.Δϵij_s[ix, 2]
        mp.ϵij_s[ix, 4] += mp.Δϵij_s[ix, 4]
        mp.ϵij_w[ix, 1] += mp.Δϵij_w[ix, 1]
        mp.ϵij_w[ix, 2] += mp.Δϵij_w[ix, 2]
        mp.ϵij_w[ix, 4] += mp.Δϵij_w[ix, 4]
        ## update water pressure
        # mp.σw[ix] += (Kw/mp.porosity[ix])*(
        #     (FNUM_1-mp.porosity[ix])*(mp.Δϵij_s[ix, 1]+mp.Δϵij_s[ix, 2])+
        #             mp.porosity[ix] *(mp.Δϵij_w[ix, 1]+mp.Δϵij_w[ix, 2])
        # )
        ## update volume and density
        deco = (FNUM_1+mp.Δϵij_s[ix, 1]+mp.Δϵij_s[ix, 2])
        mp.vol[ix] *= deco
        # mp.ρs[ix]  /= deco
        # mp.ρw[ix]  /= deco
        # mp.porosity[ix] = FNUM_1-(FNUM_1-mp.porosity[ix])/deco
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
        bc::VBoundary3D{T1, T2}, Ti::T2, ΔT, ::Val{:TS})

Description:
---
MPM update procedure with two-phase single-point model. (3D)
"""
@views function procedure!(args::     Args3D{T1, T2},
                           grid::     Grid3D{T1, T2},
                           mp  :: Particle3D{T1, T2},
                           bc  ::VBoundary3D{T1, T2}, 
                           Ti  ::T2,
                           ΔT  ::T2,
                               ::Val{:TS}) where {T1, T2}
    # specific values 
    FNUM_0 = T2(0.0); FNUM_1  = T2(1.0)
    INUM_2 = T1(2)  ; FNUM_13 = T2(1/3)
    # compute particle info
    mp.Ms .= mp.vol.*mp.ρs
    mp.Mw .= mp.vol.*mp.ρw
    mp.Mi .= mp.vol.*((FNUM_1.-mp.porosity).*mp.ρs.+mp.porosity.*mp.ρw)
    mp.Ps .= mp.Ms.*mp.Vs
    mp.Pw .= mp.Mw.*mp.Vw
    # reset particle values
    fill!(mp.tmp_Nx , FNUM_0)
    fill!(mp.tmp_Ny , FNUM_0)
    fill!(mp.tmp_Nz , FNUM_0)
    fill!(mp.tmp_dNx, FNUM_0)
    fill!(mp.tmp_dNy, FNUM_0)
    fill!(mp.tmp_dNz, FNUM_0)
    fill!(mp.∂Fs    , FNUM_0)
    fill!(mp.∂Fw    , FNUM_0)
    # reset grid values
    fill!(grid.Ms   , FNUM_0)
    fill!(grid.Mw   , FNUM_0)
    fill!(grid.Mi   , FNUM_0)
    fill!(grid.Ps   , FNUM_0)
    fill!(grid.Pw   , FNUM_0)
    fill!(grid.Fdrag, FNUM_0)
    fill!(grid.Fw   , FNUM_0)
    fill!(grid.Fs   , FNUM_0)
    fill!(grid.σm   , FNUM_0)
    fill!(grid.vol  , FNUM_0)
    fill!(grid.σw   , FNUM_0)
    # compute p2c, p2n, and shape function values
    basis_function!(grid, mp, Val(args.basis))
    # get current gravity (elastic loading)
    Ti<args.Te ? (gravity=args.gravity/args.Te*Ti) : (gravity=args.gravity)
    # particle to node mapping procedure
    gridF  = zeros(grid.node_num)
    maxval = grid.init[findmin(abs.(grid.init[:, 3].-maximum(mp.init[:, 3])))[2], 3]
    idx    = findall(i->(  grid.init[i, 3]==maxval)&&(
                        (0≤grid.init[i, 1]≤0.2)&&(0≤grid.init[i, 2]≤0.2)), 1:grid.node_num)
    gridF[idx] .= -40/length(idx)
    for j in 1:mp.NIC, i in 1:mp.num
        p2n = mp.p2n[i, j]
        ∂Nx = mp.∂Nx[i, j]
        ∂Ny = mp.∂Ny[i, j]
        ∂Nz = mp.∂Nz[i, j]
        por = mp.porosity[i]
        Ni  = mp.Ni[i, j]
        # scattering mass from particle to node
        grid.Ms[p2n] += Ni*mp.Ms[i]*(FNUM_1-por)
        grid.Mw[p2n] += Ni*mp.Mw[i]
        grid.Mi[p2n] += Ni*mp.Mw[i]*(       por)
        # scattering momentum from particle to node
        grid.Ps[p2n, 1] += Ni*mp.Ps[i, 1]*(FNUM_1-por)
        grid.Ps[p2n, 2] += Ni*mp.Ps[i, 2]*(FNUM_1-por)
        grid.Ps[p2n, 3] += Ni*mp.Ps[i, 3]*(FNUM_1-por)
        grid.Pw[p2n, 1] += Ni*mp.Pw[i, 1]*(       por)
        grid.Pw[p2n, 2] += Ni*mp.Pw[i, 2]*(       por)
        grid.Pw[p2n, 3] += Ni*mp.Pw[i, 3]*(       por)
        # compute nodal drag force
        # tmp = -(mp.Mw[i]*gravity*mp.porosity[i])/mp.k[i]
        # grid.Fdrag[p2n, 1] += Ni*tmp
        # grid.Fdrag[p2n, 2] += Ni*tmp
        # grid.Fdrag[p2n, 3] += Ni*tmp
        # compute nodal force (fw = fw_trac+fw_grav-fw_int) for water
        grid.Fw[p2n, 1] += -mp.vol[i]*∂Nx*mp.σw[i]
        grid.Fw[p2n, 2] += -mp.vol[i]*∂Ny*mp.σw[i]
        grid.Fw[p2n, 3] += -mp.vol[i]*∂Nz*mp.σw[i]#+Ni*mp.Mw[i]*gravity
        # compute nodal force for solid
        grid.Fs[p2n, 1] += -mp.vol[i]*(∂Nx*(mp.σij[i, 1]+mp.σw[i])+
                                       ∂Ny* mp.σij[i, 4]+
                                       ∂Nz* mp.σij[i, 6])
        grid.Fs[p2n, 2] += -mp.vol[i]*(∂Nx* mp.σij[i, 4]+
                                       ∂Ny*(mp.σij[i, 2]+mp.σw[i])+
                                       ∂Nz* mp.σij[i, 5])
        grid.Fs[p2n, 3] += -mp.vol[i]*(∂Nx* mp.σij[i, 6]+
                                       ∂Ny* mp.σij[i, 5]+
                                       ∂Nz*(mp.σij[i, 3]+mp.σw[i]))#+Ni*mp.Mi[i]*gravity
    end
    grid.Fs[:, 3] .+= gridF
    # compute nodal velocity
    grid.Vs .= grid.Ps./grid.Ms
    grid.Vw .= grid.Pw./grid.Mi
    grid.Vs[iszero.(grid.Ms), :] .= FNUM_0
    grid.Vw[iszero.(grid.Mi), :] .= FNUM_0
    # compute nodal drag force
    # grid.Fdrag .= grid.Fdrag.*(grid.Vw.-grid.Vs)
    # damping force for water
    # grid.Fw_damp .= -args.ζ.*sign.(grid.Vw).*sqrt.(
    #     grid.Fw[:, 1].^INUM_2.+grid.Fw[:, 2].^INUM_2.+grid.Fw[:, 3].^INUM_2
    # )
    # grid.Fw .= grid.Fw.+grid.Fw_damp.-grid.Fdrag
    # compute nodal accelaration for water
    grid.a_w .= grid.Fw./grid.Mw
    grid.a_w[iszero.(grid.Mw), :] .= FNUM_0
    # damping force for solid
    # grid.Fs_damp .= -args.ζ.*sign.(grid.Vs).*sqrt.(
    #     (grid.Fs[:, 1].-grid.Fw[:, 1]).^INUM_2.+
    #     (grid.Fs[:, 2].-grid.Fw[:, 2]).^INUM_2.+
    #     (grid.Fs[:, 3].-grid.Fw[:, 3]).^INUM_2
    # ) .+ grid.Fw_damp
    # compute nodal total force for mixture
    grid.Fs .= grid.Fs.-grid.Mi.*grid.a_w#.+grid.Fs_damp
    # compute nodal accelaration for solid
    grid.a_s .= grid.Fs./grid.Ms
    grid.a_s[iszero.(grid.Ms), :] .= FNUM_0
    # update nodal velocity
    grid.Vs_T .= (grid.Ps.+grid.Fs*ΔT)./grid.Ms
    grid.Vw_T .= (grid.Pw.+grid.Fw*ΔT)./grid.Mi
    grid.Vs_T[iszero.(grid.Ms), :] .= FNUM_0
    grid.Vw_T[iszero.(grid.Mi), :] .= FNUM_0
    # apply boundary conditions
    grid.Vs_T[bc.Vx_s_Idx, 1] .= bc.Vx_s_Val
    grid.Vs_T[bc.Vy_s_Idx, 2] .= bc.Vy_s_Val
    grid.Vs_T[bc.Vz_s_Idx, 3] .= bc.Vz_s_Val
    grid.Vw_T[bc.Vx_w_Idx, 1] .= bc.Vx_w_Val
    grid.Vw_T[bc.Vy_w_Idx, 2] .= bc.Vy_w_Val
    grid.Vw_T[bc.Vz_w_Idx, 3] .= bc.Vz_w_Val
    # update particle position & velocity
    for i in 1:mp.num
        tmp_vx_s1 = tmp_vx_s2 = tmp_vy_s1 = tmp_vy_s2 = tmp_vz_s1 = tmp_vz_s2 = FNUM_0
        tmp_vx_w1 = tmp_vx_w2 = tmp_vy_w1 = tmp_vy_w2 = tmp_vz_w1 = tmp_vz_w2 = FNUM_0
        for j in 1:mp.NIC
            Ni  = mp.Ni[i, j]
            p2n = mp.p2n[i, j]
            mp.pos[i, 1] += ΔT*(Ni*grid.Vs_T[p2n, 1])
            mp.pos[i, 2] += ΔT*(Ni*grid.Vs_T[p2n, 2])
            mp.pos[i, 3] += ΔT*(Ni*grid.Vs_T[p2n, 3])
            tmp_vx_s1 += Ni*(grid.Vs_T[p2n, 1]-grid.Vs[p2n, 1])
            tmp_vx_s2 += Ni* grid.Vs_T[p2n, 1]
            tmp_vy_s1 += Ni*(grid.Vs_T[p2n, 2]-grid.Vs[p2n, 2])
            tmp_vy_s2 += Ni* grid.Vs_T[p2n, 2]
            tmp_vz_s1 += Ni*(grid.Vs_T[p2n, 3]-grid.Vs[p2n, 3])
            tmp_vz_s2 += Ni* grid.Vs_T[p2n, 3]
            tmp_vx_w1 += Ni*(grid.Vw_T[p2n, 1]-grid.Vw[p2n, 1])
            tmp_vx_w2 += Ni* grid.Vw_T[p2n, 1]
            tmp_vy_w1 += Ni*(grid.Vw_T[p2n, 2]-grid.Vw[p2n, 2])
            tmp_vy_w2 += Ni* grid.Vw_T[p2n, 2]
            tmp_vz_w1 += Ni*(grid.Vw_T[p2n, 3]-grid.Vw[p2n, 3])
            tmp_vz_w2 += Ni* grid.Vw_T[p2n, 3]
        end
        mp.Vs[i, 1] = args.FLIP*(mp.Vs[i, 1]+tmp_vx_s1)+args.PIC*tmp_vx_s2
        mp.Vs[i, 2] = args.FLIP*(mp.Vs[i, 2]+tmp_vy_s1)+args.PIC*tmp_vy_s2
        mp.Vs[i, 3] = args.FLIP*(mp.Vs[i, 3]+tmp_vz_s1)+args.PIC*tmp_vz_s2
        mp.Vw[i, 1] = args.FLIP*(mp.Vw[i, 1]+tmp_vx_w1)+args.PIC*tmp_vx_w2
        mp.Vw[i, 2] = args.FLIP*(mp.Vw[i, 2]+tmp_vy_w1)+args.PIC*tmp_vy_w2
        mp.Vw[i, 3] = args.FLIP*(mp.Vw[i, 3]+tmp_vz_w1)+args.PIC*tmp_vz_w2
        # update CFL conditions
        # sqr = sqrt((mp.E[i]+mp.Kw[i]*mp.porosity[i])/(mp.ρs[i]*(FNUM_1-mp.porosity[i])+
        #                                               mp.ρw[i]*        mp.porosity[i]))
        # cd_sx = grid.space_x/(sqr+abs(mp.Vs[i, 1]))
        # cd_sy = grid.space_y/(sqr+abs(mp.Vs[i, 2]))
        # cd_sz = grid.space_z/(sqr+abs(mp.Vs[i, 3]))
        # cd_wx = grid.space_x/(sqr+abs(mp.Vw[i, 1]))
        # cd_wy = grid.space_y/(sqr+abs(mp.Vw[i, 2]))
        # cd_wz = grid.space_z/(sqr+abs(mp.Vw[i, 3]))
        # val   = min(cd_sx, cd_sy, cd_sz, cd_wx, cd_wy, cd_wz)
        # mp.cfl[i] = args.αT*val
        # update particle momentum
        mp.Ps[i, 1] = mp.Ms[i]*mp.Vs[i, 1]
        mp.Ps[i, 2] = mp.Ms[i]*mp.Vs[i, 2]
        mp.Ps[i, 3] = mp.Ms[i]*mp.Vs[i, 3]
        mp.Pw[i, 1] = mp.Mw[i]*mp.Vw[i, 1]
        mp.Pw[i, 2] = mp.Mw[i]*mp.Vw[i, 2]
        mp.Pw[i, 3] = mp.Mw[i]*mp.Vw[i, 3]
    end
    # update nodal momentum
    fill!(grid.Ps, FNUM_0)
    fill!(grid.Pw, FNUM_0)
    for j in 1:mp.NIC, i in 1:mp.num
        p2n = mp.p2n[i, j]
        por = mp.porosity[i]
        Ni  = mp.Ni[i, j]
        grid.Ps[p2n, 1] += Ni*mp.Ps[i, 1]*(FNUM_1-por)
        grid.Ps[p2n, 2] += Ni*mp.Ps[i, 2]*(FNUM_1-por)
        grid.Ps[p2n, 3] += Ni*mp.Ps[i, 3]*(FNUM_1-por)
        grid.Pw[p2n, 1] += Ni*mp.Pw[i, 1]*(       por)
        grid.Pw[p2n, 2] += Ni*mp.Pw[i, 2]*(       por)
        grid.Pw[p2n, 3] += Ni*mp.Pw[i, 3]*(       por)
    end
    # update nodal velocity for solid and water
    grid.Vs .= grid.Ps./grid.Ms
    grid.Vw .= grid.Pw./grid.Mi
    grid.Vs[iszero.(grid.Ms), :] .= FNUM_0
    grid.Vw[iszero.(grid.Mi), :] .= FNUM_0
    # apply boundary conditions
    grid.Vs[bc.Vx_s_Idx, 1] .= bc.Vx_s_Val
    grid.Vs[bc.Vy_s_Idx, 2] .= bc.Vy_s_Val
    grid.Vs[bc.Vz_s_Idx, 3] .= bc.Vz_s_Val
    grid.Vw[bc.Vx_w_Idx, 1] .= bc.Vx_w_Val
    grid.Vw[bc.Vy_w_Idx, 2] .= bc.Vy_w_Val
    grid.Vw[bc.Vz_w_Idx, 3] .= bc.Vz_w_Val
    # compute nodal displacement
    grid.Δd_s .= grid.Vs.*ΔT
    grid.Δd_w .= grid.Vw.*ΔT
    # compute incremental deformation gradient, update strain
    for j in 1:mp.NIC, i in 1:mp.num
        p2n = mp.p2n[i, j]
        ∂Nx = mp.∂Nx[i, j]
        ∂Ny = mp.∂Ny[i, j]
        ∂Nz = mp.∂Nz[i, j]
        # compute solid incremental deformation gradient
        mp.∂Fs[i, 1] += grid.Δd_s[p2n, 1]*∂Nx
        mp.∂Fs[i, 2] += grid.Δd_s[p2n, 1]*∂Ny
        mp.∂Fs[i, 3] += grid.Δd_s[p2n, 1]*∂Nz
        mp.∂Fs[i, 4] += grid.Δd_s[p2n, 2]*∂Nx
        mp.∂Fs[i, 5] += grid.Δd_s[p2n, 2]*∂Ny
        mp.∂Fs[i, 6] += grid.Δd_s[p2n, 2]*∂Nz
        mp.∂Fs[i, 7] += grid.Δd_s[p2n, 3]*∂Nx
        mp.∂Fs[i, 8] += grid.Δd_s[p2n, 3]*∂Ny
        mp.∂Fs[i, 9] += grid.Δd_s[p2n, 3]*∂Nz
        # compute water incremental deformation gradient
        mp.∂Fw[i, 1] += grid.Δd_w[p2n, 1]*∂Nx
        mp.∂Fw[i, 2] += grid.Δd_w[p2n, 1]*∂Ny
        mp.∂Fw[i, 3] += grid.Δd_w[p2n, 1]*∂Nz
        mp.∂Fw[i, 4] += grid.Δd_w[p2n, 2]*∂Nx
        mp.∂Fw[i, 5] += grid.Δd_w[p2n, 2]*∂Ny
        mp.∂Fw[i, 6] += grid.Δd_w[p2n, 2]*∂Nz
        mp.∂Fw[i, 7] += grid.Δd_w[p2n, 3]*∂Nx
        mp.∂Fw[i, 8] += grid.Δd_w[p2n, 3]*∂Ny
        mp.∂Fw[i, 9] += grid.Δd_w[p2n, 3]*∂Nz
    end
    for i in 1:mp.num
        # compute strain increment 
        mp.Δϵij_s[i, 1] = mp.∂Fs[i, 1]
        mp.Δϵij_s[i, 2] = mp.∂Fs[i, 5]
        mp.Δϵij_s[i, 3] = mp.∂Fs[i, 9]
        mp.Δϵij_s[i, 4] = mp.∂Fs[i, 2]+mp.∂Fs[i, 4]
        mp.Δϵij_s[i, 5] = mp.∂Fs[i, 6]+mp.∂Fs[i, 8]
        mp.Δϵij_s[i, 6] = mp.∂Fs[i, 3]+mp.∂Fs[i, 7]
        mp.Δϵij_w[i, 1] = mp.∂Fw[i, 1]
        mp.Δϵij_w[i, 2] = mp.∂Fw[i, 5]
        mp.Δϵij_w[i, 3] = mp.∂Fw[i, 9]
        mp.Δϵij_w[i, 4] = mp.∂Fw[i, 2]+mp.∂Fw[i, 4]
        mp.Δϵij_w[i, 5] = mp.∂Fw[i, 6]+mp.∂Fw[i, 8]
        mp.Δϵij_w[i, 6] = mp.∂Fw[i, 3]+mp.∂Fw[i, 7]
        # update strain tensor
        mp.ϵij_s[i, 1] += mp.Δϵij_s[i, 1]
        mp.ϵij_s[i, 2] += mp.Δϵij_s[i, 2]
        mp.ϵij_s[i, 3] += mp.Δϵij_s[i, 3]
        mp.ϵij_s[i, 4] += mp.Δϵij_s[i, 4]
        mp.ϵij_s[i, 5] += mp.Δϵij_s[i, 5]
        mp.ϵij_s[i, 6] += mp.Δϵij_s[i, 6] 
        mp.ϵij_w[i, 1] += mp.Δϵij_w[i, 1]
        mp.ϵij_w[i, 2] += mp.Δϵij_w[i, 2]
        mp.ϵij_w[i, 3] += mp.Δϵij_w[i, 3]
        mp.ϵij_w[i, 4] += mp.Δϵij_w[i, 4]
        mp.ϵij_w[i, 5] += mp.Δϵij_w[i, 5]
        mp.ϵij_w[i, 6] += mp.Δϵij_w[i, 6] 
        # update pore water pressure
        mp.σw[i] += (mp.Kw[i]/mp.porosity[i])*(
            (FNUM_1-mp.porosity[i])*(mp.Δϵij_s[i, 1]+mp.Δϵij_s[i, 2]+mp.Δϵij_s[i, 3])+
            (       mp.porosity[i])*(mp.Δϵij_w[i, 1]+mp.Δϵij_w[i, 2]+mp.Δϵij_w[i, 3]))
        # update jacobian matrix
        mp.J[i]   = FNUM_1+mp.Δϵij_s[i, 1]+mp.Δϵij_s[i, 2]+mp.Δϵij_s[i, 3]
        mp.vol[i] = mp.J[i]*mp.vol[i]
        # update porosity
        mp.porosity[i] = FNUM_1-(FNUM_1-mp.porosity[i])/mp.J[i]
    end
    # update by the constitutive model
    constitutive!(args, mp, Ti)
    # volumn locking
    if args.vollock == true
        for i in 1:mp.num
            p2c = mp.p2c[i]
            grid.σm[p2c]  += mp.vol[i]*mp.σm[i]
            grid.σw[p2c]  += mp.vol[i]*mp.σw[i]
            grid.vol[p2c] += mp.vol[i]
        end
        for i in 1:mp.num
            p2c = mp.p2c[i]
            σm  = grid.σm[p2c]/grid.vol[p2c]
            σw  = grid.σw[p2c]/grid.vol[p2c]
            mp.σij[i, 1] = mp.sij[i, 1]+σm
            mp.σij[i, 2] = mp.sij[i, 2]+σm
            mp.σij[i, 3] = mp.sij[i, 3]+σm
            mp.σij[i, 4] = mp.sij[i, 4]
            mp.σij[i, 5] = mp.sij[i, 5]
            mp.σij[i, 6] = mp.sij[i, 6]
            mp.σw[i] = σw
            # update mean stress tensor
            mp.σm[i] = (mp.σij[i, 1]+mp.σij[i, 2]+mp.σij[i, 3])*FNUM_13
            # update deviatoric stress tensor
            mp.sij[i, 1] = mp.σij[i, 1]-mp.σm[i]
            mp.sij[i, 2] = mp.σij[i, 2]-mp.σm[i]
            mp.sij[i, 3] = mp.σij[i, 3]-mp.σm[i]
            mp.sij[i, 4] = mp.σij[i, 4]
            mp.sij[i, 5] = mp.σij[i, 5]
            mp.sij[i, 6] = mp.σij[i, 6]
        end
    end
    return nothing
end
