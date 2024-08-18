#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : particle.jl                                                                |
|  Description: Type system for particle in MaterialPointSolver.jl                         |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Struct     : 1. Particle2D                                                              |
|               2. GPUParticle2D                                                           |
|               3. Particle3D                                                              |
|               4. GPUParticle3D                                                           |
|               5. Base.show                                                               |
+==========================================================================================#

"""
    Particle2D{T1, T2}

Description:
---
This struct will save the values for 2D material particle.
"""
@kwdef struct Particle2D{T1, T2} <: KernelParticle2D{T1, T2}
    num     ::T1 = 0
    phase   ::T1
    NIC     ::T1 = 16
    space_x ::T2
    space_y ::T2
    p2c     ::Array{T1, 1} = [0]
    p2n     ::Array{Int32, 2} = [0 0]
    pos     ::Array{T2, 2}
    σm      ::Array{T2, 1} = [0]
    J       ::Array{T2, 1} = [0]
    epII    ::Array{T2, 1} = [0]
    epK     ::Array{T2, 1} = [0]
    vol     ::Array{T2, 1} = [0]
    vol_init::Array{T2, 1} = [0]
    Ms      ::Array{T2, 1} = [0]
    Mw      ::Array{T2, 1} = [0]
    Mi      ::Array{T2, 1} = [0]
    porosity::Array{T2, 1} = [0]
    cfl     ::Array{T2, 1} = [0]
    ρs      ::Array{T2, 1}
    ρs_init ::Array{T2, 1} = [0]
    ρw      ::Array{T2, 1} = [0]
    ρw_init ::Array{T2, 1} = [0]  
    init    ::Array{T2, 2} = [0 0]
    σw      ::Array{T2, 1} = [0]
    σij     ::Array{T2, 2} = [0 0]
    ϵij_s   ::Array{T2, 2} = [0 0]
    ϵij_w   ::Array{T2, 2} = [0 0]
    Δϵij_s  ::Array{T2, 2} = [0 0]
    Δϵij_w  ::Array{T2, 2} = [0 0]
    sij     ::Array{T2, 2} = [0 0]
    Vs      ::Array{T2, 2} = [0 0]
    Vw      ::Array{T2, 2} = [0 0]
    Ps      ::Array{T2, 2} = [0 0]
    Pw      ::Array{T2, 2} = [0 0]
    Ni      ::Array{T2, 2} = [0 0]
    ∂Nx     ::Array{T2, 2} = [0 0]
    ∂Ny     ::Array{T2, 2} = [0 0]
    ΔFs     ::Array{T2, 2} = [0 0]
    ΔFw     ::Array{T2, 2} = [0 0]
    F       ::Array{T2, 2} = [0 0]
    function Particle2D{T1, T2}(num, phase, NIC, space_x, space_y, p2c, p2n, pos, σm, J, 
        epII, epK, vol, vol_init, Ms, Mw, Mi, porosity, cfl, ρs, ρs_init, ρw, ρw_init, init, 
        σw, σij, ϵij_s, ϵij_w, Δϵij_s, Δϵij_w, sij, Vs, Vw, Ps, Pw, Ni, ∂Nx, ∂Ny, ΔFs, ΔFw, 
        F) where {T1, T2}
        # necessary input check
        DoF = 2
        NIC==4||NIC==16       ? nothing : error("Nodes in Cell can only be 4 or 16.")
        space_x!=zeros(T2, 1) ? nothing : error("Particle x_space is 0."            )
        space_y!=zeros(T2, 1) ? nothing : error("Particle y_space is 0."            )
        # particles properties setup
        num      = size(pos, 1)
        J        = ones(T2, num)
        vol_init = repeat([space_x*space_y], num)
        ρs_init  = copy(ρs)
        ρw_init  = copy(ρw)
        F        = repeat(T2[1 0 0 1] , num)
        vol      = copy(vol_init)
        init     = copy(pos)
        phase==1 ? (num_new=1  ; DoF_new=1  ) : 
        phase==2 ? (num_new=num; DoF_new=DoF) : nothing
        p2c      = Array{T1, 1}(calloc, num             )
        Ms       = Array{T1, 1}(calloc, num             )
        Mw       = Array{T1, 1}(calloc, num_new         )
        Mi       = Array{T1, 1}(calloc, num_new         )
        epII     = Array{T2, 1}(calloc, num             )
        epK      = Array{T2, 1}(calloc, num             )
        σw       = Array{T2, 1}(calloc, num_new         )
        σm       = Array{T2, 1}(calloc, num             )
        cfl      = Array{T2, 1}(calloc, num             )
        ϵij_s    = Array{T2, 2}(calloc, num    , 4      )
        ϵij_w    = Array{T2, 2}(calloc, num_new, 4      )
        σij      = Array{T2, 2}(calloc, num    , 4      )
        Δϵij_s   = Array{T2, 2}(calloc, num    , 4      )
        Δϵij_w   = Array{T2, 2}(calloc, num_new, 4      )
        sij      = Array{T2, 2}(calloc, num    , 4      )
        ΔFs      = Array{T2, 2}(calloc, num    , 4      )
        ΔFw      = Array{T2, 2}(calloc, num_new, 4      )
        p2n      = Array{Int32, 2}(calloc, num , NIC    )
        Ps       = Array{T2, 2}(calloc, num    , DoF    )
        Pw       = Array{T2, 2}(calloc, num_new, DoF_new)
        Vs       = Array{T2, 2}(calloc, num    , DoF    )
        Vw       = Array{T2, 2}(calloc, num_new, DoF_new)
        Ni       = Array{T2, 2}(calloc, num    , NIC    )
        ∂Nx      = Array{T2, 2}(calloc, num    , NIC    )
        ∂Ny      = Array{T2, 2}(calloc, num    , NIC    )
        # update struct
        new(num, phase, NIC, space_x, space_y, p2c, p2n, pos, σm, J, epII, epK, vol, 
            vol_init, Ms, Mw, Mi, porosity, cfl, ρs, ρs_init, ρw, ρw_init, init, σw, σij, 
            ϵij_s, ϵij_w, Δϵij_s, Δϵij_w, sij, Vs, Vw, Ps, Pw, Ni, ∂Nx, ∂Ny, ΔFs, ΔFw, F)
    end
end

"""
    struct GPUParticle2D{T1, T2, T3<:AbstractArray,
                                 T4<:AbstractArray,
                                 T5<:AbstractArray,
                                 T6<:AbstractArray}

Description:
---
Particle2D GPU struct. See [`Particle2D`](@ref) for more details.
"""
struct GPUParticle2D{T1, T2, T3<:AbstractArray,
                             T4<:AbstractArray,
                             T5<:AbstractArray,
                             T6<:AbstractArray} <: KernelParticle2D{T1, T2}
    num     ::T1
    phase   ::T1
    NIC     ::T1
    space_x ::T2
    space_y ::T2
    p2c     ::T3
    p2n     ::T4
    pos     ::T6
    σm      ::T5
    J       ::T5
    epII    ::T5
    epK     ::T5
    vol     ::T5
    vol_init::T5
    Ms      ::T5
    Mw      ::T5
    Mi      ::T5
    porosity::T5
    cfl     ::T5
    ρs      ::T5
    ρs_init ::T5
    ρw      ::T5
    ρw_init ::T5
    init    ::T6
    σw      ::T5
    σij     ::T6
    ϵij_s   ::T6
    ϵij_w   ::T6
    Δϵij_s  ::T6
    Δϵij_w  ::T6
    sij     ::T6
    Vs      ::T6
    Vw      ::T6
    Ps      ::T6
    Pw      ::T6
    Ni      ::T6
    ∂Nx     ::T6
    ∂Ny     ::T6
    ΔFs     ::T6
    ΔFw     ::T6
    F       ::T6
end

"""
    Particle3D{T1, T2}

Description:
---
This struct will save the values for 3D material particle.
"""
@kwdef struct Particle3D{T1, T2} <: KernelParticle3D{T1, T2}
    num     ::T1 = 0
    phase   ::T1
    NIC     ::T1 = 64
    space_x ::T2
    space_y ::T2
    space_z ::T2
    p2c     ::Array{T1, 1} = [0]
    p2n     ::Array{Int32, 2} = [0 0]
    pos     ::Array{T2, 2}
    σm      ::Array{T2, 1} = [0]
    J       ::Array{T2, 1} = [0]
    epII    ::Array{T2, 1} = [0]
    epK     ::Array{T2, 1} = [0]
    vol     ::Array{T2, 1} = [0]
    vol_init::Array{T2, 1} = [0]
    Ms      ::Array{T2, 1} = [0]
    Mw      ::Array{T2, 1} = [0]
    Mi      ::Array{T2, 1} = [0]
    porosity::Array{T2, 1} = [0]
    cfl     ::Array{T2, 1} = [0]
    ρs      ::Array{T2, 1}
    ρs_init ::Array{T2, 1} = [0]
    ρw      ::Array{T2, 1} = [0]
    ρw_init ::Array{T2, 1} = [0]
    init    ::Array{T2, 2} = [0 0]
    σw      ::Array{T2, 1} = [0]
    σij     ::Array{T2, 2} = [0 0]
    ϵij_s   ::Array{T2, 2} = [0 0]
    ϵij_w   ::Array{T2, 2} = [0 0]
    Δϵij_s  ::Array{T2, 2} = [0 0]
    Δϵij_w  ::Array{T2, 2} = [0 0]
    sij     ::Array{T2, 2} = [0 0]
    Vs      ::Array{T2, 2} = [0 0]
    Vw      ::Array{T2, 2} = [0 0]
    Ps      ::Array{T2, 2} = [0 0]
    Pw      ::Array{T2, 2} = [0 0]
    Ni      ::Array{T2, 2} = [0 0]
    ∂Nx     ::Array{T2, 2} = [0 0]
    ∂Ny     ::Array{T2, 2} = [0 0]
    ∂Nz     ::Array{T2, 2} = [0 0]
    ΔFs     ::Array{T2, 2} = [0 0]
    ΔFw     ::Array{T2, 2} = [0 0]
    F       ::Array{T2, 2} = [0 0]
    function Particle3D{T1, T2}(num, phase, NIC, space_x, space_y, space_z, p2c, p2n, pos, 
        σm, J, epII, epK, vol, vol_init, Ms, Mw, Mi, porosity, cfl, ρs, ρs_init, ρw, 
        ρw_init, init, σw, σij, ϵij_s, ϵij_w, Δϵij_s, Δϵij_w, sij, Vs, Vw, Ps, Pw, Ni, ∂Nx, 
        ∂Ny, ∂Nz, ΔFs, ΔFw, F) where {T1, T2}
        # necessary input check
        DoF = 3
        NIC==8||NIC==64 ? nothing : error("Nodes in Cell can only be 8 or 64." )
        space_x!=T2(0)  ? nothing : error("Particle x_space is 0."             )
        space_y!=T2(0)  ? nothing : error("Particle y_space is 0."             )
        space_z!=T2(0)  ? nothing : error("Particle z_space is 0."             )
        # particles properties setup
        num      = size(pos, 1)
        vol_init = repeat([space_x*space_y*space_z], num)
        ρs_init  = copy(ρs)
        ρw_init  = copy(ρw)
        F        = repeat(T2[1 0 0 0 1 0 0 0 1] , num)
        J        = ones(T2, num)
        vol      = copy(vol_init)
        init     = copy(pos)
        phase==1 ? (num_new=1  ; DoF_new=1  ) : 
        phase==2 ? (num_new=num; DoF_new=DoF) : nothing
        p2c      = Array{T1, 1}(calloc, num             )
        Ms       = Array{T1, 1}(calloc, num             )
        Mw       = Array{T1, 1}(calloc, num_new         )
        Mi       = Array{T1, 1}(calloc, num_new         )
        epII     = Array{T2, 1}(calloc, num             )
        epK      = Array{T2, 1}(calloc, num             )
        σw       = Array{T2, 1}(calloc, num_new         )
        σm       = Array{T2, 1}(calloc, num             )
        cfl      = Array{T2, 1}(calloc, num             )
        ϵij_s    = Array{T2, 2}(calloc, num    , 6      )
        ϵij_w    = Array{T2, 2}(calloc, num_new, 6      )
        σij      = Array{T2, 2}(calloc, num    , 6      )
        Δϵij_s   = Array{T2, 2}(calloc, num    , 6      )
        Δϵij_w   = Array{T2, 2}(calloc, num_new, 6      )   
        sij      = Array{T2, 2}(calloc, num    , 6      )
        ΔFs      = Array{T2, 2}(calloc, num    , 9      )
        ΔFw      = Array{T2, 2}(calloc, num_new, 9      )
        p2n      = Array{Int32, 2}(calloc, num , NIC    ) 
        Ps       = Array{T2, 2}(calloc, num    , DoF    )
        Pw       = Array{T2, 2}(calloc, num_new, DoF_new)
        Vs       = Array{T2, 2}(calloc, num    , DoF    )
        Vw       = Array{T2, 2}(calloc, num_new, DoF_new)
        Ni       = Array{T2, 2}(calloc, num    , NIC    )
        ∂Nx      = Array{T2, 2}(calloc, num    , NIC    )
        ∂Ny      = Array{T2, 2}(calloc, num    , NIC    )
        ∂Nz      = Array{T2, 2}(calloc, num    , NIC    )
        # update struct
        new(num, phase, NIC, space_x, space_y, space_z, p2c, p2n, pos, σm, J, epII, epK, 
            vol, vol_init, Ms, Mw, Mi, porosity, cfl, ρs, ρs_init, ρw, ρw_init, init, σw, 
            σij, ϵij_s, ϵij_w, Δϵij_s, Δϵij_w, sij, Vs, Vw, Ps, Pw, Ni, ∂Nx, ∂Ny, ∂Nz, ΔFs, 
            ΔFw, F)
    end
end

"""
    struct GPUParticle3D{T1, T2, T3<:AbstractArray,
                                 T4<:AbstractArray,
                                 T5<:AbstractArray,
                                 T6<:AbstractArray}

Description:
---
Particle3D GPU struct. See [`Particle3D`](@ref) for more details.
"""
struct GPUParticle3D{T1, T2, T3<:AbstractArray,
                             T4<:AbstractArray,
                             T5<:AbstractArray,
                             T6<:AbstractArray} <: KernelParticle3D{T1, T2}
    num     ::T1
    phase   ::T1
    NIC     ::T1
    space_x ::T2
    space_y ::T2
    space_z ::T2
    p2c     ::T3
    p2n     ::T4
    pos     ::T6
    σm      ::T5
    J       ::T5
    epII    ::T5
    epK     ::T5
    vol     ::T5
    vol_init::T5
    Ms      ::T5
    Mw      ::T5
    Mi      ::T5
    porosity::T5
    cfl     ::T5
    ρs      ::T5
    ρs_init ::T5
    ρw      ::T5
    ρw_init ::T5
    init    ::T6
    σw      ::T5
    σij     ::T6
    ϵij_s   ::T6
    ϵij_w   ::T6
    Δϵij_s  ::T6
    Δϵij_w  ::T6
    sij     ::T6
    Vs      ::T6
    Vw      ::T6
    Ps      ::T6
    Pw      ::T6
    Ni      ::T6
    ∂Nx     ::T6
    ∂Ny     ::T6
    ∂Nz     ::T6
    ΔFs     ::T6
    ΔFw     ::T6
    F       ::T6
end

function Base.show(io::IO, mp::PARTICLE)
      print(io, typeof(mp)                    , "\n")
      print(io, "─"^length(string(typeof(mp))), "\n")
      print(io, "particle: ", mp.num          , "\n")
end