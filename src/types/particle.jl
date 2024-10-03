#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : particle.jl                                                                |
|  Description: Type system for particle in MaterialPointSolver.jl                         |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  License    : MIT License                                                                |
+==========================================================================================#

export AbstractParticle
export DeviceParticle, DeviceParticle2D, DeviceParticle3D
export Particle2D, Particle3D
export UserParticle2D, UserParticle3D
export UserParticleExtra

abstract type AbstractParticle end
abstract type DeviceParticle{T1, T2} <: AbstractParticle end
abstract type DeviceParticle2D{T1, T2} <: DeviceParticle{T1, T2} end
abstract type DeviceParticle3D{T1, T2} <: DeviceParticle{T1, T2} end
abstract type UserParticleExtra end

struct TempParticleExtra{T1<:AbstractArray} <: UserParticleExtra
    i::T1
end

@user_struct TempParticleExtra

#=-----------------------------------------------------------------------------------------#
|    2D Particle System                                                                    |
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓=#
struct Particle2D{T1, T2,
    T3 <: AbstractArray, # Array{Int  , 1}
    T4 <: AbstractArray, # Array{Int  , 2}
    T5 <: AbstractArray, # Array{Float, 1}
    T6 <: AbstractArray, # Array{Float, 2}
    T7 <: UserParticleExtra,
} <: DeviceParticle2D{T1, T2}
    phase :: T1
    np    :: T1
    NIC   :: T1
    dx    :: T2
    dy    :: T2
    p2c   :: T3
    p2n   :: T4
    ξ     :: T6
    ξ0    :: T6
    σm    :: T5
    ϵq    :: T5
    ϵk    :: T5
    ϵv    :: T5
    Ω     :: T5
    Ω0    :: T5
    ms    :: T5
    mw    :: T5
    mi    :: T5
    n     :: T5
    ρs    :: T5
    ρs0   :: T5
    ρw    :: T5
    ρw0   :: T5
    σw    :: T5
    σij   :: T6
    ϵijs  :: T6
    ϵijw  :: T6
    Δϵijs :: T6
    Δϵijw :: T6
    sij   :: T6
    vs    :: T6
    vw    :: T6
    ps    :: T6
    pw    :: T6
    Nij   :: T6
    ∂Nx   :: T6
    ∂Ny   :: T6
    ΔFs   :: T6
    ΔFw   :: T6
    F     :: T6
    ext   :: T7
end

@user_struct Particle2D

function UserParticle2D(; ϵ="FP64", phase=1, NIC=16, dx, dy, ξ, n=[0], ρs, ρw=[0], ext=0)
    # input check
    dx > 0 && dy > 0 || throw(ArgumentError("dx and dy should be positive"))
    # default values
    phase = phase in [1, 2] ? phase : 1
    NIC = NIC in [4, 16] ? NIC : 16
    ext = ext == 0 ? TempParticleExtra(rand(2)) : ext
    ϵ == ϵ in ["FP64", "FP32"] ? ϵ : "FP64"
    T1 = ϵ == "FP64" ? Int64 : Int32
    T2 = ϵ == "FP64" ? Float64 : Float32
    # particles properties setup
    np  = size(ξ, 1)
    Ω0  = repeat([dx * dy], np)
    Ω   = copy(Ω0)
    F   = repeat(T2[1 0 0 1] , np)
    ξ0  = copy(ξ)
    DoF = 2
    size(ξ, 2) == DoF || throw(ArgumentError("ξ should have $(DoF) columns"))
    phase == 1 ? (np_new = 1 ; DoF_new = 1  ) :
    phase == 2 ? (np_new = np; DoF_new = DoF) : nothing
    ρs0   = copy(ρs)
    ρw0   = copy(ρw)
    p2c   = zeros(T1, np             )
    ms    = zeros(T1, np             )
    mw    = zeros(T1, np_new         )
    mi    = zeros(T1, np_new         )
    ϵq    = zeros(T2, np             )
    ϵk    = zeros(T2, np             )
    ϵv    = zeros(T2, np             )
    σw    = zeros(T2, np_new         )
    σm    = zeros(T2, np             )
    ϵijs  = zeros(T2, np    , 4      )
    ϵijw  = zeros(T2, np_new, 4      )
    σij   = zeros(T2, np    , 4      )
    Δϵijs = zeros(T2, np    , 4      )
    Δϵijw = zeros(T2, np_new, 4      )
    sij   = zeros(T2, np    , 4      )
    ΔFs   = zeros(T2, np    , 4      )
    ΔFw   = zeros(T2, np_new, 4      )
    p2n   = zeros(T1, np    , NIC    )
    ps    = zeros(T2, np    , DoF    )
    pw    = zeros(T2, np_new, DoF_new)
    vs    = zeros(T2, np    , DoF    )
    vw    = zeros(T2, np_new, DoF_new)
    Nij   = zeros(T2, np    , NIC    )
    ∂Nx   = zeros(T2, np    , NIC    )
    ∂Ny   = zeros(T2, np    , NIC    )

    tmp = Particle2D{T1, T2, AbstractArray{T1, 1}, AbstractArray{T1, 2}, 
        AbstractArray{T2, 1}, AbstractArray{T2, 2}, UserParticleExtra}(phase, np, NIC, dx, 
        dy, p2c, p2n, ξ, ξ0, σm, ϵq, ϵk, ϵv, Ω, Ω0, ms, mw, mi, n, ρs, ρs0, ρw, ρw0, σw, 
        σij, ϵijs, ϵijw, Δϵijs, Δϵijw, sij, vs, vw, ps, pw, Nij, ∂Nx, ∂Ny, ΔFs, ΔFw, F, ext)
    return user_adapt(Array, tmp)
end

function Base.show(io::IO, mp::T) where {T<:DeviceParticle2D}
    typeof(mp).parameters[2]==Float64 ? precision="FP64" : 
    typeof(mp).parameters[2]==Float32 ? precision="FP32" : nothing
    # printer
    print(io, "DeviceParticle2D:"                        , "\n")
    print(io, "┬", "─" ^ 16                              , "\n")
    print(io, "├─ ", "phase  : ", mp.phase               , "\n")
    print(io, "├─ ", "NIC    : ", mp.NIC                 , "\n")
    print(io, "├─ ", "ϵ      : ", precision              , "\n")
    print(io, "├─ ", "dx - dy: ", "$(mp.dx) - $(mp.dy)"  , "\n")
    print(io, "└─ ", "np     : ", @sprintf("%.2e", mp.np), "\n")
    return nothing
end
#=↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑=#



#=-----------------------------------------------------------------------------------------#
|    3D Particle System                                                                    |
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓=#
struct Particle3D{T1, T2,
    T3 <: AbstractArray, # Array{Int  , 1}
    T4 <: AbstractArray, # Array{Int  , 2}
    T5 <: AbstractArray, # Array{Float, 1}
    T6 <: AbstractArray, # Array{Float, 2}
    T7 <: UserParticleExtra,
} <: DeviceParticle3D{T1, T2}
    phase :: T1
    np    :: T1
    NIC   :: T1
    dx    :: T2
    dy    :: T2
    dz    :: T2
    p2c   :: T3
    p2n   :: T4
    ξ     :: T6
    ξ0    :: T6
    σm    :: T5
    ϵq    :: T5
    ϵk    :: T5
    ϵv    :: T5
    Ω     :: T5
    Ω0    :: T5
    ms    :: T5
    mw    :: T5
    mi    :: T5
    n     :: T5
    ρs    :: T5
    ρs0   :: T5
    ρw    :: T5
    ρw0   :: T5
    σw    :: T5
    σij   :: T6
    ϵijs  :: T6
    ϵijw  :: T6
    Δϵijs :: T6
    Δϵijw :: T6
    sij   :: T6
    vs    :: T6
    vw    :: T6
    ps    :: T6
    pw    :: T6
    Nij   :: T6
    ∂Nx   :: T6
    ∂Ny   :: T6
    ∂Nz   :: T6
    ΔFs   :: T6
    ΔFw   :: T6
    F     :: T6
    ext   :: T7
end

@user_struct Particle3D

function UserParticle3D(; ϵ="FP64", phase=1, NIC=16, dx, dy, dz, ξ, n=[0], ρs, ρw=[0], 
    ext=0)
    # input check
    dx > 0 && dy > 0 && dz > 0 || 
        throw(ArgumentError("dx, dy, and dz should be positive"))
    # default values
    phase = phase in [1, 2] ? phase : 1
    NIC = NIC in [8, 64] ? NIC : 64
    ext = ext == 0 ? TempParticleExtra(rand(2)) : ext
    ϵ == ϵ in ["FP64", "FP32"] ? ϵ : "FP64"
    T1 = ϵ == "FP64" ? Int64 : Int32
    T2 = ϵ == "FP64" ? Float64 : Float32
    # particles properties setup
    np  = size(ξ, 1)
    Ω0  = repeat([dx * dy * dz], np)
    Ω   = copy(Ω0)
    F   = repeat(T2[1 0 0 0 1 0 0 0 1] , np)
    ξ0  = copy(ξ)
    DoF = 3
    size(ξ, 2) == DoF || throw(ArgumentError("ξ should have $(DoF) columns"))
    phase == 1 ? (np_new = 1 ; DoF_new = 1  ) :
    phase == 2 ? (np_new = np; DoF_new = DoF) : nothing
    ρs0   = copy(ρs)
    ρw0   = copy(ρw)
    p2c   = zeros(T1, np             )
    ms    = zeros(T1, np             )
    mw    = zeros(T1, np_new         )
    mi    = zeros(T1, np_new         )
    ϵq    = zeros(T2, np             )
    ϵk    = zeros(T2, np             )
    ϵv    = zeros(T2, np             )
    σw    = zeros(T2, np_new         )
    σm    = zeros(T2, np             )
    ϵijs  = zeros(T2, np    , 6      )
    ϵijw  = zeros(T2, np_new, 6      )
    σij   = zeros(T2, np    , 6      )
    Δϵijs = zeros(T2, np    , 6      )
    Δϵijw = zeros(T2, np_new, 6      )
    sij   = zeros(T2, np    , 6      )
    ΔFs   = zeros(T2, np    , 9      )
    ΔFw   = zeros(T2, np_new, 9      )
    p2n   = zeros(T1, np    , NIC    )
    ps    = zeros(T2, np    , DoF    )
    pw    = zeros(T2, np_new, DoF_new)
    vs    = zeros(T2, np    , DoF    )
    vw    = zeros(T2, np_new, DoF_new)
    Nij   = zeros(T2, np    , NIC    )
    ∂Nx   = zeros(T2, np    , NIC    )
    ∂Ny   = zeros(T2, np    , NIC    )
    ∂Nz   = zeros(T2, np    , NIC    )

    tmp = Particle3D{T1, T2, AbstractArray{T1, 1}, AbstractArray{T1, 2}, 
        AbstractArray{T2, 1}, AbstractArray{T2, 2}, UserParticleExtra}(phase, np, NIC, dx, 
        dy, dz, p2c, p2n, ξ, ξ0, σm, ϵq, ϵk, ϵv, Ω, Ω0, ms, mw, mi, n, ρs, ρs0, ρw, ρw0, σw, 
        σij, ϵijs, ϵijw, Δϵijs, Δϵijw, sij, vs, vw, ps, pw, Nij, ∂Nx, ∂Ny, ∂Nz, ΔFs, ΔFw, F, 
        ext)
    return user_adapt(Array, tmp)
end

function Base.show(io::IO, mp::T) where {T<:DeviceParticle3D}
    typeof(mp).parameters[2]==Float64 ? precision="FP64" : 
    typeof(mp).parameters[2]==Float32 ? precision="FP32" : nothing
    # printer
    print(io, "DeviceParticle3D:"                                  , "\n")
    print(io, "┬", "─" ^ 16                                        , "\n")
    print(io, "├─ ", "phase  : ", mp.phase                         , "\n")
    print(io, "├─ ", "NIC    : ", mp.NIC                           , "\n")
    print(io, "├─ ", "ϵ      : ", precision                        , "\n")
    print(io, "├─ ", "d x-y-z: ", "$(mp.dx) - $(mp.dy) - $(mp.dz)" , "\n")
    print(io, "└─ ", "np     : ", @sprintf("%.2e", mp.np)          , "\n")
    return nothing
end
#=↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑=#