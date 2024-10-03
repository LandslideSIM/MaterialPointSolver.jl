#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : property.jl                                                                |
|  Description: Type system for property in MaterialPointSolver.jl                         |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  License    : MIT License                                                                |
+==========================================================================================#

export AbstractProperty
export DeviceProperty
export Property
export UserProperty
export UserPropertyExtra

abstract type AbstractProperty end
abstract type DeviceProperty{T1, T2} <: AbstractProperty end
abstract type UserPropertyExtra end

struct TempPropertyExtra{T1<:AbstractArray} <: UserPropertyExtra
    i::T1
end

@user_struct TempPropertyExtra

#=-----------------------------------------------------------------------------------------#
|    2D Property System                                                                    |
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓=#
struct Property{T1, T2, 
    T3 <: AbstractArray, # Array{Int  , 1}
    T4 <: AbstractArray, # Array{Float, 1}
    T5 <: UserPropertyExtra
} <: DeviceProperty{T1, T2}
    tmp1 :: T1
    tmp2 :: T2
    nid  :: T3
    ν    :: T4
    Es   :: T4
    Gs   :: T4
    Ks   :: T4
    Kw   :: T4
    k    :: T4
    σt   :: T4
    ϕ    :: T4
    ϕr   :: T4
    ψ    :: T4     
    c    :: T4
    cr   :: T4
    Hp   :: T4
    ext  :: T5
end

@user_struct Property

function UserProperty(; ϵ="FP64", nid, ν, Es, Gs, Ks, Kw=[0], k=[0], σt=[0], ϕ=[0], ϕr=[0], 
    ψ=[0], c=[0], cr=[0], Hp=[0], ext=0)
    # input check
    length(Es) == length(Gs) == length(Ks) == length(ν) || 
        throw(ArgumentError("The length of Es, Gs, Ks, and ν must be the same."))
    length(unique(nid)) == length(Es) || 
        throw(ArgumentError("nid layer must be the same as the length of properties."))

    ext = ext == 0 ? TempPropertyExtra(rand(2)) : ext
    ϵ == ϵ in ["FP64", "FP32"] ? ϵ : "FP64"
    T1 = ϵ == "FP64" ? Int64 : Int32
    T2 = ϵ == "FP64" ? Float64 : Float32
    tmp = Property{T1, T2, AbstractArray{T1, 1}, AbstractArray{T2, 1}, UserPropertyExtra}(
        0, 0.0, nid, ν, Es, Gs, Ks, Kw, k, σt, ϕ, ϕr, ψ, c, cr, Hp, ext)
    return adapt(Array, tmp)
end

function Base.show(io::IO, attr::T) where {T<:DeviceProperty}
    typeof(attr).parameters[2]==Float64 ? precision="FP64" : 
    typeof(attr).parameters[2]==Float32 ? precision="FP32" : nothing
    # printer
    print(io, "DeviceProperty:"      , "\n")
    print(io, "┬", "─" ^ 14          , "\n")
    print(io, "└─ ", "ϵ: ", precision, "\n")
    return nothing
end
#=↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑=#