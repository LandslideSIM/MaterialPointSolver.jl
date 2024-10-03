#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : boundary.jl                                                                |
|  Description: Type system for boundary in MaterialPointSolver.jl                         |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  License    : MIT License                                                                |
+==========================================================================================#

export AbstractBoundary
export DeviceVBoundary, DeviceVBoundary2D, DeviceVBoundary3D
export VBoundary2D, VBoundary3D
export UserVBoundary2D, UserVBoundary3D
export UserBoundaryExtra

abstract type AbstractBoundary end
abstract type DeviceVBoundary{T1, T2} <: AbstractBoundary end
abstract type DeviceVBoundary2D{T1, T2} <: DeviceVBoundary{T1, T2} end
abstract type DeviceVBoundary3D{T1, T2} <: DeviceVBoundary{T1, T2} end
abstract type UserBoundaryExtra end

struct TempBoundaryExtra{T1<:AbstractArray} <: UserBoundaryExtra
    i::T1
end

@user_struct TempBoundaryExtra

#=-----------------------------------------------------------------------------------------#
|    2D Boundary System                                                                    |
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓=#
struct VBoundary2D{T1, T2,
    T3 <: AbstractArray, # Array{Int  , 1}
    T4 <: AbstractArray, # Array{Float, 1}
    T5 <: UserBoundaryExtra
} <: DeviceVBoundary2D{T1, T2}
    vx_s_idx :: T3
    vx_s_val :: T4
    vy_s_idx :: T3
    vy_s_val :: T4
    vx_w_idx :: T3
    vx_w_val :: T4
    vy_w_idx :: T3
    vy_w_val :: T4
    smlength :: T2
    tmp1     :: T1
    tmp2     :: T2
    ext      :: T5
end

@user_struct VBoundary2D

function UserVBoundary2D(; ϵ="FP64", vx_s_idx, vx_s_val, vy_s_idx, vy_s_val, vx_w_idx=[0], 
    vx_w_val=[0], vy_w_idx=[0], vy_w_val=[0], smlength=0, tmp1=0, tmp2=0, ext=0)
    # input check
    length(vx_s_idx) == length(vx_s_val) && length(vy_s_idx) == length(vy_s_val) &&
    length(vx_w_idx) == length(vx_w_val) && length(vy_w_idx) == length(vy_w_val) ||
        throw(ArgumentError("The length of vx_s_idx, vx_s_val, vy_s_idx, vy_s_val, vx_w_idx, 
            vx_w_val, vy_w_idx, vy_s_val should be the same."))
    ext = ext == 0 ? TempBoundaryExtra(rand(2)) : ext
    ϵ == ϵ in ["FP64", "FP32"] ? ϵ : "FP64"
    T1 = ϵ == "FP64" ? Int64 : Int32
    T2 = ϵ == "FP64" ? Float64 : Float32
    tmp = VBoundary2D{T1, T2, AbstractArray{T1, 1}, AbstractArray{T2, 1}, 
        UserBoundaryExtra}(vx_s_idx, vx_s_val, vy_s_idx, vy_s_val, vx_w_idx, vx_w_val, 
        vy_w_idx, vy_w_val, smlength, tmp1, tmp2, ext)
    return user_adapt(Array, tmp)
end

function Base.show(io::IO, bc::T) where {T<:DeviceVBoundary2D}
    typeof(bc).parameters[2]==Float64 ? precision="FP64" : 
    typeof(bc).parameters[2]==Float32 ? precision="FP32" : nothing
    # printer
    print(io, "DeviceVBoundary2D:", "\n")
    print(io, "┬", "─" ^ 17       , "\n")
    print(io, "└─ ", "ϵ: "        , precision, "\n")
    return nothing
end
#=↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑=#



#=-----------------------------------------------------------------------------------------#
|    3D Boundary System                                                                    |
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓=#
struct VBoundary3D{T1, T2,
    T3 <: AbstractArray, # Array{Int  , 1}
    T4 <: AbstractArray, # Array{Float, 1}
    T5 <: UserBoundaryExtra
} <: DeviceVBoundary3D{T1, T2}
    vx_s_idx :: T3
    vx_s_val :: T4
    vy_s_idx :: T3
    vy_s_val :: T4
    vz_s_idx :: T3
    vz_s_val :: T4
    vx_w_idx :: T3
    vx_w_val :: T4
    vy_w_idx :: T3
    vy_w_val :: T4
    vz_w_idx :: T3
    vz_w_val :: T4
    smlength :: T2
    tmp1     :: T1
    tmp2     :: T2
    ext      :: T5
end

@user_struct VBoundary3D

function UserVBoundary3D(; ϵ="FP64", vx_s_idx, vx_s_val, vy_s_idx, vy_s_val, vz_s_idx, 
    vz_s_val, vx_w_idx=[0], vx_w_val=[0], vy_w_idx=[0], vy_w_val=[0], vz_w_idx=[0], 
    vz_w_val=[0], smlength=0, tmp1=0, tmp2=0, ext=0)
    # input check
    length(vx_s_idx) == length(vx_s_val) && length(vy_s_idx) == length(vy_s_val) &&
    length(vz_s_idx) == length(vz_s_val) && length(vz_w_idx) == length(vz_w_val) &&
    length(vx_w_idx) == length(vx_w_val) && length(vy_w_idx) == length(vy_w_val) ||
        throw(ArgumentError("The length of vx_s_idx, vx_s_val, vy_s_idx, vy_s_val, vz_s_idx, 
            vz_s_val, vx_w_idx, vx_w_val, vy_w_idx, vy_s_val, vz_w_idx, vz_w_val should be 
            the same."))
    ext = ext == 0 ? TempBoundaryExtra(rand(2)) : ext
    ϵ == ϵ in ["FP64", "FP32"] ? ϵ : "FP64"
    T1 = ϵ == "FP64" ? Int64 : Int32
    T2 = ϵ == "FP64" ? Float64 : Float32
    tmp = VBoundary3D{T1, T2, AbstractArray{T1, 1}, AbstractArray{T2, 1}, 
        UserBoundaryExtra}(vx_s_idx, vx_s_val, vy_s_idx, vy_s_val, vz_s_idx, vz_s_val, 
        vx_w_idx, vx_w_val, vy_w_idx, vy_w_val, vz_w_idx, vz_w_val, smlength, tmp1, tmp2, 
        ext)
    return user_adapt(Array, tmp)
end

function Base.show(io::IO, bc::T) where {T<:DeviceVBoundary3D}
    typeof(bc).parameters[2]==Float64 ? precision="FP64" : 
    typeof(bc).parameters[2]==Float32 ? precision="FP32" : nothing
    # printer
    print(io, "DeviceVBoundary3D:", "\n")
    print(io, "┬", "─" ^ 17       , "\n")
    print(io, "└─ ", "ϵ: "        , precision, "\n")
    return nothing
end
#=↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑=#