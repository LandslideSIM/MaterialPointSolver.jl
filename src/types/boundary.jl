#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : boundary.jl                                                                |
|  Description: Type system for boundary conditions in MaterialPointSolver.jl              |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Struct     : 1. VBoundary2D                                                             |
|               2. KernelVBoundary2D                                                       |
|               3. VBoundary3D                                                             |
|               4. KernelVBoundary3D                                                       |
|               5. Base.show                                                               |
+==========================================================================================#

export VBoundary2D, VBoundary3D
export GPUVBoundary2D, GPUVBoundary3D

"""
    VBoundary2D{T1, T2}

Description:
---
This struct will save the values for 2D velocity boundary conditions.
"""
@kwdef struct VBoundary2D{T1, T2} <: KernelBoundary2D{T1, T2}
    Vx_s_Idx::Array{T1, 1} = [0]
    Vx_s_Val::Array{T2, 1} = [0]
    Vy_s_Idx::Array{T1, 1} = [0]
    Vy_s_Val::Array{T2, 1} = [0]
    Vx_w_Idx::Array{T1, 1} = [0]
    Vx_w_Val::Array{T2, 1} = [0]
    Vy_w_Idx::Array{T1, 1} = [0]
    Vy_w_Val::Array{T2, 1} = [0]
    smdomain::Array{T2, 1} = [0, 0, 0, 0]
    smlength::T2           = T2(0)
    tmp1    ::T1           = T1(0)
    tmp2    ::T2           = T2(0)
end

"""
    struct GPUVBoundary2D{T1, T2, T3<:AbstractArray, T4<:AbstractArray}

Description:
---
VBoundary2D GPU struct. See [`VBoundary2D`](@ref) for more details.
"""
struct GPUVBoundary2D{T1,
                      T2,
                      T3<:AbstractArray, 
                      T4<:AbstractArray} <: KernelBoundary2D{T1, T2}
    Vx_s_Idx::T3
    Vx_s_Val::T4
    Vy_s_Idx::T3
    Vy_s_Val::T4
    Vx_w_Idx::T3
    Vx_w_Val::T4
    Vy_w_Idx::T3
    Vy_w_Val::T4
    smdomain::T4
    smlength::T2
    tmp1    ::T1
    tmp2    ::T2
end

"""
    VBoundary3D{T1, T2}

Description:
---
This struct will save the values for 3D velocity boundary conditions.
"""
@kwdef struct VBoundary3D{T1, T2} <: KernelBoundary3D{T1, T2}
    Vx_s_Idx::Array{T1, 1} = [0]
    Vx_s_Val::Array{T2, 1} = [0]
    Vy_s_Idx::Array{T1, 1} = [0]
    Vy_s_Val::Array{T2, 1} = [0]
    Vz_s_Idx::Array{T1, 1} = [0]
    Vz_s_Val::Array{T2, 1} = [0]
    Vx_w_Idx::Array{T1, 1} = [0]
    Vx_w_Val::Array{T2, 1} = [0]
    Vy_w_Idx::Array{T1, 1} = [0]
    Vy_w_Val::Array{T2, 1} = [0]
    Vz_w_Idx::Array{T1, 1} = [0]
    Vz_w_Val::Array{T2, 1} = [0]
    tmp1::T1               = T1(0)
    tmp2::T2               = T2(0)
end

"""
    struct GPUVBoundary3D{T1, T2, T3<:AbstractArray, T4<:AbstractArray}

Description:
---
VBoundary3D GPU struct. See [`VBoundary3D`](@ref) for more details.
"""
struct GPUVBoundary3D{T1,
                      T2,
                      T3<:AbstractArray, 
                      T4<:AbstractArray} <: KernelBoundary3D{T1, T2}
    Vx_s_Idx::T3
    Vx_s_Val::T4
    Vy_s_Idx::T3
    Vy_s_Val::T4
    Vz_s_Idx::T3
    Vz_s_Val::T4
    Vx_w_Idx::T3
    Vx_w_Val::T4
    Vy_w_Idx::T3
    Vy_w_Val::T4
    Vz_w_Idx::T3
    Vz_w_Val::T4
    tmp1    ::T1
    tmp2    ::T2
end

function Base.show(io::IO, bc::BOUNDARY)
    print(io, typeof(bc)                    , "\n")
    print(io, "─"^length(string(typeof(bc))), "\n")
    print(io, "velocity boundary"           , "\n")
end