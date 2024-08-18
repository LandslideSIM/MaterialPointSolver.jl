#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : property.jl                                                                |
|  Description: Type system for particle's property in MaterialPointSolver.jl              |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Struct     : 1. ParticleProperty                                                        |
|               2. GPUParticleProperty                                                     |
|               3. Base.show                                                               |
+==========================================================================================#

"""
    ParticleProperty{T1, T2}

Description:
---
This struct will save the material properties for material particle.
"""
@kwdef struct ParticleProperty{T1, T2} <: KernelParticleProperty{T1, T2}
    layer::Array{T1, 1}
    ν    ::Array{T2, 1}
    E    ::Array{T2, 1}
    G    ::Array{T2, 1}
    Ks   ::Array{T2, 1}
    Kw   ::Array{T2, 1} = [0]
    k    ::Array{T2, 1} = [0]
    σt   ::Array{T2, 1} = [0] # tensile strength
    ϕ    ::Array{T2, 1} = [0] # friction angle ϕ
    ψ    ::Array{T2, 1} = [0] # dilation angle ψ     
    c    ::Array{T2, 1} = [0] # cohesion
    cr   ::Array{T2, 1} = [0] # residual cohesion
    Hp   ::Array{T2, 1} = [0] # softening modulus
    tmp1 ::T1           = T1(0)
    tmp2 ::T2           = T2(0)
    function ParticleProperty{T1, T2}(layer, ν, E, G, Ks, Kw, k, σt, ϕ, ψ, c, cr, Hp, tmp1, 
        tmp2) where {T1, T2}
        minimum(layer) == 1 ? nothing : error("Wrong layer index.")
        new(layer, ν, E, G, Ks, Kw, k, σt, ϕ, ψ, c, cr, Hp, tmp1, tmp2)
    end
end

"""
    GPUParticleProperty{T1, T2, 
                        T3<:AbstractArray, 
                        T4<:AbstractArray} <: KernelParticleProperty{T1, T2}

Description:
---
ParticleProperty GPU struct. See [`ParticleProperty`](@ref) for more details.
"""
struct GPUParticleProperty{T1, T2, T3<:AbstractArray, T4<:AbstractArray} <: KernelParticleProperty{T1, T2}
    layer::T3
    ν    ::T4
    E    ::T4
    G    ::T4
    Ks   ::T4
    Kw   ::T4
    k    ::T4
    σt   ::T4
    ϕ    ::T4
    ψ    ::T4     
    c    ::T4
    cr   ::T4
    Hp   ::T4
    tmp1 ::T1
    tmp2 ::T2
end

function Base.show(io::IO, pts_attr::PROPERTY)
    print(io, typeof(pts_attr)                    , "\n")
    print(io, "─"^length(string(typeof(pts_attr))), "\n")
    print(io, "material partition: ", maximum(pts_attr.layer), "\n")
end