#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : type.jl                                                                    |
|  Description: Type system in MPMSolver.jl                                                |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
+==========================================================================================#

# abstract types
export MODELARGS, GRID, PARTICLE, PROPERTY, BOUNDARY
# Parent types
abstract type MODELARGS end
abstract type      GRID end
abstract type  PARTICLE end
abstract type  BOUNDARY end
abstract type  PROPERTY end

# kernel types for user
export KernelGrid2D, KernelGrid3D, KernelParticle2D, KernelParticle3D, 
       KernelParticleProperty, KernelBoundary2D, KernelBoundary3D
# Child types for 2D and 3D
abstract type           KernelGrid2D{T1, T2} <:     GRID end
abstract type           KernelGrid3D{T1, T2} <:     GRID end
abstract type       KernelParticle2D{T1, T2} <: PARTICLE end
abstract type       KernelParticle3D{T1, T2} <: PARTICLE end
abstract type       KernelBoundary2D{T1, T2} <: BOUNDARY end
abstract type       KernelBoundary3D{T1, T2} <: BOUNDARY end
abstract type KernelParticleProperty{T1, T2} <: PROPERTY end

include(joinpath(@__DIR__, "types/modelargs.jl"))
include(joinpath(@__DIR__, "types/grid.jl"     ))
include(joinpath(@__DIR__, "types/particle.jl" ))
include(joinpath(@__DIR__, "types/boundary.jl" ))
include(joinpath(@__DIR__, "types/property.jl" ))

# union device concreate types
export GPUGRID, GPUPARTICLE, GPUPARTICLEPROPERTY, GPUBOUNDARY
const GPUGRID             = Union{     GPUGrid2D,      GPUGrid3D}
const GPUPARTICLE         = Union{ GPUParticle2D,  GPUParticle3D}
const GPUBOUNDARY         = Union{GPUVBoundary2D, GPUVBoundary3D}
const GPUPARTICLEPROPERTY = Union{GPUParticleProperty}

# union concreate types for 2/3D 
export MODELARGSD, GRIDD, PARTICLED, PROPERTYD, BOUNDARYD
const MODELARGSD{T1, T2} = Union{          Args2D{T1, T2},      Args3D{T1, T2}}
const      GRIDD{T1, T2} = Union{          Grid2D{T1, T2},      Grid3D{T1, T2}}
const  PARTICLED{T1, T2} = Union{      Particle2D{T1, T2},  Particle3D{T1, T2}}
const  PROPERTYD{T1, T2} = Union{ParticleProperty{T1, T2}                     }
const  BOUNDARYD{T1, T2} = Union{     VBoundary2D{T1, T2}, VBoundary3D{T1, T2}}

Adapt.@adapt_structure GPUGrid2D
Adapt.@adapt_structure GPUGrid3D
Adapt.@adapt_structure GPUParticle2D
Adapt.@adapt_structure GPUParticle3D
Adapt.@adapt_structure GPUVBoundary2D
Adapt.@adapt_structure GPUVBoundary3D
Adapt.@adapt_structure GPUParticleProperty

StructTypes.StructType(::Type{          Args2D}) = StructTypes.Mutable()
StructTypes.StructType(::Type{          Args3D}) = StructTypes.Mutable()
StructTypes.StructType(::Type{          Grid2D}) = StructTypes.Struct()
StructTypes.StructType(::Type{          Grid3D}) = StructTypes.Struct()
StructTypes.StructType(::Type{      Particle2D}) = StructTypes.Struct()
StructTypes.StructType(::Type{      Particle3D}) = StructTypes.Struct()
StructTypes.StructType(::Type{ParticleProperty}) = StructTypes.Struct()
StructTypes.StructType(::Type{     VBoundary2D}) = StructTypes.Struct()
StructTypes.StructType(::Type{     VBoundary3D}) = StructTypes.Struct()