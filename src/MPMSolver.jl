#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : MPMSolver.jl                                                               |
|  Description: Module file of MPMSolver.jl                                                |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
+==========================================================================================#

# import other packages in MPMSolver.jl
using Adapt
using ArrayAllocators
using BenchmarkTools
using CUDA
using CUDA: i32
using Dates
using DelimitedFiles
using HDF5
using Printf
using ProgressMeter
using WriteVTK

# disable scalar operation in GPU
CUDA.allowscalar(false)

include(joinpath(@__DIR__, "types.jl"       ))
include(joinpath(@__DIR__, "toolkits.jl"    ))
include(joinpath(@__DIR__, "solver.jl"      ))
include(joinpath(@__DIR__, "postprocess.jl" ))
include(joinpath(@__DIR__, "constitutive.jl"))
include(joinpath(@__DIR__, "basis/linear.jl"))
include(joinpath(@__DIR__, "basis/uGIMP.jl" ))

include(joinpath(@__DIR__, "device/types_d.jl"       ))
include(joinpath(@__DIR__, "device/toolkits_d.jl"    ))
include(joinpath(@__DIR__, "device/solver_d.jl"      ))
include(joinpath(@__DIR__, "device/constitutive_d.jl"))

"""
    simulate!(args::ARGS, grid::GRID, mp::PARTICLE, bc::VBC)

Description:
---
This function is the main function of the MPM solver, user has to pre-define the data of
    `args`, `grid`, `mp`, and `bc`, they are the parameters, background grid, material
    points, and boundary conditions (2/3D).
"""
function simulate!(args::ARGS, grid::GRID, mp::PARTICLE, bc::VBC)
    info_print(args, mp)                             # terminal info
    model_info(args, grid, mp)                       # export model info
    solver!(args, grid, mp, bc, Val(args.device))    # MPM solver
    perf(args, grid, mp)                             # performance summary
    args.animation==true ? animation(args) : nothing # generate animation files (ParaView)
    return nothing
end
