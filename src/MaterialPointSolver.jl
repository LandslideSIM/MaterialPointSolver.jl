#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : MaterialPointSolver.jl                                                     |
|  Description: Module file of MaterialPointSolver.jl                                      |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
+==========================================================================================#

module MaterialPointSolver

using Adapt, ArrayAllocators, BenchmarkTools, Dates, DelimitedFiles, HDF5, JSON3, 
      KernelAbstractions, Printf, ProgressMeter, StructTypes, SysInfo, WriteVTK # PrecompileTools

import KernelAbstractions.synchronize as KAsync
import KernelAbstractions.Extras.LoopInfo.@unroll as @KAunroll
import Suppressor: @suppress

const assets_dir = joinpath(@__DIR__, "../docs/assets")
const fontcmu    = joinpath(assets_dir, "fonts/cmu.ttf")
const fonttnr    = joinpath(assets_dir, "fonts/tnr.ttf")

# export functions
export materialpointsolver!
export submit_work!
export JSON3
export sysinfo
export @KAatomic, @KAunroll, KAsync
export @suppress

macro KAatomic(expr)
    esc(quote
        KernelAbstractions.@atomic :monotonic $expr
    end)
end

include(joinpath(@__DIR__, "type.jl"    ))
include(joinpath(@__DIR__, "toolkit.jl" ))
include(joinpath(@__DIR__, "solver.jl"  ))

function print_welcome_message()
    print("\e[1;1H\e[2J")
    @info"""\e[1;31mA high-performance MPM solver in Julia 🚀\e[0
    version    : v0.2.0
    affiliation: Risk Group, UNIL-ISTE
    """
    println()
    println("\e[1;33m⚠\e[0m \e[0;33mplease try to warm up before simulating:\e[0m\n
\e[1;33mhelp?>\e[0m warmup")
    println()
end

function __init__()
    print_welcome_message()
end

"""
    materialpointsolver!(args::MODELARGS, grid::GRID, mp::PARTICLE, pts_attr::PROPERTY, 
        bc::BOUNDARY; workflow::Function=procedure!)

Description:
---
This function is the main function of the MPM solver, user has to pre-define the data of
    `args`, `grid`, `mp`, 'pts_attr' and `bc`, they are the model configuration, background 
    grid, material points, particle property and boundary conditions (2/3D).
"""
function materialpointsolver!(args    ::MODELARGS, 
                              grid    ::GRID, 
                              mp      ::PARTICLE, 
                              pts_attr::PROPERTY,
                              bc      ::BOUNDARY; 
                              workflow::Function=procedure!)
    info_print(args, grid, mp) # terminal info
    submit_work!(args, grid, mp, pts_attr, bc, workflow) # MPM solver
    perf(args, grid, mp) # performance summary
    # generate animation files (ParaView)
    args.animation==true ? animation(args) : nothing 
    return nothing
end

# @setup_workload begin
#     # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
#     # precompile file and potentially make loading faster.
#     @compile_workload begin
#         # all calls in this block will be precompiled, regardless of whether
#         # they belong to your package or not (on Julia 1.8 and higher)

#     end
# end

end