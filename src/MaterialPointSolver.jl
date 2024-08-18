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

using Adapt, AMDGPU, ArrayAllocators, BenchmarkTools, CpuId, CUDA, Dates, DelimitedFiles, HDF5, 
      JSON3, KernelAbstractions, Printf, #=PrecompileTools, =#ProgressMeter, StructTypes, 
      WriteVTK
#import KernelAbstractions.@atomic as @KAatomic 
import KernelAbstractions.synchronize as KAsync
import Suppressor: @suppress

const assets_dir = joinpath(@__DIR__, "../docs/src/assets")
const fontcmu    = joinpath(assets_dir, "fonts/cmu.ttf")
const fonttnr    = joinpath(assets_dir, "fonts/tnr.ttf")

macro KAatomic(expr)
    esc(quote
        KernelAbstractions.@atomic :monotonic $expr
    end)
end

# export structs
export MODELARGS, GRID, PARTICLE, PROPERTY, BOUNDARY
export GPUGRID, GPUPARTICLE, GPUPARTICLEPROPERTY, GPUBOUNDARY
export Args2D, Args3D, Grid2D, Grid3D, Particle2D, Particle3D, ParticleProperty, 
       VBoundary2D, VBoundary3D
export GPUGrid2D, GPUGrid3D, GPUParticle2D, GPUParticle3D, GPUParticleProperty, 
       GPUVBoundary2D, GPUVBoundary3D
export KernelGrid2D, KernelGrid3D, KernelParticle2D, KernelParticle3D, 
       KernelParticleProperty, KernelBoundary2D, KernelBoundary3D
# export functions
export materialpointsolver!
export submit_work!
export JSON3
export @KAatomic, KAsync
export @suppress

include(joinpath(@__DIR__, "type.jl"    ))
include(joinpath(@__DIR__, "toolkit.jl" ))
include(joinpath(@__DIR__, "solver.jl"  ))

function print_welcome_message()
    print("\e[1;1H\e[2J")
    println("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║        ⭐ Welcome to \e[1;34mM\e[0material\e[1;31mP\e[0moint\e[1;32mS\e[0molver.jl ⭐        ║
    ║           \e[1;35m─────────────────────────────────\e[0m           ║
    ║                                                       ║
    ║ \e[1;34mVersion    :\e[0m v0.1.0                                   ║
    ║ \e[1;31mDescription:\e[0m A high-performance MPM solver in Julia   ║
    ║ \e[1;32mStart Date :\e[0m 01/01/2022                               ║
    ║ \e[1;35mAffiliation:\e[0m Risk Group, UNIL-ISTE                    ║
    ║                                                       ║
    ║ \e[1;33mTips: \e[0;33m⚠ please try to warm up before simulating\e[0m       ║
    ║ \e[1;33m───────────────────────────────────────────────       \e[0m║
    ║ \e[1;33mhelp?>\e[0m MaterialPointSolver\e[0;31m.\e[0;36m\e[0m\e[0;36mwarmup\e[0;33m(\e[0mdevicetype; ID\e[0;31m=\e[0;35m0\e[0m)   ║
    ║\e[1;33m\e[0m   1). \e[0;34mdevicetype\e[0m can be one of \e[0;34m:CUDA\e[0m, \e[0;34m:ROCm\e[0m, and \e[0;34m:CPU\e[0m ║
    ║\e[1;33m\e[0m   2). \e[0;34mID\e[0m (optional) is \e[0;35m0\e[0m by default                   ║
    ║\e[1;32m julia>\e[0m MaterialPointSolver\e[0;31m.\e[0;36m\e[0m\e[0;36mwarmup\e[0;33m(\e[0;35m:CUDA\e[0;33m)\e[0m              ║
    ╚═══════════════════════════════════════════════════════╝
    \n""")
end


function __init__()
    CUDA.allowscalar(false) # disable scalar operation in GPU
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