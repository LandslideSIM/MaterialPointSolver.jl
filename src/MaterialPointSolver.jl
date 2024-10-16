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

using Adapt, BenchmarkTools, Dates, DelimitedFiles, HDF5, KernelAbstractions, Printf, 
      ProgressMeter, SysInfo, WriteVTK # PrecompileTools

import KernelAbstractions.synchronize as KAsync
import KernelAbstractions.Extras: @unroll as @KAunroll
import Suppressor: @suppress
import Adapt.adapt as user_adapt
import Adapt.@adapt_structure as @user_struct

const assets_dir = joinpath(@__DIR__  , "../docs/assets" )
const cmu        = joinpath(assets_dir, "fonts/cmu.ttf"  )
const cmui       = joinpath(assets_dir, "fonts/cmui.ttf" )
const cmub       = joinpath(assets_dir, "fonts/cmub.ttf" )
const cmuib      = joinpath(assets_dir, "fonts/cmuib.ttf")
const tnr        = joinpath(assets_dir, "fonts/tnr.ttf"  )
const tnri       = joinpath(assets_dir, "fonts/tnri.ttf" )
const tnrb       = joinpath(assets_dir, "fonts/tnrb.ttf" )
const tnrib      = joinpath(assets_dir, "fonts/tnrib.ttf")

macro KAatomic(expr)
    esc(quote
        KernelAbstractions.@atomic :monotonic $expr
    end)
end

# export functions
export materialpointsolver!
export @KAatomic, @KAunroll, KAsync
export @suppress
export user_adapt, @user_struct

include(joinpath(@__DIR__, "type.jl"   ))
include(joinpath(@__DIR__, "toolkit.jl" ))
include(joinpath(@__DIR__, "solver.jl"  ))

include(joinpath(@__DIR__, "extension/frictionExt.jl"))

"""
    materialpointsolver!(args::DeviceArgs{T1, T2}, grid::DeviceGrid{T1, T2}, 
        mp::DeviceParticle{T1, T2}, attr::DeviceProperty{T1, T2}, 
        bc::DeviceVBoundary{T1, T2}; workflow::Function=procedure!)

Description:
---
This function is the main function of the MPM solver, user has to pre-define the data of
    `args`, `grid`, `mp`, 'attr' and `bc`, they are the model configuration, background 
    grid, material points, particle property and boundary conditions (2/3D).
"""
function materialpointsolver!(
    args    ::     DeviceArgs{T1, T2}, 
    grid    ::     DeviceGrid{T1, T2}, 
    mp      :: DeviceParticle{T1, T2}, 
    attr    :: DeviceProperty{T1, T2},
    bc      ::DeviceVBoundary{T1, T2}; 
    workflow::Function=procedure!
) where {T1, T2}
    info_print(args, grid, mp) # terminal info
    submit_work!(args, grid, mp, attr, bc, workflow) # MPM solver
    perf(args) # performance summary
    # generate animation files (ParaView)
    args.animation==true ? animation(args) : nothing 
    return nothing
end

end