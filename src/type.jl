#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : MaterialPointSolver.jl                                                     |
|  Description: Type system of MaterialPointSolver.jl                                      |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
+==========================================================================================#

include(joinpath(@__DIR__, "types/modelargs.jl"))
include(joinpath(@__DIR__, "types/grid.jl"     ))
include(joinpath(@__DIR__, "types/particle.jl" ))
include(joinpath(@__DIR__, "types/property.jl" ))
include(joinpath(@__DIR__, "types/boundary.jl" ))