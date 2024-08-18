#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : toolkit.jl                                                                 |
|  Description: Some helper functions for MPMSolver.jl                                     |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : info_page                                                                  |
+==========================================================================================#

export cfl
export host2device, device2host!, device2hostinfo!, clean_gpu!, GPUbandwidth, getBackend, 
       getDArray
export linearBasis, uGIMPbasis
export savevtu, animation
export info_print, perf, format_seconds, progressinfo, updatepb!
export json2args2D, json2args3D, json2grid2D, json2grid3D, json2particle2D, json2particle3D, 
       json2particleproperty, json2vboundary2D, json2vboundary3D, export_model, 
       import_model, check_datasize, @memcheck
export getP2N_linear, getP2N_uGIMP
export cpu_info
export grf_gc!, grf_ec!
export meshbuilder

include(joinpath(@__DIR__, "toolkits/cfltimestep.jl" ))
include(joinpath(@__DIR__, "toolkits/gpuhelpfunc.jl" ))
include(joinpath(@__DIR__, "toolkits/mpbasisfunc.jl" ))
include(joinpath(@__DIR__, "toolkits/postprocess.jl" ))
include(joinpath(@__DIR__, "toolkits/terminaltxt.jl" ))
include(joinpath(@__DIR__, "toolkits/modelinfo.jl"   ))
include(joinpath(@__DIR__, "toolkits/warmup.jl"      ))
include(joinpath(@__DIR__, "toolkits/p2nindex.jl"    ))
include(joinpath(@__DIR__, "toolkits/hardwareinfo.jl"))
include(joinpath(@__DIR__, "toolkits/randomfield.jl" ))
include(joinpath(@__DIR__, "toolkits/meshbuilder.jl" ))