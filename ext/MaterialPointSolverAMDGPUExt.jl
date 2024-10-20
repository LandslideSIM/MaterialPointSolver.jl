module MaterialPointSolverAMDGPUExt

using BenchmarkTools
using AMDGPU
using KernelAbstractions
using Printf
using MaterialPointSolver

# rewrite with ROCm
import MaterialPointSolver: host2device, device2host!, clean_device!, Tpeak, getBackend,
    warmup, grf_gc!, grf_ec!, getArray, getparticle

# just import the function
import MaterialPointSolver: gmsh_mesh3D, pts_in_polyhedron!

include(joinpath(@__DIR__, "AMDExt/devicehelpfunc_amd.jl"))
include(joinpath(@__DIR__, "AMDExt/warmup_amd.jl"        ))
include(joinpath(@__DIR__, "AMDExt/randomfield_amd.jl"   ))
include(joinpath(@__DIR__, "AMDExt/meshbuilder_amd.jl"   ))

end