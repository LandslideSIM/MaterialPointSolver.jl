module MaterialPointSolverAMDGPUExt

using BenchmarkTools
using AMDGPU
using KernelAbstractions
using MaterialPointSolver
using Printf

import MaterialPointSolver: host2device, device2host!, clean_device!, Tpeak, getBackend,
    warmup, grf_gc!, grf_ec!, getArray

include(joinpath(@__DIR__, "AMDExt/devicehelpfunc_amd.jl"))
include(joinpath(@__DIR__, "AMDExt/warmup_amd.jl"        ))
include(joinpath(@__DIR__, "AMDExt/randomfield_amd.jl"   ))

end