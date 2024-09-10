module MaterialPointSolverCUDAExt

using BenchmarkTools
using CUDA
using KernelAbstractions
using MaterialPointSolver
using Printf

import MaterialPointSolver: host2device, device2host!, clean_device!, Tpeak, getBackend,
    warmup, grf_gc!, grf_ec!, getArray

CUDA.allowscalar(false) # disable scalar operation in GPU

include(joinpath(@__DIR__, "CUDAExt/devicehelpfunc_cuda.jl"))
include(joinpath(@__DIR__, "CUDAExt/warmup_cuda.jl"        ))
include(joinpath(@__DIR__, "CUDAExt/randomfield_cuda.jl"   ))

end