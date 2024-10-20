module MaterialPointSolverCUDAExt

using BenchmarkTools
using CUDA
using KernelAbstractions
using Printf
using MaterialPointSolver

# rewrite with CUDA
import MaterialPointSolver: host2device, device2host!, clean_device!, Tpeak, getBackend,
    warmup, grf_gc!, grf_ec!, getArray, getparticle

# just import the function
import MaterialPointSolver: gmsh_mesh3D, pts_in_polyhedron!

CUDA.allowscalar(false) # disable scalar operation in GPU

include(joinpath(@__DIR__, "CUDAExt/devicehelpfunc_cuda.jl"))
include(joinpath(@__DIR__, "CUDAExt/warmup_cuda.jl"        ))
include(joinpath(@__DIR__, "CUDAExt/randomfield_cuda.jl"   ))
include(joinpath(@__DIR__, "CUDAExt/meshbuilder_cuda.jl"   ))

end