#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : warmup.jl                                                                  |
|  Description: The minimal example of executing core functionality.                       |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : warmup                                                                     |
+==========================================================================================#

# We do not export `warmup` since BenchmarkTools.jl also has a same function name.

"""
    warmup(Val{:CPU}; ID=0)

Description:
---
The minimal example of executing core functionality is used to reduce the first-time running
time. `devicetype` can be one of `:CPU`, `:CUDA`, or `:ROCm` (more backends will be support
in the future).

Examples:
---

`warmup(Val(:CPU))` 

or 

`warmup(Val(:CUDA), ID=0)`.

> Note that on the GPU from AMD, the device id start from `1`.
"""
function warmup(::Val{:CPU}; ID::Int=0)
    rtsdir = joinpath(tempdir(), "MaterialPointSolverTEMP_$(ID)/")
    args = UserArgs2D(Ttol=1, ΔT=0.1, constitutive=:linearelastic, project_name="test", 
        project_path=rtsdir, gravity=0.01, device=:CPU)
    grid = UserGrid2D(x1=-5, x2=5, y1=-5, y2=5, dx=1, dy=1)
    mp   = UserParticle2D(dx=0.5, dy=0.5, ξ=rand(5, 2), ρs=rand(5))
    attr = UserProperty(nid=rand([1], 5), ν=[0.1], Es=[1.0], Gs=[1.0], Ks=[1.0])
    bc   = UserVBoundary2D(vx_s_idx=ones(grid.ni), vx_s_val=zeros(grid.ni), 
        vy_s_idx=ones(grid.ni), vy_s_val=zeros(grid.ni), smlength=0, tmp1=0, tmp2=0, ext=0)
    # MPM solver
    @info "warming up on :CPU [$(ID)] 🔥"
    @suppress begin
        materialpointsolver!(args, grid, mp, attr, bc)
    end
    rm(rtsdir, recursive=true, force=true)
    return nothing
end