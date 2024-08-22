using MaterialPointSolver
using KernelAbstractions
using Test
using JSON3

@testset "CPU device tests" begin
    @test true begin
        # model configuration
        rtsdir            = joinpath(@__DIR__, "outputs")
        devicebackend     = CPU()
        mpmbasis          = :linear
        init_basis        = mpmbasis
        init_NIC          = 16
        init_grid_space_x = 0.25
        init_grid_space_y = 0.25
        init_T            = 3
        init_gravity      = -9.8
        init_ζs           = 0
        init_ρs           = 1e3
        init_ν            = 0
        init_E            = 9e7
        init_grid_range_x = [-1, 6]
        init_grid_range_y = [-1, 8]
        init_mp_in_space  = 2
        init_project_name = "2d_cantilever_beam"
        init_project_path = joinpath(rtsdir, init_project_name)
        init_constitutive = :hyperelastic
        init_G            = init_E/(2*(1+  init_ν))
        init_Ks           = init_E/(3*(1-2*init_ν))
        init_Te           = 0
        init_ΔT           = 0.5*init_grid_space_x/sqrt(init_E/init_ρs)
        init_phase        = 1
        init_scheme       = :MUSL
        iInt              = Int64
        iFloat            = Float64
    end
end

@testset "NVIDIA GPU device tests" begin
    @test 1==1
end

@testset "AMD GPU device test" begin
    @test 1==1
end

@testset "Apple GPU device test" begin
    @test 1==1
end

@testset "Intel GPU device test" begin
    @test 1==1
end

@testset "other functions" begin
    @test 1==1
end