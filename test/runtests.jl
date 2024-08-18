using CairoMakie

include(joinpath(@__DIR__, "../src/MPMSolver.jl"))
const assets = joinpath(@__DIR__, "../assets")  # code assets
const rtsdir = "/home/zhuo/Workbench/outputs/"  # result path

info_page()
@info "initialization done"