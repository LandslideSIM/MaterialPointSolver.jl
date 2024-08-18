#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : constitutive.jl                                                            |
|  Description: Constitutive model for different materials                                 |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. constitutive! [2D]                                                      |
|               2. constitutive! [3D]                                                      |
+==========================================================================================#

# import different constitutive models
include(joinpath(@__DIR__, "material/hyperelastic.jl" ))
include(joinpath(@__DIR__, "material/linearelastic.jl"))
include(joinpath(@__DIR__, "material/druckerprager.jl"))
include(joinpath(@__DIR__, "material/mohrcoulomb.jl"  ))
include(joinpath(@__DIR__, "material/taitwater.jl"    ))

"""
    constitutive!(args::Args2D{T1, T2}, mp::Particle2D{T1, T2}, Ti::T2)

Description:
---
Update strain and stress according to different constitutive models.
"""
@views function constitutive!(args::    Args2D{T1, T2},
                              mp  ::Particle2D{T1, T2},
                              Ti  ::T2) where {T1, T2}
    if  args.constitutive == :hyperelastic
        hyperelastic!(mp)
    elseif args.constitutive == :linearelastic
        linearelastic!(mp)
    elseif args.constitutive == :druckerprager
        linearelastic!(mp)
        Ti≥args.Te ? druckerprager!(mp) : nothing
    elseif args.constitutive == :mohrcoulomb
        linearelastic!(mp)
        Ti≥args.Te ? mohrcoulomb!(mp)   : nothing
    elseif args.constitutive == :taitwater
        taitwater!(mp)
    end
    return nothing
end

"""
    constitutive!(args::Args3D{T1, T2}, mp::Particle3D{T1, T2})

Description:
---
Update strain and stress according to different constitutive models.
"""
@views function constitutive!(args::    Args3D{T1, T2},
                              mp  ::Particle3D{T1, T2},
                              Ti  ::T2) where {T1, T2}
    if  args.constitutive == :hyperelastic
        hyperelastic!(mp)
    elseif args.constitutive == :linearelastic
        linearelastic!(mp)
    elseif args.constitutive == :druckerprager
        linearelastic!(mp)
        Ti≥args.Te ? druckerprager!(mp) : nothing
    elseif args.constitutive == :mohrcoulomb
        linearelastic!(mp)
        Ti≥args.Te ? mohrcoulomb!(mp)   : nothing
    elseif args.constitutive == :taitwater
        taitwater!(mp)
    end
    return nothing
end