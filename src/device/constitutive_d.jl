#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : constitutive_d.jl                                                          |
|  Description: Constitutive model for different materials                                 |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. constitutive! [2D GPU]                                                  |
|               2. constitutive! [3D GPU]                                                  |
+==========================================================================================#

include(joinpath(@__DIR__, "material/linearelastic_d.jl"))
include(joinpath(@__DIR__, "material/druckerprager_d.jl"))
include(joinpath(@__DIR__, "material/mohrcoulomb_d.jl"  ))
include(joinpath(@__DIR__, "material/hyperelastic_d.jl" ))
include(joinpath(@__DIR__, "material/taitwater_d.jl"    ))

"""
    constitutive!(args::Args2D{T1, T2}, cu_mp::GPUParticle2D{T1, T2}, Ti::T2, 
        OccAPI::NamedTuple)

Description:
---
Update strain and stress according to different constitutive models.
"""
@views function constitutive!(args  ::       Args2D{T1, T2},
                              cu_mp ::GPUParticle2D{T1, T2},
                              Ti    ::T2,
                              OccAPI::NamedTuple) where {T1, T2}
    if args.constitutive==:hyperelastic
        @cuda threads=OccAPI.hyE_t blocks=OccAPI.hyE_b hyE!(cu_mp)
    elseif args.constitutive==:linearelastic
        @cuda threads=OccAPI.liE_t blocks=OccAPI.liE_b liE!(cu_mp)
    elseif args.constitutive==:druckerprager
        @cuda threads=OccAPI.liE_t blocks=OccAPI.liE_b liE!(cu_mp)
        Ti≥args.Te ? (
            @cuda threads=OccAPI.dpP_t blocks=OccAPI.dpP_b dpP!(cu_mp);
        ) : nothing
    elseif args.constitutive==:mohrcoulomb
        @cuda threads=OccAPI.liE_t blocks=OccAPI.liE_b liE!(cu_mp)
        Ti≥args.Te ? (
            @cuda threads=OccAPI.mcP_t blocks=OccAPI.mcP_b mcP!(cu_mp);
        ) : nothing
    elseif args.constitutive==:taitwater
        @cuda threads=OccAPI.twP_t blocks=OccAPI.twP_b twP!(cu_mp)
    end
    return nothing
end

"""
    constitutive!(args::Args3D{T1, T2}, cu_mp::GPUParticle3D{T1, T2}, Ti::T2, 
        OccAPI::NamedTuple)

Description:
---
Update strain and stress according to different constitutive models.
"""
@views function constitutive!(args  ::       Args3D{T1, T2},
                              cu_mp ::GPUParticle3D{T1, T2},
                              Ti    ::T2,
                              OccAPI::NamedTuple) where {T1, T2}
    if args.constitutive==:hyperelastic
        @cuda threads=OccAPI.hyE_t blocks=OccAPI.hyE_b hyE!(cu_mp)
    elseif args.constitutive==:linearelastic
        @cuda threads=OccAPI.liE_t blocks=OccAPI.liE_b liE!(cu_mp)
    elseif args.constitutive==:druckerprager
        @cuda threads=OccAPI.liE_t blocks=OccAPI.liE_b liE!(cu_mp)
        Ti≥args.Te ? (
            @cuda threads=OccAPI.dpP_t blocks=OccAPI.dpP_b dpP!(cu_mp);
        ) : nothing
    elseif args.constitutive==:mohrcoulomb
        @cuda threads=OccAPI.liE_t blocks=OccAPI.liE_b liE!(cu_mp)
        Ti≥args.Te ? (
            @cuda threads=OccAPI.mcP_t blocks=OccAPI.mcP_b mcP!(cu_mp);
        ) : nothing
    elseif args.constitutive==:taitwater
        @cuda threads=OccAPI.twP_t blocks=OccAPI.twP_b twP!(cu_mp)
    end
    return nothing
end
