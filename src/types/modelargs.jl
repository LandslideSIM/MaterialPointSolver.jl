#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : modelargs.jl                                                               |
|  Description: Type system for args in MaterialPointSolver.jl                             |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Struct     : 1. Args2D                                                                  |
|               2. Args3D                                                                  |
|               2. Base.show                                                               |
+==========================================================================================#

"""
    Args2D{T1, T2} <: MODELARGS

Description:
---
This struct is used to store the parameters of the solver, it is a mutable struct, The 
    parameters are listed below:

1. `Ttol`        : simulation time (s)
2. `Te`          : elastic loading time (s)
3. `ΔT`          : time step (s)
4. `time_step`   : time step mode (:fixed/:auto)
5. `FLIP`        : FLIP ratio (0~1)
6. `PIC`         : PIC ratio (0~1)
7. `constitutive`: constitutive model (:linearelastic/:hyperelastic/:druckerprager/:mohrcoulomb/:taitwater)
8. `basis`       : basis function (:uGIMP/:linear)
9. `animation`   : animation switch (true/false)
10. `hdf5`       : HDF5 switch (true/false)
11. `hdf5_step`  : HDF5 step (Int)
12. `MVL`        : mitigate volume locking (true/false)
13. `device`     : device (:CPU/:CUDA)
14. `coupling`   : coupling mode (:OS/:TS)
15. `progressbar`: progress bar switch (true/false)
16. `gravity`    : gravity (m/s²)
17. `ζs`         : solid phase damping ratio (0~1)
18. `ζw`         : water phase damping ratio (0~1)
19. `αT`         : time step factor (0~1)
20. `iter_num`   : iteration number (Int)
21. `end_time`   : end time (s)
22. `start_time` : start time (s)
23. `project_name`: project name (String)
24. `project_path`: project path (String)
"""
@kwdef mutable struct Args2D{T1, T2} <: MODELARGS
    const Ttol        ::T2
    const Te          ::T2
          ΔT          ::T2
    const time_step   ::Symbol  = :auto
    const FLIP        ::T2
    const PIC         ::T2
    const constitutive::Symbol
    const basis       ::Symbol
    const animation   ::Bool    = false
    const hdf5        ::Bool    = false
    const hdf5_step   ::T1      = T1(1)
    const MVL         ::Bool    = false
    const device      ::Symbol  = :CPU
    const coupling    ::Symbol  = :OS
    const scheme      ::Symbol  = :MUSL
    const progressbar ::Bool    = true
    const gravity     ::T2      = T2(-9.8)
    const ζs          ::T2      = T2(0)
    const ζw          ::T2      = T2(0)
    const αT          ::T2      = T2(0.5)
          iter_num    ::Int64   = Int64(0)
          end_time    ::Float64 = Float64(0)
          start_time  ::Float64 = Float64(0)
    const project_name::String
    const project_path::String
    function Args2D{T1, T2}(Ttol, Te, ΔT, time_step, FLIP, PIC, constitutive, basis, 
        animation, hdf5, hdf5_step, MVL, device, coupling, scheme, progressbar, gravity, ζs, 
        ζw, αT, iter_num, end_time, start_time, project_name, project_path) where {T1, T2}
        # project default value
        project_path = abspath(project_path)
        rm(project_path, recursive=true, force=true); mkpath(project_path)
        cop_set = [:OS, :TS]
        bas_set = [:uGIMP, :linear]
        dev_set = [:CPU, :CUDA, :ROCm, :oneAPI, :Metal]
        tis_set = [:fixed, :auto]
        # parameter check
        0<Ttol                  ? nothing : error("Simulation time cannot be ≤0 s."         )
        ΔT≤Ttol                 ? nothing : error("Time step cannot be >$(Ttol)s."          )
        0≤Te≤Ttol               ? nothing : error("Elastic loading time must in 0~$(Ttol)s.")
        basis in bas_set        ? nothing : error("Cannot find $(basis) basis function."    )
        (FLIP+PIC)==1           ? nothing : error("FLIP+PIC must be 1."                     )
        0≤αT≤1                  ? nothing : error("Time step factor error."                 )
        0≤ζs≤1                  ? nothing : error("Out of range 0≤ζs≤1."                    )
        0≤ζw≤1                  ? nothing : error("Out of range 0≤ζw≤1."                    )
        project_name≠""         ? nothing : error("Empty project name."                     )
        coupling in cop_set     ? nothing : error("Coupling mode is wrong."                 )
        device in dev_set       ? nothing : error("Cannot find $(device) device."           )
        time_step in tis_set    ? nothing : error("$(time_step) time step is not allowed."  )
        (animation==true)&&(hdf5==false) ? (hdf5=true; @warn "HDF5 forced ON due to the animation") : nothing
        (hdf5==true)&&(hdf5_step≤0)  ? error("HDF5 step cannot be ≤0.") : nothing
        # update struct
        new(Ttol, Te, ΔT, time_step, FLIP, PIC, constitutive, basis, animation, hdf5, 
            hdf5_step, MVL, device, coupling, scheme, progressbar, gravity, ζs, ζw, αT, 
            iter_num, end_time, start_time, project_name, project_path)
    end
end

"""
    Args3D{T1, T2} <: MODELARGS

Description:
---
3D version of `Args`, which has the same fields as [`Args2D`](@ref).
"""
@kwdef mutable struct Args3D{T1, T2} <: MODELARGS
    const Ttol        ::T2
    const Te          ::T2
          ΔT          ::T2
    const time_step   ::Symbol  = :auto
    const FLIP        ::T2
    const PIC         ::T2
    const constitutive::Symbol
    const basis       ::Symbol
    const animation   ::Bool    = false
    const hdf5        ::Bool    = false
    const hdf5_step   ::T1      = T1(1)
    const MVL         ::Bool    = false
    const device      ::Symbol  = :CPU
    const coupling    ::Symbol  = :OS
    const scheme      ::Symbol  = :MUSL
    const progressbar ::Bool    = true
    const gravity     ::T2      = T2(-9.8)
    const ζs          ::T2      = T2(0)
    const ζw          ::T2      = T2(0)
    const αT          ::T2      = T2(0.5)
          iter_num    ::Int64   = Int64(0)
          end_time    ::Float64 = Float64(0)
          start_time  ::Float64 = Float64(0)
    const project_name::String
    const project_path::String
    function Args3D{T1, T2}(Ttol, Te, ΔT, time_step, FLIP, PIC, constitutive, basis, 
        animation, hdf5, hdf5_step, MVL, device, coupling, scheme, progressbar, gravity, ζs, 
        ζw, αT, iter_num, end_time, start_time, project_name, project_path) where {T1, T2}
        # project default value
        project_path = abspath(project_path)
        rm(project_path, recursive=true, force=true); mkpath(project_path)
        cop_set = [:OS, :TS]
        bas_set = [:uGIMP, :linear]
        dev_set = [:CPU, :CUDA, :ROCm, :oneAPI, :Metal]
        tis_set = [:fixed, :auto]
        # parameter check
        0<Ttol                  ? nothing : error("Simulation time cannot be ≤0 s."         )
        ΔT≤Ttol                 ? nothing : error("Time step cannot be >$(Ttol)s."          )
        0≤Te≤Ttol               ? nothing : error("Elastic loading time must in 0~$(Ttol)s.")
        basis in bas_set        ? nothing : error("Cannot find $(basis) basis function."    )
        (FLIP+PIC)==1           ? nothing : error("FLIP+PIC must be 1."                     )
        0≤αT≤1                  ? nothing : error("Time step factor error."                 )
        0≤ζs≤1                  ? nothing : error("Out of range 0≤ζs≤1."                    )
        0≤ζw≤1                  ? nothing : error("Out of range 0≤ζw≤1."                    )
        project_name≠""         ? nothing : error("Empty project name."                     )
        coupling in cop_set     ? nothing : error("Coupling mode is wrong."                 )
        device in dev_set       ? nothing : error("Cannot find $(device) device."           )
        time_step in tis_set    ? nothing : error("$(time_step) time step is not allowed."  )
        (animation==true)&&(hdf5==false) ? (hdf5=true; @warn "HDF5 forced ON due to the animation") : nothing
        (hdf5==true)&&(hdf5_step≤0)  ? error("HDF5 step cannot be ≤0.") : nothing
        # update struct
        new(Ttol, Te, ΔT, time_step, FLIP, PIC, constitutive, basis, animation, hdf5, 
            hdf5_step, MVL, device, coupling, scheme, progressbar, gravity, ζs, ζw, αT, 
            iter_num, end_time, start_time, project_name, project_path)
    end
end

function Base.show(io::IO, args::MODELARGS)
    typeof(args).parameters[2]==Float64 ? precision="FP64" : 
    typeof(args).parameters[2]==Float32 ? precision="FP32" : nothing
    # printer
    print(io, typeof(args)                       , "\n")
    print(io, "─"^length(string(typeof(args)))   , "\n")
    print(io, "project name    : ", args.project_name, "\n")
    print(io, "project path    : ", args.project_path, "\n")
    print(io, "precision       : ", precision        , "\n")
    print(io, "constitutive    : ", args.constitutive, "\n")
    print(io, "basis method    : ", args.basis       , "\n")
    print(io, "mitigate vollock: ", args.MVL     , "\n")
    print(io, "coupling scheme : ", args.coupling    , "\n")
    return nothing
end