#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : modelargs.jl                                                               |
|  Description: Type system for modelargs in MaterialPointSolver.jl                        |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  License    : MIT License                                                                |
+==========================================================================================#

export AbstractArgs
export DeviceArgs, DeviceArgs2D, DeviceArgs3D
export Args2D, Args3D
export UserArgs2D, UserArgs3D
export UserArgsExtra

abstract type AbstractArgs end
abstract type DeviceArgs{T1, T2} <: AbstractArgs end
abstract type DeviceArgs2D{T1, T2} <: DeviceArgs{T1, T2} end
abstract type DeviceArgs3D{T1, T2} <: DeviceArgs{T1, T2} end
abstract type UserArgsExtra end

struct TempArgsExtra{T1} <: UserArgsExtra
    i::T1
end

#=-----------------------------------------------------------------------------------------#
|    2D Args System                                                                        |
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓=#
mutable struct Args2D{T1, T2, T3<:UserArgsExtra} <: DeviceArgs2D{T1, T2}
    const Ttol        ::T2
    const Te          ::T2
          ΔT          ::T2
    const time_step   ::Symbol
    const FLIP        ::T2
    const PIC         ::T2
    const constitutive::Symbol
    const basis       ::Symbol
    const animation   ::Bool
    const hdf5        ::Bool
    const hdf5_step   ::T1
    const MVL         ::Bool
    const device      ::Symbol
    const coupling    ::Symbol
    const scheme      ::Symbol
    const va          ::Symbol
    const progressbar ::Bool
    const gravity     ::T2
    const ζs          ::T2
    const ζw          ::T2
    const αT          ::T2
          iter_num    ::Int64
          end_time    ::Float64
          start_time  ::Float64
    const project_name::String
    const project_path::String
          ext         ::T3
end

function UserArgs2D(; Ttol, Te=0, ΔT, time_step=:fixed, FLIP=1, PIC=0, constitutive, 
    basis=:uGIMP, animation=false, hdf5=false, hdf5_step=1, MVL=false, device=:CPU, 
    coupling=:OS, scheme=:MUSL, va=:a, progressbar=true, gravity=-9.8, ζs=0, ζw=0, αT=0.5, 
    iter_num=0, end_time=0, start_time=0, project_name, project_path, ext=0, ϵ="FP64")
    T1 = ϵ=="FP32" ? Int32   : Int64
    T2 = ϵ=="FP32" ? Float32 : Float64
    # project default value
    folderdir = joinpath(abspath(project_path), project_name)
    mkpath(folderdir); rm(folderdir, recursive=true, force=true); mkpath(folderdir)
    cop_set = [:OS, :TS]
    bas_set = [:uGIMP, :linear, :gslinear]
    dev_set = [:CPU, :CUDA, :ROCm, :oneAPI, :Metal]
    tis_set = [:fixed, :auto]
    v_a_set = [:v, :a]
    # parameter check
    0<Ttol                || error("Simulation time cannot be ≤0 s."         )
    ΔT≤Ttol               || error("Time step cannot be >$(Ttol)s."          )
    0≤Te≤Ttol             || error("Elastic loading time must in 0~$(Ttol)s.")
    basis in bas_set      || error("Cannot find $(basis) basis function."    )
    (FLIP+PIC)==1         || error("FLIP+PIC must be 1."                     )
    0≤αT≤1                || error("Time step factor error."                 )
    0≤ζs≤1                || error("Out of range 0≤ζs≤1."                    )
    0≤ζw≤1                || error("Out of range 0≤ζw≤1."                    )
    project_name≠""       || error("Empty project name."                     )
    coupling in cop_set   || error("Coupling mode is wrong."                 )
    device in dev_set     || error("Cannot find $(device) device."           )
    time_step in tis_set  || error("$(time_step) time step is not allowed."  )
    va in v_a_set         || error("Cannot find $(va) velocity update mode." )
    (animation==true)&&(hdf5==false) ? 
        (hdf5=true; @warn "HDF5 forced ON due to the animation") : nothing 
    (hdf5==true)&&(hdf5_step≤0) ? error("HDF5 step cannot be ≤0.") : nothing
    tmp = ext == 0 ? TempArgsExtra(0) : ext

    return Args2D{T1, T2, UserArgsExtra}(Ttol, Te, ΔT, time_step, FLIP, PIC, constitutive, 
        basis, animation, hdf5, hdf5_step, MVL, device, coupling, scheme, va, progressbar, 
        gravity, ζs, ζw, αT, iter_num, end_time, start_time, project_name, project_path, 
        tmp)
end

function Base.show(io::IO, args::T) where {T<:DeviceArgs2D}
    typeof(args).parameters[2]==Float64 ? precision="FP64" : 
    typeof(args).parameters[2]==Float32 ? precision="FP32" : nothing
    # printer
    print(io, "DeviceArgs2D:"                               , "\n")
    print(io, "┬", "─" ^ 12                                 , "\n")
    print(io, "├─ ", "project name    : ", args.project_name, "\n")
    print(io, "├─ ", "project path    : ", args.project_path, "\n")
    print(io, "├─ ", "precision       : ", precision        , "\n")
    print(io, "├─ ", "constitutive    : ", args.constitutive, "\n")
    print(io, "├─ ", "basis method    : ", args.basis       , "\n")
    print(io, "├─ ", "mitigate vollock: ", args.MVL         , "\n")
    print(io, "└─ ", "coupling scheme : ", args.coupling    , "\n")
    return nothing
end
#=↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑=#



#=-----------------------------------------------------------------------------------------#
|    3D Args System                                                                        |
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓=#
mutable struct Args3D{T1, T2, T3<:UserArgsExtra} <: DeviceArgs3D{T1, T2}
    const Ttol        ::T2
    const Te          ::T2
          ΔT          ::T2
    const time_step   ::Symbol
    const FLIP        ::T2
    const PIC         ::T2
    const constitutive::Symbol
    const basis       ::Symbol
    const animation   ::Bool
    const hdf5        ::Bool
    const hdf5_step   ::T1
    const MVL         ::Bool
    const device      ::Symbol
    const coupling    ::Symbol
    const scheme      ::Symbol
    const va          ::Symbol
    const progressbar ::Bool
    const gravity     ::T2
    const ζs          ::T2
    const ζw          ::T2
    const αT          ::T2
          iter_num    ::Int64
          end_time    ::Float64
          start_time  ::Float64
    const project_name::String
    const project_path::String
          ext         ::T3
end

function UserArgs3D(; Ttol, Te=0, ΔT, time_step=:fixed, FLIP=1, PIC=0, constitutive, 
    basis=:uGIMP, animation=false, hdf5=false, hdf5_step=1, MVL=false, device=:CPU, 
    coupling=:OS, scheme=:MUSL, va=:a, progressbar=true, gravity=-9.8, ζs=0, ζw=0, αT=0.5, 
    iter_num=0, end_time=0, start_time=0, project_name, project_path, ext=0, ϵ="FP64")
    T1 = ϵ=="FP32" ? Int32   : Int64
    T2 = ϵ=="FP32" ? Float32 : Float64
    # project default value
    folderdir = joinpath(abspath(project_path), project_name)
    mkpath(folderdir); rm(folderdir, recursive=true, force=true); mkpath(folderdir)
    cop_set = [:OS, :TS]
    bas_set = [:uGIMP, :linear, :gslinear]
    dev_set = [:CPU, :CUDA, :ROCm, :oneAPI, :Metal]
    tis_set = [:fixed, :auto]
    v_a_set = [:v, :a]
    # parameter check
    0<Ttol                || error("Simulation time cannot be ≤0 s."         )
    ΔT≤Ttol               || error("Time step cannot be >$(Ttol)s."          )
    0≤Te≤Ttol             || error("Elastic loading time must in 0~$(Ttol)s.")
    basis in bas_set      || error("Cannot find $(basis) basis function."    )
    (FLIP+PIC)==1         || error("FLIP+PIC must be 1."                     )
    0≤αT≤1                || error("Time step factor error."                 )
    0≤ζs≤1                || error("Out of range 0≤ζs≤1."                    )
    0≤ζw≤1                || error("Out of range 0≤ζw≤1."                    )
    project_name≠""       || error("Empty project name."                     )
    coupling in cop_set   || error("Coupling mode is wrong."                 )
    device in dev_set     || error("Cannot find $(device) device."           )
    time_step in tis_set  || error("$(time_step) time step is not allowed."  )
    va in v_a_set         || error("Cannot find $(va) velocity update mode." )
    (animation==true)&&(hdf5==false) ? 
        (hdf5=true; @warn "HDF5 forced ON due to the animation") : nothing 
    (hdf5==true)&&(hdf5_step≤0) ? error("HDF5 step cannot be ≤0.") : nothing
    tmp = ext == 0 ? TempArgsExtra(0) : ext

    return Args3D{T1, T2, UserArgsExtra}(Ttol, Te, ΔT, time_step, FLIP, PIC, constitutive, 
        basis, animation, hdf5, hdf5_step, MVL, device, coupling, scheme, va, progressbar, 
        gravity, ζs, ζw, αT, iter_num, end_time, start_time, project_name, project_path, 
        tmp)
end

function Base.show(io::IO, args::T) where {T<:DeviceArgs3D}
    typeof(args).parameters[2]==Float64 ? precision="FP64" : 
    typeof(args).parameters[2]==Float32 ? precision="FP32" : nothing
    # printer
    print(io, "DeviceArgs3D:"                               , "\n")
    print(io, "┬", "─" ^ 12                                 , "\n")
    print(io, "├─ ", "project name    : ", args.project_name, "\n")
    print(io, "├─ ", "project path    : ", args.project_path, "\n")
    print(io, "├─ ", "precision       : ", precision        , "\n")
    print(io, "├─ ", "constitutive    : ", args.constitutive, "\n")
    print(io, "├─ ", "basis method    : ", args.basis       , "\n")
    print(io, "├─ ", "mitigate vollock: ", args.MVL         , "\n")
    print(io, "└─ ", "coupling scheme : ", args.coupling    , "\n")
    return nothing
end
#=↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑=#