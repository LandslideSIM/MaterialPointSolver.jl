#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : fictionExt.jl                                                              |
|  Description: implementation of friction in MaterialPointSolver.jl                       |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 07/10/2024                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
+==========================================================================================#

module Friction

using ..MaterialPointSolver

export SolidBasal2D, SolidBasal3D 
export UserSolidBasal2D, UserSolidBasal3D

struct SolidBasal2D{T1, T2,
    T3 <: AbstractArray,
    T4 <: AbstractArray,
    T5 <: AbstractArray
} <: UserGridExtra
    phase :: T1
    x1    :: T2
    x2    :: T2
    y1    :: T2
    y2    :: T2
    dx    :: T2
    dy    :: T2
    nnx   :: T1
    nny   :: T1
    ni    :: T1
    NIC   :: T1
    ξ     :: T5
    ncx   :: T1
    ncy   :: T1
    nc    :: T1
    p2nD  :: T3
    σm    :: T4
    σw    :: T4
    Ω     :: T4
    ms    :: T4
    mw    :: T4
    mi    :: T4
    ps    :: T5
    pw    :: T5
    vs    :: T5
    vw    :: T5
    vsT   :: T5
    vwT   :: T5
    fs    :: T5
    fw    :: T5
    fd    :: T5
    as    :: T5
    aw    :: T5
    Δus   :: T5
    Δuw   :: T5
    ∂m    :: T5
end

@user_struct SolidBasal2D

function UserSolidBasal2D(grid::DeviceGrid2D{T1, T2}) where {T1, T2}
    tmp = SolidBasal2D{T1, T2, AbstractArray{T1, 2}, AbstractArray{T2, 1}, 
        AbstractArray{T2, 2}}(grid.phase, grid.x1, grid.x2, grid.y1, grid.y2, grid.dx,
        grid.dy, grid.nnx, grid.nny, grid.ni, grid.NIC, grid.ξ, grid.ncx, grid.ncy,
        grid.nc, grid.p2nD, grid.σm, grid.σw, grid.Ω, grid.ms, grid.mw, grid.mi,
        grid.ps, grid.pw, grid.vs, grid.vw, grid.vsT, grid.vwT, grid.fs, grid.fw,
        grid.fd, grid.as, grid.aw, grid.Δus, grid.Δuw, zeros(T2, grid.ni, 2))
    return user_adapt(Array, tmp)
end

struct SolidBasal3D{T1, T2,
    T3 <: AbstractArray,
    T4 <: AbstractArray,
    T5 <: AbstractArray
} <: UserGridExtra
    phase :: T1
    x1    :: T2
    x2    :: T2
    y1    :: T2
    y2    :: T2
    z1    :: T2
    z2    :: T2
    dx    :: T2
    dy    :: T2
    dz    :: T2
    nnx   :: T1
    nny   :: T1
    nnz   :: T1
    ni    :: T1
    NIC   :: T1
    ξ     :: T5
    ncx   :: T1
    ncy   :: T1
    ncz   :: T1
    nc    :: T1
    p2nD  :: T3
    σm    :: T4
    σw    :: T4
    Ω     :: T4
    ms    :: T4
    mw    :: T4
    mi    :: T4
    ps    :: T5
    pw    :: T5
    vs    :: T5
    vw    :: T5
    vsT   :: T5
    vwT   :: T5
    fs    :: T5
    fw    :: T5
    fd    :: T5
    as    :: T5
    aw    :: T5
    Δus   :: T5
    Δuw   :: T5
    ∂m    :: T5
end

@user_struct SolidBasal3D

function UserSolidBasal3D(grid::DeviceGrid3D{T1, T2}) where {T1, T2}
    tmp = SolidBasal3D{T1, T2, AbstractArray{T1, 2}, AbstractArray{T2, 1}, 
        AbstractArray{T2, 2}}(grid.phase, grid.x1, grid.x2, grid.y1, grid.y2, grid.z1, 
        grid.z2, grid.dx, grid.dy, grid.dz, grid.nnx, grid.nny, grid.nnz, grid.ni, grid.NIC, 
        grid.ξ, grid.ncx, grid.ncy, grid.ncz, grid.nc, grid.p2nD, grid.σm, grid.σw, grid.Ω, 
        grid.ms, grid.mw, grid.mi, grid.ps, grid.pw, grid.vs, grid.vw, grid.vsT, grid.vwT, 
        grid.fs, grid.fw, grid.fd, grid.as, grid.aw, grid.Δus, grid.Δuw, 
        zeros(T2, grid.ni, 3))
    return user_adapt(Array, tmp)
end

end