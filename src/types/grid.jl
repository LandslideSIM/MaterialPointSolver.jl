#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : grid.jl                                                                    |
|  Description: Type system for grid in MaterialPointSolver.jl                             |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  License    : MIT License                                                                |
+==========================================================================================#

export AbstractGrid
export DeviceGrid, DeviceGrid2D, DeviceGrid3D
export Grid2D, Grid3D
export UserGrid2D, UserGrid3D
export UserGridExtra

abstract type AbstractGrid end
abstract type DeviceGrid{T1, T2} <: AbstractGrid end
abstract type DeviceGrid2D{T1, T2} <: DeviceGrid{T1, T2} end
abstract type DeviceGrid3D{T1, T2} <: DeviceGrid{T1, T2} end
abstract type UserGridExtra end

struct TempGridExtra{T1<:AbstractArray} <: UserGridExtra
    i::T1
end

@user_struct TempGridExtra

#=-----------------------------------------------------------------------------------------#
|    2D Grid System                                                                        |
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓=#
struct Grid2D{T1, T2,
    T3 <: AbstractArray, # Array{Int  , 2} 
    T4 <: AbstractArray, # Array{Float, 1}
    T5 <: AbstractArray, # Array{Float, 2}
    T6 <: UserGridExtra
} <: DeviceGrid2D{T1, T2}
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
    ext   :: T6
end

@user_struct Grid2D

function UserGrid2D(; ϵ="FP64", phase=1, x1, x2, y1, y2, dx, dy, NIC=16, ext=0)
    # input check
    x1 < x2 || throw(ArgumentError("x1 should be less than x2"))
    y1 < y2 || throw(ArgumentError("y1 should be less than y2"))
    dx > 0 && dy > 0 || throw(ArgumentError("dx and dy should be positive"))
    # default values
    phase = phase in [1, 2] ? phase : 1
    NIC = NIC in [4, 16] ? NIC : 16
    ext = ext == 0 ? TempGridExtra(rand(2)) : ext
    ϵ == ϵ in ["FP64", "FP32"] ? ϵ : "FP64"
    T1 = ϵ == "FP64" ? Int64 : Int32
    T2 = ϵ == "FP64" ? Float64 : Float32
    # set the nodes in background grid
    vx = x1:dx:x2 |> collect
    vy = y1:dy:y2 |> collect
    sort!(vx); sort!(vy, rev=true) # vy should from largest to smallest
    nnx = length(vx); vx = reshape(vx, 1, nnx)
    nny = length(vy); vy = reshape(vy, nny, 1)
    ni  = nny*nnx
    x   = repeat(vx, nny, 1) |> vec
    y   = repeat(vy, 1, nnx) |> vec
    ξ   = hcat(x, y)
    # set the cells in background grid
    ncx = nnx - 1
    ncy = nny - 1
    nc  = ncy * ncx
    # grid properties setup
    DoF = 2
    phase == 1 ? (nc_new = 1 ; ni_new = 1 ; DoF_new = 1  ) : 
    phase == 2 ? (nc_new = nc; ni_new = ni; DoF_new = DoF) : nothing
    σm  = zeros(T2, nc             )
    σw  = zeros(T2, nc_new         )
    Ω   = zeros(T2, nc             )
    ms  = zeros(T2, ni             )
    mw  = zeros(T2, ni_new         )
    mi  = zeros(T2, ni_new         )
    ps  = zeros(T2, ni    , DoF    )
    pw  = zeros(T2, ni_new, DoF_new)
    vs  = zeros(T2, ni    , DoF    )
    vw  = zeros(T2, ni_new, DoF_new)
    vsT = zeros(T2, ni    , DoF    )
    vwT = zeros(T2, ni_new, DoF_new)
    as  = zeros(T2, ni    , DoF    )
    aw  = zeros(T2, ni_new, DoF_new)
    fs  = zeros(T2, ni    , DoF    )
    fw  = zeros(T2, ni_new, DoF_new)
    fd  = zeros(T2, ni_new, DoF_new)
    Δus = zeros(T2, ni    , DoF    )
    Δuw = zeros(T2, ni_new, DoF_new)
    # set the computing cell to node topology
    if NIC == 4
        p2nD = T1.([0 1; 0 0; nny 1; nny 0])
    elseif NIC == 16
        p2nD = T1.([
            -nny   2; -nny   1; -nny   0; -nny   -1       
                0   2;    0   1;    0   0;    0   -1     
                nny   2;  nny   1;  nny   0;  nny   -1       
                nny*2 2;  nny*2 1;  nny*2 0;  nny*2 -1 
        ])
    end

    tmp = Grid2D{T1, T2, AbstractArray{T1, 2}, AbstractArray{T2, 1}, AbstractArray{T2, 2}, 
        UserGridExtra}(phase, x1, x2, y1, y2, dx, dy, nnx, nny, ni, NIC, ξ, ncx, ncy, nc, 
        p2nD, σm, σw, Ω, ms, mw, mi, ps, pw, vs, vw, vsT, vwT, fs, fw, fd, as, aw, Δus, Δuw,
        ext)
    return adapt(Array, tmp)
end

function Base.show(io::IO, grid::T) where {T<:DeviceGrid2D}
    typeof(grid).parameters[2]==Float64 ? precision="FP64" : 
    typeof(grid).parameters[2]==Float32 ? precision="FP32" : nothing
    # printer
    print(io, "DeviceGrid2D:"                              , "\n")
    print(io, "┬", "─" ^ 12                                , "\n")
    print(io, "├─ ", "phase  : ", grid.phase               , "\n")
    print(io, "├─ ", "NIC    : ", grid.NIC                 , "\n")
    print(io, "├─ ", "ϵ      : ", precision                , "\n")
    print(io, "├─ ", "x1 - x2: ", "$(grid.x1) - $(grid.x2)", "\n")
    print(io, "├─ ", "y1 - y2: ", "$(grid.y1) - $(grid.y2)", "\n")
    print(io, "├─ ", "dx - dy: ", "$(grid.dx) - $(grid.dy)", "\n")
    print(io, "├─ ", "nc     : ", @sprintf("%.2e", grid.nc), "\n")
    print(io, "└─ ", "ni     : ", @sprintf("%.2e", grid.ni), "\n")
    return nothing
end
#=↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑=#



#=-----------------------------------------------------------------------------------------#
|    3D Grid System                                                                        |
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓=#
struct Grid3D{T1, T2,
    T3 <: AbstractArray, # Array{Int  , 2} 
    T4 <: AbstractArray, # Array{Float, 1}
    T5 <: AbstractArray, # Array{Float, 2}
    T6 <: UserGridExtra
} <: DeviceGrid3D{T1, T2}
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
    ext   :: T6
end

@user_struct Grid3D

function UserGrid3D(; ϵ="FP64", phase=1, x1, x2, y1, y2, z1, z2, dx, dy, dz, NIC=64, ext=0)
    # input check
    x1 < x2 || throw(ArgumentError("x1 should be less than x2"))
    y1 < y2 || throw(ArgumentError("y1 should be less than y2"))
    z1 < z2 || throw(ArgumentError("z1 should be less than z2"))
    dx > 0 && dy > 0 && dz > 0 || 
        throw(ArgumentError("dx, dy, and dz should be positive"))
    # default values
    phase = phase in [1, 2] ? phase : 1
    NIC = NIC in [8, 64] ? NIC : 64
    ext = ext == 0 ? TempGridExtra(rand(2)) : ext
    ϵ == ϵ in ["FP64", "FP32"] ? ϵ : "FP64"
    T1 = ϵ == "FP64" ? Int64 : Int32
    T2 = ϵ == "FP64" ? Float64 : Float32
    # set the nodes in background grid
    vx = x1:dx:x2 |> collect
    vy = y1:dy:y2 |> collect
    vz = z1:dz:z2 |> collect
    m, n, o = length(vy), length(vx), length(vz)
    vx  = reshape(vx, 1, n, 1)
    vy  = reshape(vy, m, 1, 1)
    vz  = reshape(vz, 1, 1, o)
    om  = ones(Int, m)
    on  = ones(Int, n)
    oo  = ones(Int, o)
    x   = vec(vx[om, :, oo])
    y   = vec(vy[:, on, oo])
    z   = vec(vz[om, on, :])
    ξ   = hcat(x, y, z)
    nnx = length(vx)
    nny = length(vy)
    nnz = length(vz)
    ni  = nnx * nny * nnz
    # set the cells in background grid
    ncx = nnx - 1
    ncy = nny - 1
    ncz = nnz - 1
    nc  = ncx * ncy * ncz
    # grid properties setup
    DoF = 3
    phase == 1 ? (nc_new = 1 ; ni_new = 1 ; DoF_new = 1  ) : 
    phase == 2 ? (nc_new = nc; ni_new = ni; DoF_new = DoF) : nothing
    σm  = zeros(T2, nc             )
    σw  = zeros(T2, nc_new         )
    Ω   = zeros(T2, nc             )
    ms  = zeros(T2, ni             )
    mw  = zeros(T2, ni_new         )
    mi  = zeros(T2, ni_new         )
    ps  = zeros(T2, ni    , DoF    )
    pw  = zeros(T2, ni_new, DoF_new)
    vs  = zeros(T2, ni    , DoF    )
    vw  = zeros(T2, ni_new, DoF_new)
    vsT = zeros(T2, ni    , DoF    )
    vwT = zeros(T2, ni_new, DoF_new)
    as  = zeros(T2, ni    , DoF    )
    aw  = zeros(T2, ni_new, DoF_new)
    fs  = zeros(T2, ni    , DoF    )
    fw  = zeros(T2, ni_new, DoF_new)
    fd  = zeros(T2, ni_new, DoF_new)
    Δus = zeros(T2, ni    , DoF    )
    Δuw = zeros(T2, ni_new, DoF_new)
    # set the computing cell to node topology
    if NIC == 8
        p2nD = T1.([
            -1 -1 -1; -1 0 -1; 0 0 -1; 0 -1 -1
            -1 -1  0; -1 0  0; 0 0  0; 0 -1  0
        ])
    elseif NIC == 64
        p2nD = T1.([
            -1 -1 -1 +0; -1 -0 -1 +0; -0 -0 -1 +0; -0 -1 -1 +0; -0 -2 -1 +0; -1 -2 -1 +0 
            -2 -2 -1 +0; -2 -1 -1 +0; -2 -0 -1 +0; -2 +1 -1 +0; -1 +1 -1 +0; -0 +1 -1 +0 
            +1 +1 -1 +0; +1 -0 -1 +0; +1 -1 -1 +0; +1 -2 -1 +0; -1 -1 -1 +1; -1 -0 -1 +1 
            -0 -0 -1 +1; -0 -1 -1 +1; -0 -2 -1 +1; -1 -2 -1 +1; -2 -2 -1 +1; -2 -1 -1 +1 
            -2 -0 -1 +1; -2 +1 -1 +1; -1 +1 -1 +1; -0 +1 -1 +1; +1 +1 -1 +1; +1 -0 -1 +1 
            +1 -1 -1 +1; +1 -2 -1 +1; -1 -1 -0 +1; -1 -0 -0 +1; -0 -0 -0 +1; -0 -1 -0 +1 
            -0 -2 -0 +1; -1 -2 -0 +1; -2 -2 -0 +1; -2 -1 -0 +1; -2 -0 -0 +1; -2 +1 -0 +1 
            -1 +1 -0 +1; -0 +1 -0 +1; +1 +1 -0 +1; +1 -0 -0 +1; +1 -1 -0 +1; +1 -2 -0 +1 
            -1 -1 -0 +2; -1 -0 -0 +2; -0 -0 -0 +2; -0 -1 -0 +2; -0 -2 -0 +2; -1 -2 -0 +2 
            -2 -2 -0 +2; -2 -1 -0 +2; -2 -0 -0 +2; -2 +1 -0 +2; -1 +1 -0 +2; -0 +1 -0 +2 
            +1 +1 -0 +2; +1 -0 -0 +2; +1 -1 -0 +2; +1 -2 -0 +2
        ])
    end

    tmp = Grid3D{T1, T2, AbstractArray{T1, 2}, AbstractArray{T2, 1}, AbstractArray{T2, 2}, 
        UserGridExtra}(phase, x1, x2, y1, y2, z1, z2, dx, dy, dz, nnx, nny, nnz, ni, NIC, ξ, 
        ncx, ncy, ncz, nc, p2nD, σm, σw, Ω, ms, mw, mi, ps, pw, vs, vw, vsT, vwT, fs, fw, 
        fd, as, aw, Δus, Δuw, ext)
    return adapt(Array, tmp)
end

function Base.show(io::IO, grid::T) where {T<:DeviceGrid3D}
    typeof(grid).parameters[2]==Float64 ? precision="FP64" : 
    typeof(grid).parameters[2]==Float32 ? precision="FP32" : nothing
    # printer
    print(io, "DeviceGrid3D:"                                           , "\n")
    print(io, "┬", "─" ^ 12                                             , "\n")
    print(io, "├─ ", "phase  : ", grid.phase                            , "\n")
    print(io, "├─ ", "NIC    : ", grid.NIC                              , "\n")
    print(io, "├─ ", "ϵ      : ", precision                             , "\n")
    print(io, "├─ ", "x1 - x2: ", "$(grid.x1) - $(grid.x2)"             , "\n")
    print(io, "├─ ", "y1 - y2: ", "$(grid.y1) - $(grid.y2)"             , "\n")
    print(io, "├─ ", "z1 - z2: ", "$(grid.z1) - $(grid.z2)"             , "\n")
    print(io, "├─ ", "d x-y-z: ", "$(grid.dx) - $(grid.dy) - $(grid.dz)", "\n")
    print(io, "├─ ", "nc     : ", @sprintf("%.2e", grid.nc)             , "\n")
    print(io, "└─ ", "ni     : ", @sprintf("%.2e", grid.ni)             , "\n")
    return nothing
end
#=↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑=#