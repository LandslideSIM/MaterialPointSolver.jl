#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : meshbuilder.jl                                                             |
|  Description: Generate structured mesh in 2&3D space                                     |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : meshbuilder()                                                              |
+==========================================================================================#

function meshbuilder(x::AbstractRange, y::AbstractRange)
    x_tmp = repeat(x', length(y), 1) |> vec
    y_tmp = repeat(y , 1, length(x)) |> vec
    return x_tmp, y_tmp
end

function meshbuilder(x::AbstractRange, y::AbstractRange, z::AbstractRange)
    vx      = x |> collect
    vy      = y |> collect
    vz      = z |> collect
    m, n, o = length(vy), length(vx), length(vz)
    vx      = reshape(vx, 1, n, 1)
    vy      = reshape(vy, m, 1, 1)
    vz      = reshape(vz, 1, 1, o)
    om      = ones(Int, m)
    on      = ones(Int, n)
    oo      = ones(Int, o)
    x_tmp   = vec(vx[om, :, oo])
    y_tmp   = vec(vy[:, on, oo])
    z_tmp   = vec(vz[om, on, :])
    return x_tmp, y_tmp, z_tmp
end