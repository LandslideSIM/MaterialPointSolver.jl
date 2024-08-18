#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : randomfield.jl                                                             |
|  Description: Gaussian Random Field with 1) Gaussian covariance generation and           |
|               2) Gaussian Random Field with exponential covariance generation            |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. grf_gc                                                                  |
|               2. grf_ec                                                                  |
|  References : Räss, Ludovic, Dmitriy Kolyukhin, and Alexander Minakov. "Efficient        |
|               parallel random field generator for large 3-D geophysical problems."       |
|               Computers & geosciences 131 (2019): 158-169.                               |
+==========================================================================================#

@kernel inbounds=true function compute1!(Yf, a, b, V1, V2, V3, pts)
    i = @index(Global)
    if i ≤ length(Yf)
        tmp = pts[i, 1] * V1 + pts[i, 2] * V2 + pts[i, 3] * V3
        Yf[i] += a * sin(tmp) + b * cos(tmp)
    end
end

@kernel inbounds=true function compute2!(Yf, C)
    i = @index(Global)
    if i ≤ length(Yf)
        Yf[i] = Yf[i] * C
    end
end

@kernel inbounds=true function compute3!(Yf, max_val, min_val, rl, rr)
    i = @index(Global)
    if i ≤ length(Yf)
        Yf[i] = ((Yf[i] - min_val) / (max_val - min_val)) * (rr - rl) + rl
    end
end

"""
    grf_gc!(Yf, pts, rl, rr; precision=Float64, If=2.0, k_m=100.0, Nh=5000, sf=1)

Description:
---
`Yf` is the scalar array to store the generated random field, `pts` is the coordinates of
the particles size = [pts_num, 3]. `rl`/`rr` are the left/right boundary of the new range.

```txt
sf  标准偏差
If  相关长度
Nh  内部参数, 谐波数
k_m 波数的最大值
```
"""
function grf_gc!(Yf, pts, rl, rr; precision=Float64, If=2.0, k_m=100.0, Nh=5000, sf=1)
    T2    = precision
    C     = sf / sqrt(Nh) |> T2
    rl    = min(rl, rr)   |> T2
    rr    = max(rl, rr)   |> T2
    c_pts = CuArray(T2.(pts))
    c_Yf  = CuArray(T2.(Yf))
    dev   = CUDABackend()
    @info "GPU data ready"
    for _ = 1:Nh
        fi = T2(6.283185) * rand(T2)
        lf = T2(1.128379) * T2(If)
        k  = T2(0)
        flag = true
        while flag
            k = T2(k_m) * rand(T2)
            d = k * k * exp(-0.5 * k * k) |> T2
            flag = rand(T2) * T2(0.735759) >= d
        end
        k = T2(1.414214) * k / lf
        theta = acos(1 - 2 * rand(T2))
        V1 = k * sin(fi) * sin(theta) |> T2
        V2 = k * cos(fi) * sin(theta) |> T2
        V3 = k * cos(theta)           |> T2
        a = randn(T2) 
        b = randn(T2)
        compute1!(dev)(ndrange=length(c_Yf), c_Yf, a, b, V1, V2, V3, c_pts)
    end
    compute2!(dev)(ndrange=length(c_Yf), c_Yf, C)
    max_val = reduce(max, c_Yf)
    min_val = reduce(min, c_Yf)
    compute3!(dev)(ndrange=length(c_Yf), c_Yf, max_val, min_val, rl, rr)
    @info "downloading data from GPU"
    Yf .= Array(c_Yf)
end

"""
    grf_ec!(Yf, pts, rl, rr; precision=Float64, If=[10.0, 8.0, 5.0], Nh=5000, sf=1)

Description:
---
`Yf` is the scalar array to store the generated random field, `pts` is the coordinates of
the particles size = [pts_num, 3]. `rl`/`rr` are the left/right boundary of the new range.

```txt
sf  标准偏差
If  指数协方差的相关长度 [x, y, z]
Nh  内部参数, 谐波数
```
"""
function grf_ec!(Yf, pts, rl, rr; precision=Float64, If=[10.0, 8.0, 5.0], Nh=5000, sf=1)
    T2    = precision
    C     = sf / sqrt(Nh) |> T2
    rl    = min(rl, rr)   |> T2
    rr    = max(rl, rr)   |> T2
    c_pts = CuArray(T2.(pts))
    c_Yf  = CuArray(T2.(Yf))
    dev   = CUDABackend()
    @info "GPU data ready"
    for _ = 1:Nh
        fi = T2(6.283185) * rand(T2)
        k  = T2(0)
        flag = true
        while flag
            k = tan(T2(1.570796) * rand())
            d = (k * k) / (T2(1.0) + (k * k))
            if rand() < d
                flag = false
            end
        end
        theta = acos(1 - 2 * rand())
        V1 = k * sin(fi) * sin(theta) / If[1] |> T2
        V2 = k * cos(fi) * sin(theta) / If[2] |> T2
        V3 = k * cos(theta) / If[3]           |> T2
        a = randn(T2)
        b = randn(T2)
        compute1!(dev)(ndrange=length(c_Yf), c_Yf, a, b, V1, V2, V3, c_pts)
    end
    compute2!(dev)(ndrange=length(c_Yf), c_Yf, C)
    max_val = reduce(max, c_Yf)
    min_val = reduce(min, c_Yf)
    compute3!(dev)(ndrange=length(c_Yf), c_Yf, max_val, min_val, rl, rr)
    @info "downloading data from GPU"
    Yf .= Array(c_Yf)
end