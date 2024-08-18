using CUDA
using KernelAbstractions
using BenchmarkTools

@kernel inbounds = true function MWEkernel1!(a, b, c)
    ix = @index(Global)
    if ix ≤ size(a, 1)
        tmp1 = c[ix, 1]
        tmp2 = c[ix, 2]
        tmp3 = c[ix, 3]
        tmp4 = c[ix, 4]
        t1, t2 = callf1(a[ix, 1], a[ix, 2], a[ix, 3], a[ix, 4], a[ix, 5], a[ix, 6])
        t3, t4 = callf2(b[ix, 1], b[ix, 2], b[ix, 3], b[ix, 4], b[ix, 5], b[ix, 6])
        t5, t6 = callf3(tmp1, tmp2, tmp3, tmp4)
        a[ix, 1] = abs(t1 - t2) > 10 ? 1 : abs(t1 - t2)
        a[ix, 2] = abs(t3 - t4) > 10 ? 1 : abs(t3 - t4)
        a[ix, 3] = abs(t5 - t6) > 10 ? 1 : abs(t5 - t6)
    end
end


@kernel inbounds = true function MWEkernel2!(a, b, c)
    ix = @index(Global)
    if ix ≤ size(a, 1)
        tmp1 = c[ix, 1]
        tmp2 = c[ix, 2]
        tmp3 = c[ix, 3]
        tmp4 = c[ix, 4]

        tmp11 = a[ix, 1] + sqrt(a[ix, 1]) + a[ix, 1] ^ 3 - sin(a[ix, 1])
        tmp12 = a[ix, 2] + sqrt(a[ix, 2]) + a[ix, 2] ^ 4 - cos(a[ix, 2])
        tmp13 = a[ix, 3] + sqrt(a[ix, 3]) + a[ix, 3] ^ 5 - sin(a[ix, 3])
        tmp14 = a[ix, 4] + sqrt(a[ix, 4]) + a[ix, 4] ^ 6 - cos(a[ix, 4])
        tmp15 = a[ix, 5] + sqrt(a[ix, 5]) + a[ix, 5] ^ 7 - tan(a[ix, 5])
        tmp16 = a[ix, 6] + sqrt(a[ix, 6]) + a[ix, 6] ^ 2 - tan(a[ix, 6])
        t1 = tmp11+tmp12+tmp13
        t2 = tmp14+tmp15+tmp16

        tmp21 = b[ix, 2] + sqrt(b[ix, 1]) + b[ix, 1] ^ 3 - sin(b[ix, 1])
        tmp22 = b[ix, 1] + sqrt(b[ix, 2]) + b[ix, 2] ^ 4 - cos(b[ix, 2])
        tmp23 = b[ix, 4] + sqrt(b[ix, 3]) + b[ix, 3] ^ 5 - sin(b[ix, 3])
        tmp24 = b[ix, 3] + sqrt(b[ix, 4]) + b[ix, 4] ^ 6 - cos(b[ix, 4])
        tmp25 = b[ix, 6] + sqrt(b[ix, 5]) + b[ix, 5] ^ 7 - tan(b[ix, 5])
        tmp26 = b[ix, 1] + sqrt(b[ix, 6]) + b[ix, 6] ^ 2 - tan(b[ix, 6])
        t3 = tmp21+tmp22+tmp23
        t4 = tmp24+tmp25+tmp26 

        if tmp1 < 0.5
            tm1 = tmp1 + sqrt(tmp1) + tmp1 ^ 3 - sin(tmp1)
            tm2 = tmp2 + sqrt(tmp2) + tmp2 ^ 4 - cos(tmp2)
            tm3 = tmp3 + sqrt(tmp3) + tmp3 ^ 5 - sin(tmp3)
            tm4 = tmp4 + sqrt(tmp4) + tmp4 ^ 6 - cos(tmp4)
        elseif tmp1 > 0.7
            tm1 = tmp1 + sqrt(tmp1) + tmp1 ^ 3 - sin(tmp1)
            tm2 = tmp3 + sqrt(tmp2) + tmp2 ^ 4 - cos(tmp2)
            tm3 = tmp2 + sqrt(tmp3) + tmp3 ^ 5 - sin(tmp3)
            tm4 = tmp4 + sqrt(tmp4) + tmp4 ^ 6 - cos(tmp4)
        else
            tm1 = 2
            tm2 = 3
            tm3 = 4
            tm4 = 5
        end
        t5 = tm1+tm2
        t6 = tm3+tm4

        a[ix, 1] = abs(t1 - t2) > 10 ? 1 : abs(t1 - t2)
        a[ix, 2] = abs(t3 - t4) > 10 ? 1 : abs(t3 - t4)
        a[ix, 3] = abs(t5 - t6) > 10 ? 1 : abs(t5 - t6)
    end
end

@inline function callf1(a1, a2, a3, a4, a5, a6)
    tmp1 = a1 + sqrt(a1) + a1 ^ 3 - sin(a1)
    tmp2 = a2 + sqrt(a2) + a2 ^ 4 - cos(a2)
    tmp3 = a3 + sqrt(a3) + a3 ^ 5 - sin(a3)
    tmp4 = a4 + sqrt(a4) + a4 ^ 6 - cos(a4)
    tmp5 = a5 + sqrt(a5) + a5 ^ 7 - tan(a5)
    tmp6 = a6 + sqrt(a6) + a6 ^ 2 - tan(a6)
    return tmp1+tmp2+tmp3, tmp4+tmp5+tmp6
end

@inline function callf2(a1, a2, a3, a4, a5, a6)
    tmp1 = a2 + sqrt(a1) + a1 ^ 3 - sin(a1)
    tmp2 = a1 + sqrt(a2) + a2 ^ 4 - cos(a2)
    tmp3 = a4 + sqrt(a3) + a3 ^ 5 - sin(a3)
    tmp4 = a3 + sqrt(a4) + a4 ^ 6 - cos(a4)
    tmp5 = a6 + sqrt(a5) + a5 ^ 7 - tan(a5)
    tmp6 = a1 + sqrt(a6) + a6 ^ 2 - tan(a6)
    return tmp1+tmp2+tmp3, tmp4+tmp5+tmp6
end

@inline function callf3(a1, a2, a3, a4)
    if a1 < 0.5
        tmp1 = a1 + sqrt(a1) + a1 ^ 3 - sin(a1)
        tmp2 = a2 + sqrt(a2) + a2 ^ 4 - cos(a2)
        tmp3 = a3 + sqrt(a3) + a3 ^ 5 - sin(a3)
        tmp4 = a4 + sqrt(a4) + a4 ^ 6 - cos(a4)
    elseif a1 > 0.7
        tmp1 = a1 + sqrt(a1) + a1 ^ 3 - sin(a1)
        tmp2 = a3 + sqrt(a2) + a2 ^ 4 - cos(a2)
        tmp3 = a2 + sqrt(a3) + a3 ^ 5 - sin(a3)
        tmp4 = a4 + sqrt(a4) + a4 ^ 6 - cos(a4)
    else
        tmp1 = 2
        tmp2 = 3
        tmp3 = 4
        tmp4 = 5
    end
    return tmp1+tmp2, tmp3+tmp4
end

n = 102400000
a = CUDA.rand(n, 6)
b = CUDA.rand(n, 6)
c = CUDA.rand(n, 4)

MWEkernel1!(CUDABackend())(ndrange=n, a, b, c)
@benchmark CUDA.@sync MWEkernel1!($CUDABackend())(ndrange=$n, $a, $b, $c)
#=
v100:
BenchmarkTools.Trial: 230 samples with 1 evaluation.
 Range (min … max):  21.657 ms … 21.863 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     21.729 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   21.735 ms ± 37.302 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

              ▃ ▁▆▄▆▁▁█▃▁▁▂▃ ▄▆▂ ▁▃                            
  ▃▁▃▁▁▃▃▄▄▇█▃█▇████████████▆██████▆▄▄▇▇▄▇▃▆▄▃▆▄▄▄█▄▃▁▇▄▃▁▆▁▃ ▄
  21.7 ms         Histogram: frequency by time        21.8 ms <

 Memory estimate: 2.45 KiB, allocs estimate: 62.

rtx3090:
BenchmarkTools.Trial: 295 samples with 1 evaluation.
 Range (min … max):  15.764 ms …  17.418 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     16.925 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   16.950 ms ± 153.605 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

                                         ▁▆ ▂█   ▆              
  ▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▁▄██▅██▄▃▆█▅▅▃▃▃▁▃▁▁▃▃▂ ▃
  15.8 ms         Histogram: frequency by time         17.4 ms <

 Memory estimate: 2.45 KiB, allocs estimate: 62.
=#

MWEkernel2!(CUDABackend())(ndrange=n, a, b, c)
@benchmark CUDA.@sync MWEkernel2!($CUDABackend())(ndrange=$n, $a, $b, $c)
#=
v100:
BenchmarkTools.Trial: 171 samples with 1 evaluation.
 Range (min … max):  29.204 ms … 29.415 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     29.300 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   29.297 ms ± 34.494 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

                               ▃ ▃▂   █ ▄ ▃ ▂    ▂             
  ▃▁▁▁▁▃▃▃▁▁▃▁▅█▆▅▃▃▆▅▃▇█▆▃▆▅▆▆█▇███▆▆███▃█▆█▅█▇▇██▅▁▇▇▁▅▅▃▁▆ ▃
  29.2 ms         Histogram: frequency by time        29.4 ms <

 Memory estimate: 2.45 KiB, allocs estimate: 62.

rtx3090:
BenchmarkTools.Trial: 263 samples with 1 evaluation.
 Range (min … max):  17.946 ms …  20.262 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     19.021 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   19.064 ms ± 195.571 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

                                ▇▃ █▇▁ ▃▄  ▄                    
  ▄▁▁▁▁▁▁▁▁▁▁▁▁▄▁▁▁▁▁▁▁▁▁▁▁▁▄▁▁▄██▆███▇██▇▆█▇▇▆▁▇▄▄▁▁▁▄▁▁▁▁▁▁▄ ▆
  17.9 ms       Histogram: log(frequency) by time      19.8 ms <

 Memory estimate: 2.45 KiB, allocs estimate: 62.
=#