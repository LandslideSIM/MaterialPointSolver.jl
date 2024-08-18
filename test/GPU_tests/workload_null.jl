using KernelAbstractions
using CUDA
using BenchmarkTools

@kernel inbounds = true function test1!(a)
    ix = @index(Global)
    if ix ≤ size(a, 1)
        if a[ix] != 0
            a[ix] = sin(a[ix]) + sqrt(abs(a[ix])) - a[ix] * 2 + a[ix] ^ 3
        end
    end
end

@kernel inbounds = true function test2!(a)
    ix = @index(Global)
    if ix ≤ size(a, 1)
        a[ix] = sin(a[ix]) + sqrt(abs(a[ix])) - a[ix] * 2 + a[ix] ^ 3
    end
end

@kernel inbounds = true function test3!(a)
    ix = @index(Global)
    if ix ≤ Int(1024000000 / 2)
        a[ix] = sin(a[ix]) + sqrt(abs(a[ix])) - a[ix] * 2 + a[ix] ^ 3
    end
end

n = 1024000000
a = CUDA.rand(n)
a[Int(n / 2) : end] .= 0
b = CUDA.rand(Int(n / 2))


@benchmark CUDA.@sync test1!($CUDABackend())(ndrange=$size(a, 1), $a)
#=
BenchmarkTools.Trial: 600 samples with 1 evaluation.
 Range (min … max):  8.324 ms …  8.519 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     8.335 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   8.337 ms ± 14.957 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

      ▁▄▅██▃▁                                                 
  ▂▃▄▇████████▅▄▃▂▃▂▂▂▃▃▂▃▂▂▁▂▂▂▂▂▁▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂ ▃
  8.32 ms        Histogram: frequency by time         8.4 ms <

 Memory estimate: 1.56 KiB, allocs estimate: 51.
=#

@benchmark CUDA.@sync test2!($CUDABackend())(ndrange=$size(a, 1), $a)
#=
BenchmarkTools.Trial: 457 samples with 1 evaluation.
 Range (min … max):  10.921 ms … 11.030 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     10.940 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   10.944 ms ± 16.128 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

      ▁▂▆▇█▃▂▁     ▃ ▁      ▁▁                                 
  ▃▅▄▇██████████▅▇▇███▆▆▆▇▅▇██▇▇▇▄▄▂▃▃▃▂▁▂▂▂▃▃▄▃▁▂▁▁▁▂▁▁▁▁▁▁▃ ▄
  10.9 ms         Histogram: frequency by time          11 ms <

 Memory estimate: 1.56 KiB, allocs estimate: 51.
=#

@benchmark CUDA.@sync test3!($CUDABackend())(ndrange=$size(b, 1), $b)
#=
BenchmarkTools.Trial: 912 samples with 1 evaluation.
 Range (min … max):  5.446 ms …  8.110 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     5.465 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   5.473 ms ± 92.808 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

   ▄▅█▃    ▂▄▃     ▂▁      ▁                                  
  ▆█████▇▅▅███▆▅▅▅▆██▇▆▆▄▇██▆▄▄▃▃▄▃▃▃▃▂▁▃▂▂▂▂▁▁▂▂▁▁▁▁▂▁▁▂▂▁▂ ▄
  5.45 ms        Histogram: frequency by time        5.54 ms <

 Memory estimate: 1.56 KiB, allocs estimate: 51.
=#