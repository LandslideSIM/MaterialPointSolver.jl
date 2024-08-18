using CUDA
using KernelAbstractions
using BenchmarkTools
using Printf

macro KAatomic(expr)
    esc(quote
        KernelAbstractions.@atomic :monotonic $expr
    end)
end

CUDA.device!(1)

@kernel inbounds = true function atomic_add!(a, b, c)
    ix = @index(Global)
    if ix ≤ size(a, 1)
        @KAatomic a[b[ix], 1] += c[ix, 1]
        @KAatomic a[b[ix], 2] += c[ix, 2]
        @KAatomic a[b[ix], 3] += c[ix, 3]
    end
end

@kernel inbounds = true function normal_add!(a, b, c)
    ix = @index(Global)
    if ix ≤ size(a, 1)
        a[b[ix], 1] += c[ix, 1]
        a[b[ix], 2] += c[ix, 2]
        a[b[ix], 3] += c[ix, 3]
    end
end

function normal_add_host!(a, b, c)
    @inbounds for ix in 1:size(a, 1)
        a[b[ix], 1] += c[ix, 1]
        a[b[ix], 2] += c[ix, 2]
        a[b[ix], 3] += c[ix, 3]
    end
end

function no_race_condition()
    @inbounds for i in 1:1:4
        n = 102400000*i |> Int64
        a = CUDA.zeros(n, 3)
        b = cu(collect(1:1:n) .|> Int32)
        c = CUDA.rand(n, 3)

        time1 = @sprintf("%.2f", 1e3 * @belapsed begin
            normal_add!($(CUDABackend()))(ndrange=$n, $a, $b, $c)
            CUDA.synchronize()
        end)
        time2 = @sprintf("%.2f", 1e3 * @belapsed begin
            atomic_add!($(CUDABackend()))(ndrange=$n, $a, $b, $c)
            CUDA.synchronize()
        end)

        @info """data size $n
        --------------------------
        normal_add took $time1 ms
        atomic_add took $time2 ms
        """
        println()

        CUDA.unsafe_free!(a)
        CUDA.unsafe_free!(b)
        CUDA.unsafe_free!(c)
        CUDA.reclaim()
    end
    return nothing
end

function race_condition()
    @inbounds for i in 1:1:4
        n = 102400000*i |> Int64
        a = CUDA.zeros(n, 3)
        b = cu(ones(Int32, n))
        c = CUDA.rand(n, 3)
        A = Array(a)
        B = Array(b)
        C = Array(c)

        time1 = @sprintf("%.2f", 1e3 * @belapsed begin
            atomic_add!($(CUDABackend()))(ndrange=$n, $a, $b, $c)
            CUDA.synchronize()
        end)

        time2 = @sprintf("%.2f", 1e3 * @belapsed begin
            normal_add_host!($A, $B, $C)
        end)

        @info """data size $n
        --------------------------
        atomic_add took $time1 ms
        normal_host took $time2 ms
        """
        println()

        CUDA.unsafe_free!(a)
        CUDA.unsafe_free!(b)
        CUDA.unsafe_free!(c)
        CUDA.reclaim()
    end
    return nothing
end

no_race_condition()
race_condition()