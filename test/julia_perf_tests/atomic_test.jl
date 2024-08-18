using CUDA
using BenchmarkTools

n = 2^25

function no_atomic_host!(a, b)
    for i in eachindex(a)
        a[b[i]] = b[i]^2
    end
    return nothing
end

function no_atomic_gpu!(a, b)
    function kernel!(a, b)
        ix = (blockIdx().x-1)*blockDim().x+threadIdx().x
        if ix ≤ length(a)
            a[b[ix]] = b[ix]^2
        end
        return nothing
    end
    @cuda threads=1024 blocks=length(a)÷1024 kernel!(a, b)
    return nothing
end

begin
    a   = zeros(n)
    b   = collect(1:n)
    no_atomic_host!(a, b)
    # rst = @belapsed no_atomic_host!($a, $b)
    # 0.02940021
end

begin
    da = CuArray(zeros(n))
    db = CuArray(collect(1:n))
    CUDA.@sync no_atomic_gpu!(da, db)
    rst = @belapsed CUDA.@sync no_atomic_gpu!($da, $db)
    # 0.000647056
end

a ≈ Array(da)
b ≈ Array(db)
@info "t1 done"




#####################################################

function atomic_host!(a, b)
    for i in eachindex(a)
        a[b[i]] += b[i]^2
    end
    return nothing
end

function atomic_gpu!(a, b)
    function kernel!(a, b)
        ix = (blockIdx().x-1)*blockDim().x+threadIdx().x
        if ix ≤ length(a)
            CUDA.@atomic a[b[ix]] = +(a[b[ix]], b[ix]^2)
        end
        return nothing
    end
    @cuda threads=1024 blocks=length(a)÷1024 kernel!(a, b)
    return nothing
end


begin
    rer = 16*4
    a   = zeros(n)
    b   = vec(repeat(collect(1:Int(n/rer)), 1, rer))
    atomic_host!(a, b)
    #rst = @belapsed atomic_host!($a, $b)
    # 0.029888254
end

begin
    rer = 16*4
    da  = CuArray(zeros(n))
    db  = CuArray(vec(repeat(collect(1:Int(n/rer)), 1, rer)))
    CUDA.@sync no_atomic_gpu!(da, db)
    rst = @belapsed CUDA.@sync no_atomic_gpu!($da, $db)
    # 0.000528016
end

a ≈ Array(da)
b ≈ Array(db)