using CUDA
using BenchmarkTools
using KernelAbstractions
using Printf
using NVTX

const dev_src = 0
const dev_dst = 1
const n       = 1024

@kernel inbounds = true function test1!(x, y)
    ix = @index(Global)
    x[ix] = y[ix]
end

function main1!(x, y)
    Mem.maybe_enable_peer_access(CuDevice(dev_src), CuDevice(dev_dst))
    CUDA.device!(dev_src)
    exeti = @belapsed begin
        test1!($CUDABackend())(ndrange=$(size(x)), $x, $y)
        CUDA.synchronize()
    end
    content = @sprintf("%.2e", exeti)
    @info "CUDA P2P $content s"
    return nothing
end

function main2!(x, y)
    CUDA.device!(dev_src)
    exeti = @belapsed begin
        copyto!($x, $y)
        CUDA.synchronize()
    end
    content = @sprintf("%.2e", exeti)
    @info "CUDA.jl built-in P2P $content s"
    return nothing
end

CUDA.device!(dev_src)
x = CUDA.zeros(n, n, n)
CUDA.device!(dev_dst)
y = CUDA.ones(n, n, n)

# main1!(x, y)


# main2!(x, y)


Mem.maybe_enable_peer_access(CuDevice(dev_src), CuDevice(dev_dst))
CUDA.device!(dev_src)
test1!(CUDABackend())(ndrange=size(x), x, y)

CUDA.device!(dev_src)
NVTX.@range "copy" test1!(CUDABackend())(ndrange=size(x), x, y)