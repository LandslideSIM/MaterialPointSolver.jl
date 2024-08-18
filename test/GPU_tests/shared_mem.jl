using KernelAbstractions
using CUDA

@kernel inbounds=true function shared_kernel!(a, b)
    ix = @index(Global)
    i = @index(Local) 
    N = @uniform @groupsize()[1]
    shared_mem = @localmem eltype(a) N+1
    
    shared_mem[i] = a[ix]
    @synchronize

    b[ix] = shared_mem[i]
end

n = 1024000
a = CUDA.rand(n)
b = CUDA.zeros(n)
shared_kernel!(CUDABackend(), (1024))(ndrange=n, a, b)