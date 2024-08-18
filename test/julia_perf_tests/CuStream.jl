using NVTX
using CUDA
device!(5)

function kernelA(a)
    i = threadIdx().x
    a[i] *= 2
    return nothing
end

function kernelB(b)
    i = threadIdx().x
    b[i] += 3
    return nothing
end

# 初始化两个向量
N = 256
a = CUDA.fill(1.0f0, N)
b = CUDA.fill(1.0f0, N)

# 创建两个流
stream1 = CuStream()
stream2 = CuStream()

#= 在不同的流上执行核函数
function test1(N, a, b, stream1, stream2)
        @sync begin
            @async begin
                NVTX.@range "stream_1_kernel" @cuda threads=N stream=stream1 kernelA(a)
            end
            @async begin
                NVTX.@range "stream_2_kernel" @cuda threads=N stream=stream2 kernelB(b)
            end
        end
    # 同步流，确保操作完成
    CUDA.@sync(stream1)
    CUDA.@sync(stream2)
    # 检查结果
    a_host = Array(a)  # 将a从设备内存复制到主机内存
    b_host = Array(b)  # 将b从设备内存复制到主机内存
    return a_host, b_host
end

CUDA.@profile external=true a_host, b_host = test1(N, a, b, stream1, stream2)
# a_host, b_host = test1(N, a, b, stream1, stream2)
# println("a after kernelA: ", a_host)
# println("b after kernelB: ", b_host)
=#

function test2(N, a, b)
    NVTX.@range "kernel_1" @cuda threads=N kernelA(a)
    NVTX.@range "kernel_2" @cuda threads=N kernelB(b)
    # copy the results back to host
    a_host = Array(a)
    b_host = Array(b)
    return a_host, b_host
end

CUDA.@profile external=true a_host, b_host = test2(N, a, b)