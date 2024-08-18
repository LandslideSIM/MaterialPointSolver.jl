#=
这个文件是为了测试mpm自动时间步中需要得到最大值的操作的性能。

问题： 
    在V100上进行循环迭代时，内存分配异常且显著慢于cpu版本

当前状态：
    2023.10.27
        在升级到nvidia driver 12.1之后，V100解决了这个问题 ↓
            单循环在测试结果：
            ┌ Info:  
            │ devc1 = 0.0011030903682110395 s
            │ devc2 = 0.0010523640179818064 s
            │ devc3 = 0.0010353740353871773 s
            └ host1 = 0.08485888022033898 s
            多次循环中的测试结果：
            ┌ Info:  
            │ devc1 = 10.923959787 s
            │ devc2 = 10.429344512 s
            │ devc3 = 10.804434995 s
            └ host1 = 830.609115429 s
=#

using CUDA
using BenchmarkTools

data_size = 1e8 |> Int
data_host = rand(Float64, data_size)
data_devc = CuArray(data_host)

function devc_code_ver1(data_devc)
    return findmax(data_devc)[1]
end

function devc_code_ver2(data_devc)
    return reduce(max, data_devc)
end

function devc_code_ver3(data_devc)
    return maximum(data_devc)
end

function host_code_ver1(data_host)
    return maximum(data_host)
end


devc1 = @benchmark CUDA.@sync @inbounds devc_code_ver1($data_devc)
devc2 = @benchmark CUDA.@sync @inbounds devc_code_ver2($data_devc)
devc3 = @benchmark CUDA.@sync @inbounds devc_code_ver3($data_devc)
host1 = @benchmark            @inbounds host_code_ver1($data_host)

@info """ 
devc1 = $(mean(devc1).time/1e9) s
devc2 = $(mean(devc2).time/1e9) s
devc3 = $(mean(devc3).time/1e9) s
host1 = $(mean(host1).time/1e9) s
"""


function iter1(data_devc)
    for i in 1:10000
        devc_code_ver1(data_devc)
    end
    return nothing
end

function iter2(data_devc)
    for i in 1:10000
        devc_code_ver2(data_devc)
    end
    return nothing
end

function iter3(data_devc)
    for i in 1:10000
        devc_code_ver3(data_devc)
    end
    return nothing
end

function iter4(data_host)
    for i in 1:10000
        host_code_ver1(data_host)
    end
    return nothing
end

#=
devc1 = @benchmark CUDA.@sync @inbounds iter1($data_devc)
devc2 = @benchmark CUDA.@sync @inbounds iter2($data_devc)
devc3 = @benchmark CUDA.@sync @inbounds iter3($data_devc)
host1 = @benchmark            @inbounds iter4($data_host)

@info """ 
devc1 = $(mean(devc1).time/1e9) s
devc2 = $(mean(devc2).time/1e9) s
devc3 = $(mean(devc3).time/1e9) s
host1 = $(mean(host1).time/1e9) s
"""
=#