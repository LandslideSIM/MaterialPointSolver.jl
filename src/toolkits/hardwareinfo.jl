#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : hardwareinfo.jl                                                            |
|  Description: check the information of CPU or GPU                                        |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. cpu_info                                                                |
|               2. gpu_info                                                                |
|               3. gpus_info                                                               |
+==========================================================================================#

function cpu_info(; verbose=false)
    display(CpuId.cpuinfo())
    verbose == true ? (println(); display(CpuId.cpufeaturetable())) : nothing
    return nothing
end

# todo:
# single GPU
# multi GPU