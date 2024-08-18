#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : hardwareinfo.jl                                                            |
|  Description: check the information of CPU or GPU                                        |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. cpuinfo                                                                 |
+==========================================================================================#

export cpuinfo

const cpuinfo()  = SysInfo.sysinfo() 
# todo:
# single GPU
# multi GPU