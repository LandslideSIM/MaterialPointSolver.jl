#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : randomfield_oneAPI.jl                                                      |
|  Description: Gaussian Random Field with 1) Gaussian covariance generation and           |
|               2) Gaussian Random Field with exponential covariance generation            |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. grf_gc!                                                                 |
|               2. grf_ec!                                                                 |
|  References : Räss, Ludovic, Dmitriy Kolyukhin, and Alexander Minakov. "Efficient        |
|               parallel random field generator for large 3-D geophysical problems."       |
|               Computers & geosciences 131 (2019): 158-169.                               |
+==========================================================================================#
