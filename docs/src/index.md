# MaterialPointSolver Manual

[![CI](https://github.com/LandslideSIM/MaterialPointSolver.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/LandslideSIM/MaterialPointSolver.jl/actions/workflows/ci.yml) 
[![codecov](https://codecov.io/gh/LandslideSIM/MaterialPointSolver.jl/branch/master/graph/badge.svg?token=5P4XHD79HN)](https://codecov.io/gh/LandslideSIM/MaterialPointSolver.jl) 
[![](https://img.shields.io/badge/docs-stable-blue.svg?logo=quicklook)](https://LandslideSIM.github.io/MaterialPointSolver.jl/stable)

[![](https://img.shields.io/badge/NVIDIA-CUDA-green.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![](https://img.shields.io/badge/AMD-ROCm-red.svg?logo=amd)](https://www.amd.com/en/products/software/rocm.html)
[![](https://img.shields.io/badge/Intel-oneAPI-blue.svg?logo=intel)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
[![](https://img.shields.io/badge/Apple-Metal-purple.svg?logo=apple)](https://developer.apple.com/metal/)


<p>
This package provides a high-performance implementation of the Material Point Method (MPM) in <a href="https://julialang.org" target="_blank"><img src="https://raw.githubusercontent.com/JuliaLang/julia-logo-graphics/master/images/julia.ico" width="16em"> Julia Language</a>. Currently, we are focusing on the run-out processes of landslides, there are two built-in schemes, i.e., one-phase single-point MPM and two-phase single-point MPM, but users also can create/modify their own update schemes by using this package.

While primarily focused on GPU(s) performance, extensive tests show MPMSolver.jl also delivers notable CPU efficiency, often faster than MATLAB's vectorized implementations. It's crucial to recognize that CPUs' limited threading may reduce performance when using synchronization methods like atomic or spin locks without modifing the MPM algorithm or data structure. Therefore, for peak CPU performance, using a single-thread mode is recommended.
</p>

## Installation ⚙️

Just type `]` in Julia's  `REPL`:

```julia
julia> ]
(@1.10) Pkg> add MaterialPointSolver
```

## Features ✨

- [x] sMPM/GIMP
- [x] One/Two-phase single-point MPM
- [x] Mitigates volume locking
- [x] Linear-Elastic/Mohr-Coulomb/Drucker-Prager/Neo-Hookean material
- [x] Multi-threads supported
- [ ] GPU / Multi-GPU
- [x] Backend-agnostic (KernelAbstractions.jl)
- [x] Visualization (Paraview)

## Citation ❤
If you use MaterialPointSolver.jl in your research, please consider to cite this paper:

```bib
@article{zubov2021neuralpde,
  title={NeuralPDE: Automating Physics-Informed Neural Networks (PINNs) with Error Approximations},
  author={Zubov, Kirill and McCarthy, Zoe and Ma, Yingbo and Calisto, Francesco and Pagliarino, Valerio and Azeglio, Simone and Bottero, Luca and Luj{\'a}n, Emmanuel and Sulzer, Valentin and Bharambe, Ashutosh and others},
  journal={arXiv preprint arXiv:2107.09443},
  year={2021}
}
```


## Acknowledgement 👍

This project is sponserd by [Risk Group | Université de Lausanne](https://wp.unil.ch/risk/) and [China Scholarship Council [中国国家留学基金管理委员会]](https://www.csc.edu.cn/).

Many thanks to **Prof. Dr. Michel Jaboyedoff**[^1], **Prof. Dr. Yury podladchikov**[^1][^2], **Dr. Marc-Henri Derron**[^1], **Prof. Dr. Gang Mei**[^3] and **Dr. Wyser Emmanuel**[^1] for their help. ❤

<br>

[^1]:
    Institute of Earth Sciences, University of Lausanne, 1015 Lausanne, Switzerland
[^2]:
    Swiss Geocomputing Center, University of Lausanne, 1015 Lausanne, Switzerland
[^3]:
    School of Engineering and Technology, China University of Geosciences (Beijing), 100083, Beijing, China