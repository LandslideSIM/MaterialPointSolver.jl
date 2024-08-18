# One-phase Single-point I/O Counts

- [One-phase Single-point I/O Counts](#one-phase-single-point-io-counts)
  - [Variables in the simulation](#variables-in-the-simulation)
  - [I/O counts in different procedure](#io-counts-in-different-procedure)
    - [N\_io on the background grid](#n_io-on-the-background-grid)
    - [N\_io on the particles](#n_io-on-the-particles)
  - [Summary](#summary)

## Variables in the simulation

|     var name      | value |
| :---------------: | :---: |
|   grid_node_num   | `Nn`  |
|   particle_num    | `Np`  |
|     cell_num      | `Ne`  |
|   node_in_cell    | `NIC` |
| degree_of_freedom | `DoF` |

## I/O counts in different procedure

### N_io on the background grid

|  grid   |       count        | category |         note         |
| :-----: | :----------------: | :------: | :------------------: |
|   Ms    |        `Nn`        | unknown  |         mass         |
|   Ps    |      `Nn*DoF`      | unknown  |       momentum       |
|  fext   |      `Nn*DoF`      | unknown  |    external force    |
|  fint   |      `Nn*DoF`      | unknown  |    internal force    |
|  ftol   |      `Nn*DoF`      | unknown  |     total force      |
|  fdamp  |      `Nn*DoF`      | unknown  |    damping force     |
|  Vs_T   |      `Nn*DoF`      | unknown  | velocity tmp (MUSL)  |
|   Vs    |      `Nn*DoF`      | unknown  |       velocity       |
|  Δd_s   |      `Nn*DoF`      | unknown  |     displacement     |
|   vol   |        `Ne`        | unknown  |        volume        |
|   σm    |        `Ne`        | unknown  |     mean stress      |
|   pos   |      `Nn*DoF`      |  known   |     coordinates      |
|   c2n   |      `Ne*NIC`      |  known   | cell to node (uGIMP) |
| D total | `36*Nn+Ne*(4+NIC)` |    -     |          2D          |
| D total | `53*Nn+Ne*(4+NIC)` |    -     |          3D          |

### N_io on the particles

| particle |       count        | category |                       note                        |
| :------: | :----------------: | :------: | :-----------------------------------------------: |
|    Ms    |        `Np`        | unknown  |                       mass                        |
|   vol    |        `Np`        | unknown  |                      volume                       |
|    ρs    |        `Np`        | unknown  |                      density                      |
|    Ps    |      `Np*DoF`      | unknown  |                     momentum                      |
|    Vs    |      `Np*DoF`      | unknown  |                     velocity                      |
|    Ni    |      `Np*NIC`      | unknown  |              basis function (uGIMP)               |
|    ∂N    |    `Np*NIC*DoF`    | unknown  |                  ∂ basis (uGIMP)                  |
|    Δd    |    `Np*NIC*DoF`    | unknown  |                   displacement                    |
|    F     |       `Np*i`       | unknown  |    deformation matrix <br> 2D: `i=4` 3D: `i=9`    |
|    ∂F    |       `Np*i`       | unknown  |            ∂F <br> 2D: `i=4` 3D: `i=9`            |
|   σij    |       `Np*i`       | unknown  |      stress tensor <br> 2D: `i=4` 3D: `i=6`       |
|  ϵij_s   |       `Np*i`       | unknown  |      strain tensor <br> 2D: `i=4` 3D: `i=6`       |
|  Δϵij_s  |       `Np*i`       | unknown  |      Δstrain tensor <br> 2D: `i=4` 3D: `i=6`      |
|   sij    |       `Np*i`       | unknown  | deviatoric stress tensor <br> 2D: `i=4` 3D: `i=6` |
|   spin   |       `Np*i`       | unknown  |       spin tensor <br> 2D: `i=1` 3D: `i=3`        |
|    σm    |        `Np`        | unknown  |                    mean stress                    |
|   epII   |        `Np`        | unknown  |             equivalent plastic strain             |
|    J     |        `Np`        | unknown  |          determinant of jacobian matrix           |
|   epK    |        `Np`        | unknown  |                 volumetric strain                 |
|   pos    |      `Np*DoF`      | unknown  |                    coordinates                    |
|   p2n    |      `Np*NIC`      | unknown  |             particle to node (uGIMP)              |
|   p2c    |        `Np`        | unknown  |             particle to cell (uGIMP)              |
|    E     |        `Np`        |  known   |                  elastic modulus                  |
|    ψ     |        `Np`        |  known   |                  dilation angle                   |
|    ϕ     |        `Np`        |  known   |                  friction angle                   |
|    c     |        `Np`        |  known   |                 cohesion strength                 |
|    G     |        `Np`        |  known   |                   shear modulus                   |
|    ν     |        `Np`        |  known   |                   possion ratio                   |
|    K     |        `Np`        |  known   |                   bulk modulus                    |
|    σt    |        `Np`        |  known   |                 tensile strength                  |
| D total  | `86*Np+12*Np*NIC`  |    -     |                        2D                         |
| D total  | `132*Np+16*Np*NIC` |    -     |                        3D                         |

## Summary

| dimension |                total                |            note             |
| :-------: | :---------------------------------: | :-------------------------: |
|    2D     | `36*Nn+Ne*(4+NIC)+86*Np+12*Np*NIC`  | `Aeff=total*precision*1e-9` |
|    3D     | `53*Nn+Ne*(4+NIC)+132*Np+16*Np*NIC` | `Aeff=total*precision*1e-9` |
